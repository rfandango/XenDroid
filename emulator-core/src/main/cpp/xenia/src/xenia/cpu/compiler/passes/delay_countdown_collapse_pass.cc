/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/compiler/passes/delay_countdown_collapse_pass.h"

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/ppc/ppc_context.h"

DEFINE_bool(
    collapse_memory_delay_spins, true,
    "Collapse guest delay-countdown self-loops whose loop counter lives in "
    "memory (a stack slot) instead of CTR - `while (--*slot != 0) db16cyc...` - "
    "into one body pass, the exact exit state (slot = 0, flags as if the loop "
    "ran out), and one adaptive backoff. The memory-counter sibling of "
    "collapse_ctr_spin_loops: the guest compiler sometimes spills the delay "
    "counter instead of using mtctr. Requires a delay_execution sled in the "
    "body as proof of delay intent, a thread-private stack counter, and every "
    "exit-state value provably computable, else the loop is left alone. "
    "Detection is default-deny and rare in practice; every accepted collapse "
    "is logged. Disable for a title that depends on the elided wall time.",
    "CPU");

DEFINE_bool(
    log_delay_collapse_rejects, false,
    "Log every candidate delay-countdown self-loop that "
    "collapse_memory_delay_spins rejects, with the guest address and the "
    "failing predicate. Diagnostic aid; noisy during precompilation. Accepted "
    "collapses are always logged regardless of this setting.",
    "CPU");

namespace xe {
namespace cpu {
namespace compiler {
namespace passes {

// TODO(benvanik): remove when enums redefined.
using namespace xe::cpu::hir;

using xe::cpu::hir::Block;
using xe::cpu::hir::Edge;
using xe::cpu::hir::HIRBuilder;
using xe::cpu::hir::Instr;
using xe::cpu::hir::Value;

namespace {

// Emitted host wait that stands in for the whole elided countdown.
constexpr uint32_t kBackoffUnits = 32;

// A lone db16cyc is an ordinary SMT priority hint; a real delay sled is a run
// of them. Requiring several keeps the pattern from matching an incidental
// hint that happens to sit in a counted loop.
constexpr size_t kMinDelaySled = 4;

uint64_t MaskToType(uint64_t v, TypeName type) {
  switch (type) {
    case INT8_TYPE:
      return v & 0xFFull;
    case INT16_TYPE:
      return v & 0xFFFFull;
    case INT32_TYPE:
      return v & 0xFFFFFFFFull;
    default:
      return v;
  }
}

uint64_t BswapToType(uint64_t v, TypeName type) {
  switch (type) {
    case INT16_TYPE:
      return __builtin_bswap16(uint16_t(v));
    case INT32_TYPE:
      return __builtin_bswap32(uint32_t(v));
    case INT64_TYPE:
      return __builtin_bswap64(v);
    default:  // INT8: byte swap is identity
      return MaskToType(v, type);
  }
}

// Concretely evaluates the value cone of `v` given that every load of the
// counter slot yields `c_pre` (loads sequenced before the decrement store) or
// `c_post` (loads after it). Leaves that are not constants or mapped counter
// loads fail the evaluation - callers treat failure as "cannot prove, reject".
// Only the op set a compiler-generated countdown actually uses is modelled.
bool EvalCounterCone(Value* v, uint64_t c_pre, uint64_t c_post,
                     const std::vector<std::pair<Instr*, bool>>& counter_loads,
                     int depth, uint64_t* out) {
  if (!v) {
    return false;
  }
  if (v->IsConstant()) {
    *out = MaskToType(v->constant.u64, v->type);
    return true;
  }
  if (depth > 16 || !v->def) {
    return false;
  }
  Instr* d = v->def;
  for (const auto& cl : counter_loads) {
    if (cl.first == d) {
      *out = MaskToType(cl.second ? c_post : c_pre, v->type);
      return true;
    }
  }
  const OpcodeInfo* op = d->opcode;
  uint64_t a = 0, b = 0;
  if (op == &OPCODE_ZERO_EXTEND_info || op == &OPCODE_TRUNCATE_info) {
    if (!EvalCounterCone(d->src1.value, c_pre, c_post, counter_loads, depth + 1,
                         &a)) {
      return false;
    }
    *out = MaskToType(a, v->type);
    return true;
  }
  if (op == &OPCODE_BYTE_SWAP_info) {
    if (!EvalCounterCone(d->src1.value, c_pre, c_post, counter_loads, depth + 1,
                         &a)) {
      return false;
    }
    *out = BswapToType(a, v->type);
    return true;
  }
  if (op == &OPCODE_ADD_info || op == &OPCODE_SUB_info) {
    if (!EvalCounterCone(d->src1.value, c_pre, c_post, counter_loads, depth + 1,
                         &a) ||
        !EvalCounterCone(d->src2.value, c_pre, c_post, counter_loads, depth + 1,
                         &b)) {
      return false;
    }
    *out = MaskToType(op == &OPCODE_ADD_info ? a + b : a - b, v->type);
    return true;
  }
  if (op == &OPCODE_COMPARE_EQ_info || op == &OPCODE_COMPARE_NE_info) {
    if (!EvalCounterCone(d->src1.value, c_pre, c_post, counter_loads, depth + 1,
                         &a) ||
        !EvalCounterCone(d->src2.value, c_pre, c_post, counter_loads, depth + 1,
                         &b)) {
      return false;
    }
    *out = (op == &OPCODE_COMPARE_EQ_info) ? (a == b) : (a != b);
    return true;
  }
  return false;
}

// True if the value cone of `v` reaches any counter-slot load (so its value
// varies with the iteration). Non-counter leaves (constants, context loads,
// anything else) are iteration-invariant here because the loop body may not
// store to context slots it also reads (checked by the caller).
bool ConeTouchesCounter(Value* v,
                        const std::vector<std::pair<Instr*, bool>>& counter_loads,
                        int depth) {
  if (!v || v->IsConstant() || !v->def) {
    return false;
  }
  if (depth > 16) {
    // Fail SAFE: an over-deep cone is treated as counter-derived, which routes
    // it into EvalCounterCone whose own depth cutoff then rejects the collapse
    // (a false "invariant" here would silently skip an exit-state fixup).
    return true;
  }
  for (const auto& cl : counter_loads) {
    if (cl.first == v->def) {
      return true;
    }
  }
  // Only recurse into operands that really are Values: srcN.value is a union
  // shared with immediates/offsets/labels, and srcN_use is non-null exactly
  // when the operand is a tracked Value.
  Instr* d = v->def;
  return (d->src1_use &&
          ConeTouchesCounter(d->src1.value, counter_loads, depth + 1)) ||
         (d->src2_use &&
          ConeTouchesCounter(d->src2.value, counter_loads, depth + 1)) ||
         (d->src3_use &&
          ConeTouchesCounter(d->src3.value, counter_loads, depth + 1));
}

Value* MakeTypedConstant(HIRBuilder* builder, TypeName type, uint64_t value) {
  switch (type) {
    case INT8_TYPE:
      return builder->LoadConstantUint8(uint8_t(value));
    case INT16_TYPE:
      return builder->LoadConstantUint16(uint16_t(value));
    case INT32_TYPE:
      return builder->LoadConstantUint32(uint32_t(value));
    case INT64_TYPE:
      return builder->LoadConstantUint64(value);
    default:
      return nullptr;
  }
}

}  // namespace

DelayCountdownCollapsePass::DelayCountdownCollapsePass() : CompilerPass() {}

DelayCountdownCollapsePass::~DelayCountdownCollapsePass() {}

bool DelayCountdownCollapsePass::Run(HIRBuilder* builder) {
  if (!cvars::collapse_memory_delay_spins) {
    return true;
  }
  for (auto block = builder->first_block(); block; block = block->next) {
    // Transforms insert instructions via the builder, whose AppendInstr may
    // lazily create one empty block at the list tail (harmless: no labels,
    // edges, or instrs survive there after MoveBefore); the walk itself only
    // ever appends past the current position, so iteration stays valid.
    TryCollapseDelayCountdown(builder, block);
  }
  return true;
}

// Matches `do { db16cyc...; *slot = *slot - 1; } while (*slot != 0)` on a
// thread-private slot: reads no external state, trip count fixed at entry, so
// the exit state is compile-time computable. Runs the body once, replaces the
// loop-back branch with one adaptive backoff and forces the exit state.
bool DelayCountdownCollapsePass::TryCollapseDelayCountdown(HIRBuilder* builder,
                                                           Block* block) {
  bool has_self_edge = false;
  for (Edge* e = block->outgoing_edge_head; e; e = e->outgoing_next) {
    if (e->dest == block) {
      has_self_edge = true;
      break;
    }
  }
  if (!has_self_edge) {
    return false;
  }

  uint64_t guest_address = 0;
  for (Instr* instr = block->instr_head; instr; instr = instr->next) {
    if (instr->opcode == &OPCODE_SOURCE_OFFSET_info) {
      guest_address = instr->src1.offset;
      break;
    }
  }
  auto reject = [&](const char* reason) {
    if (cvars::log_delay_collapse_rejects) {
      XELOGI("DelayCollapse: self-loop at guest {:08X} rejected: {}",
             guest_address, reason);
    }
    return false;
  };

  // The conditional self-branch. The block may continue with post-loop code
  // after it - mid-block branches are normal in this HIR.
  Instr* loop_branch = nullptr;
  for (Instr* instr = block->instr_head; instr; instr = instr->next) {
    Label* target = nullptr;
    if (instr->opcode == &OPCODE_BRANCH_info) {
      target = instr->src1.label;
    } else if (instr->opcode == &OPCODE_BRANCH_TRUE_info ||
               instr->opcode == &OPCODE_BRANCH_FALSE_info) {
      target = instr->src2.label;
    } else {
      continue;
    }
    if (!target || target->block != block) {
      continue;
    }
    if (instr->opcode == &OPCODE_BRANCH_info || loop_branch) {
      return false;  // unconditional self-branch / multiple: not this shape
    }
    loop_branch = instr;
  }
  if (!loop_branch) {
    return false;
  }
  if (!loop_branch->next) {
    return reject("loop-back branch terminates the block");
  }

  // Walk the loop body classifying every real op. Default-deny: anything not
  // explicitly expected disqualifies the loop.
  std::vector<Instr*> delays;
  std::vector<std::pair<Instr*, bool>> counter_loads;  // (load, after-store)
  std::vector<Instr*> ctx_stores;
  std::vector<std::pair<uint64_t, uint32_t>> loaded_ctx, stored_ctx;
  Instr* counter_store = nullptr;
  for (Instr* instr = block->instr_head; instr != loop_branch;
       instr = instr->next) {
    if (instr->IsFake()) {
      continue;
    }
    const OpcodeInfo* op = instr->opcode;
    if (op == &OPCODE_DELAY_EXECUTION_info) {
      delays.push_back(instr);
      continue;
    }
    if (op == &OPCODE_CHECK_PREEMPT_info) {
      // A safepoint marker with no data effect. PreemptCheckInjectionPass runs
      // before this pass and puts one at the head of every loop it finds, so
      // rejecting it here would disqualify every candidate whenever the
      // cooperative scheduler is compiled in. Collapsing the loop removes the
      // need for the safepoint; the one surviving body pass keeps its check.
      continue;
    }
    if (op == &OPCODE_LOAD_OFFSET_info) {
      counter_loads.emplace_back(instr, counter_store != nullptr);
      continue;
    }
    if (op == &OPCODE_STORE_OFFSET_info) {
      if (counter_store) {
        return reject("multiple memory stores");
      }
      counter_store = instr;
      continue;
    }
    if (op == &OPCODE_LOAD_CONTEXT_info) {
      loaded_ctx.emplace_back(instr->src1.offset,
                              uint32_t(GetTypeSize(instr->dest->type)));
      continue;
    }
    if (op == &OPCODE_STORE_CONTEXT_info) {
      ctx_stores.push_back(instr);
      stored_ctx.emplace_back(instr->src1.offset,
                              uint32_t(GetTypeSize(instr->src2.value->type)));
      continue;
    }
    if (op == &OPCODE_ZERO_EXTEND_info || op == &OPCODE_TRUNCATE_info ||
        op == &OPCODE_SIGN_EXTEND_info || op == &OPCODE_ADD_info ||
        op == &OPCODE_SUB_info || op == &OPCODE_COMPARE_EQ_info ||
        op == &OPCODE_COMPARE_NE_info || op == &OPCODE_ASSIGN_info ||
        op == &OPCODE_BYTE_SWAP_info) {
      // Pure value ops of the countdown idiom. BYTE_SWAP appears explicitly at
      // this pass position (MemorySequenceCombination fuses it into the
      // load/store only later in the pipeline) and is modelled by the
      // evaluator.
      continue;
    }
    if (cvars::log_delay_collapse_rejects) {
      XELOGI(
          "DelayCollapse: self-loop at guest {:08X} rejected: unexpected "
          "opcode in delay-loop body: {}",
          guest_address, GetOpcodeName(op));
    }
    return false;
  }
  if (delays.size() < kMinDelaySled) {
    return reject("delay sled too short - not a delay loop");
  }
  if (!counter_store || counter_loads.empty()) {
    return reject("no memory countdown");
  }

  // Loop-invariance of context state: no slot both read and written (the base
  // pointer for the counter is typically re-loaded from context each
  // iteration; a store to that slot would change the counter address).
  for (const auto& s : stored_ctx) {
    for (const auto& l : loaded_ctx) {
      if (s.first < l.first + l.second && l.first < s.first + s.second) {
        return reject("context slot both read and written");
      }
    }
  }

  // All memory accesses must hit ONE slot: same base SSA value, equal constant
  // offsets, same access size.
  Value* base = counter_store->src1.value;
  Value* store_off = counter_store->src2.value;
  if (!store_off || !store_off->IsConstant()) {
    return reject("counter offset not constant");
  }
  // Thread-privacy guard: the slot must be a guest stack local (base from
  // r1/SP). CTR is architecturally private, but a memory word could be shared -
  // a heap counter doubling as a cross-thread progress signal would break a
  // reader of the intermediate values this elides.
  constexpr uint64_t kSpOffset = offsetof(xe::cpu::ppc::PPCContext, r) + 1 * 8;
  if (!base->def || base->def->opcode != &OPCODE_LOAD_CONTEXT_info ||
      base->def->src1.offset != kSpOffset) {
    return reject("counter base is not the guest stack pointer");
  }
  const uint64_t slot_off = MaskToType(store_off->constant.u64, store_off->type);
  const TypeName slot_type = counter_store->src3.value->type;
  // INT8/16/32 only: for these the evaluator's masked arithmetic is exact for
  // every entry value including wrap. An INT64 slot whose compare was narrowed
  // (trunc) would diverge for entry values >= 2^32.
  if (slot_type != INT8_TYPE && slot_type != INT16_TYPE &&
      slot_type != INT32_TYPE) {
    return reject("counter slot is not a 32-bit-or-narrower integer");
  }
  for (const auto& cl : counter_loads) {
    Value* loff = cl.first->src2.value;
    if (cl.first->src1.value != base || !loff || !loff->IsConstant() ||
        MaskToType(loff->constant.u64, loff->type) != slot_off ||
        cl.first->dest->type != slot_type) {
      return reject("memory access outside the counter slot");
    }
  }

  // Body SSA values must not escape the loop segment: a use after loop_branch
  // (in the post-loop code of this block) would observe the single-pass value
  // where the original observed the exit-iteration value. The frontend's
  // context-round-trip convention makes this shape unlikely, but do not depend
  // on it.
  {
    std::vector<const Instr*> body_instrs;
    for (Instr* instr = block->instr_head; instr != loop_branch;
         instr = instr->next) {
      body_instrs.push_back(instr);
    }
    auto in_body = [&](const Instr* i) {
      if (i == loop_branch) {
        return true;
      }
      for (const Instr* b : body_instrs) {
        if (b == i) {
          return true;
        }
      }
      return false;
    };
    for (const Instr* bi : body_instrs) {
      if (!bi->dest) {
        continue;
      }
      for (Value::Use* use = bi->dest->use_head; use; use = use->next) {
        if (!in_body(use->instr)) {
          return reject("body value used outside the loop segment");
        }
      }
    }
  }

  // The exit condition must be a single top-level compare over a cone with no
  // nested compares and no add/sub mixing two counter-dependent operands. With
  // a plain +/-1 recurrence this makes the probe set below a sound
  // certificate (an even-coefficient combination could have a second root the
  // probes cannot see).
  {
    Value* cond_root = loop_branch->src1.value;
    if (!cond_root || !cond_root->def ||
        (cond_root->def->opcode != &OPCODE_COMPARE_EQ_info &&
         cond_root->def->opcode != &OPCODE_COMPARE_NE_info)) {
      return reject("exit condition is not a plain compare");
    }
    std::function<bool(Value*, int)> cone_ok = [&](Value* v,
                                                   int depth) -> bool {
      if (!v || v->IsConstant() || !v->def || depth > 16) {
        return true;  // leaves are fine; depth overflow rejected by eval later
      }
      const OpcodeInfo* op = v->def->opcode;
      if (op == &OPCODE_COMPARE_EQ_info || op == &OPCODE_COMPARE_NE_info) {
        return false;  // nested compare inside the cone
      }
      if (op == &OPCODE_ADD_info || op == &OPCODE_SUB_info) {
        if (ConeTouchesCounter(v->def->src1.value, counter_loads, 0) &&
            ConeTouchesCounter(v->def->src2.value, counter_loads, 0)) {
          return false;  // both operands counter-dependent: coefficient != +-1
        }
      }
      return (!v->def->src1_use || cone_ok(v->def->src1.value, depth + 1)) &&
             (!v->def->src2_use || cone_ok(v->def->src2.value, depth + 1)) &&
             (!v->def->src3_use || cone_ok(v->def->src3.value, depth + 1));
    };
    if (!cone_ok(cond_root->def->src1.value, 0) ||
        !cone_ok(cond_root->def->src2.value, 0)) {
      return reject("exit-condition cone too complex to certify");
    }
  }

  // Structural proof by concrete probes: the stored value must be exactly
  // (counter - 1) and the loop must exit when it reaches 0. The counter may be
  // stored byte-swapped, so both representations are probed. bswap(0)==0, so
  // the forced slot value is 0 either way.
  const bool is_branch_true = loop_branch->opcode == &OPCODE_BRANCH_TRUE_info;
  auto probe_convention = [&](bool swapped) {
    auto raw = [&](uint64_t n) {
      return swapped ? BswapToType(n, slot_type) : MaskToType(n, slot_type);
    };
    uint64_t v = 0;
    for (uint64_t probe : {7ull, 100ull}) {
      if (!EvalCounterCone(counter_store->src3.value, raw(probe),
                           raw(probe - 1), counter_loads, 0, &v) ||
          v != raw(probe - 1)) {
        return false;
      }
    }
    auto loops_at = [&](uint64_t n, bool* out_loops) {
      uint64_t cond = 0;
      if (!EvalCounterCone(loop_branch->src1.value, raw(n), raw(n - 1),
                           counter_loads, 0, &cond)) {
        return false;
      }
      *out_loops = is_branch_true ? cond != 0 : cond == 0;
      return true;
    };
    bool loops = false;
    if (!loops_at(1, &loops) || loops) {
      return false;
    }
    for (uint64_t probe : {2ull, 3ull, 100ull}) {
      if (!loops_at(probe, &loops) || !loops) {
        return false;
      }
    }
    return true;
  };
  bool swapped_counter = false;
  if (!probe_convention(false)) {
    if (probe_convention(true)) {
      swapped_counter = true;
    } else {
      return reject("not a plain decrement-to-zero countdown");
    }
  }
  auto raw_final = [&](uint64_t n) {
    return swapped_counter ? BswapToType(n, slot_type)
                           : MaskToType(n, slot_type);
  };

  // Every counter-derived context store must be computable at the exit
  // iteration (entry counter 1, post-store 0). Invariant stores are correct
  // from the single body pass and need no fixup.
  std::vector<std::pair<Instr*, uint64_t>> forced_ctx;
  for (Instr* cs : ctx_stores) {
    Value* sv = cs->src2.value;
    if (!ConeTouchesCounter(sv, counter_loads, 0)) {
      continue;
    }
    uint64_t final_value = 0;
    if (!EvalCounterCone(sv, raw_final(1), raw_final(0), counter_loads, 0,
                         &final_value)) {
      return reject("counter-derived context store not computable at exit");
    }
    if (sv->type != INT8_TYPE && sv->type != INT16_TYPE &&
        sv->type != INT32_TYPE && sv->type != INT64_TYPE) {
      return reject("counter-derived store of a non-integer type");
    }
    // The forced fixup lands after the whole body; a LATER body store into an
    // overlapping context range would be re-clobbered by it, inverting their
    // original order.
    const uint64_t cs_off = cs->src1.offset;
    const uint32_t cs_size = uint32_t(GetTypeSize(sv->type));
    for (Instr* later = cs->next; later != loop_branch; later = later->next) {
      if (later->IsFake() || later->opcode != &OPCODE_STORE_CONTEXT_info) {
        continue;
      }
      const uint64_t lo = later->src1.offset;
      const uint32_t ls = uint32_t(GetTypeSize(later->src2.value->type));
      if (cs_off < lo + ls && lo < cs_off + cs_size) {
        return reject("counter-derived store overlaps a later context store");
      }
    }
    forced_ctx.emplace_back(cs, final_value);
  }

  // --- All checks passed: collapse. ---
  Instr* anchor = loop_branch->next;

  // One adaptive backoff replaces the whole countdown; the body has already
  // run once by the time it executes. The loop is gone, so this backoff runs
  // once and returns - it never parks a thread inside a live loop.
  loop_branch->Replace(&OPCODE_SPIN_BACKOFF_info, 0);
  loop_branch->src1.offset = kBackoffUnits;
  loop_branch->src2.value = nullptr;
  loop_branch->src3.value = nullptr;

  // Force the exact exit state: counter slot = 0 (with the original store's
  // flags, so byte-swap variants stay equivalent - swapped zero is zero), then
  // every counter-derived context slot at its exit value.
  Value* zero = MakeTypedConstant(builder, slot_type, 0);
  if (zero) {
    builder->StoreOffset(base, store_off, zero, counter_store->flags);
    builder->last_instr()->MoveBefore(anchor);
  }
  for (const auto& fc : forced_ctx) {
    Value* fv =
        MakeTypedConstant(builder, fc.first->src2.value->type, fc.second);
    if (fv) {
      builder->StoreContext(fc.first->src1.offset, fv);
      builder->last_instr()->MoveBefore(anchor);
    }
  }

  // The block no longer loops back.
  builder->RemoveEdge(block, block);
  // Always logged: this transform elides real wall time, so every accepted
  // collapse is meant to be audited against the caller having a real-clock
  // backstop before the cvar is enabled for a title.
  XELOGI(
      "DelayCollapse: collapsed memory-delay countdown at guest {:08X} "
      "(sled={} ctx_fixups={})",
      guest_address, delays.size(), forced_ctx.size());
  return true;
}

}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe

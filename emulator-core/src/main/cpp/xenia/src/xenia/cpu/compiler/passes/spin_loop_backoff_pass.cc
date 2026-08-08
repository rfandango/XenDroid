/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/compiler/passes/spin_loop_backoff_pass.h"

#include <algorithm>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/ppc/ppc_context.h"

DEFINE_bool(
    collapse_ctr_spin_loops, true,
    "Collapse constant-trip-count bdnz spin-backoff loops (the XDK spin-wait "
    "primitive: mtctr small-constant + a sled of priority-hint nops + bdnz) "
    "into a single bounded host backoff. Guest semantics are preserved (CTR "
    "reads 0 after the loop); only the per-iteration CTR round trip through "
    "guest context is eliminated. Disable to translate such loops literally.",
    "CPU");

DEFINE_bool(
    log_spin_loop_rejects, false,
    "Log every candidate bdnz self-loop that collapse_ctr_spin_loops rejects, "
    "with the guest address and the first failing predicate condition. "
    "Diagnostic aid; noisy during shader-storage precompilation.",
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

// Offset of the CTR slot in the guest context.
const uint64_t kCtrOffset = offsetof(xe::cpu::ppc::PPCContext, ctr);
// Trip counts above this are assumed to be timing-calibration loops and keep
// their real form.
const uint64_t kMaxTripCount = 128;
// Upper bound on emitted host backoff units (trip_count * sled_length,
// clamped).
const uint64_t kMaxBackoffUnits = 256;

}  // namespace

SpinLoopBackoffPass::SpinLoopBackoffPass() : CompilerPass() {}

SpinLoopBackoffPass::~SpinLoopBackoffPass() {}

bool SpinLoopBackoffPass::Run(HIRBuilder* builder) {
  if (!cvars::collapse_ctr_spin_loops) {
    return true;
  }
  for (auto block = builder->first_block(); block; block = block->next) {
    TryCollapseLoop(builder, block);
  }
  return true;
}

// Walks the predecessor block backwards looking for the store that defines
// CTR on entry to the loop. Only returns true if that store provably writes a
// compile-time INT64 constant and nothing after it could change CTR.
bool SpinLoopBackoffPass::FindConstantCtrStore(Block* pred, Block* loop_block,
                                               uint64_t* out_trip_count) {
  // The predecessor is itself an extended block, so the branch that enters
  // the loop can sit mid-block with the not-taken path's code after it. A CTR
  // store found after that branch never executes on the path entering the
  // loop - anchor the backward scan at the entry branch, not the block tail.
  Instr* entry_branch = nullptr;
  for (Instr* instr = pred->instr_head; instr; instr = instr->next) {
    Label* target = nullptr;
    if (instr->opcode == &OPCODE_BRANCH_info) {
      target = instr->src1.label;
    } else if (instr->opcode == &OPCODE_BRANCH_TRUE_info ||
               instr->opcode == &OPCODE_BRANCH_FALSE_info) {
      target = instr->src2.label;
    } else {
      continue;
    }
    if (target && target->block == loop_block) {
      if (entry_branch) {
        // Two branches into the loop from one predecessor: ambiguous path.
        return false;
      }
      entry_branch = instr;
    }
  }
  if (!entry_branch) {
    // The edge has no surviving branch (stale after constant folding); the
    // entry path - and with it the trip count - cannot be proven.
    return false;
  }
  for (Instr* instr = entry_branch->prev; instr; instr = instr->prev) {
    if (instr->opcode == &OPCODE_STORE_CONTEXT_info) {
      Value* stored = instr->src2.value;
      if (instr->src1.offset == kCtrOffset) {
        if (!stored->IsConstant() || stored->type != INT64_TYPE) {
          return false;
        }
        *out_trip_count = stored->AsUint64();
        return true;
      }
      // A wider store to a lower offset could alias the CTR slot. None of the
      // PPCContext layouts do this in practice, but stay conservative.
      if (instr->src1.offset < kCtrOffset + 8 &&
          instr->src1.offset + GetTypeSize(stored->type) > kCtrOffset) {
        return false;
      }
      continue;
    }
    // Calls and other volatile operations may write CTR behind our back
    // (callees run guest code against the same context). Any doubt means the
    // loop keeps its real form.
    if (instr->opcode->flags & (OPCODE_FLAG_BRANCH | OPCODE_FLAG_VOLATILE)) {
      return false;
    }
  }
  // Reached the head of the predecessor without finding a CTR store; the
  // trip count is not proven within this block.
  return false;
}

bool SpinLoopBackoffPass::TryCollapseLoop(HIRBuilder* builder, Block* block) {
  // The loop head must not be the entry block: the entry is reachable without
  // traversing any edge, so the trip count could not be proven from a
  // predecessor.
  if (block == builder->first_block()) {
    return false;
  }

  // Candidacy by CFG ground truth: the block loops back to itself. The edges
  // come from ControlFlowAnalysisPass and survive whatever the simplifiers do
  // to the branch instructions themselves.
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

  // From here on this IS a candidate self-loop; when diagnostics are enabled,
  // report exactly which predicate condition rejects it.
  uint64_t guest_address = 0;
  if (cvars::log_spin_loop_rejects) {
    for (Instr* instr = block->instr_head; instr; instr = instr->next) {
      if (instr->opcode == &OPCODE_SOURCE_OFFSET_info) {
        guest_address = instr->src1.offset;
        break;
      }
    }
  }
  auto reject = [&](const char* reason, const char* detail) {
    if (cvars::log_spin_loop_rejects) {
      XELOGI("SpinLoopBackoff: self-loop at guest {:08X} rejected: {}{}",
             guest_address, reason, detail ? detail : "");
    }
    return false;
  };

  // Locate the branch that creates the self edge. HIR blocks are extended -
  // delimited by labels, not by branches - so the loop-back conditional
  // usually sits mid-block, with the post-loop fall-through code following it
  // in the SAME block (nothing jumps to that code, so it never gets a label).
  Instr* branch_true = nullptr;
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
    if (instr->opcode == &OPCODE_BRANCH_info) {
      return reject("self-branch is unconditional (infinite loop)", nullptr);
    }
    if (instr->opcode == &OPCODE_BRANCH_FALSE_info) {
      return reject("self-branch is branch_false", nullptr);
    }
    if (branch_true) {
      return reject("multiple self-branches", nullptr);
    }
    branch_true = instr;
  }
  if (!branch_true) {
    return reject("self edge exists but no self-targeting branch found",
                  nullptr);
  }

  // Exactly two incoming edges: the self edge and one predecessor.
  Edge* in0 = block->incoming_edge_head;
  if (!in0 || !in0->incoming_next || in0->incoming_next->incoming_next) {
    return reject("incoming edge count != 2", nullptr);
  }
  Edge* in1 = in0->incoming_next;
  Block* pred = nullptr;
  if (in0->src == block && in1->src != block) {
    pred = in1->src;
  } else if (in1->src == block && in0->src != block) {
    pred = in0->src;
  } else {
    return reject("incoming edges are not self + one predecessor", nullptr);
  }

  // No constraint on outgoing edges: any edges beyond the self edge belong to
  // branches in the post-loop code, which the rewrite leaves untouched. The
  // loop portion itself is proven single-exit by the body whitelist below
  // (a branch before the loop-back is a non-whitelisted opcode).

  // Body whitelist scan: the loop body may contain only the CTR
  // decrement/compare/branch machinery, delay_execution (the db16cyc
  // priority-hint sled) and fake (nop/comment/source_offset/context_barrier)
  // instructions. Anything else - any load, store, call, or unexpected
  // operand - disqualifies the loop.
  Instr* load_ctr = nullptr;
  Instr* sub = nullptr;
  Instr* store_ctr = nullptr;
  Instr* truncate = nullptr;
  Instr* compare = nullptr;
  std::vector<Instr*> delays;
  for (Instr* instr = block->instr_head; instr != branch_true;
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
      // A safepoint marker with no data effect. PreemptCheckInjectionPass
      // puts one at the head of every loop, so rejecting it here silently
      // disabled this collapse whenever the cooperative scheduler was on.
      // Collapsing removes the loop and the safepoint with it.
      continue;
    }
    if (op == &OPCODE_LOAD_CONTEXT_info) {
      if (load_ctr || instr->src1.offset != kCtrOffset ||
          instr->dest->type != INT64_TYPE) {
        return reject("unexpected LOAD_CONTEXT shape", nullptr);
      }
      load_ctr = instr;
      continue;
    }
    if (op == &OPCODE_SUB_info) {
      if (sub || instr->flags != 0 || !load_ctr ||
          instr->src1.value != load_ctr->dest ||
          !instr->src2.value->IsConstant() ||
          instr->src2.value->type != INT64_TYPE ||
          instr->src2.value->AsUint64() != 1) {
        return reject("unexpected SUB shape", nullptr);
      }
      sub = instr;
      continue;
    }
    if (op == &OPCODE_STORE_CONTEXT_info) {
      if (store_ctr || instr->src1.offset != kCtrOffset || !sub ||
          instr->src2.value != sub->dest) {
        return reject("unexpected STORE_CONTEXT shape", nullptr);
      }
      store_ctr = instr;
      continue;
    }
    if (op == &OPCODE_TRUNCATE_info) {
      if (truncate || !sub || instr->src1.value != sub->dest ||
          instr->dest->type != INT32_TYPE) {
        return reject("unexpected TRUNCATE shape", nullptr);
      }
      truncate = instr;
      continue;
    }
    if (op == &OPCODE_COMPARE_NE_info) {
      if (compare || !truncate || instr->src1.value != truncate->dest ||
          !instr->src2.value->IsConstantZero()) {
        return reject("unexpected COMPARE_NE shape", nullptr);
      }
      compare = instr;
      continue;
    }
    return reject("non-whitelisted opcode in body: ", GetOpcodeName(op));
  }
  if (!load_ctr || !sub || !store_ctr || !truncate || !compare) {
    return reject("CTR decrement chain incomplete", nullptr);
  }
  if (branch_true->src1.value != compare->dest) {
    return reject("branch condition is not the CTR compare", nullptr);
  }

  // Every value the loop body defines must be consumed entirely within the
  // CTR chain itself. The post-loop code lives in this same extended block,
  // so a block-identity check is not enough here.
  const Instr* const consumers[] = {sub, store_ctr, truncate, compare,
                                    branch_true};
  Instr* const defs[] = {load_ctr, sub, truncate, compare};
  for (Instr* def : defs) {
    for (auto use = def->dest->use_head; use; use = use->next) {
      bool in_chain = false;
      for (const Instr* consumer : consumers) {
        if (use->instr == consumer) {
          in_chain = true;
          break;
        }
      }
      if (!in_chain) {
        return reject("loop-defined value used outside the chain", nullptr);
      }
    }
  }

  // The trip count must be a proven compile-time constant reaching the loop
  // via the predecessor, in [1, kMaxTripCount]. Dynamic or large counts (for
  // example timing-calibration loops) keep the real loop. A count of 0 would
  // wrap CTR and iterate 2^32 times; never collapse it.
  uint64_t trip_count = 0;
  if (!FindConstantCtrStore(pred, block, &trip_count)) {
    return reject("trip count not a proven constant in the predecessor",
                  nullptr);
  }
  if (trip_count == 0 || trip_count > kMaxTripCount) {
    return reject("trip count out of range", nullptr);
  }

  // --- All checks passed: rewrite the loop. ---
  uint64_t sled_length = std::max<uint64_t>(delays.size(), 1);
  uint64_t units = std::min(trip_count * sled_length, kMaxBackoffUnits);

  // bdnz exits this loop with CTR == 0; preserve that architectural effect by
  // re-pointing the existing CTR store at a constant zero.
  store_ctr->set_src2(builder->LoadConstantUint64(0));

  // Replace the loop-back conditional branch with the bounded host backoff.
  // It stays in place, so the post-loop code after it runs exactly once.
  branch_true->Replace(&OPCODE_SPIN_BACKOFF_info, 0);
  branch_true->src1.offset = units;
  branch_true->src2.value = nullptr;
  branch_true->src3.value = nullptr;

  // Drop the now-dead CTR round trip and the per-iteration delay sled.
  compare->UnlinkAndNOP();
  truncate->UnlinkAndNOP();
  sub->UnlinkAndNOP();
  load_ctr->UnlinkAndNOP();
  for (Instr* delay : delays) {
    delay->UnlinkAndNOP();
  }

  // The block no longer loops: remove the self edge. RemoveEdge re-flags the
  // sole remaining (predecessor) incoming edge as dominating. The trailing
  // unconditional branch to the fall-through successor is left untouched.
  builder->RemoveEdge(block, block);
  if (cvars::log_spin_loop_rejects) {
    XELOGI(
        "SpinLoopBackoff: collapsed self-loop at guest {:08X}: trips={} "
        "sled={} units={}",
        guest_address, trip_count, delays.size(), units);
  }
  return true;
}

}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe

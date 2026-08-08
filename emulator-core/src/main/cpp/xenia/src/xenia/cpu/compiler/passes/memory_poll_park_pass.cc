/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/compiler/passes/memory_poll_park_pass.h"

#include <unordered_set>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/hir/hir_builder.h"

DEFINE_bool(
    park_memory_poll_loops, true,
    "Give indefinite guest memory-poll loops (load, test, branch back - the "
    "shape GPU fence and frame waits take) the same adaptive spin-then-park "
    "backoff collapsed spin loops use, so a long wait stops burning a core. "
    "Nothing is removed or reordered: every iteration still runs, a young "
    "wait still spins cheap, only a wait that has proven long sleeps in "
    "bounded slices. Loops that read the clock, store, call or use atomics "
    "keep their real form.",
    "CPU");

DEFINE_bool(log_memory_poll_park, false,
            "Log every candidate memory-poll self-loop this pass rejects, "
            "with the guest address and the failing condition. Accepted "
            "loops are always logged.",
            "CPU");

namespace xe {
namespace cpu {
namespace compiler {
namespace passes {

using namespace xe::cpu::hir;

using xe::cpu::hir::Block;
using xe::cpu::hir::Edge;
using xe::cpu::hir::HIRBuilder;
using xe::cpu::hir::Instr;
using xe::cpu::hir::Value;

MemoryPollParkPass::MemoryPollParkPass() : CompilerPass() {}

MemoryPollParkPass::~MemoryPollParkPass() {}

bool MemoryPollParkPass::Run(HIRBuilder* builder) {
  if (!cvars::park_memory_poll_loops) {
    return true;
  }
  for (auto block = builder->first_block(); block; block = block->next) {
    TryInstrumentLoop(builder, block);
  }
  return true;
}

namespace {

// Pure data ops a poll body may massage the loaded value with. Each is
// time-invariant and side-effect free, so slowing the iteration cannot
// change what the loop computes.
bool IsAllowedDataOp(const OpcodeInfo* op) {
  return op == &OPCODE_ASSIGN_info || op == &OPCODE_CAST_info ||
         op == &OPCODE_ROTATE_LEFT_info || op == &OPCODE_SELECT_info ||
         op == &OPCODE_BYTE_SWAP_info || op == &OPCODE_TRUNCATE_info ||
         op == &OPCODE_ZERO_EXTEND_info || op == &OPCODE_SIGN_EXTEND_info ||
         op == &OPCODE_AND_info || op == &OPCODE_OR_info ||
         op == &OPCODE_XOR_info || op == &OPCODE_NOT_info ||
         op == &OPCODE_SHL_info || op == &OPCODE_SHR_info ||
         op == &OPCODE_SHA_info || op == &OPCODE_ADD_info ||
         op == &OPCODE_SUB_info || op == &OPCODE_COMPARE_EQ_info ||
         op == &OPCODE_COMPARE_NE_info || op == &OPCODE_COMPARE_SLT_info ||
         op == &OPCODE_COMPARE_SLE_info || op == &OPCODE_COMPARE_SGT_info ||
         op == &OPCODE_COMPARE_SGE_info || op == &OPCODE_COMPARE_ULT_info ||
         op == &OPCODE_COMPARE_ULE_info || op == &OPCODE_COMPARE_UGT_info ||
         op == &OPCODE_COMPARE_UGE_info;
}

}  // namespace

bool MemoryPollParkPass::TryInstrumentLoop(HIRBuilder* builder, Block* block) {
  // Candidacy by CFG ground truth, exactly like the collapse passes. Entry
  // blocks are eligible: real poll loops sit there (the guard that excluded
  // them elsewhere cost weeks once already).
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
  auto reject = [&](const char* reason, const char* detail) {
    if (cvars::log_memory_poll_park) {
      XELOGI("MemoryPollPark: self-loop at guest {:08X} rejected: {}{}",
             guest_address, reason, detail ? detail : "");
    }
    return false;
  };

  // The loop-back branch. Conditional either way around: polls exit on
  // "value changed" (branch_true on NE) or "flag still clear" (branch_false
  // on EQ) depending on how the guest compiler leaned.
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
    if (instr->opcode == &OPCODE_BRANCH_info) {
      return reject("self-branch is unconditional", nullptr);
    }
    if (loop_branch) {
      return reject("multiple self-branches", nullptr);
    }
    loop_branch = instr;
  }
  if (!loop_branch) {
    return reject("self edge exists but no self-targeting branch", nullptr);
  }

  // Body scan up to the loop-back branch. Default-deny: anything outside the
  // allowlist keeps the loop untouched. Stores to guest MEMORY, calls,
  // atomics and clock reads disqualify; context traffic is this thread's own
  // architectural state and merely repeats, so it is safe to slow down.
  bool saw_load = false;
  size_t delay_count = 0;
  std::unordered_set<const Value*> load_tainted;
  for (Instr* instr = block->instr_head; instr != loop_branch;
       instr = instr->next) {
    if (instr->IsFake()) {
      continue;
    }
    const OpcodeInfo* op = instr->opcode;
    if (op == &OPCODE_DELAY_EXECUTION_info) {
      ++delay_count;
      continue;
    }
    if (op == &OPCODE_CHECK_PREEMPT_info ||
        op == &OPCODE_MEMORY_BARRIER_info ||
        op == &OPCODE_LOAD_CONTEXT_info ||
        op == &OPCODE_STORE_CONTEXT_info) {
      continue;
    }
    if (op == &OPCODE_LOAD_info || op == &OPCODE_LOAD_OFFSET_info) {
      saw_load = true;
      load_tainted.insert(instr->dest);
      continue;
    }
    if (IsAllowedDataOp(op)) {
      // Propagate "derived from the polled load" through the data chain.
      if (instr->dest) {
        Value* srcs[3] = {nullptr, nullptr, nullptr};
        uint32_t sig = op->signature;
        if (GET_OPCODE_SIG_TYPE_SRC1(sig) == OPCODE_SIG_TYPE_V) {
          srcs[0] = instr->src1.value;
        }
        if (GET_OPCODE_SIG_TYPE_SRC2(sig) == OPCODE_SIG_TYPE_V) {
          srcs[1] = instr->src2.value;
        }
        if (GET_OPCODE_SIG_TYPE_SRC3(sig) == OPCODE_SIG_TYPE_V) {
          srcs[2] = instr->src3.value;
        }
        for (Value* src : srcs) {
          if (src && load_tainted.count(src)) {
            load_tainted.insert(instr->dest);
            break;
          }
        }
      }
      continue;
    }
    return reject("non-whitelisted opcode in body: ", GetOpcodeName(op));
  }
  // Two acceptable shapes: a memory poll must exit on the value it loads; a
  // delay sled (db16cyc run with a GPR counter) exists to burn time, so
  // parking in it is its purpose. A lone hint does not qualify.
  constexpr size_t kMinDelaySled = 4;
  bool is_delay_loop = delay_count >= kMinDelaySled;
  if (!saw_load && !is_delay_loop) {
    return reject("no memory load and no delay sled in body", nullptr);
  }
  if (saw_load && !is_delay_loop &&
      !load_tainted.count(loop_branch->src1.value)) {
    return reject("branch condition not derived from the polled load",
                  nullptr);
  }

  // Inject the adaptive backoff ahead of the loop-back branch, feeding the
  // same spin-then-park escalation collapsed spin loops already use. It runs
  // on the exiting iteration too, so a parked wait pays at most one extra
  // sleep quantum of exit latency - the same order as its polling cadence.
  Instr* backoff = builder->SpinBackoff(1);
  backoff->MoveBefore(loop_branch);

  XELOGI("MemoryPollPark: parked poll loop at guest {:08X}", guest_address);
  return true;
}

}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe

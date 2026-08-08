/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_COMPILER_PASSES_DELAY_COUNTDOWN_COLLAPSE_PASS_H_
#define XENIA_CPU_COMPILER_PASSES_DELAY_COUNTDOWN_COLLAPSE_PASS_H_

#include "xenia/cpu/compiler/compiler_pass.h"

namespace xe {
namespace cpu {
namespace compiler {
namespace passes {

// Collapses guest delay countdowns whose counter is spilled to memory instead
// of CTR, hiding them from SpinLoopBackoffPass's CTR collapse:
//
//   loop: db16cyc x8 ; lwz r11,0x50(r1) ; subi r11,r11,1
//         stw r11,0x50(r1) ; cmpwi r11,0 ; bne loop
//
// Runs the body once, replaces the loop-back branch with one bounded host
// backoff, then forces the exact exit state.
//
// Does NOT throttle live poll loops: an earlier version parked inside a
// still-live poll, so its producer never signalled and NFS Most Wanted
// deadlocked. Only a countdown waiting on nothing is safe - thread-private
// stack slot, body reads no other memory.
//
// Default-deny. Must run with ControlFlowAnalysisPass edges valid and before
// MemorySequenceCombinationPass, which fuses the byte swaps modelled here.
class DelayCountdownCollapsePass : public CompilerPass {
 public:
  DelayCountdownCollapsePass();
  ~DelayCountdownCollapsePass() override;

  bool Run(hir::HIRBuilder* builder) override;

 private:
  bool TryCollapseDelayCountdown(hir::HIRBuilder* builder, hir::Block* block);
};

}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_COMPILER_PASSES_DELAY_COUNTDOWN_COLLAPSE_PASS_H_

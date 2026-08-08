/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_COMPILER_PASSES_MEMORY_POLL_PARK_PASS_H_
#define XENIA_CPU_COMPILER_PASSES_MEMORY_POLL_PARK_PASS_H_

#include "xenia/cpu/compiler/compiler_pass.h"

namespace xe {
namespace cpu {
namespace compiler {
namespace passes {

// Injects an adaptive backoff into indefinite guest memory-poll self-loops
// (load, test, branch back) so a long wait parks instead of burning a core.
// Removes nothing - every iteration still runs, only slower once the wait has
// proven long. Loops using LOAD_CLOCK, stores, calls or atomics are skipped.
class MemoryPollParkPass : public CompilerPass {
 public:
  MemoryPollParkPass();
  ~MemoryPollParkPass() override;

  bool Run(hir::HIRBuilder* builder) override;

 private:
  bool TryInstrumentLoop(hir::HIRBuilder* builder, hir::Block* block);
};

}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_COMPILER_PASSES_MEMORY_POLL_PARK_PASS_H_

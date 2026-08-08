/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */
#include "xenia/base/platform.h"
#include "xenia/kernel/kernel_flags.h"

#if XE_PLATFORM_xendroid
DEFINE_bool(headless, true,
            "Don't display any UI, using defaults for prompts as needed.",
            "UI");
#else
DEFINE_bool(headless, false,
            "Don't display any UI, using defaults for prompts as needed.",
            "UI");
#endif
DEFINE_bool(log_high_frequency_kernel_calls, false,
            "Log kernel calls with the kHighFrequency tag.", "Logging");
DEFINE_bool(
    guest_scheduler, true,
    "Run guest threads as cooperative fibers driven by an in-kernel scheduler "
    "instead of mapping each to its own host OS thread. Requires a restart to "
    "take effect.",
    "Kernel");
UPDATE_from_bool(guest_scheduler, 2026, 8, 1, 13, false);
DEFINE_uint32(
    guest_scheduler_quantum_us, 1000,
    "Cooperative-scheduler timeslice in microseconds. A guest fiber running "
    "this long yields at its next JIT safepoint so co-resident fibers on the "
    "same dispatch thread make progress. Lower is fairer but switches more.",
    "Kernel");

# AI Agent Guidance

This document is intended for AI agents (LLMs, automation tools) to quickly orient themselves within the XenDroid codebase.

## Project DNA
XenDroid is a fork of Xenia (specifically Xenia Edge) adapted for Android ARM64. It follows a "frontend/backend" separation.

## Quick Map

| Area | Location |
|---|---|
| **Entry Point (App)** | `app/src/main/java/xendroid/compose/MainActivity.kt` |
| **Emulator Process** | `app/src/main/java/xendroid/compose/EmulatorHostActivity.kt` |
| **JNI Bridge (Java)** | `emulator-core/src/main/java/xendroid/emulator/Emulator.java` |
| **JNI Bridge (C++)** | `emulator-core/src/main/cpp/xendroid_emu.cpp` |
| **Build Logic** | `build.gradle`, `app/build.gradle`, `emulator-core/build.gradle` |
| **CMake Config** | `emulator-core/src/main/cpp/CMakeLists.txt` |
| **Game Specifics** | `GAME_COMPAT.md` |

## Key Symbols

- **`xendroid.compose.core.EmulatorSession`**: Manages the lifecycle of a single emulation run.
- **`xendroid.emulator.Emulator`**: The static JNI wrapper.
- **`xe::Emulator`**: The core native class (from upstream Xenia).
- **`EmulatorHostActivity`**: The Activity that owns the `:emu` process.

## Navigation Tips

1.  **Large Submodule**: `emulator-core/src/main/cpp/xenia-canary/` contains the massive upstream Xenia codebase. Most Android-specific native logic is *outside* this folder, in `emulator-core/src/main/cpp/`.
2.  **Config Flow**: Settings are passed from Kotlin `ConfigStore` -> `Emulator.java` -> C++ `xe::base::Config`.
3.  **Vulkan**: Look at `xe_android_util.cpp` and `vulkan_graphics_system.cpp` (upstream) for surface integration.

## Common Tasks for Agents

- **Adding a Setting**: Add to `ConfigStore.kt` (Kotlin) and ensure it's picked up in `xendroid_emu.cpp` via `CVar` mappings.
- **Debugging Crashes**: Start by checking `xe.log` and searching for `LOG_TAG "xendroid_native"`.
- **UI Tweaks**: The UI is entirely Jetpack Compose, located in `app/src/main/java/xendroid/compose/ui/`.

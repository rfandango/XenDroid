# XenDroid Architecture

XenDroid is structured to provide a high-performance Xbox 360 emulation experience on Android by combining a modern Kotlin/Compose frontend with a battle-tested C++ backend.

## Module Structure

The project is divided into two primary Gradle modules:

### 1. `:app` (Frontend)
- **Language**: Kotlin, Jetpack Compose.
- **Responsibility**: User interface, game library management, settings, input mapping, and process orchestration.
- **Key Components**:
    - `MainActivity`: The main entry point and UI host.
    - `ConfigStore`: Handles persistence of emulator and app settings.
    - `GamepadController`: Manages physical controller and touch overlay inputs.

### 2. `:emulator-core` (Backend)
- **Language**: C++, Java (JNI wrappers).
- **Responsibility**: Xbox 360 CPU/GPU emulation, audio, and system services.
- **Key Components**:
    - `xendroid_emu.cpp`: The main entry point for the native emulator instance.
    - `Emulator.java`: The JNI bridge allowing the Kotlin frontend to control the native core.
    - `xe_android_util.cpp`: Android-specific utilities (logging, asset management).

## Dual-Process Model

XenDroid utilizes a dual-process architecture for enhanced stability:
- **Main Process (`xendroid.compose`)**: Runs the UI and library.
- **Emulator Process (`:emu`)**: Hosted by `EmulatorHostActivity`. This process runs the actual emulation.

**Benefits:**
- If the emulator crashes due to a GPU hang or native exception, the main app remains running.
- Clean memory state for every game launch.
- Precise control over the Vulkan surface lifecycle.

## Emulation Flow

1. **Launch**: `MainActivity` starts `EmulatorHostActivity` with a `game_uri`.
2. **Setup**: `EmulatorHostActivity` initializes the `EmulatorRuntime` and prepares the Vulkan `SurfaceView`.
3. **Native Init**: The native `Emulator::Setup` is called via JNI, loading the global and per-game configs.
4. **Boot**: The guest executable (XEX/ISO) is loaded, and the JIT compiler begins translating PowerPC instructions to ARM64.
5. **Execution**: The GPU thread translates guest D3D12-like commands into Vulkan calls for the Adreno GPU.

## JNI Boundary

The boundary is kept thin to minimize overhead. Core commands (Start, Stop, Pause) and hardware events (Input, Surface changes) are the primary data types crossing the bridge.

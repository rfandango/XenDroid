# User Guide

This guide covers the setup and usage of XenDroid on your Android device.

## Requirements

### Hardware
- **SoC**: Snapdragon 8 Gen 2 or higher.
- **GPU**: Adreno 740 or higher.
- **RAM**: 8GB+ recommended.

### Software
- Android 13 or higher.
- **Custom Vulkan Drivers**: Highly recommended for performance and stability (e.g., Turnip drivers).

## Installation

1. Download the latest APK from the [official releases](https://github.com/rfandango/XenDroid/releases).
2. Install the APK on your device.
3. Grant "All Files Access" permission when prompted. This is required to load ROMs from your storage.

## Folder Structure

XenDroid uses the following directory for its data:
`/sdcard/Android/data/xendroid.compose/files/xendroid/`

- `config/`: Global and per-game configuration files.
- `patches/`: Game patches (`.patch.toml`).
- `content/`: Guest save data and installed content.
- `logs/`: Emulator logs (`xe.log`).

## Loading Games

Supported formats: `.iso`, `.zar`, `.xex`, and STFS containers.

1. Launch XenDroid and navigate to the **Library** tab.
2. Select the folder containing your Xbox 360 games.
3. Tap a game icon to launch.

## Input & Controllers

- **Physical Controllers**: Most Android-compatible gamepads are supported via HID.
- **Touch Overlay**: A customizable touch controller is available for handheld play.
- **Input Mapping**: Configure buttons in **Settings > Input**.

## Using Custom Drivers

1. Download a driver ZIP (e.g., from [Adreno-Tools-Drivers](https://github.com/StevenMXZ/Adreno-Tools-Drivers/releases)).
2. In XenDroid, go to **Settings > Vulkan > Custom Vulkan Driver**.
3. Select the downloaded ZIP file.
4. Restart the app for changes to take effect.

# Troubleshooting

Common issues and solutions for XenDroid.

## Performance & Crashes

### GPU Hangs / Device Loss
If a game crashes and you see "GPU Fault" or `VK_ERROR_DEVICE_LOST` in the logs:
- **Solution**: Try a different Vulkan driver. Turnip drivers are often more stable than proprietary Qualcomm drivers for emulation.
- **Workaround**: Check [GAME_COMPAT.md](../GAME_COMPAT.md) for game-specific settings like `depth_float24_convert_in_pixel_shader = true`.

### Black Screen at Boot
- **Reason**: Often caused by incompatible ROMs or missing dependencies.
- **Check**: Ensure the game is in a supported format and not corrupted.
- **Log**: Check `xe.log` for "Failed to load" messages.

### Audio Stuttering
- **Solution**: Set the audio backend to `aaudio` in settings (usually the default).
- **Performance**: Audio stuttering is often a symptom of the CPU/GPU being throttled or unable to keep up with emulation speed.

## Graphics Issues

### Missing Character Models
- **Fix**: For Team Ninja titles (like Ninja Gaiden), set `clear_memory_page_state = true` in the per-game config.

### Transparent Ground/Floors
- **Status**: Known issue in titles like Fable II on the Vulkan backend. Refer to [GAME_COMPAT.md](../GAME_COMPAT.md) for the latest research.

## Getting Help

### Extracting Logs
If you need to report a bug, providing a log is essential:
1. Enable **Detailed Logging** in Settings.
2. Reproduce the issue.
3. Locate `xe.log` in `/sdcard/Android/data/xendroid.compose/files/xendroid/`.
4. Share the log on the [Xenia Discord](https://discord.gg/xenia).

### Discord
The primary place for support is the `xenia-android` channel on the Xenia Discord server.

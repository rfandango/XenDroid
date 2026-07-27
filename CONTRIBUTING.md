# Contributing to XenDroid

Thank you for your interest in contributing to XenDroid! As a community-driven project, we welcome contributions that help improve the emulator's performance, compatibility, and user experience.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone --recurse-submodules https://github.com/YOUR_USERNAME/XenDroid.git
    ```
    *Note: The `--recurse-submodules` flag is critical as the emulator core depends on a specific fork of Xenia.*
3.  **Set up the build environment** by following the instructions in [BUILD.md](BUILD.md).

## Development Workflow

### Repository Structure
- `app/`: Kotlin/Jetpack Compose frontend.
- `emulator-core/`: Native C++ code and JNI bindings.
- `docs/`: Additional documentation and integration guides.

### Coding Standards
- **Kotlin**: Follow the [official Kotlin style guide](https://kotlinlang.org/docs/coding-conventions.html). Use Jetpack Compose best practices for UI.
- **C++**: Follow the existing Xenia coding style. Maintain compatibility with the upstream [Xenia Edge](https://github.com/has207/xenia-edge) where possible.
- **Resources**: Optimize assets (images, icons) for Android to keep the APK size manageable.

### Submitting Changes
1.  Create a feature branch for your changes.
2.  Ensure your code builds successfully (both `:app` and `:emulator-core`).
3.  Test your changes on a real device (Snapdragon 8 Gen 2 or higher recommended).
4.  Submit a Pull Request with a clear description of the changes and what they solve.

## Reporting Issues
As per our [Issue Policy](README.md#issue-policy), GitHub issues are currently limited to contributors. Please use the `xenia-android` channel on the Xenia Discord for bug reports and feedback.

## License
By contributing, you agree that your contributions will be licensed under the same terms as the project (check the LICENSE file for details).

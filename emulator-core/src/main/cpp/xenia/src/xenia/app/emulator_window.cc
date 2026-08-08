/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/emulator_window.h"

#include "third_party/imgui/imgui.h"
#include "third_party/stb/stb_image_write.h"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
#endif
#include "third_party/tomlplusplus/toml.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "xenia/base/assert.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/debugging.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/profiling.h"
#include "xenia/base/system.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/processor.h"
#include "xenia/emulator.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/xam/profile_manager.h"
#include "xenia/kernel/xam/xam_module.h"
#include "xenia/kernel/xam/xam_state.h"
#include "xenia/config.h"
#include "xenia/kernel/xconfig.h"
#include "xenia/ui/file_picker.h"
#include "xenia/ui/graphics_provider.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/ui/imgui_host_notification.h"
#include "xenia/ui/immediate_drawer.h"
#include "xenia/ui/presenter.h"
#include "xenia/ui/ui_event.h"
#include "xenia/ui/virtual_key.h"

#include "version.h"

DECLARE_bool(debug);

DECLARE_string(hid);

DECLARE_bool(guide_button);

DECLARE_bool(clear_memory_page_state);
DECLARE_bool(memexport_await_fences);

DECLARE_string(readback_resolve);


DEFINE_bool(fullscreen, false, "Whether to launch the emulator in fullscreen.",
            "Display");

DEFINE_bool(controller_hotkeys, false, "Hotkeys for Xbox and PS controllers.",
            "General");

DEFINE_string(
    postprocess_antialiasing, "",
    "Post-processing anti-aliasing effect to apply to the image output of the "
    "game.\n"
    "Using post-process anti-aliasing is heavily recommended when AMD "
    "FidelityFX Contrast Adaptive Sharpening or Super Resolution 1.0 is "
    "active.\n"
    "Use: [none, fxaa, fxaa_extreme]\n"
    " none (or any value not listed here):\n"
    "  Don't alter the original image.\n"
    " fxaa:\n"
    "  NVIDIA Fast Approximate Anti-Aliasing 3.11, normal quality preset (12)."
    "\n"
    " fxaa_extreme:\n"
    "  NVIDIA Fast Approximate Anti-Aliasing 3.11, extreme quality preset "
    "(39).",
    "Display");
DEFINE_string(
    postprocess_scaling_and_sharpening, "",
    "Post-processing effect to use for resampling and/or sharpening of the "
    "final display output.\n"
    "Use: [bilinear, cas, fsr]\n"
    " bilinear (or any value not listed here):\n"
    "  Original image at 1:1, simple bilinear stretching for resampling.\n"
    " cas:\n"
    "  Use AMD FidelityFX Contrast Adaptive Sharpening (CAS) for sharpening "
    "at scaling factors of up to 2x2, with additional bilinear stretching for "
    "larger factors.\n"
    " fsr:\n"
    "  Use AMD FidelityFX Super Resolution 1.0 (FSR) for highest-quality "
    "upscaling, or AMD FidelityFX Contrast Adaptive Sharpening for sharpening "
    "while not scaling or downsampling.\n"
    "  For scaling by factors of more than 2x2, multiple FSR passes are done.",
    "Display");
DEFINE_double(
    postprocess_ffx_cas_additional_sharpness,
    xe::ui::Presenter::GuestOutputPaintConfig::kCasAdditionalSharpnessDefault,
    "Additional sharpness for AMD FidelityFX Contrast Adaptive Sharpening "
    "(CAS), from 0 to 1.\n"
    "Higher is sharper.",
    "Display");
DEFINE_uint32(
    postprocess_ffx_fsr_max_upsampling_passes,
    xe::ui::Presenter::GuestOutputPaintConfig::kFsrMaxUpscalingPassesMax,
    "Maximum number of upsampling passes performed in AMD FidelityFX Super "
    "Resolution 1.0 (FSR) before falling back to bilinear stretching after the "
    "final pass.\n"
    "Each pass upscales only to up to 2x2 the previous size. If the game "
    "outputs a 1280x720 image, 1 pass will upscale it to up to 2560x1440 "
    "(below 4K), after 2 passes it will be upscaled to a maximum of 5120x2880 "
    "(including 3840x2160 for 4K), and so on.\n"
    "This variable has no effect if the display resolution isn't very high, "
    "but may be reduced on resolutions like 4K or 8K in case the performance "
    "impact of multiple FSR upsampling passes is too high, or if softer edges "
    "are desired.\n"
    "The default value is the maximum internally supported by Xenia.",
    "Display");
DEFINE_double(
    postprocess_ffx_fsr_sharpness_reduction,
    xe::ui::Presenter::GuestOutputPaintConfig::kFsrSharpnessReductionDefault,
    "Sharpness reduction for AMD FidelityFX Super Resolution 1.0 (FSR), in "
    "stops.\n"
    "Lower is sharper.",
    "Display");
// Dithering to 8bpc is enabled by default since the effect is minor, only
// effects what can't be shown normally by host displays, and nothing is changed
// by it for 8bpc source without resampling.
DEFINE_bool(
    postprocess_dither, true,
    "Dither the final image output from the internal precision to 8 bits per "
    "channel so gradients are smoother.\n"
    "On a 10bpc display, the lower 2 bits will still be kept, but noise will "
    "be added to them - disabling may be recommended for 10bpc, but it "
    "depends on the 10bpc displaying capabilities of the actual display used.",
    "Display");

DEFINE_int32(recent_titles_entry_amount, 10,
             "Allows user to define how many titles is saved in list of "
             "recently played titles.",
             "General");
DEFINE_bool(disable_doubleclick_fullscreen, false,
            "Allows the user to disable the behavior where a fast double-click "
            "causes Xenia to enter fullscreen mode.",
            "General");

namespace xe {
namespace app {

using xe::ui::FileDropEvent;
using xe::ui::KeyEvent;
using xe::ui::MenuItem;
using xe::ui::UIEvent;

using namespace xe::hid;
using namespace xe::gpu;

constexpr std::string_view kRecentlyPlayedTitlesFilename = "recent.toml";
constexpr std::string_view kBaseTitle = "Xenia-canary";

EmulatorWindow::EmulatorWindow(Emulator* emulator,
                               ui::WindowedAppContext& app_context,
                               uint32_t width, uint32_t height)
    : emulator_(emulator),
      app_context_(app_context),
      window_listener_(*this),
      window_(ui::Window::Create(app_context, kBaseTitle, width, height)),
      imgui_drawer_(
          std::make_unique<ui::ImGuiDrawer>(window_.get(), kZOrderImGui)) {
  base_title_ = std::string(kBaseTitle) +
#ifdef DEBUG
#if _NO_DEBUG_HEAP == 1
                " DEBUG"
#else
                " CHECKED"
#endif
#endif
                " ("
#ifdef XE_BUILD_IS_PR
                "PR#" XE_BUILD_PR_NUMBER " - "
#endif
                XE_BUILD_BRANCH "@" XE_BUILD_COMMIT_SHORT " on " XE_BUILD_DATE
                ")";

  LoadRecentlyLaunchedTitles();
}

std::unique_ptr<EmulatorWindow> EmulatorWindow::Create(
    Emulator* emulator, ui::WindowedAppContext& app_context, uint32_t width,
    uint32_t height) {
  assert_true(app_context.IsInUIThread());
  std::unique_ptr<EmulatorWindow> emulator_window(
      new EmulatorWindow(emulator, app_context, width, height));
  if (!emulator_window->Initialize()) {
    return nullptr;
  }
  return emulator_window;
}

EmulatorWindow::~EmulatorWindow() {
  // Notify the ImGui drawer that the immediate drawer is being destroyed.
  ShutdownGraphicsSystemPresenterPainting();
}

ui::Presenter* EmulatorWindow::GetGraphicsSystemPresenter() const {
  gpu::GraphicsSystem* graphics_system = emulator_->graphics_system();
  return graphics_system ? graphics_system->presenter() : nullptr;
}

void EmulatorWindow::SetupGraphicsSystemPresenterPainting() {
  ShutdownGraphicsSystemPresenterPainting();

  if (!window_) {
    return;
  }

  ui::Presenter* presenter = GetGraphicsSystemPresenter();
  if (!presenter) {
    return;
  }

#if XE_PLATFORM_xendroid
  XELOGI("SetupGraphicsSystemPresenterPainting: window={} presenter={}",
         static_cast<void*>(window_.get()), static_cast<void*>(presenter));
#endif

  ApplyDisplayConfigForCvars();

  window_->SetPresenter(presenter);

  immediate_drawer_ =
      emulator_->graphics_system()->provider()->CreateImmediateDrawer();
  if (immediate_drawer_) {
    immediate_drawer_->SetPresenter(presenter);
    imgui_drawer_->SetPresenterAndImmediateDrawer(presenter,
                                                  immediate_drawer_.get());
    Profiler::SetUserIO(kZOrderProfiler, window_.get(), presenter,
                        immediate_drawer_.get());
  }
}

void EmulatorWindow::ShutdownGraphicsSystemPresenterPainting() {
  Profiler::SetUserIO(kZOrderProfiler, window_.get(), nullptr, nullptr);
  imgui_drawer_->SetPresenterAndImmediateDrawer(nullptr, nullptr);
  immediate_drawer_.reset();
  if (window_) {
    window_->SetPresenter(nullptr);
  }
}

void EmulatorWindow::OnEmulatorInitialized() {
  emulator_initialized_ = true;
  window_->SetMainMenuEnabled(true);
  // When the user can see that the emulator isn't initializing anymore (the
  // menu isn't disabled), enter fullscreen if requested.
  if (cvars::fullscreen) {
    SetFullscreen(true);
  }

  if (IsUseNexusForGameBarEnabled()) {
    XELOGE(
        "Xbox Gamebar Enabled, using BACK button instead of GUIDE for "
        "controller hotkeys!!!");
  }

  // Create a thread to listen for controller hotkeys.
  if (cvars::controller_hotkeys) {
    Gamepad_HotKeys_Listener =
        threading::Thread::Create({}, [&] { GamepadHotKeys(); });
    Gamepad_HotKeys_Listener->set_name("Gamepad HotKeys Listener");
  }
}

void EmulatorWindow::EmulatorWindowListener::OnClosing(ui::UIEvent& e) {
  emulator_window_.app_context_.QuitFromUIThread();
}

void EmulatorWindow::EmulatorWindowListener::OnFileDrop(ui::FileDropEvent& e) {
  emulator_window_.FileDrop(e.filename());
}

void EmulatorWindow::EmulatorWindowListener::OnKeyDown(ui::KeyEvent& e) {
  emulator_window_.OnKeyDown(e);
}

void EmulatorWindow::EmulatorWindowListener::OnMouseDown(ui::MouseEvent& e) {
  emulator_window_.OnMouseDown(e);
}

void EmulatorWindow::EmulatorWindowListener::OnMouseUp(ui::MouseEvent& e) {
  emulator_window_.OnMouseUp(e);
}

void EmulatorWindow::EmulatorWindowListener::OnUsbDeviceChanged(
    bool is_arrival) {
  if (!emulator_window_.emulator()) {
    return;
  }

  if (!emulator_window_.emulator()->input_system()) {
    return;
  }

  auto* portal = emulator_window_.emulator()->input_system()->GetPortal();
  if (!portal) {
    return;
  }

  if (is_arrival) {
    portal->OnDeviceArrival();
  } else {
    portal->OnDeviceRemoval();
  }
}

// XenDroid: DisplayConfigGameConfigLoadCallback::PostGameConfigLoad removed —
// edge deleted Emulator::GameConfigLoadCallback. Intentionally not re-hooked on
// Android (no live per-game cvar overlay; display config is reapplied on every
// launch/relaunch via SetupGraphicsSystemPresenterPainting). See emulator_window.h.
//
// XenDroid: the display / content-install / XMP config dialog OnDraw
// implementations were dropped here -- the native ImGui config dialogs are
// vestigial on Android (Compose owns the real UI). The cvar helpers below
// (GetSwapPostEffectForCvarValue / GetGuestOutputPaintConfigForCvars /
// ApplyDisplayConfigForCvars / ...) are KEPT: they still feed
// SetupGraphicsSystemPresenterPainting().

bool EmulatorWindow::Initialize() {
  window_->AddListener(&window_listener_);
  window_->AddInputListener(&window_listener_, kZOrderEmulatorWindowInput);

  // Main menu.
  // FIXME: This code is really messy.
  auto main_menu = MenuItem::Create(MenuItem::Type::kNormal);
  auto file_menu = MenuItem::Create(MenuItem::Type::kPopup, "&File");
  auto recent_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Open Recent");
  auto zar_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Zar Package");
  FillRecentlyLaunchedTitlesMenu(recent_menu.get());
  {
    file_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Open...", "Ctrl+O",
                         std::bind(&EmulatorWindow::FileOpen, this)));
    file_menu->AddChild(std::move(recent_menu));
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    zar_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Create",
                         std::bind(&EmulatorWindow::CreateZarchive, this)));
    zar_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Extract",
                         std::bind(&EmulatorWindow::ExtractZarchive, this)));
    file_menu->AddChild(std::move(zar_menu));
#ifdef DEBUG
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    file_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Close",
                         std::bind(&EmulatorWindow::FileClose, this)));
#endif  // #ifdef DEBUG
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    file_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Show content directory...",
        std::bind(&EmulatorWindow::ShowContentDirectory, this)));
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    file_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "E&xit", "Alt+F4",
                         [this]() { window_->RequestClose(); }));
  }
  main_menu->AddChild(std::move(file_menu));

  // CPU menu.
  auto cpu_menu = MenuItem::Create(MenuItem::Type::kPopup, "&CPU");
  {
    cpu_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Reset Time Scalar", "Numpad *",
        std::bind(&EmulatorWindow::CpuTimeScalarReset, this)));
    cpu_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Time Scalar /= 2", "Numpad -",
        std::bind(&EmulatorWindow::CpuTimeScalarSetHalf, this)));
    cpu_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Time Scalar *= 2", "Numpad +",
        std::bind(&EmulatorWindow::CpuTimeScalarSetDouble, this)));
  }
  cpu_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    cpu_menu->AddChild(MenuItem::Create(MenuItem::Type::kString,
                                        "Toggle Profiler &Display", "F3",
                                        []() { Profiler::ToggleDisplay(); }));
    cpu_menu->AddChild(MenuItem::Create(MenuItem::Type::kString,
                                        "&Pause/Resume Profiler", "`",
                                        []() { Profiler::TogglePause(); }));
  }
  cpu_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    cpu_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Break and Show Guest Debugger",
        "Pause/Break", std::bind(&EmulatorWindow::CpuBreakIntoDebugger, this)));
    cpu_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Break into Host Debugger",
        "Ctrl+Pause/Break",
        std::bind(&EmulatorWindow::CpuBreakIntoHostDebugger, this)));
  }
  main_menu->AddChild(std::move(cpu_menu));

  // GPU menu.
  auto gpu_menu = MenuItem::Create(MenuItem::Type::kPopup, "&GPU");
  {
    gpu_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Trace Frame", "F4",
                         std::bind(&EmulatorWindow::GpuTraceFrame, this)));
  }
  gpu_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    gpu_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Clear Runtime Caches", "F5",
                         std::bind(&EmulatorWindow::GpuClearCaches, this)));
  }
  main_menu->AddChild(std::move(gpu_menu));

  // Display menu.
  auto display_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Display");
  {
    display_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Fullscreen", "F11",
                         std::bind(&EmulatorWindow::ToggleFullscreen, this)));
    display_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Take Screenshot", "F12",
                         std::bind(&EmulatorWindow::TakeScreenshot, this)));
  }
  main_menu->AddChild(std::move(display_menu));

  // HID menu.
  auto hid_menu = MenuItem::Create(MenuItem::Type::kPopup, "&HID");
  {
    hid_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Toggle controller vibration", "",
        std::bind(&EmulatorWindow::ToggleControllerVibration, this)));
    hid_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Display controller hotkeys", "",
        std::bind(&EmulatorWindow::DisplayHotKeysConfig, this)));
  }
  main_menu->AddChild(std::move(hid_menu));

  // Help menu.
  auto help_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Help");
  {
    help_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "FA&Q...", "F1",
                         std::bind(&EmulatorWindow::ShowFAQ, this)));
    help_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    help_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Game &compatibility...",
                         std::bind(&EmulatorWindow::ShowCompatibility, this)));
    help_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    help_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Build commit on GitHub...", "F2",
        std::bind(&EmulatorWindow::ShowBuildCommit, this)));
    help_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Recent changes on GitHub...", []() {
          LaunchWebBrowser(
              "https://github.com/xenia-canary/xenia-canary/"
              "compare/" XE_BUILD_COMMIT "..." XE_BUILD_BRANCH);
        }));
    help_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    help_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&About...",
        []() { LaunchWebBrowser("https://xenia.jp/about/"); }));
  }
  main_menu->AddChild(std::move(help_menu));

  window_->SetMainMenu(std::move(main_menu));

  window_->SetMainMenuEnabled(false);

  UpdateTitle();

  if (!window_->Open()) {
    XELOGE("Failed to open the platform window");
    return false;
  }

  Profiler::SetUserIO(kZOrderProfiler, window_.get(), nullptr, nullptr);

  return true;
}

const char* EmulatorWindow::GetCvarValueForSwapPostEffect(
    gpu::CommandProcessor::SwapPostEffect effect) {
  switch (effect) {
    case gpu::CommandProcessor::SwapPostEffect::kFxaa:
      return "fxaa";
    case gpu::CommandProcessor::SwapPostEffect::kFxaaExtreme:
      return "fxaa_extreme";
    default:
      return "";
  }
}

gpu::CommandProcessor::SwapPostEffect
EmulatorWindow::GetSwapPostEffectForCvarValue(const std::string& cvar_value) {
  if (cvar_value == GetCvarValueForSwapPostEffect(
                        gpu::CommandProcessor::SwapPostEffect::kFxaa)) {
    return gpu::CommandProcessor::SwapPostEffect::kFxaa;
  }
  if (cvar_value == GetCvarValueForSwapPostEffect(
                        gpu::CommandProcessor::SwapPostEffect::kFxaaExtreme)) {
    return gpu::CommandProcessor::SwapPostEffect::kFxaaExtreme;
  }
  return gpu::CommandProcessor::SwapPostEffect::kNone;
}

const char* EmulatorWindow::GetCvarValueForGuestOutputPaintEffect(
    ui::Presenter::GuestOutputPaintConfig::Effect effect) {
  switch (effect) {
    case ui::Presenter::GuestOutputPaintConfig::Effect::kCas:
      return "cas";
    case ui::Presenter::GuestOutputPaintConfig::Effect::kFsr:
      return "fsr";
    default:
      return "";
  }
}

ui::Presenter::GuestOutputPaintConfig::Effect
EmulatorWindow::GetGuestOutputPaintEffectForCvarValue(
    const std::string& cvar_value) {
  if (cvar_value == GetCvarValueForGuestOutputPaintEffect(
                        ui::Presenter::GuestOutputPaintConfig::Effect::kCas)) {
    return ui::Presenter::GuestOutputPaintConfig::Effect::kCas;
  }
  if (cvar_value == GetCvarValueForGuestOutputPaintEffect(
                        ui::Presenter::GuestOutputPaintConfig::Effect::kFsr)) {
    return ui::Presenter::GuestOutputPaintConfig::Effect::kFsr;
  }
  return ui::Presenter::GuestOutputPaintConfig::Effect::kBilinear;
}

ui::Presenter::GuestOutputPaintConfig
EmulatorWindow::GetGuestOutputPaintConfigForCvars() {
  ui::Presenter::GuestOutputPaintConfig paint_config;
  paint_config.SetAllowOverscanCutoff(true);
  paint_config.SetEffect(GetGuestOutputPaintEffectForCvarValue(
      cvars::postprocess_scaling_and_sharpening));
  paint_config.SetCasAdditionalSharpness(
      float(cvars::postprocess_ffx_cas_additional_sharpness));
  paint_config.SetFsrMaxUpsamplingPasses(
      cvars::postprocess_ffx_fsr_max_upsampling_passes);
  paint_config.SetFsrSharpnessReduction(
      float(cvars::postprocess_ffx_fsr_sharpness_reduction));
  paint_config.SetDither(cvars::postprocess_dither);
  return paint_config;
}

void EmulatorWindow::ApplyDisplayConfigForCvars() {
  gpu::GraphicsSystem* graphics_system = emulator_->graphics_system();
  if (!graphics_system) {
    return;
  }

  gpu::CommandProcessor* command_processor =
      graphics_system->command_processor();
  if (command_processor) {
    command_processor->SetDesiredSwapPostEffect(
        GetSwapPostEffectForCvarValue(cvars::postprocess_antialiasing));
  }

  ui::Presenter* presenter = graphics_system->presenter();
  if (presenter) {
    presenter->SetGuestOutputPaintConfigFromUIThread(
        GetGuestOutputPaintConfigForCvars());
  }
}

void EmulatorWindow::OnKeyDown(ui::KeyEvent& e) {
  if (!emulator_initialized_) {
    return;
  }

  switch (e.virtual_key()) {
    case ui::VirtualKey::kO: {
      if (!e.is_ctrl_pressed()) {
        return;
      }
      FileOpen();
    } break;
    case ui::VirtualKey::kMultiply: {
      CpuTimeScalarReset();
    } break;
    case ui::VirtualKey::kSubtract: {
      CpuTimeScalarSetHalf();
    } break;
    case ui::VirtualKey::kAdd: {
      CpuTimeScalarSetDouble();
    } break;

    case ui::VirtualKey::kF3: {
      Profiler::ToggleDisplay();
    } break;

    case ui::VirtualKey::kF4: {
      GpuTraceFrame();
    } break;
    case ui::VirtualKey::kF5: {
      GpuClearCaches();
    } break;

    case ui::VirtualKey::kF11: {
      ToggleFullscreen();
    } break;
    case ui::VirtualKey::kF12: {
      TakeScreenshot();
    } break;

    case ui::VirtualKey::kEscape: {
      // Allow users to escape fullscreen (but not enter it).
      if (!window_->IsFullscreen()) {
        return;
      }
      SetFullscreen(false);
    } break;

#ifdef DEBUG
    case ui::VirtualKey::kF7: {
      // Save to file
      // TODO: Choose path based on user input, or from options
      // TODO: Spawn a new thread to do this.
      emulator()->SaveToFile("test.sav");
    } break;
    case ui::VirtualKey::kF8: {
      // Restore from file
      // TODO: Choose path from user
      // TODO: Spawn a new thread to do this.
      emulator()->RestoreFromFile("test.sav");
    } break;
#endif  // #ifdef DEBUG

    case ui::VirtualKey::kPause: {
      CpuBreakIntoDebugger();
    } break;
    case ui::VirtualKey::kCancel: {
      CpuBreakIntoHostDebugger();
    } break;

    case ui::VirtualKey::kF1: {
      ShowFAQ();
    } break;

    case ui::VirtualKey::kF2: {
      ShowBuildCommit();
    } break;

    case ui::VirtualKey::kF9: {
      RunPreviouslyPlayedTitle();
    } break;

    default:
      return;
  }

  e.set_handled(true);
}

void EmulatorWindow::OnMouseDown(const ui::MouseEvent& e) {
  if (imgui_drawer_->HasOpenDialogs()) {
    return;
  }

  if (e.button() == ui::MouseEvent::Button::kLeft) {
    ToggleFullscreenOnDoubleClick();
  }
}

void EmulatorWindow::OnMouseUp(const ui::MouseEvent& e) {
  last_mouse_up = steady_clock::now();
}

void EmulatorWindow::TakeScreenshot() {
  xe::ui::RawImage image;

  imgui_drawer_->EnableNotifications(false);

  if (!GetGraphicsSystemPresenter()->CaptureGuestOutput(image) ||
      GetGraphicsSystemPresenter() == nullptr) {
    XELOGE("Failed to capture guest output for screenshot");
    return;
  }

  imgui_drawer_->EnableNotifications(true);
  ExportScreenshot(image);
}

void EmulatorWindow::ExportScreenshot(const xe::ui::RawImage& image) {
  auto t = std::time(nullptr);

  // The format is: Year-Month-DayTHours-Minutes-Seconds based off ISO 8601
  std::string datetime =
      fmt::format("{:%Y-%m-%dT%H-%M-%S}", *std::localtime(&t));

  // Get the title id of the game because some titles contain characters that
  // cannot be used as a directory
  std::string title_id;
  if (emulator()->title_id()) {
    title_id = fmt::format("{:08X}", emulator()->title_id());
  } else {
    XELOGE("Failed to get the current title id");
    return;
  }

  // Find where xenia.exe or xenia_canary.exe is located and create a
  // screenshots folder
  auto screenshot_path =
      (xe::filesystem::GetExecutableFolder() / "screenshots" / title_id);

  if (!std::filesystem::exists(screenshot_path)) {
    std::filesystem::create_directories(screenshot_path);
  }

  std::string filename = fmt::format("{} - {}.png", title_id, datetime);
  SaveImage(screenshot_path / filename, image);

  const std::string notification_text =
      fmt::format("Screenshot saved: {}", filename);

  app_context_.CallInUIThread([&, notification_text]() {
    new xe::ui::HostNotificationWindow(imgui_drawer(), "Screenshot Created!",
                                       notification_text, 0);
  });
}

// Converts a RawImage into a PNG file
void EmulatorWindow::SaveImage(const std::filesystem::path& filepath,
                               const xe::ui::RawImage& image) {
  auto file = std::ofstream(filepath, std::ios::binary);
  if (!file.is_open()) {
    XELOGE("Failed to open file for writing: {}", filepath);
    return;
  }

  auto result = stbi_write_png_to_func(
      [](void* context, void* data, int size) {
        auto file = reinterpret_cast<std::ofstream*>(context);
        file->write(reinterpret_cast<const char*>(data), size);
      },
      &file, image.width, image.height, 4, image.data.data(),
      (int)image.stride);
  if (result == 0) {
    XELOGE("Failed to write PNG to file: {}", filepath);
    return;
  }
}

void EmulatorWindow::ToggleFullscreenOnDoubleClick() {
  if (cvars::disable_doubleclick_fullscreen) {
    return;
  }

  // this function tests if user has double clicked.
  // if double click was achieved the fullscreen gets toggled
  const auto now = steady_clock::now();  // current mouse event time
  constexpr int16_t mouse_down_max_threshold = 250;
  constexpr int16_t mouse_up_max_threshold = 250;
  constexpr int16_t mouse_up_down_max_delta = 100;
  // max delta to prevent 'chaining' of double clicks with next mouse events

  const auto last_mouse_down_delta = diff_in_ms(now, last_mouse_down);
  if (last_mouse_down_delta >= mouse_down_max_threshold) {
    last_mouse_down = now;
    return;
  }

  const auto last_mouse_up_delta = diff_in_ms(now, last_mouse_up);
  const auto mouse_event_deltas = diff_in_ms(last_mouse_up, last_mouse_down);
  if (last_mouse_up_delta >= mouse_up_max_threshold) {
    return;
  }

  if (mouse_event_deltas < mouse_up_down_max_delta) {
    ToggleFullscreen();
  }
}

void EmulatorWindow::FileDrop(const std::filesystem::path& path) {
  if (!emulator_initialized_) {
    return;
  }

  RunTitle(path);
}

void EmulatorWindow::FileOpen() {
  std::filesystem::path path;

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kFile);
  file_picker->set_multi_selection(false);
  file_picker->set_title("Select Content Package");
  file_picker->set_extensions({
      {"Supported Files", "*.iso;*.xex;*.zar;*.*"},
      {"Disc Image (*.iso)", "*.iso"},
      {"Disc Archive (*.zar)", "*.zar"},
      {"Xbox Executable (*.xex)", "*.xex"},
      //{"Content Package (*.xcp)", "*.xcp" },
      {"All Files (*.*)", "*.*"},
  });
  if (file_picker->Show(window_.get())) {
    auto selected_files = file_picker->selected_files();
    if (!selected_files.empty()) {
      path = selected_files[0];
    }
    // Only run the title if a file is selected
    RunTitle(path);
  }
}

void EmulatorWindow::FileClose() { emulator_->TerminateTitle(); }

void EmulatorWindow::ExtractZarchive() {
  std::vector<std::filesystem::path> zarchive_files;
  std::filesystem::path extract_dir;

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kFile);
  file_picker->set_multi_selection(true);
  file_picker->set_title("Select Zar Package");
  file_picker->set_extensions({
      {"Zarchive Files (*.zar)", "*.zar"},
  });

  if (file_picker->Show(window_.get())) {
    zarchive_files = file_picker->selected_files();
  }

  if (zarchive_files.empty()) {
    return;
  }

  file_picker->set_type(ui::FilePicker::Type::kDirectory);
  file_picker->set_title("Select Directory to Extract");

  if (file_picker->Show(window_.get())) {
    extract_dir = file_picker->selected_files().front();
  }

  if (extract_dir.empty()) {
    return;
  }

  std::string extract_overview = "";

  for (auto& zarchive_file_path : zarchive_files) {
    extract_overview += "\n" + path_to_utf8(zarchive_file_path);
  }

  app_context_.CallInUIThread([&]() {
    new xe::ui::HostNotificationWindow(imgui_drawer(), "Extracting...",
                                       string_util::trim(extract_overview), 0);
  });

  auto run = [this, extract_dir, zarchive_files]() -> void {
    std::string summary = "";

    for (auto& zarchive_file_path : zarchive_files) {
      // Normalize the path and make absolute.
      auto abs_path = std::filesystem::absolute(zarchive_file_path);
      std::filesystem::path abs_extract_dir;

      if (zarchive_files.size() > 1) {
        abs_extract_dir =
            std::filesystem::absolute((extract_dir / abs_path.stem()));
      } else {
        abs_extract_dir = std::filesystem::absolute(extract_dir);
      }

      XELOGI("Extracting zar package: {}\n",
             zarchive_file_path.filename().string());

      X_STATUS result = X_STATUS_NOT_IMPLEMENTED;

      if (result != X_STATUS_SUCCESS) {
        std::error_code ec;

        if (!std::filesystem::is_empty(abs_extract_dir)) {
          std::filesystem::remove(abs_extract_dir, ec);
        }

        summary += fmt::format("\nFailed: {}", zarchive_file_path);

        XELOGE("Failed to extract Zarchive package.", result);
      } else {
        summary += fmt::format("\nSuccess: {}", abs_extract_dir);
      }
    }

    new xe::ui::HostNotificationWindow(imgui_drawer(), "Zar Extraction Summary",
                                       string_util::trim(summary), 0);
  };

  auto zarThread = std::thread(run);
  zarThread.detach();
}

void EmulatorWindow::CreateZarchive() {
  std::vector<std::filesystem::path> content_dirs;
  std::filesystem::path zarchive_dir;

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kDirectory);
  file_picker->set_multi_selection(true);
  file_picker->set_title("Select Contents");

  if (file_picker->Show(window_.get())) {
    content_dirs = file_picker->selected_files();
  }

  if (content_dirs.empty()) {
    return;
  }

  if (content_dirs.size() == 1) {
    file_picker->set_mode(ui::FilePicker::Mode::kSave);
    file_picker->set_type(ui::FilePicker::Type::kFile);
    file_picker->set_multi_selection(false);
    file_picker->set_file_name(content_dirs.front().filename().string());
    file_picker->set_default_extension("zar");
    file_picker->set_title("Zarchive File");
    file_picker->set_extensions({
        {"Zarchive File (*.zar)", "*.zar"},
    });
  } else {
    file_picker->set_title("Output Directory");
  }

  if (file_picker->Show(window_.get())) {
    zarchive_dir = file_picker->selected_files().front();
  }

  if (zarchive_dir.empty()) {
    return;
  }

  std::string create_overview = "";

  std::map<std::filesystem::path, std::filesystem::path> zarchive_files{};

  for (auto& content_path : content_dirs) {
    // Normalize the path and make absolute.
    auto abs_content_dir = std::filesystem::absolute(content_path);
    std::filesystem::path abs_zarchive_file;

    if (content_dirs.size() > 1) {
      abs_zarchive_file = std::filesystem::absolute(
          (zarchive_dir / abs_content_dir.filename().concat(".zar")));
    } else {
      abs_zarchive_file = std::filesystem::absolute(zarchive_dir);
    }

    zarchive_files[content_path] = abs_zarchive_file;

    create_overview += "\n" + path_to_utf8(abs_zarchive_file);
  }

  app_context_.CallInUIThread([&]() {
    new xe::ui::HostNotificationWindow(imgui_drawer(), "Creating...",
                                       string_util::trim(create_overview), 0);
  });

  auto run = [this, zarchive_files]() -> void {
    std::string summary = "";

    for (auto const& [content_path, zarchive_file] : zarchive_files) {
      // Normalize the path and make absolute.
      auto abs_content_dir = std::filesystem::absolute(content_path);

      XELOGI("Creating zar package: {}\n", zarchive_file.filename().string());

      X_STATUS result = X_STATUS_NOT_IMPLEMENTED;

      if (result != X_ERROR_SUCCESS) {
        std::error_code ec;

        // delete incomplete output file
        std::filesystem::remove(zarchive_file, ec);

        summary += fmt::format("\nFailed: {}", abs_content_dir);

        XELOGE("Failed to create Zarchive package.", result);
      } else {
        summary += fmt::format("\nSuccess: {}", zarchive_file);
      }
    }

    new xe::ui::HostNotificationWindow(imgui_drawer(), "Zar Creation Summary",
                                       string_util::trim(summary), 0);
  };

  auto zarThread = std::thread(run);
  zarThread.detach();
}

void EmulatorWindow::ShowContentDirectory() {
  auto content_root = emulator_->content_root();

  if (!std::filesystem::exists(content_root)) {
    std::filesystem::create_directories(content_root);
  }

  LaunchFileExplorer(content_root);
}

void EmulatorWindow::CpuTimeScalarReset() {
  Clock::set_guest_time_scalar(1.0);
  UpdateTitle();
}

void EmulatorWindow::CpuTimeScalarSetHalf() {
  Clock::set_guest_time_scalar(Clock::guest_time_scalar() / 2.0);
  UpdateTitle();
}

void EmulatorWindow::CpuTimeScalarSetDouble() {
  Clock::set_guest_time_scalar(Clock::guest_time_scalar() * 2.0);
  UpdateTitle();
}

void EmulatorWindow::CpuBreakIntoDebugger() {
  if (!cvars::debug) {
    xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(), "Xenia Debugger",
                                        "Xenia must be launched with the "
                                        "--debug flag in order to enable "
                                        "debugging.");
    return;
  }
  auto processor = emulator()->processor();
  if (processor->execution_state() == cpu::ExecutionState::kRunning) {
    // Currently running, so interrupt (and show the debugger).
    processor->Pause();
  } else {
    // Not running, so just bring the debugger into focus.
    processor->ShowDebugger();
  }
}

void EmulatorWindow::CpuBreakIntoHostDebugger() { xe::debugging::Break(); }

void EmulatorWindow::GpuTraceFrame() {
  emulator()->graphics_system()->RequestFrameTrace();
}

void EmulatorWindow::GpuClearCaches() {
  emulator()->graphics_system()->ClearCaches();
}

void EmulatorWindow::SetFullscreen(bool fullscreen_) {
  if (window_->IsFullscreen() == fullscreen_) {
    return;
  }

  OVERRIDE_bool(fullscreen, fullscreen_);

  window_->SetFullscreen(fullscreen_);
  window_->SetCursorVisibility(fullscreen_
                                   ? ui::Window::CursorVisibility::kAutoHidden
                                   : ui::Window::CursorVisibility::kVisible);
}

void EmulatorWindow::ToggleFullscreen() {
  SetFullscreen(!window_->IsFullscreen());
}

void EmulatorWindow::ToggleControllerVibration() {
  auto input_sys = emulator()->input_system();
  if (input_sys) {
    auto input_lock = input_sys->lock();

    input_sys->ToggleVibration();

    if (emulator_->kernel_state()) {
      emulator_->kernel_state()->BroadcastNotification(
          kXNotificationSystemProfileSettingChanged,
          static_cast<uint32_t>(input_sys->GetConnectedSlots().count()));
    }
  }
}

void EmulatorWindow::ShowCompatibility() {
  const std::string_view base_url =
      "https://github.com/xenia-canary/game-compatibility/issues";
  std::string url;
  // Avoid searching for a title ID of "00000000".
  uint32_t title_id = emulator_->title_id();
  if (!title_id) {
    url = base_url;
  } else {
    url = fmt::format("{}?q=is%3Aissue+is%3Aopen+{:08X}", base_url, title_id);
  }
  LaunchWebBrowser(url);
}

void EmulatorWindow::ShowFAQ() {
  LaunchWebBrowser("https://github.com/xenia-canary/xenia-canary/wiki/FAQ");
}

void EmulatorWindow::ShowBuildCommit() {
#ifdef XE_BUILD_IS_PR
  LaunchWebBrowser(
      "https://github.com/xenia-canary/xenia-canary/pull/" XE_BUILD_PR_NUMBER);
#else
  LaunchWebBrowser(
      "https://github.com/xenia-canary/xenia-canary/commit/" XE_BUILD_COMMIT);
#endif
}

void EmulatorWindow::UpdateTitle() {
  xe::StringBuffer sb;
  sb.Append(base_title_);

  // Title information, if available
  if (emulator()->is_title_open()) {
    sb.AppendFormat(" | [{:08X}", emulator()->title_id());
    auto title_version = emulator()->title_version();
    if (!title_version.empty()) {
      sb.Append(" v");
      sb.Append(title_version);
    }
    sb.Append("]");

    auto title_name = emulator()->title_name();
    if (!title_name.empty()) {
      sb.Append(" ");
      sb.Append(title_name);
    }
  }

  // Graphics system name, if available
  auto graphics_system = emulator()->graphics_system();
  if (graphics_system) {
    auto graphics_name = graphics_system->name();
    if (!graphics_name.empty()) {
      sb.Append(" <");
      sb.Append(graphics_name);
      sb.Append(">");
    }
  }

  if (Clock::guest_time_scalar() != 1.0) {
    sb.AppendFormat(" (@{:.2f}x)", Clock::guest_time_scalar());
  }

  if (initializing_shader_storage_) {
    sb.Append(" (Preloading shaders\u2026)");
  }

  patcher::Patcher* patcher = emulator()->patcher();
  if (patcher && patcher->IsAnyPatchApplied()) {
    sb.Append(" [Patches Applied]");
  }

  patcher::PluginLoader* pluginloader = emulator()->plugin_loader();
  if (pluginloader && pluginloader->IsAnyPluginLoaded()) {
    sb.Append(" [Plugins Loaded]");
  }

  window_->SetTitle(sb.to_string_view());
}

void EmulatorWindow::SetInitializingShaderStorage(bool initializing) {
  if (initializing_shader_storage_ == initializing) {
    return;
  }
  initializing_shader_storage_ = initializing;
  UpdateTitle();
}

// Notes:
// SDL and XInput both support the guide button
//
// Assumes titles do not use the guide button.
// For titles that do such as dashboards these titles could be excluded based on
// their title ID.
//
// Xbox Gamebar:
// If the Xbox Gamebar overlay is enabled Windows will consume the guide
// button's input, this can be seen using hid-demo.
//
// Workaround: Detect if the Xbox Gamebar overlay is enabled then use the BACK
// button instead of the GUIDE button. Therefore BACK and GUIDE are reserved
// buttons for hotkeys.
//
// This is not an issue with DualShock controllers because Windows will not
// open the gamebar overlay using the PlayStation menu button.
//
// Xbox One S Controller:
// The guide button on this controller is very buggy no idea why.
// Using xinput usually registers after a double tap.
// Doesn't work at all using SDL.
// Needs more testing.
//
// Steam:
// If guide button focus is enabled steam will open.
// Steam uses BACK + GUIDE to open an On-Screen keyboard, however this is not a
// problem since both these buttons are reserved.
const std::map<int, EmulatorWindow::ControllerHotKey> controller_hotkey_map = {
    // Must use the Guide Button for all pass through hotkeys
    {X_INPUT_GAMEPAD_A | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ReadbackResolve,
         "A + Guide = Toggle Readback Resolve", true)},
    {X_INPUT_GAMEPAD_B | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleLogging,
         "B + Guide = Toggle between loglevel set in config and the 'Disabled' "
         "loglevel.",
         true, true)},
    {X_INPUT_GAMEPAD_Y | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleFullscreen,
         "Y + Guide = Toggle Fullscreen", true)},
    {X_INPUT_GAMEPAD_X | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ClearMemoryPageState,
         "X + Guide = Toggle Clear Memory Page State", true)},

    {X_INPUT_GAMEPAD_RIGHT_SHOULDER | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ClearGPUCache,
         "Right Shoulder + Guide = Clear GPU Cache", true)},
    {X_INPUT_GAMEPAD_LEFT_SHOULDER | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleControllerVibration,
         "Left Shoulder + Guide = Toggle Controller Vibration", true)},

    // CPU Time Scalar with no rumble feedback
    {X_INPUT_GAMEPAD_DPAD_DOWN | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::CpuTimeScalarSetHalf,
         "D-PAD Down + Guide = Half CPU Scalar")},
    {X_INPUT_GAMEPAD_DPAD_UP | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::CpuTimeScalarSetDouble,
         "D-PAD Up + Guide = Double CPU Scalar")},
    {X_INPUT_GAMEPAD_DPAD_RIGHT | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::CpuTimeScalarReset,
         "D-PAD Right + Guide = Reset CPU Scalar")},

    // non-pass through hotkeys
    {X_INPUT_GAMEPAD_Y, EmulatorWindow::ControllerHotKey(
                            EmulatorWindow::ButtonFunctions::ToggleFullscreen,
                            "Y = Toggle Fullscreen", true, false)},
    {X_INPUT_GAMEPAD_START, EmulatorWindow::ControllerHotKey(
                                EmulatorWindow::ButtonFunctions::RunTitle,
                                "Start = Run Selected Title", false, false)},
    {X_INPUT_GAMEPAD_BACK | X_INPUT_GAMEPAD_START,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleLogging,
         "Back + Start = Toggle between loglevel set in config and the "
         "'Disabled' loglevel.",
         false, false)},
    {X_INPUT_GAMEPAD_DPAD_DOWN,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::IncTitleSelect,
         "D-PAD Down = Title Selection +1", true, false)},
    {X_INPUT_GAMEPAD_DPAD_UP,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::DecTitleSelect,
         "D-PAD Up = Title Selection -1", true, false)}};

EmulatorWindow::ControllerHotKey EmulatorWindow::ProcessControllerHotkey(
    int buttons) {
  // Default return value
  EmulatorWindow::ControllerHotKey Unknown_hotkey = {};

  if (buttons == 0) {
    return Unknown_hotkey;
  }

  if (disable_hotkeys_.load()) {
    return Unknown_hotkey;
  }

  // Hotkey cool-down to prevent toggling too fast
  constexpr std::chrono::milliseconds delay(75);

  // If the Xbox Gamebar is enabled or the Guide button is disabled then
  // replace the Guide button with the Back button without redeclaring the key
  // mappings
  if (IsUseNexusForGameBarEnabled() || !cvars::guide_button) {
    if ((buttons & X_INPUT_GAMEPAD_BACK) == X_INPUT_GAMEPAD_BACK) {
      buttons &= ~X_INPUT_GAMEPAD_BACK;
      buttons |= X_INPUT_GAMEPAD_GUIDE;
    }
  }

  auto it = controller_hotkey_map.find(buttons);
  if (it == controller_hotkey_map.end()) {
    return Unknown_hotkey;
  }

  // Do not activate hotkeys that are not intended for activation during
  // gameplay
  if (emulator_->is_title_open()) {
    // If non-pass through (menu hoykeys) or hotkeys disabled then return
    if (!it->second.title_passthru || !cvars::controller_hotkeys) {
      return Unknown_hotkey;
    }
  }

  std::string notificationTitle = "";
  std::string notificationDesc = "";

  EmulatorWindow::ControllerHotKey button_combination = it->second;

  switch (button_combination.function) {
    case ButtonFunctions::ToggleFullscreen:
      app_context().CallInUIThread([this]() { ToggleFullscreen(); });

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::RunTitle: {
      if (selected_title_index == -1) {
        selected_title_index++;
      }

      if (selected_title_index < recently_launched_titles_.size()) {
        app_context().CallInUIThread([this]() {
          RunTitle(
              recently_launched_titles_[selected_title_index].path_to_file);
        });
      }
    } break;
    case ButtonFunctions::ClearMemoryPageState:
      ToggleGPUSetting(GPUSetting::ClearMemoryPageState);

      // Assume the user wants ClearCaches as well
      if (cvars::clear_memory_page_state) {
        GpuClearCaches();
      }

      notificationTitle = "Toggle Clear Memory Page State";
      notificationDesc =
          cvars::clear_memory_page_state ? "Enabled" : "Disabled";

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::ReadbackResolve:
      CycleReadbackResolve();

      notificationTitle = "Readback Resolve Mode";
      notificationDesc = cvars::readback_resolve;

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::CpuTimeScalarSetHalf:
      CpuTimeScalarSetHalf();

      notificationTitle = "Time Scalar";
      notificationDesc =
          fmt::format("Decreased to {}", Clock::guest_time_scalar());
      break;
    case ButtonFunctions::CpuTimeScalarSetDouble:
      CpuTimeScalarSetDouble();

      notificationTitle = "Time Scalar";
      notificationDesc =
          fmt::format("Increased to {}", Clock::guest_time_scalar());
      break;
    case ButtonFunctions::CpuTimeScalarReset:
      CpuTimeScalarReset();

      notificationTitle = "Time Scalar";
      notificationDesc = fmt::format("Reset to {}", Clock::guest_time_scalar());
      break;
    case ButtonFunctions::ClearGPUCache:
      GpuClearCaches();

      notificationTitle = "Clear GPU Cache";
      notificationDesc = "Complete";

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::ToggleControllerVibration: {
      ToggleControllerVibration();

      bool vibration = false;

      auto input_sys = emulator()->input_system();
      if (input_sys) {
        vibration = input_sys->GetVibrationCvar();
      }

      notificationTitle = "Toggle Controller Vibration";
      notificationDesc = vibration ? "Enabled" : "Disabled";

      // Extra Sleep
      xe::threading::Sleep(delay);
    } break;
    case ButtonFunctions::IncTitleSelect:
      selected_title_index++;
      break;
    case ButtonFunctions::DecTitleSelect:
      selected_title_index--;
      break;
    case ButtonFunctions::ToggleLogging: {
      logging::ToggleLogLevel();

      notificationTitle = "Toggle Logging";

      LogLevel level = static_cast<LogLevel>(logging::internal::GetLogLevel());
      notificationDesc = level == LogLevel::Disabled ? "Disabled" : "Enabled";
    } break;
    case ButtonFunctions::Unknown:
    default:
      break;
  }

  if ((button_combination.function == ButtonFunctions::IncTitleSelect ||
       button_combination.function == ButtonFunctions::DecTitleSelect) &&
      recently_launched_titles_.size() > 0) {
    selected_title_index =
        std::clamp(selected_title_index, 0,
                   static_cast<int32_t>(recently_launched_titles_.size() - 1));

    // Must clear dialogs to prevent stacking
    ClearDialogs();

    // Titles may contain Unicode characters such as At World’s End
    // Must use ImGUI font that can render these Unicode characters
    std::string title_name;

    // Use filename if title name is empty
    if (recently_launched_titles_[selected_title_index].title_name.empty()) {
      title_name = recently_launched_titles_[selected_title_index]
                       .path_to_file.filename()
                       .string();
    } else {
      title_name = recently_launched_titles_[selected_title_index].title_name;
    }

    std::string title = fmt::format(
        "{}: {}\n\n{}", selected_title_index + 1, title_name,
        controller_hotkey_map.find(X_INPUT_GAMEPAD_START)->second.pretty);

    xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(), "Title Selection",
                                        title);
  }

  if (!notificationTitle.empty()) {
    app_context_.CallInUIThread(
        [imgui_drawer = imgui_drawer(), notificationTitle, notificationDesc]() {
          new xe::ui::HostNotificationWindow(imgui_drawer, notificationTitle,
                                             notificationDesc, 0);
        });
  }

  xe::threading::Sleep(delay);

  return it->second;
}

void EmulatorWindow::VibrateController(xe::hid::InputSystem* input_sys,
                                       uint32_t user_index,
                                       bool toggle_rumble) {
  constexpr std::chrono::milliseconds rumble_duration(100);

  // Hold lock while sleeping this thread for the duration of the rumble,
  // otherwise the rumble may fail.
  auto input_lock = input_sys->lock();

  X_INPUT_VIBRATION vibration = {};

  vibration.left_motor_speed = toggle_rumble ? UINT16_MAX : 0;
  vibration.right_motor_speed = toggle_rumble ? UINT16_MAX : 0;

  input_sys->SetState(user_index, &vibration);

  // Vibration duration
  if (toggle_rumble) {
    xe::threading::Sleep(rumble_duration);
  }
}

void EmulatorWindow::GamepadHotKeys() {
  X_INPUT_STATE state;

  constexpr std::chrono::milliseconds thread_delay(75);

  auto input_sys = emulator_->input_system();

  if (input_sys) {
    while (true) {
      // Collect controller states while holding the lock
      std::array<std::pair<bool, X_INPUT_STATE>, XUserMaxUserCount>
          controller_states;
      {
        auto input_lock = input_sys->lock();
        for (uint32_t user_index = 0; user_index < XUserMaxUserCount;
             ++user_index) {
          X_RESULT result = input_sys->GetState(
              user_index, X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD, &state);
          controller_states[user_index] = {result == X_ERROR_SUCCESS, state};
        }
      }  // Lock is released here when input_lock goes out of scope

      // Process hotkeys without holding the lock
      for (uint32_t user_index = 0; user_index < XUserMaxUserCount;
           ++user_index) {
        if (controller_states[user_index].first) {
          if (ProcessControllerHotkey(
                  controller_states[user_index].second.gamepad.buttons)
                  .rumble) {
            // Enable Vibration
            VibrateController(input_sys, user_index, true);

            // Disable Vibration
            VibrateController(input_sys, user_index, false);
          }
        }
      }

      xe::threading::Sleep(thread_delay);
    }
  }
}

void EmulatorWindow::ToggleGPUSetting(gpu::GPUSetting setting) {
  const char* cvar_name = nullptr;
  bool new_value = false;

  switch (setting) {
    case GPUSetting::ClearMemoryPageState:
      new_value = !cvars::clear_memory_page_state;
      SaveGPUSetting(GPUSetting::ClearMemoryPageState, new_value);
      cvar_name = "clear_memory_page_state";
      break;
    case GPUSetting::MemexportAwaitFences:
      new_value = !cvars::memexport_await_fences;
      SaveGPUSetting(GPUSetting::MemexportAwaitFences, new_value);
      cvar_name = "memexport_await_fences";
      break;
  }

  // Persist to the per-game config through the fork's helper, which already
  // does upstream's load/insert/save and the title-open check.
  if (cvar_name) {
    config::SaveGameConfigSetting(emulator_, "GPU", cvar_name, new_value);
  }
}

void EmulatorWindow::CycleReadbackResolve() {
  gpu::GraphicsSystem* graphics_system = emulator_->graphics_system();
  if (!graphics_system) {
    return;
  }

  gpu::CommandProcessor* command_processor =
      graphics_system->command_processor();
  if (!command_processor) {
    return;
  }

  gpu::ReadbackResolveMode current =
      command_processor->GetReadbackResolveMode();
  gpu::ReadbackResolveMode next;
  switch (current) {
    case gpu::ReadbackResolveMode::kDisabled:
      next = gpu::ReadbackResolveMode::kFast;
      break;
    case gpu::ReadbackResolveMode::kFast:
      next = gpu::ReadbackResolveMode::kAll;
      break;
    default:
      next = gpu::ReadbackResolveMode::kDisabled;
      break;
  }
}

void EmulatorWindow::DisplayHotKeysConfig() {
  std::string msg = "";
  std::string msg_passthru = "";

  bool guide_enabled = !IsUseNexusForGameBarEnabled() && cvars::guide_button;

  for (auto const& [key, val] : controller_hotkey_map) {
    std::string pretty_text = val.pretty;

    if (!guide_enabled) {
      pretty_text = std::regex_replace(
          pretty_text, std::regex("Guide", std::regex_constants::icase),
          "Back");
    }

    if (emulator_->is_title_open() && !val.title_passthru) {
      pretty_text += " (Disabled)";
    }

    if (val.title_passthru && !cvars::controller_hotkeys) {
      pretty_text += " (Disabled)";
    }

    if (val.title_passthru) {
      msg += pretty_text + "\n";
    } else {
      msg_passthru += pretty_text + "\n";
    }
  }

  // Add Title
  msg.insert(0, "Gameplay Hotkeys\n");

  // Prepend non-passthru hotkeys
  msg_passthru += "\n";
  msg.insert(0, msg_passthru);
  msg += "\n";

  msg += "Readback Resolve: " + cvars::readback_resolve;
  msg += "\n";

  msg += "Clear Memory Page State: " +
         xe::string_util::BoolToString(cvars::clear_memory_page_state);
  msg += "\n";

  msg += "Controller Hotkeys: " +
         xe::string_util::BoolToString(cvars::controller_hotkeys);

  ClearDialogs();
  xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(), "Controller Hotkeys",
                                      msg);
}

std::string EmulatorWindow::CanonicalizeFileExtension(
    const std::filesystem::path& path) {
  return xe::utf8::lower_ascii(xe::path_to_utf8(path.extension()));
}

xe::X_STATUS EmulatorWindow::RunTitle(
    const std::filesystem::path& path_to_file) {
  std::error_code ec = {};
  bool titleExists = std::filesystem::exists(path_to_file, ec);

  if (path_to_file.empty() || !titleExists) {
    std::string log_msg =
        fmt::format("Failed to launch title path is {}.",
                    path_to_file.empty() ? "empty" : "invalid");

    if (!path_to_file.empty() && !titleExists) {
      log_msg.append(fmt::format("\nProvided Path: {}", path_to_file));
    }

    if (ec) {
      log_msg.append(fmt::format("\nExtended message info: {} ({:08X})",
                                 ec.message(), ec.value()));
    }

    XELOGE("{}", log_msg);

    ClearDialogs();

    xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(),
                                        "Title Launch Failed!", log_msg);

    return X_STATUS_NO_SUCH_FILE;
  }

  if (emulator_->is_title_open()) {
    // Terminate the current title and start a new title.
    // if (emulator_->TerminateTitle() == X_STATUS_SUCCESS) {
    //   return RunTitle(path);
    // }

    return X_STATUS_UNSUCCESSFUL;
  }

  // Prevent crashing the emulator by not loading a game if a game is already
  // loaded.
  auto abs_path = std::filesystem::absolute(path_to_file);

  auto extension = CanonicalizeFileExtension(abs_path);

  if (extension == ".7z" || extension == ".zip" || extension == ".rar" ||
      extension == ".tar" || extension == ".gz") {
    xe::ShowSimpleMessageBox(
        xe::SimpleMessageBoxType::Error,
        fmt::format(
            "Unsupported format!\n"
            "Xenia does not support running software in an archived format."));

    return X_STATUS_UNSUCCESSFUL;
  }

  auto result = emulator_->LaunchPath(abs_path);

  disable_hotkeys_ = false;

  ClearDialogs();

  if (result) {
    XELOGE("Failed to launch target: {:08X}", result);

    xe::ui::ImGuiDialog::ShowMessageBox(
        imgui_drawer_.get(), "Title Launch Failed!",
        "Failed to launch title.\n\nCheck xenia.log for technical details.");

    emulator_->file_system()->Clear();
  } else {
    AddRecentlyLaunchedTitle(path_to_file, emulator_->title_name());

    auto xam =
        emulator_->kernel_state()->GetKernelModule<kernel::xam::XamModule>(
            "xam.xex");

    xam->loader_data().host_path = xe::path_to_utf8(abs_path);
  }

  return result;
}

void EmulatorWindow::RunPreviouslyPlayedTitle() {
  if (recently_launched_titles_.size() >= 1) {
    RunTitle(recently_launched_titles_[0].path_to_file);
  }
}

void EmulatorWindow::FillRecentlyLaunchedTitlesMenu(
    xe::ui::MenuItem* recent_menu) {
  for (int i = 0; i < recently_launched_titles_.size(); ++i) {
    std::string hotkey = (i == 0) ? "F9" : "";

    const RecentTitleEntry& entry = recently_launched_titles_[i];
    const std::string item_text = entry.title_name.empty()
                                      ? entry.path_to_file.string()
                                      : entry.title_name;

    recent_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, item_text, hotkey,
        std::bind(&EmulatorWindow::RunTitle, this, entry.path_to_file)));
  }
}

void EmulatorWindow::LoadRecentlyLaunchedTitles() {
  std::ifstream file(emulator()->storage_root() /
                     kRecentlyPlayedTitlesFilename);
  if (!file.is_open()) {
    return;
  }

  toml::parse_result parsed_file;
  try {
    parsed_file = toml::parse(file);
  } catch (toml::parse_error& exception) {
    XELOGE("Cannot parse file: recent.toml. Error: {}", exception.what());
    return;
  }

  if (parsed_file.is_table()) {
    for (const auto& [index, entry] : *parsed_file.as_table()) {
      if (!entry.is_table()) {
        continue;
      }

      const toml::table* entry_table = entry.as_table();

      std::string title_name =
          entry_table->get_as<std::string>("title_name")->get();
      std::string path = entry_table->get_as<std::string>("path")->get();
      std::time_t last_run_time =
          entry_table->get_as<int64_t>("last_run_time")->get();

      std::error_code ec = {};
      if (path.empty() || !std::filesystem::exists(path, ec)) {
        continue;
      }

      recently_launched_titles_.push_back({title_name, path, last_run_time});
    }
  }
}

void EmulatorWindow::AddRecentlyLaunchedTitle(
    std::filesystem::path path_to_file, std::string title_name) {
  if (cvars::recent_titles_entry_amount <= 0) {
    return;
  }

  // Check if game is already on list and pop it to front
  auto entry_index = std::find_if(recently_launched_titles_.cbegin(),
                                  recently_launched_titles_.cend(),
                                  [&title_name](const RecentTitleEntry& entry) {
                                    return entry.title_name == title_name;
                                  });
  if (entry_index != recently_launched_titles_.cend()) {
    recently_launched_titles_.erase(entry_index);
  }

  recently_launched_titles_.insert(recently_launched_titles_.cbegin(),
                                   {title_name, path_to_file, time(nullptr)});
  // Serialize to toml
  auto toml_table = toml::table();

  uint8_t index = 0;
  for (const RecentTitleEntry& entry : recently_launched_titles_) {
    auto entry_table = toml::table();

    // Fill entry under specific index.
    std::string str_path = xe::path_to_utf8(entry.path_to_file);
    entry_table.insert("title_name", entry.title_name);
    entry_table.insert("path", str_path);
    entry_table.insert("last_run_time", entry.last_run_time);

    toml_table.insert(std::to_string(index++), entry_table);

    if (index >= cvars::recent_titles_entry_amount) {
      break;
    }
  }
  // Open and write serialized data.
  std::ofstream file(emulator()->storage_root() / kRecentlyPlayedTitlesFilename,
                     std::ofstream::trunc);
  file << toml_table;
  file.close();
}

void EmulatorWindow::ClearDialogs() {
  imgui_drawer_.get()->ClearDialogs();
  emulator_->kernel_state()->xam_state()->is_xam_dialog_present_.store(false);
}

}  // namespace app
}  // namespace xe

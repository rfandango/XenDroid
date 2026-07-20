package xendroid.compose.settings

import xendroid.compose.settings.Setting.*

private fun desc(n: String) = SettingDescriptions.byName[n] ?: ""
private fun b(s: String, n: String, t: String, d: Boolean) = Setting.Bool(s, n, t, d, desc(n))
private fun i(s: String, n: String, t: String, d: Int, lo: Int, hi: Int) =
    Setting.IntRange(s, n, t, d, lo, hi, desc(n))
private fun l(s: String, n: String, t: String, d: String, vararg o: Pair<String, String>) =
    Setting.ListChoice(s, n, t, d, o.map { ListOption(it.first, it.second) }, desc(n))

object SettingsSchema {

    val categories: List<SettingsCategory> = listOf(

        SettingsCategory("Vulkan", listOf(
            b("Vulkan", "vulkan_sparse_shared_memory", "Sparse shared memory", true),
            b("Vulkan", "vulkan_log_debug_messages", "Log debug messages", false),
            b("Vulkan", "vulkan_validation", "Validation layers", false),
            b("Vulkan", "vulkan_renderdoc_capture", "RenderDoc capture layer", false),
            b("Vulkan", "vulkan_allow_present_mode_immediate", "Allow present mode: immediate", true),
            b("Vulkan", "vulkan_allow_present_mode_mailbox", "Allow present mode: mailbox", true),
            b("Vulkan", "vulkan_allow_present_mode_fifo_relaxed", "Allow present mode: FIFO relaxed", true),
            b("Vulkan", "vulkan_async_skip_draws", "Async skip draws", true),
            b("Vulkan", "vulkan_dynamic_pipeline_state", "Extended dynamic state", true),
            // Numeric thread count (0 = synchronous, 1..5 = explicit) -> a slider, not a
            // dropdown. Native cvar also accepts -1 (auto = 75% of cores), but the template
            // ships 4 and a 0..5 slider is the intended UX.
            i("Vulkan", "vulkan_pipeline_creation_threads", "Pipeline creation threads", 4, 0, 5),
            Action("Vulkan", "vulkan_lib_path", "Custom Vulkan driver", "", desc("vulkan_lib_path")),
            b("Vulkan", "adrenotools_force_max_clocks", "Force max GPU clocks (adrenotools)", false),
            // TU_DEBUG flags for the Turnip (Mesa freedreno) driver. Empty = driver default
            // (GMEM/tiled, fast); 'sysmem' forces untiled rendering (much slower) but avoids a
            // class of Adreno GPU hangs. Applied at vkCreateInstance, so it takes effect on the
            // next game launch.
            l("Vulkan", "turnip_debug", "Turnip debug mode", "",
                "" to "None (no TU_DEBUG flags, GMEM)", "sysmem" to "sysmem (untiled, slower)",
                "sysmem,nolrz" to "sysmem + nolrz (LRZ off, perf diagnostic)",
                "sysmem,noubwc" to "sysmem + noubwc (UBWC off, perf diagnostic)"),
            // Cross-draw texture/sampler descriptor-set reuse (perf). Master toggle gates reuse
            // on/off; the edge toggle (only when reuse is on) picks edge's bitmask gate vs
            // XenDroid's content-hash gate for A/B. Three-way: off / on+hash / on+edge.
            b("Vulkan", "vulkan_cache_texture_descriptors", "Cache texture descriptors", true),
            b("Vulkan", "vulkan_texture_descriptor_reuse_edge", "Texture descriptor gate: edge (off = hash)", false),
        )),

        SettingsCategory("Video", listOf(
            b("Video", "widescreen", "Widescreen", true),
            l("Video", "video_standard", "Video standard", "1",
                "1" to "NTSC", "2" to "NTSC-J", "3" to "PAL-60"),
            l("Video", "internal_display_resolution", "Internal display resolution", "8",
                "0" to "640x480", "1" to "640x576", "2" to "720x480", "3" to "720x576",
                "4" to "800x600", "5" to "848x480", "6" to "1024x768", "7" to "1152x864",
                "8" to "1280x720", "9" to "1280x768", "10" to "1280x960", "11" to "1280x1024",
                "12" to "1360x768", "13" to "1440x900", "14" to "1680x1050", "15" to "1920x540",
                "16" to "1920x1080"),
            b("Video", "use_50Hz_mode", "Use 50Hz mode", false),
            l("Video", "avpack", "AV pack", "8",
                "0" to "PAL-60 Component (SD)", "1" to "Unused", "2" to "PAL-60 SCART",
                "3" to "480p Component (HD)", "4" to "HDMI+A", "5" to "PAL-60 Composite/S-Video",
                "6" to "VGA", "7" to "TV PAL-60", "8" to "HDMI"),
            b("Video", "interlaced", "Interlaced", false),
            b("Video", "enable_3d_mode", "Enable 3D mode", false),
        )),

        SettingsCategory("UI", listOf(
            b("UI", "show_achievement_notification", "Show achievement notification", true),
            b("UI", "storage_selection_dialog", "Storage selection dialog", false),
            b("UI", "headless", "Headless", true),
        )),

        SettingsCategory("Storage", listOf(
            b("Storage", "mount_scratch", "Mount scratch", false),
            b("Storage", "mount_cache", "Mount cache", false),
        )),

        SettingsCategory("Kernel", listOf(
            b("Kernel", "staging_mode", "Staging mode", false),
            b("Kernel", "log_high_frequency_kernel_calls", "Log high-frequency kernel calls", false),
            l("Kernel", "kernel_display_gamma_type", "Display gamma type", "2",
                "0" to "linear", "1" to "sRGB (CRT)", "2" to "BT.709 (HDTV)"),
            b("Kernel", "ignore_thread_affinities", "Ignore thread affinities", true),
            b("Kernel", "kernel_pix", "Kernel PIX", false),
            b("Kernel", "kernel_cert_monitor", "Kernel cert monitor", false),
            b("Kernel", "ignore_thread_priorities", "Ignore thread priorities", true),
            b("Kernel", "allow_incompatible_title_update", "Allow incompatible title update", true),
            b("Kernel", "apply_title_update", "Apply title update", true),
            b("Kernel", "kernel_debug_monitor", "Kernel debug monitor", false),
        )),

        SettingsCategory("HID", listOf(
            l("HID", "hid", "HID backend", "android",
                "android" to "android", "nop" to "nop"),
        )),

        SettingsCategory("Memory", listOf(
            i("Memory", "mmap_address_high", "mmap address high", 8, 2, 63),
            b("Memory", "scribble_heap", "Scribble heap", false),
            b("Memory", "protect_zero", "Protect zero page", true),
            b("Memory", "writable_executable_memory", "Writable executable memory", true),
            b("Memory", "protect_on_release", "Protect on release", false),
            b("Memory", "ignore_offset_for_ranged_allocations", "Ignore offset for ranged allocations", false),
        )),

        SettingsCategory("XConfig", listOf(
            l("XConfig", "user_language", "User language", "1",
                "1" to "en", "2" to "ja", "3" to "de", "4" to "fr", "5" to "es", "6" to "it",
                "7" to "ko", "8" to "zh", "9" to "pt", "11" to "pl", "12" to "ru", "13" to "sv",
                "14" to "tr", "15" to "nb", "16" to "nl", "17" to "zh"), // value 10 skipped; 8 & 17 both zh
            l("XConfig", "user_country", "User country", "103", *userCountryOptions()),
        )),

        SettingsCategory("Display", listOf(
            b("Display", "present_letterbox", "Present letterbox", true),
            b("Display", "postprocess_dither", "Postprocess dither", true),
            l("Display", "postprocess_scaling_and_sharpening", "Scaling & sharpening", "",
                "bilinear" to "bilinear", "cas" to "cas", "fsr" to "fsr"),  // "" => bilinear (no selection)
            b("Display", "present_render_pass_clear", "Present render-pass clear", true),
            l("Display", "postprocess_antialiasing", "Antialiasing", "",
                "none" to "none", "fxaa" to "fxaa", "fxaa_extreme" to "fxaa_extreme"), // "" => none
            // host_present_from_non_ui_thread intentionally NOT exposed: it MUST be true on
            // Android (forced in xendroid_emu.cpp after config load) -- false black-screens the
            // app, so there is no valid user choice to make.
            b("Display", "show_debug_overlay", "Show debug overlay", false),
        )),

        SettingsCategory("GPU", listOf(
            b("GPU", "guest_display_refresh_cap", "Cap guest display refresh (VSync)", true),
            b("GPU", "store_shaders", "Store shaders", true),
            b("GPU", "resolve_resolution_scale_fill_half_pixel_offset", "Resolve scale: fill half-pixel offset", true),
            // readback_resolve is a STRING cvar (NOT a bool): CPU readback of render-to-texture
            // resolve results. uma=read the mapped shared-memory buffer directly on unified-memory
            // GPUs (Adreno), no GPU staging copy - falls back to fast when the buffer is not
            // host-visible (cvar default); fast=copy every frame; some=skip copy on cache hit;
            // full=wait for GPU (accurate but a GPU-CPU sync stall); none=disable readback (some
            // games render better without it, and it avoids the stall).
            l("GPU", "readback_resolve", "Readback resolve", "uma",
                "uma" to "UMA (direct map, no copy)",
                "fast" to "Fast (copy every frame)", "some" to "Some (skip copy on cache hit)",
                "full" to "Full (wait for GPU, slow)", "none" to "None (disabled)"),
            // How guest occlusion queries (PM4 EVENT_WRITE_ZPD) are serviced. 'fake' fabricates a
            // result with zero GPU-query overhead (fastest; some effects e.g. lens flares may look
            // slightly wrong); 'fast'/'fast-alt' issue real async Vulkan queries without stalling;
            // 'strict' issues a real query and stalls the command thread until the GPU answers (most
            // accurate, slowest). Live (SetZPDMode) -- takes effect without relaunch.
            l("GPU", "occlusion_query", "Occlusion query mode", "fast",
                "fake" to "Fake (no GPU query, fastest)", "fast" to "Fast (async query, no stall)",
                "fast-alt" to "Fast-alt (async, precise zeros)", "strict" to "Strict (real query, stalls)"),
            // Mid-frame command-buffer split: if >0, end+submit every N real draws so the GPU
            // overlaps rendering with CPU command-building instead of idling until swap. 0 = one
            // submission per frame (off). ~half the per-frame draw count is a good start; too-small
            // values hurt tiled GPUs.
            i("GPU", "vulkan_mid_frame_submission_draws", "Mid-frame submission (draws, 0=off)", 0, 0, 4096),
            b("GPU", "snorm16_render_target_full_range", "snorm16 render target full range", true),
            // min == the real TOML default (384); a higher floor would silently coerce the default up.
            i("GPU", "texture_cache_memory_limit_soft", "Texture cache soft limit (MB)", 384, 384, 4096),
            b("GPU", "native_2x_msaa", "Native 2x MSAA", true),
            l("GPU", "render_target_path", "Render target path", "performance",
                "performance" to "performance", "accuracy" to "accuracy"),
            b("GPU", "half_pixel_offset", "Half-pixel offset", true),
            b("GPU", "log_ringbuffer_kickoff_initiator_bts", "Log ringbuffer kickoff initiator BTs", false),
            i("GPU", "texture_cache_memory_limit_hard", "Texture cache hard limit (MB)", 768, 512, 4096),
            b("GPU", "gpu_allow_invalid_fetch_constants", "Allow invalid fetch constants", true),
            b("GPU", "log_guest_driven_gpu_register_written_values", "Log guest-driven GPU register writes", false),
            b("GPU", "trace_gpu_stream", "Trace GPU stream", false),
            b("GPU", "force_convert_quad_lists_to_triangle_lists", "Convert quad lists to triangle lists", false),
            b("GPU", "execute_unclipped_draw_vs_on_cpu_with_scissor", "Unclipped draw VS on CPU (scissor)", false),
            b("GPU", "mrt_edram_used_range_clamp_to_min", "MRT EDRAM used-range clamp to min", true),
            b("GPU", "execute_unclipped_draw_vs_on_cpu", "Unclipped draw VS on CPU", true),
            b("GPU", "readback_memexport", "Readback memexport", false),
            b("GPU", "force_convert_triangle_fans_to_lists", "Convert triangle fans to lists", false),
            b("GPU", "non_seamless_cube_map", "Non-seamless cube map", true),
            b("GPU", "depth_float24_round", "Depth float24 round", false),
            b("GPU", "clear_memory_page_state", "Clear memory page state", false),
            b("GPU", "depth_transfer_not_equal_test", "Depth transfer not-equal test", true),
            b("GPU", "execute_unclipped_draw_vs_on_cpu_for_psi_render_backend", "Unclipped draw VS on CPU (PSI backend)", true),
            b("GPU", "draw_resolution_scaled_texture_offsets", "Draw resolution-scaled texture offsets", true),
            l("GPU", "gpu", "GPU backend", "vulkan",
                "vulkan" to "vulkan", "null" to "null"),
            b("GPU", "depth_float24_convert_in_pixel_shader", "Depth float24 convert in pixel shader", false),
            b("GPU", "value_convert_7e3_8888_reuse", "value_convert_7e3_8888_reuse", true),
        )),

        SettingsCategory("CPU", listOf(
            b("CPU", "validate_hir", "Validate HIR", false),
            b("CPU", "trace_function_references", "Trace function references", false),
            b("CPU", "trace_function_coverage", "Trace function coverage", false),
            b("CPU", "store_all_context_values", "Store all context values", false),
            b("CPU", "break_condition_truncate", "Break condition truncate", true),
            b("CPU", "clock_source_raw", "Clock source raw", false),
            b("CPU", "break_on_unimplemented_instructions", "Break on unimplemented instructions", true),
            b("CPU", "break_on_start", "Break on start", false),
            b("CPU", "inline_mmio_access", "Inline MMIO access", true),
            b("CPU", "clock_no_scaling", "Clock no scaling", false),
        )),

        SettingsCategory("Logging", listOf(
            b("Logging", "log_string_format_kernel_calls", "Log string-format kernel calls", false),
            l("Logging", "log_level", "Log level", "2",
                "0" to "error", "1" to "warning", "2" to "info", "3" to "debug"),
            b("Logging", "flush_log", "Flush log", true),
            i("Logging", "log_sessions_keep", "Shelved log sessions to keep", 4, 1, 16),
            Action("Logging", "dump_session_logs", "Export session logs to Downloads", "", desc("dump_session_logs")),
        )),

        SettingsCategory("Content", listOf(
            l("Content", "license_mask", "License mask", "0",
                "0" to "disable", "1" to "first", "-1" to "all"),
        )),

        SettingsCategory("General", listOf(
            i("General", "time_scalar", "Time scalar", 1, 1, 8),
            b("General", "allow_plugins", "Allow plugins", false),
            b("General", "apply_patches", "Apply patches", true),
        )),

        SettingsCategory("APU", listOf(
            i("APU", "xmp_default_volume", "XMP default volume", 70, 0, 100),
            b("APU", "ffmpeg_verbose", "FFmpeg verbose", false),
            b("APU", "mute", "Mute", false),
            i("APU", "apu_max_queued_frames", "Max queued frames", 8, 4, 64),
            b("APU", "enable_xmp", "Enable XMP", true),
            l("APU", "xma_decoder", "XMA decoder", "new",
                "new" to "new", "old" to "old", "master" to "master", "fake" to "fake"),
            b("APU", "use_dedicated_xma_thread", "Use dedicated XMA thread", true),
            l("APU", "apu", "APU backend", "aaudio",
                "nop" to "nop", "aaudio" to "aaudio", "opensles" to "opensles"),
        )),
    )

    val allSettings: List<Setting> = categories.flatMap { it.settings }
    val byKey: Map<String, Setting> = allSettings.associateBy { it.key }
}

// user_country: values 1..109 with 17 and 94 skipped; labels are ISO country codes
// (107 entries). Order matches arrays.xml (es_arr_xconfig_user_country, transcribed
// verbatim from app/src/main/res/values/arrays.xml). Default 103 = US.
private val USER_COUNTRY_ISO: List<String> = listOf(
    "AE","AL","AM","AR","AT","AU","AZ","BE","BG","BH","BN","BO","BR","BY","BZ","CA",
    "CH","CL","CN","CO","CR","CZ","DE","DK","DO","DZ","EC","EE","EG","ES","FI","FO",
    "FR","GB","GE","GR","GT","HK","HN","HR","HU","ID","IE","IL","IN","IQ","IR","IS",
    "IT","JM","JO","JP","KE","KG","KR","KW","KZ","LB","LI","LT","LU","LV","LY","MA",
    "MC","MK","MN","MO","MV","MX","MY","NI","NL","NO","NZ","OM","PA","PE","PH","PK",
    "PL","PR","PT","PY","QA","RO","RU","SA","SE","SG","SI","SK","SV","SY","TH","TN",
    "TR","TT","TW","UA","US","UY","UZ","VE","VN","YE","ZA",
)

private fun userCountryOptions(): Array<Pair<String, String>> {
    val out = ArrayList<Pair<String, String>>(107)
    var iso = 0
    var value = 1
    while (iso < USER_COUNTRY_ISO.size) {
        if (value == 17 || value == 94) { value++; continue }
        out += value.toString() to USER_COUNTRY_ISO[iso]
        value++; iso++
    }
    return out.toTypedArray()
}

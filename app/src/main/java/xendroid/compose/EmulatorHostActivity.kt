package xendroid.compose

import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.util.Log
import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import xendroid.compose.core.EmuProcessLink
import xendroid.compose.core.EmulatorRuntime
import xendroid.compose.core.EmulatorSession
import xendroid.compose.core.SessionLogs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.lifecycle.lifecycleScope
import android.content.res.Configuration
import android.widget.Toast
import android.os.VibrationEffect
import android.os.Vibrator
import android.preference.PreferenceManager
import androidx.activity.compose.BackHandler
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.viewinterop.AndroidView
import xendroid.compose.ui.theme.xendroidTheme
import xendroid.compose.gamepad.GamepadConfigDto
import xendroid.compose.gamepad.GamepadController
import xendroid.compose.gamepad.GamepadOverlay
import xendroid.compose.gamepad.Kc
import xendroid.compose.gamepad.rememberAutoHide
import xendroid.compose.data.GameButtons
import xendroid.compose.data.KeymapStore
import xendroid.compose.settings.ConfigStore

/**
 * The :emu emulator host (separate process; see manifest). Reads game_uri from the
 * Intent, performs the PRE-surface native setup, hosts a Vulkan SurfaceView, and drives
 * the EXACT surface->boot ordering (SP0 makes destroy/recreate safe). Hardware input
 * (controller + keyboard) only; the on-screen touch pad is SP4. onDestroy hard-kills
 * the process. Mirrors legacy xendroid.compose.EmulatorActivity.
 */
class EmulatorHostActivity : ComponentActivity(), SurfaceHolder.Callback {

    companion object {
        private const val TAG = "EmuHost"
        const val EXTRA_GAME_URI = "game_uri"   // matches GameLibraryViewModel.EXTRA_GAME_URI

        // Default Android-KeyEvent -> VirtualControl KEY_CODE map (KeyMapConfig defaults).
        // Mirrored locally: VirtualControl lives in :app, not on the :app-compose classpath.
        private const val KC_DPAD_LEFT = 0;  private const val KC_DPAD_UP = 1
        private const val KC_DPAD_RIGHT = 2; private const val KC_DPAD_DOWN = 3
        private const val KC_A = 4; private const val KC_B = 5
        private const val KC_X = 6; private const val KC_Y = 7
        private const val KC_BACK = 8; private const val KC_START = 9
        private const val KC_SHOULDER_L = 10; private const val KC_SHOULDER_R = 11
        private const val KC_TRIGGER_L = 14; private const val KC_TRIGGER_R = 15
        private const val KC_LTHUMB_LEFT = 16; private const val KC_LTHUMB_UP = 17
        private const val KC_LTHUMB_RIGHT = 18; private const val KC_LTHUMB_DOWN = 19
        private const val KC_RTHUMB_LEFT = 20; private const val KC_RTHUMB_UP = 21
        private const val KC_RTHUMB_RIGHT = 22; private const val KC_RTHUMB_DOWN = 23
        private const val KEY_VALUE_UNUSED = -1
        private const val FOCUS_PAUSE_DEBOUNCE_MS = 250L
    }

    private val session = EmulatorSession()
    private var surfaceView: SurfaceView? = null
    private var started = false          // surface-callback boot guard (mirrors legacy)

    private val mainHandler = Handler(Looper.getMainLooper())
    private val pauseOnFocusLost = Runnable { if (session.booted) session.pause() }

    // SP4 on-screen touch gamepad.
    private val gamepad by lazy { GamepadController(applicationContext) }
    private var hapticsEnabled = false
    private val bootedState = mutableStateOf(false)   // gates the overlay (post-boot only)
    private val showFpsOverlay = mutableStateOf(false) // Display|show_debug_overlay (native TOML config)
    private val menuOpenState = mutableStateOf(false)  // in-game menu (back pauses + shows Quit)

    // User hardware-key bindings (Android keycode -> game KEY_CODE), loaded from
    // KeymapStore in onCreate; falls back to GameButtons.DEFAULT_LOOKUP until loaded.
    @Volatile private var keyMap: Map<Int, Int> = GameButtons.DEFAULT_LOOKUP
    private var vibrator: Vibrator? = null
    // Edge-detect state for analog triggers reported as axes (emit a digital press on threshold).
    private var lTriggerDown = false
    private var rTriggerDown = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Immersive fullscreen: hide BOTH system bars. The legacy FLAG_FULLSCREEN
        // only hid the status bar, leaving the nav buttons overlaid on the game.
        enterImmersiveMode()

        // Tie this :emu process to the main process: self-kill if the launcher dies
        // (crash / OOM / swipe). Android does not kill this sibling process for us.
        intent?.let { EmuProcessLink.bindToMainProcessDeath(it) }

        val gameUri = intent?.getStringExtra(EXTRA_GAME_URI)
        if (gameUri.isNullOrEmpty()) {
            Log.e(TAG, "No game_uri extra; finishing")
            finish(); return
        }
        if (!EmulatorRuntime.supportsVulkan) {
            Toast.makeText(this, "No Vulkan GPU; cannot boot", Toast.LENGTH_LONG).show()
            finish(); return
        }

        // Real-path (All Files Access) launch: the launch Intent carries an ABSOLUTE host
        // path in the EXTRA_GAME_URI extra (real-path is the only library mode).

        // PRE-surface native setup is async ONLY to ensureLoaded() off-main on delay-load
        // devices; the actual native setup_* calls are marshaled back to the main thread.
        lifecycleScope.launch {
            val store = KeymapStore(applicationContext)
            keyMap = withContext(Dispatchers.IO) {
                EmulatorRuntime.ensureLoaded()            // idempotent; lazy on Adreno 5xx/6xx
                store.androidToGameKey.firstOrNull() ?: GameButtons.DEFAULT_LOOKUP
            }
            // FPS overlay gate: SP2 settings live in the native TOML Config (NOT
            // SharedPreferences). Read the Display|show_debug_overlay boolean off-main
            // via ConfigStore so the existing settings toggle drives overlay visibility.
            showFpsOverlay.value = withContext(Dispatchers.IO) { readShowDebugOverlay() }

            // Guard on the LIVE All Files Access grant (revoked-in-Settings while away =>
            // fail cleanly instead of booting against a path we can no longer read).
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R ||
                !Environment.isExternalStorageManager()) {
                Toast.makeText(this@EmulatorHostActivity,
                    "All Files Access not granted; open the library first",
                    Toast.LENGTH_LONG).show()
                finish(); return@launch
            }
            // Shortcut launches skip the main process; make sure the session
            // logcat capture exists. Shelving stays a main-process concern.
            withContext(Dispatchers.IO) {
                runCatching { SessionLogs.ensureCaptureRunning() }
                    .onFailure { Log.w(TAG, "Session log capture failed", it) }
            }
            prepareNativeRealPath(gameUri)                // back on main (lifecycleScope = Main)
            installSurfaceView()
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            mainHandler.removeCallbacks(pauseOnFocusLost)
            enterImmersiveMode()
            if (session.booted && !menuOpenState.value) session.resumeIfPaused()
        } else {
            mainHandler.removeCallbacks(pauseOnFocusLost)
            mainHandler.postDelayed(pauseOnFocusLost, FOCUS_PAUSE_DEBOUNCE_MS)
        }
    }

    /** Edge-to-edge immersive: hides the status AND navigation bars, with the
     *  bars swipe-to-reveal transiently. */
    private fun enterImmersiveMode() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
    }

    /** Reads Display|show_debug_overlay from the live native TOML config (SP2 store, not
     *  SharedPreferences). Uses a read-only string snapshot so closing the single-use
     *  handle frees the native table WITHOUT re-serializing the file to disk (a plain
     *  read must not mutate the config). Best-effort: any failure defaults the gate off. */
    private fun readShowDebugOverlay(): Boolean = runCatching {
        val handle = ConfigStore(applicationContext).openLiveSnapshot()
        try {
            handle.getBool("Display", "show_debug_overlay", false)
        } finally {
            handle.closeString()
        }
    }.getOrDefault(false)

    /** Real-path (All Files Access) PRE-surface setup: setupContext -> setupGamePathReal
     *  (String setup_game_path overload => native BOOT_TYPE_WITH_PATH / Emulator::LaunchPath,
     *  mounting the right real-path device by extension) -> launch args -> uri info list.
     *  [absPath] is an absolute host path. */
    private fun prepareNativeRealPath(absPath: String) {
        session.setupContext(this)
        session.setupGamePathReal(absPath)
        session.setupLaunchArgs(arrayOf(
            "--storage_root=" + Utils.get_storage_root_path(),
            "--config=" + Application.get_global_config_file().absolutePath,
            "--log_file=" + Utils.get_log_file_path(),
            // One app session = one xe.log: emulator runs append, the shelver
            // rotates at the next app-session start.
            "--log_append=true",
        ))
        session.setupUriInfoListFile(Application.get_uri_info_list_file().absolutePath)
    }

    private fun installSurfaceView() {
        val sv = SurfaceView(this).apply {
            isFocusable = true
            isFocusableInTouchMode = true
            holder.addCallback(this@EmulatorHostActivity)
            setOnGenericMotionListener { _, ev -> onGenericMotion(ev) }
        }
        surfaceView = sv
        val compose = ComposeView(this).apply {
            setContent {
                val cfg by gamepad.config.collectAsState(initial = GamepadConfigDto())
                val landscape = resources.configuration.orientation ==
                    Configuration.ORIENTATION_LANDSCAPE
                // Recompose-trigger for boot gate: poll booted via a state hoisted in the host.
                val booted by bootedState
                val controls = remember(cfg, landscape) { gamepad.controlsFor(cfg, landscape) }
                val (visible, poke) = rememberAutoHide(cfg.globals.autoHideSeconds)
                val alpha by animateFloatAsState(
                    if (visible) cfg.globals.opacity else 0f, tween(500), label = "padAlpha")

                // Apply haptics pref each composition (cheap; reads cached field).
                LaunchedEffect(cfg.globals.hapticsEnabled) {
                    configureHaptics(cfg.globals.hapticsEnabled)
                }
                Box(Modifier.fillMaxSize()) {
                    AndroidView(factory = { sv }, modifier = Modifier.fillMaxSize())
                    // Stay MOUNTED whenever enabled (alpha drives only the DRAW): the
                    // pointerInput must keep receiving touches so the auto-hide wake tap
                    // fires, and so a held control is never unmounted mid-press (stuck).
                    if (booted && cfg.globals.enabled) {
                        GamepadOverlay(
                            controls = controls,
                            opacity = alpha,
                            onUserInteraction = poke,
                            onKeyEvent = { kc, pressed, v ->
                                if (pressed && v == Kc.VALUE_UNUSED) maybeVibrate()
                                session.keyEvent(kc, pressed, v)
                            },
                            modifier = Modifier.fillMaxSize(),
                        )
                    }
                    val showFps by showFpsOverlay
                    // The onCreate read of show_debug_overlay only saw the GLOBAL config;
                    // xenia applies any per-game override on the detached boot thread, so
                    // POLL the effective native cvar after boot. Without this, "global off
                    // + per-game on" never shows the overlay.
                    LaunchedEffect(booted) {
                        if (!booted) return@LaunchedEffect
                        while (isActive) {
                            showFpsOverlay.value = session.showDebugOverlayEnabled()
                            delay(1000)
                        }
                    }
                    FpsOverlay(
                        session = session,
                        visible = booted && showFps,
                        modifier = Modifier.fillMaxSize(),
                    )
                }

                // In-game menu: back / swipe-back PAUSES the game and opens a menu (Quit) instead
                // of leaving to the library. The dialog itself catches back / tap-outside -> resume.
                val menuOpen by menuOpenState
                BackHandler(enabled = !menuOpen) {
                    menuOpenState.value = true
                    if (session.booted) session.pause()
                }
                if (menuOpen) xendroidTheme {
                    AlertDialog(
                        onDismissRequest = {
                            menuOpenState.value = false
                            if (session.booted) session.resumeIfPaused()
                        },
                        title = { Text("Paused") },
                        confirmButton = { Button(onClick = { finish() }) { Text("Quit") } },
                    )
                }
            }
        }
        setContentView(compose)
        sv.requestFocus()
    }

    /** Honor the SAME legacy enable_vibrator pref AND the new globals flag (new wins when
     *  set, else fall back to legacy so existing users keep their setting). */
    private fun configureHaptics(globalsFlag: Boolean) {
        val legacy = PreferenceManager.getDefaultSharedPreferences(this)
            .getBoolean("enable_vibrator", false)
        hapticsEnabled = globalsFlag || legacy
        if (hapticsEnabled && vibrator == null)
            vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
    }
    private fun maybeVibrate() {
        if (!hapticsEnabled) return
        vibrator?.vibrate(VibrationEffect.createOneShot(25, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    // ---- SurfaceHolder.Callback: the load-bearing surface->boot ordering ----

    override fun surfaceCreated(holder: SurfaceHolder) {
        if (!started) {
            started = true
            session.attachSurface(holder.surface)
            try {
                session.bootOnce()
                bootedState.value = true                  // reveal the SP4 overlay post-boot
            } catch (t: RuntimeException) {
                Log.e(TAG, "boot failed", t)
                finish()                                  // fatal; single-shot core
            }
        } else {
            // Post-rotation/background re-create: re-attach, resume; NEVER boot again.
            // Stay paused if the in-game menu is open.
            session.attachSurface(holder.surface)
            if (!menuOpenState.value) session.resumeIfPaused()
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        if (!started) return
        if (width == 0 || height == 0) return
        session.changeSurface(width, height)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        if (!started) return
        // surfaceDestroyed fires BEFORE onStop here: pause first so the drain
        // below tears down a quiescent GPU. onStop's pause() is then a no-op.
        if (session.booted) session.pause()
        session.detachSurface()                           // synchronous GPU drain (SP0)
    }

    // ---- Lifecycle teardown ----

    override fun onPause() {
        super.onPause()
        session.flushGpuCaches()                          // best-effort; never throws
    }

    override fun onStop() {
        super.onStop()
        // Freeze guest CPU + GPU command worker + audio (audio goes silent) when the
        // activity is stopped (screen sleep / home / app switch). Direct, blocking,
        // main-thread call (Emulator::Pause does its own fences). Gated on booted so a
        // stop during the boot splash is a no-op, matching the keyEvent !booted guard.
        // Ordering: onPause(flushGpuCaches) -> onStop(pause) -> surfaceDestroyed(detach);
        // pausing BEFORE the swapchain teardown means no guest/GPU frames race the drain.
        mainHandler.removeCallbacks(pauseOnFocusLost)
        if (session.booted) session.pause()
    }

    override fun onStart() {
        super.onStart()
        // Mirror of onStop. resumeIfPaused() (not bare resume) stays idempotent: the
        // swapchain-recreate path already calls resumeIfPaused() in surfaceCreated
        // (line 246), so this second call is a no-op there; onStart additionally covers
        // the pure screen-sleep case where the surface was NOT destroyed. Stay paused if the
        // in-game menu is open (don't run the game behind the menu after returning).
        if (session.booted && !menuOpenState.value) session.resumeIfPaused()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Hard-kill via SIGKILL (single-shot core). killProcess skips the C++ atexit
        // static-destructor path that System.exit(0) ran, which can deadlock joining a
        // paused audio worker and wedge :emu instead of closing it.
        Process.killProcess(Process.myPid())
    }

    // ---- Hardware input -> session.keyEvent ----

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        val gameKey = keyMap[keyCode] ?: return super.onKeyDown(keyCode, event)
        if (event.repeatCount == 0) {
            session.keyEvent(gameKey, true, KEY_VALUE_UNUSED)
            return true
        }
        return super.onKeyDown(keyCode, event)            // ignore auto-repeats
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        val gameKey = keyMap[keyCode] ?: return super.onKeyUp(keyCode, event)
        session.keyEvent(gameKey, false, KEY_VALUE_UNUSED)
        return true
    }

    /** Joystick axes + hat (D-pad). Mirrors EmulatorActivity.onGenericMotion/handle_dpad. */
    private fun onGenericMotion(event: MotionEvent): Boolean {
        if (isNonDpadSource(event) && handleHat(event)) return true
        if (event.source and InputDevice.SOURCE_JOYSTICK != InputDevice.SOURCE_JOYSTICK) {
            return super.onGenericMotionEvent(event)
        }
        // Left stick
        emitAxisPair(event.getAxisValue(MotionEvent.AXIS_X),
            negKey = KC_LTHUMB_LEFT, posKey = KC_LTHUMB_RIGHT, invert = false)
        emitAxisPair(event.getAxisValue(MotionEvent.AXIS_Y),
            negKey = KC_LTHUMB_UP, posKey = KC_LTHUMB_DOWN, invert = true)   // Y inverted
        // Right stick (Z / RZ)
        emitAxisPair(event.getAxisValue(MotionEvent.AXIS_Z),
            negKey = KC_RTHUMB_LEFT, posKey = KC_RTHUMB_RIGHT, invert = false)
        emitAxisPair(event.getAxisValue(MotionEvent.AXIS_RZ),
            negKey = KC_RTHUMB_UP, posKey = KC_RTHUMB_DOWN, invert = true)
        // Triggers: digital game keys, but most pads report them as ANALOG axes (LTRIGGER/RTRIGGER,
        // or BRAKE/GAS on others). Emit a press past the threshold, edge-detected so we don't spam.
        lTriggerDown = emitTrigger(
            maxOf(event.getAxisValue(MotionEvent.AXIS_LTRIGGER), event.getAxisValue(MotionEvent.AXIS_BRAKE)),
            KC_TRIGGER_L, lTriggerDown)
        rTriggerDown = emitTrigger(
            maxOf(event.getAxisValue(MotionEvent.AXIS_RTRIGGER), event.getAxisValue(MotionEvent.AXIS_GAS)),
            KC_TRIGGER_R, rTriggerDown)
        return true
    }

    /** Analog trigger -> digital game key. Emits only on the down/up edge (>0.5 = down). */
    private fun emitTrigger(value: Float, gameKey: Int, wasDown: Boolean): Boolean {
        val down = value > 0.5f
        if (down != wasDown) session.keyEvent(gameKey, down, KEY_VALUE_UNUSED)
        return down
    }

    /** For one axis emit the opposing thumb directions, scaled to signed-short range.
     *  negative -> negKey pressed with |v|, posKey released; positive -> posKey; 0 -> both released.
     *  invert flips sign first (screen Y is up-negative; X360 up is positive). */
    private fun emitAxisPair(axis: Float, negKey: Int, posKey: Int, invert: Boolean) {
        val v = if (invert) -axis else axis
        when {
            v < 0f -> {
                session.keyEvent(posKey, false, 0)
                session.keyEvent(negKey, true, (v * 32768f).toInt())          // negative magnitude
            }
            v > 0f -> {
                session.keyEvent(negKey, false, 0)
                session.keyEvent(posKey, true, (v * 32767f).toInt())
            }
            else -> {
                session.keyEvent(negKey, false, 0)
                session.keyEvent(posKey, false, 0)
            }
        }
    }

    /** Hat axes -> D-pad. Returns true if any direction is pressed. */
    private fun handleHat(event: MotionEvent): Boolean {
        var pressed = false
        val hx = event.getAxisValue(MotionEvent.AXIS_HAT_X)
        val hy = event.getAxisValue(MotionEvent.AXIS_HAT_Y)
        when {
            hx == -1f -> { dpad(KC_DPAD_LEFT, KC_DPAD_RIGHT); pressed = true }
            hx == 1f  -> { dpad(KC_DPAD_RIGHT, KC_DPAD_LEFT); pressed = true }
        }
        when {
            hy == -1f -> { dpad(KC_DPAD_UP, KC_DPAD_DOWN); pressed = true }
            hy == 1f  -> { dpad(KC_DPAD_DOWN, KC_DPAD_UP); pressed = true }
        }
        if (pressed) return true
        // No hat direction -> release all four (legacy behavior).
        session.keyEvent(KC_DPAD_LEFT, false, KEY_VALUE_UNUSED)
        session.keyEvent(KC_DPAD_UP, false, KEY_VALUE_UNUSED)
        session.keyEvent(KC_DPAD_RIGHT, false, KEY_VALUE_UNUSED)
        session.keyEvent(KC_DPAD_DOWN, false, KEY_VALUE_UNUSED)
        return false
    }

    private fun dpad(pressCode: Int, releaseCode: Int) {
        session.keyEvent(pressCode, true, KEY_VALUE_UNUSED)
        session.keyEvent(releaseCode, false, KEY_VALUE_UNUSED)
    }

    /** Legacy isDpadDevice: TRUE when the device is NOT a SOURCE_DPAD (i.e. treat as
     *  joystick/hat). Name kept descriptive here. */
    private fun isNonDpadSource(event: MotionEvent): Boolean =
        event.source and InputDevice.SOURCE_DPAD != InputDevice.SOURCE_DPAD
}

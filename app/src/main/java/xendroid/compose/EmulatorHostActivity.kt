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
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import xendroid.compose.core.EmuProcessLink
import xendroid.compose.core.EmulatorRuntime
import xendroid.compose.core.FrontendLaunch
import xendroid.compose.core.EmulatorSession
import xendroid.compose.core.ScreenBrightnessSampler
import xendroid.compose.core.SessionLogs
import kotlin.math.abs
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
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import kotlin.math.roundToInt
import kotlinx.coroutines.flow.StateFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.viewinterop.AndroidView
import xendroid.compose.ui.disc.DiscSwapPanel
import xendroid.compose.ui.keyboard.GuestKeyboardPanel
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
        private const val KEYBOARD_POLL_MS = 150L
        const val EXTRA_GAME_URI = "game_uri"   // matches GameLibraryViewModel.EXTRA_GAME_URI
        // matches GameLibraryViewModel
        const val EXTRA_DISC_LABELS = "disc_labels"
        const val EXTRA_DISC_PATHS = "disc_paths"

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
        // Analog hardware never reports exactly 0; without this the drift
        // crosses into native on every sample.
        private const val AXIS_DEADZONE = 0.08f
        private const val FOCUS_PAUSE_DEBOUNCE_MS = 250L
    }

    private val session = EmulatorSession()
    private var surfaceView: SurfaceView? = null
    private var started = false          // surface-callback boot guard (mirrors legacy)

    private val mainHandler = Handler(Looper.getMainLooper())
    private val pauseOnFocusLost = Runnable { if (session.booted) session.pause() }

    // SP4 on-screen touch gamepad.
    private val gamepad by lazy { GamepadController(applicationContext) }
    private val brightnessSampler = ScreenBrightnessSampler()
    @Volatile private var overlayWantsBrightness = false   // sampler should run while visible
    private var hapticsEnabled = false
    private val bootedState = mutableStateOf(false)   // gates the overlay (post-boot only)
    private val showFpsOverlay = mutableStateOf(false) // Display|show_debug_overlay (native TOML config)
    private val menuOpenState = mutableStateOf(false)  // in-game menu (back pauses + shows Quit)
    private val keyboardRequestState =
        mutableStateOf<Emulator.KeyboardRequest?>(null) // guest text-entry prompt, if any
    private val discRequestState =
        mutableStateOf<Emulator.DiscSwapRequest?>(null) // guest disc-swap prompt, if any

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

        // Tie this :emu process to the main process: keep the launcher out of the
        // cached band (so lmkd stops reaping it mid-game) and notice if it dies anyway.
        EmuProcessLink.bindToMainProcess(this)

        // In-app game_uri extra, or frontend shapes (AutoStartFile, data URI).
        val gameUri = FrontendLaunch.resolveGamePath(this, intent)
        if (gameUri.isNullOrEmpty()) {
            Log.e(TAG, "No bootable game in launch intent; finishing")
            Toast.makeText(this, "XenDroid: no game in launch intent", Toast.LENGTH_LONG).show()
            finish(); return
        }
        if (!EmulatorRuntime.supportsVulkan) {
            Toast.makeText(this, "No Vulkan GPU; cannot boot", Toast.LENGTH_LONG).show()
            finish(); return
        }

        // Real-path (All Files Access) launch: the resolved gameUri is an ABSOLUTE
        // host path (real-path is the only library mode).

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
        // Gamepad play produces no touch events, so the display still times out.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
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

                // Adaptive contrast: on bright scenes the overlay tints toward grey so it stays visible.
                val overlayActive = booted && cfg.globals.enabled
                // onStart/onStop co-own the sampler thread: no PixelCopy polling while backgrounded.
                DisposableEffect(overlayActive) {
                    overlayWantsBrightness = overlayActive
                    if (overlayActive) brightnessSampler.start(sv)
                    onDispose { brightnessSampler.stop() }
                }
                val contrastState = rememberOverlayContrast(brightnessSampler.brightness)

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
                            contrast = { contrastState.value },
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

                    // The emulator blocks a dispatch thread until answered.
                    val keyboardRequest by keyboardRequestState
                    LaunchedEffect(booted) {
                        if (!booted) return@LaunchedEffect
                        while (isActive) {
                            if (keyboardRequestState.value == null) {
                                keyboardRequestState.value = session.keyboardRequest()
                            }
                            delay(KEYBOARD_POLL_MS)
                        }
                    }
                    keyboardRequest?.let { req ->
                        xendroidTheme {
                            GuestKeyboardPanel(
                                request = req,
                                onAccept = { text ->
                                    session.keyboardSubmit(req.id, true, text)
                                    keyboardRequestState.value = null
                                },
                                onCancel = {
                                    session.keyboardSubmit(req.id, false, "")
                                    keyboardRequestState.value = null
                                },
                                modifier = Modifier.fillMaxSize(),
                            )
                        }
                    }

                    // Blocks the guest thread that asked, until answered.
                    val discRequest by discRequestState
                    LaunchedEffect(booted) {
                        if (!booted) return@LaunchedEffect
                        while (isActive) {
                            if (discRequestState.value == null) {
                                discRequestState.value = session.discRequest()
                            }
                            delay(KEYBOARD_POLL_MS)
                        }
                    }
                    discRequest?.let { req ->
                        xendroidTheme {
                            DiscSwapPanel(
                                request = req,
                                onChoose = { path ->
                                    session.discSubmit(req.id, true, path)
                                    discRequestState.value = null
                                },
                                onCancel = {
                                    session.discSubmit(req.id, false, "")
                                    discRequestState.value = null
                                },
                                modifier = Modifier.fillMaxSize(),
                            )
                        }
                    }
                }

                // In-game menu: back / swipe-back PAUSES the game and opens a menu (Quit) instead
                // of leaving to the library. The dialog itself catches back / tap-outside -> resume.
                val menuOpen by menuOpenState
                // Back cancels a text prompt instead of opening the menu. The gate below
                // is what gives it priority; without it back pauses behind the prompt.
                val keyboardOpen = keyboardRequestState.value != null
                val discOpen = discRequestState.value != null
                BackHandler(enabled = keyboardOpen) {
                    keyboardRequestState.value?.let { req ->
                        session.keyboardSubmit(req.id, false, "")
                        keyboardRequestState.value = null
                    }
                }
                BackHandler(enabled = discOpen) {
                    discRequestState.value?.let { req ->
                        session.discSubmit(req.id, false, "")
                        discRequestState.value = null
                    }
                }
                BackHandler(enabled = !menuOpen && !keyboardOpen && !discOpen) {
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
                val labels = intent.getStringArrayExtra(EXTRA_DISC_LABELS)
                val paths = intent.getStringArrayExtra(EXTRA_DISC_PATHS)
                if (labels != null && paths != null && labels.isNotEmpty()) {
                    session.discSetKnown(labels.toList(), paths.toList())
                }
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
        brightnessSampler.stop()                          // no PixelCopy polling while stopped
        // Last: an orphaned core (main process died while we were on screen) hard-kills
        // here, so it does so from a paused, quiescent state.
        EmuProcessLink.setEmuForeground(false)
    }

    override fun onStart() {
        super.onStart()
        EmuProcessLink.setEmuForeground(true)
        if (overlayWantsBrightness) surfaceView?.let { brightnessSampler.start(it) }
        // Mirror of onStop. resumeIfPaused() (not bare resume) stays idempotent: the
        // swapchain-recreate path already calls resumeIfPaused() in surfaceCreated
        // (line 246), so this second call is a no-op there; onStart additionally covers
        // the pure screen-sleep case where the surface was NOT destroyed. Stay paused if the
        // in-game menu is open (don't run the game behind the menu after returning).
        if (session.booted && !menuOpenState.value) session.resumeIfPaused()
    }

    override fun onDestroy() {
        super.onDestroy()
        // A pending prompt must not hold a dispatch thread through teardown.
        keyboardRequestState.value = null
        session.keyboardCancelAll()
        discRequestState.value = null
        session.discCancelAll()
        // Hard-kill via SIGKILL (single-shot core). killProcess skips the C++ atexit
        // static-destructor path that System.exit(0) ran, which can deadlock joining a
        // paused audio worker and wedge :emu instead of closing it.
        Process.killProcess(Process.myPid())
    }

    // ---- Hardware input -> session.keyEvent ----

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        val gameKey = keyMap[keyCode]
            ?: return consumeIfGamepad(event) || super.onKeyDown(keyCode, event)
        if (event.repeatCount == 0) {
            session.keyEvent(gameKey, true, KEY_VALUE_UNUSED)
            return true
        }
        return super.onKeyDown(keyCode, event)            // ignore auto-repeats
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        val gameKey = keyMap[keyCode]
            ?: return consumeIfGamepad(event) || super.onKeyUp(keyCode, event)
        session.keyEvent(gameKey, false, KEY_VALUE_UNUSED)
        return true
    }

    /** Unmapped controller buttons (never BACK): unhandled gamepad input is what OEM
     *  overlays latch onto. */
    private fun consumeIfGamepad(event: KeyEvent): Boolean =
        event.keyCode != KeyEvent.KEYCODE_BACK &&
            (event.source and InputDevice.SOURCE_GAMEPAD == InputDevice.SOURCE_GAMEPAD ||
                event.source and InputDevice.SOURCE_JOYSTICK == InputDevice.SOURCE_JOYSTICK)

    /** The SurfaceView listener only fires while that view holds focus. */
    override fun onGenericMotionEvent(event: MotionEvent): Boolean = onGenericMotion(event)

    /** Joystick axes + hat (D-pad). Mirrors EmulatorActivity.onGenericMotion/handle_dpad. */
    private fun onGenericMotion(event: MotionEvent): Boolean {
        // No early return: sticks/triggers must still process while a hat is held.
        val hatHandled = isNonDpadSource(event) && handleHat(event)
        if (event.source and InputDevice.SOURCE_JOYSTICK != InputDevice.SOURCE_JOYSTICK) {
            return hatHandled || super.onGenericMotionEvent(event)
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
        val raw = if (invert) -axis else axis
        // Snap to exactly zero so emitAxis' equality check can match.
        val v = if (abs(raw) < AXIS_DEADZONE) 0f else raw
        when {
            v < 0f -> {
                emitAxis(posKey, false, 0)
                emitAxis(negKey, true, (v * 32768f).toInt())                  // negative magnitude
            }
            v > 0f -> {
                emitAxis(negKey, false, 0)
                emitAxis(posKey, true, (v * 32767f).toInt())
            }
            else -> {
                emitAxis(negKey, false, 0)
                emitAxis(posKey, false, 0)
            }
        }
    }

    // Last value pushed per analog code. Motion events arrive per sample (~120Hz x 4 axes)
    // and mostly repeat, so without this every sample costs ~10 JNI calls.
    private val axisPressed = BooleanArray(24)
    private val axisValue = IntArray(24) { Int.MIN_VALUE }

    private fun emitAxis(code: Int, pressed: Boolean, value: Int) {
        if (axisPressed[code] == pressed && axisValue[code] == value) return
        axisPressed[code] = pressed
        axisValue[code] = value
        session.keyEvent(code, pressed, value)
    }

    // Hat D-pad state, edge-detected: the hat only releases what IT pressed, so
    // it can't clobber a D-pad held via real KEYCODE_DPAD_* key events.
    private var hatLeft = false
    private var hatUp = false
    private var hatRight = false
    private var hatDown = false

    /** Hat axes -> D-pad; thresholds, not ==+-1f (some pads are inexact). */
    private fun handleHat(event: MotionEvent): Boolean {
        val hx = event.getAxisValue(MotionEvent.AXIS_HAT_X)
        val hy = event.getAxisValue(MotionEvent.AXIS_HAT_Y)
        val left = hx < -0.5f
        val right = hx > 0.5f
        val up = hy < -0.5f
        val down = hy > 0.5f
        if (left != hatLeft) {
            session.keyEvent(KC_DPAD_LEFT, left, KEY_VALUE_UNUSED); hatLeft = left
        }
        if (right != hatRight) {
            session.keyEvent(KC_DPAD_RIGHT, right, KEY_VALUE_UNUSED); hatRight = right
        }
        if (up != hatUp) {
            session.keyEvent(KC_DPAD_UP, up, KEY_VALUE_UNUSED); hatUp = up
        }
        if (down != hatDown) {
            session.keyEvent(KC_DPAD_DOWN, down, KEY_VALUE_UNUSED); hatDown = down
        }
        return left || right || up || down
    }

    /** Legacy isDpadDevice: TRUE when the device is NOT a SOURCE_DPAD (i.e. treat as
     *  joystick/hat). Name kept descriptive here. */
    private fun isNonDpadSource(event: MotionEvent): Boolean =
        event.source and InputDevice.SOURCE_DPAD != InputDevice.SOURCE_DPAD
}

/** Brightness -> overlay contrast, collected in a coroutine so emissions never recompose;
 *  consumers read the State in the draw phase only. The target is quantized to 1/8 steps so
 *  EMA jitter re-targets the same value instead of restarting the tween forever. */
@Composable
private fun rememberOverlayContrast(brightness: StateFlow<Float>): State<Float> {
    val anim = remember { Animatable(0f) }
    LaunchedEffect(brightness) {
        brightness.collect { b ->
            // Below ~0.5 luminance the plain overlay reads fine; ramp to full by ~0.9.
            val target = ((b - 0.5f) / 0.4f).coerceIn(0f, 1f).let { (it * 8f).roundToInt() / 8f }
            if (target != anim.targetValue) {
                launch { anim.animateTo(target, tween(400)) }   // MutatorMutex cancels the in-flight tween
            }
        }
    }
    return anim.asState()
}

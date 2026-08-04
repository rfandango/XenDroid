package xendroid.compose.core

import android.app.ActivityManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.os.Process
import androidx.core.content.getSystemService
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Ties the single-shot :emu render process to the MAIN process.
 *
 * :emu is a separate OS process forked from the zygote, so it does NOT die when the
 * main process is closed, crashes, or is OOM-killed (Android never kills sibling
 * processes for you). :emu binds [MainAliveService], which lives in the main process,
 * and that one binding does both halves of the job:
 *
 *  - BIND_IMPORTANT raises the main process to :emu's process state. Without it the
 *    launcher sits in the CACHED band for the whole session, making it the first thing
 *    lmkd reaps under memory pressure - and since lmkd picks by oom_score_adj, not by
 *    size, a 36MB launcher dies long before the 4GB game it launched.
 *  - onServiceDisconnected/onBindingDied is the liveness signal: it fires when the main
 *    process goes away, whatever the cause.
 *
 * Losing main is only fatal while :emu is NOT on screen. Hard-killing a game the user
 * is actively playing because the launcher was reaped is never the right answer; the
 * orphan this link guards against is a backgrounded/wedged core, and [killStaleEmu] is
 * the second backstop for that. So a death that arrives mid-game is remembered and
 * collected at the next onStop instead.
 */
object EmuProcessLink {
    /** :emu only, held for the process lifetime (the connection outlives the Activity). */
    private var appContext: Context? = null

    /** :emu only: mirrors the host Activity's started/stopped state. */
    private val emuForeground = AtomicBoolean(false)

    /** :emu only: the main process has died and this core is now an orphan. */
    private val mainGone = AtomicBoolean(false)

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) = Unit

        override fun onServiceDisconnected(name: ComponentName?) = onMainProcessGone()

        override fun onBindingDied(name: ComponentName?) = onMainProcessGone()
    }

    /** :emu process: bind for this process's lifetime. Call once from onCreate; a
     *  failed bind just leaves :emu unlinked, as the shortcut path always was. */
    fun bindToMainProcess(context: Context) {
        val ctx = context.applicationContext
        appContext = ctx
        runCatching {
            ctx.bindService(
                Intent(ctx, MainAliveService::class.java),
                connection,
                Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT,
            )
        }
    }

    /** :emu process: mirror of the host Activity's onStart/onStop. Backgrounding is
     *  where a main-process death that arrived mid-game is finally collected. */
    fun setEmuForeground(foreground: Boolean) {
        emuForeground.set(foreground)
        if (!foreground && mainGone.get()) Process.killProcess(Process.myPid())
    }

    private fun onMainProcessGone() {
        mainGone.set(true)
        // Stop BIND_AUTO_CREATE respawning the launcher we just outlived: main is
        // usually gone from memory pressure, and its onCreate re-runs the Vulkan probe.
        runCatching { appContext?.unbindService(connection) }
        // On screen: the user is playing. Keep going; onStop collects this.
        if (!emuForeground.get()) Process.killProcess(Process.myPid())
    }

    /** MAIN process: kill any leftover :emu before launching, so a wedged/orphaned
     *  single-shot core never blocks the next boot. */
    fun killStaleEmu(context: Context) {
        val am = context.getSystemService<ActivityManager>() ?: return
        val mine = Process.myPid()
        runCatching {
            am.runningAppProcesses.orEmpty()
                .filter { it.processName.endsWith(":emu") && it.pid != mine }
                .forEach { Process.killProcess(it.pid) }
        }
    }
}

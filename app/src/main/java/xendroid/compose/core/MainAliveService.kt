package xendroid.compose.core

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder

/**
 * Empty Service in the main process, bound by :emu for its side effects only: the
 * binding pins main out of the cached band, and its death signals main is gone.
 * See [EmuProcessLink].
 */
class MainAliveService : Service() {
    override fun onBind(intent: Intent?): IBinder = Binder()
}

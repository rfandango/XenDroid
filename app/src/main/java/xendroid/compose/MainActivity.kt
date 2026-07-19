package xendroid.compose

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import xendroid.compose.core.SessionLogs
import xendroid.compose.ui.AppNavHost
import xendroid.compose.ui.theme.xendroidTheme
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {
    companion object {
        // One rotation per main-process lifetime = the app-session boundary.
        private val sessionRotated = AtomicBoolean(false)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!sessionRotated.getAndSet(true)) {
            val appContext = applicationContext
            thread(name = "SessionLogs") {
                runCatching { SessionLogs.startAppSession(appContext) }
            }
        }
        // Application.onCreate already ran the GPU probe + (on Adreno 830) eager
        // libe.so load. Build the manual-DI container off applicationContext.
        val container = AppContainer(applicationContext)
        enableEdgeToEdge()
        setContent {
            xendroidTheme {
                // Each screen owns its own Scaffold/TopAppBar chrome (GameLibraryScreen
                // already does), so no outer Scaffold here — avoids double insets.
                AppNavHost(container)
            }
        }
    }
}

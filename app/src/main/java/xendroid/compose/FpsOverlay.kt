package xendroid.compose

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.material3.Text
import kotlinx.coroutines.delay
import java.util.Locale
import kotlin.math.roundToInt
import xendroid.compose.core.EmulatorSession

/**
 * Small, draggable FPS / frame-time readout drawn over the Vulkan SurfaceView. Shows a RenderDoc-
 * style ~1s average fps alongside the INSTANT frame time. Polls the native lock-free frame-stats
 * atomics at [pollHz].
 *
 * The user can drag it anywhere; its position is persisted in a :emu-process SharedPreferences (so
 * the same DataStore the main process uses is never touched cross-process). Visibility is owned by
 * the caller via [visible] (the show_debug_overlay setting).
 */
@Composable
fun FpsOverlay(
    session: EmulatorSession,
    visible: Boolean,
    modifier: Modifier = Modifier,
    pollHz: Int = 4,
) {
    if (!visible) return

    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("fps_overlay", Context.MODE_PRIVATE) }
    var offset by remember { mutableStateOf(Offset(prefs.getFloat("x", 0f), prefs.getFloat("y", 0f))) }
    var boxSize by remember { mutableStateOf(IntSize.Zero) }
    var fps by remember { mutableStateOf(0.0) }
    var frameMs by remember { mutableStateOf(0.0) }

    // ~4 Hz poll (250 ms) — enough for a human-readable readout while costing ~nothing.
    LaunchedEffect(pollHz) {
        val periodMs = 1000L / pollHz.coerceIn(1, 30)
        while (true) {
            fps = session.averageFps()
            frameMs = session.lastFrameTimeMs()
            delay(periodMs)
        }
    }

    Box(modifier.fillMaxSize().onSizeChanged { boxSize = it }) {
        Text(
            text = String.format(Locale.US, "FPS %.0f  ·  %.1f ms", fps, frameMs),
            color = Color.White.copy(alpha = 0.7f),       // soft, more transparent than the old yellow
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier
                .offset { IntOffset(offset.x.roundToInt(), offset.y.roundToInt()) }
                .background(Color.Black.copy(alpha = 0.28f), RoundedCornerShape(4.dp))
                .padding(horizontal = 5.dp, vertical = 1.dp)
                .pointerInput(boxSize) {
                    detectDragGestures(
                        onDrag = { change, drag ->
                            change.consume()
                            // Clamp so the readout stays fully on-screen (size = this element's px).
                            val maxX = maxOf(0f, boxSize.width.toFloat() - size.width)
                            val maxY = maxOf(0f, boxSize.height.toFloat() - size.height)
                            offset = Offset(
                                (offset.x + drag.x).coerceIn(0f, maxX),
                                (offset.y + drag.y).coerceIn(0f, maxY),
                            )
                        },
                        onDragEnd = {
                            prefs.edit().putFloat("x", offset.x).putFloat("y", offset.y).apply()
                        },
                    )
                },
        )
    }
}

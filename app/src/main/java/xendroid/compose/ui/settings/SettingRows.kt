package xendroid.compose.ui.settings

import android.app.Activity
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import xendroid.compose.Utils
import xendroid.compose.core.SessionLogs
import xendroid.compose.settings.Setting
import xendroid.compose.settings.SettingsHost

@Composable
fun SettingRow(host: SettingsHost, s: Setting, modified: Boolean, raw: String? = null) = when (s) {
    is Setting.Bool       -> BoolRow(host, s, modified)
    is Setting.IntRange   -> IntRow(host, s, modified)
    is Setting.ListChoice -> ListRow(host, s, modified)
    is Setting.Action     ->
        if (s.name == "dump_session_logs") ExportLogsRow(s)
        else DriverActionRow(host, s, modified, raw)
}

@Composable
private fun titleColor(modified: Boolean) =
    if (modified) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.onSurface

@Composable
private fun RowTitle(text: String, modified: Boolean, sub: String? = null) {
    Column {
        Text(text, color = titleColor(modified), style = MaterialTheme.typography.bodyLarge)
        if (sub != null) Text(sub, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun BoolRow(host: SettingsHost, s: Setting.Bool, modified: Boolean) {
    val checked = remember(modified) { host.currentBool(s) }
    var local by remember { mutableStateOf(checked) }
    Row(
        Modifier.fillMaxWidth().clickable { local = !local; host.onBoolChanged(s, local) }
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.weight(1f)) { RowTitle(s.title, modified) }
        Switch(checked = local, onCheckedChange = { local = it; host.onBoolChanged(s, it) })
    }
}

@Composable
private fun IntRow(host: SettingsHost, s: Setting.IntRange, modified: Boolean) {
    var showDialog by remember { mutableStateOf(false) }
    // Read live (like InheritedPreview); a remember(modified) cache stays stale where modified is
    // constant (the per-game editor always passes true).
    val current = host.currentInt(s)
    Row(Modifier.fillMaxWidth().clickable { showDialog = true }
        .padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.weight(1f)) { RowTitle(s.title, modified, sub = current.toString()) }
    }
    if (showDialog) {
        var slider by remember { mutableFloatStateOf(current.coerceIn(s.min, s.max).toFloat()) }
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(s.title) },
            text = {
                Column {
                    Text(slider.toInt().toString(), style = MaterialTheme.typography.titleLarge)
                    Slider(value = slider, onValueChange = { slider = it },
                        valueRange = s.min.toFloat()..s.max.toFloat(),
                        steps = (s.max - s.min - 1).coerceAtLeast(0))
                }
            },
            confirmButton = { TextButton(onClick = {
                host.onIntChanged(s, slider.toInt()); showDialog = false
            }) { Text("OK") } },
            dismissButton = { TextButton(onClick = { showDialog = false }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun ListRow(host: SettingsHost, s: Setting.ListChoice, modified: Boolean) {
    var showDialog by remember { mutableStateOf(false) }
    val currentValue = host.currentListValue(s)   // read live (see IntRow)
    val currentLabel = s.options.firstOrNull { it.value == currentValue }?.label
        ?: if (currentValue.isEmpty()) "(default)" else currentValue
    Row(Modifier.fillMaxWidth().clickable { showDialog = true }
        .padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.weight(1f)) { RowTitle(s.title, modified, sub = currentLabel) }
    }
    if (showDialog) {
        val listState = rememberLazyListState()
        val selectedIndex = s.options.indexOfFirst { it.value == currentValue }
        LaunchedEffect(Unit) { if (selectedIndex > 0) listState.scrollToItem(selectedIndex) }
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(s.title) },
            text = {
                LazyColumn(state = listState, modifier = Modifier.fillMaxWidth().heightIn(max = 400.dp)) {
                    items(s.options, key = { it.value }) { opt ->
                        Row(Modifier.fillMaxWidth().selectable(
                            selected = opt.value == currentValue,
                            onClick = { host.onListChanged(s, opt.value); showDialog = false },
                        ).padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                            RadioButton(selected = opt.value == currentValue, onClick = {
                                host.onListChanged(s, opt.value); showDialog = false })
                            Spacer(Modifier.width(8.dp)); Text(opt.label)
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { showDialog = false }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun ExportLogsRow(s: Setting.Action) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var busy by remember { mutableStateOf(false) }
    Row(
        Modifier.fillMaxWidth().clickable(enabled = !busy) {
            busy = true
            scope.launch {
                val dest = withContext(Dispatchers.IO) {
                    runCatching { SessionLogs.exportAll() }.getOrNull()
                }
                Toast.makeText(context,
                    dest?.let { "Logs exported to ${it.name} in Downloads" }
                        ?: "No logs to export",
                    Toast.LENGTH_LONG).show()
                busy = false
            }
        }.padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.weight(1f)) {
            RowTitle(s.title, false,
                sub = if (busy) "Exporting..." else "Shelved sessions + current run")
        }
    }
}

@Composable
private fun DriverActionRow(host: SettingsHost, s: Setting.Action, modified: Boolean, raw: String?) {
    if (!host.isCustomDriverSupported) return  // gated: not an Adreno/kgsl device
    val context = LocalContext.current
    val current = raw ?: host.currentDriverPath(s)   // snapshot-backed; recomposes on change
    // .zip picker -> install via Utils on the host Activity (see note below).
    val pickZip = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        val activity = context as? Activity ?: return@rememberLauncherForActivityResult
        if (uri != null) {
            Utils.install_custom_driver_from_zip(activity, uri) { path ->
                host.onDriverPathChanged(s, path)   // installed path
            }
        }
    }
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().clickable { pickZip.launch(arrayOf("application/zip")) }
            .padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.weight(1f)) {
                RowTitle(s.title, modified,
                    sub = current.ifEmpty { "Default" })
            }
        }
        // "" clears vulkan_lib_path -> native falls back to the system driver. (Writing a
        // literal "default" would make native try to dlopen a driver named "default".)
        TextButton(onClick = { host.onDriverPathChanged(s, "") },
            modifier = Modifier.padding(start = 8.dp)) { Text("Use default driver") }
    }
}

/**
 * Wraps a [SettingRow] with a leading override Switch for the per-game editor. When
 * [overridden] the inner control is the live editor (modified tint). When NOT overridden the
 * row still SHOWS the setting at its inherited (global) value via a disabled control, greyed,
 * so "by default" each row reads as its inherited value; flipping the Switch ON seeds the
 * override from that value, OFF clears the key. Used ONLY by the per-game screen.
 */
@Composable
fun OverrideRow(
    host: SettingsHost,
    s: Setting,
    overrideValue: String?,
    onOverrideToggle: (Boolean) -> Unit,
) {
    val overridden = overrideValue != null
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Switch(checked = overridden, onCheckedChange = onOverrideToggle,
            modifier = Modifier.padding(start = 12.dp))
        Box(Modifier.weight(1f)) {
            if (overridden) key(overrideValue) { SettingRow(host, s, modified = true) }
            else InheritedPreview(host, s)
        }
    }
}

/** The setting at its inherited value, shown DISABLED (no editor). Reads host.current* live
 *  (no remember) so it reflects the global snapshot once it has loaded. */
@Composable
private fun InheritedPreview(host: SettingsHost, s: Setting) {
    val grey = MaterialTheme.colorScheme.onSurfaceVariant
    when (s) {
        is Setting.Bool -> Row(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(s.title, color = grey, style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.weight(1f))
            Switch(checked = host.currentBool(s), onCheckedChange = null, enabled = false)
        }
        is Setting.IntRange   -> InheritedTextRow(s.title, host.currentInt(s).toString(), grey)
        is Setting.ListChoice -> {
            val v = host.currentListValue(s)
            val label = s.options.firstOrNull { it.value == v }?.label
                ?: v.ifEmpty { "(default)" }
            InheritedTextRow(s.title, label, grey)
        }
        is Setting.Action ->
            InheritedTextRow(s.title, host.currentDriverPath(s).ifEmpty { "Default" }, grey)
    }
}

@Composable
private fun InheritedTextRow(title: String, value: String, grey: Color) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp)) {
        Text(title, color = grey, style = MaterialTheme.typography.bodyLarge)
        Text(value, color = grey, style = MaterialTheme.typography.bodySmall)
    }
}

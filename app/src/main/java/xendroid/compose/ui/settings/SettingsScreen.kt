package xendroid.compose.ui.settings

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import xendroid.compose.settings.SettingValue
import xendroid.compose.settings.SettingsCategory
import xendroid.compose.settings.SettingsViewModel

/**
 * Two-level settings: an INDEX of sections (the 124-entry schema is too long for one list), and
 * a per-section DETAIL with that section's rows. Navigation between the two is internal state so
 * the single SettingsViewModel (and its config handle / flush lifecycle) is shared; the system
 * back button goes detail->index->exit via [BackHandler].
 */
@Composable
fun SettingsScreen(vm: SettingsViewModel, onBack: () -> Unit) {
    val values by vm.values.collectAsStateWithLifecycle()

    // Durable flush on pause; re-open on resume. Dispose flush = backstop.
    val owner = LocalLifecycleOwner.current
    DisposableEffect(owner) {
        val obs = LifecycleEventObserver { _, e ->
            when (e) {
                Lifecycle.Event.ON_PAUSE -> vm.flush()
                Lifecycle.Event.ON_RESUME -> vm.onResume()
                else -> {}
            }
        }
        owner.lifecycle.addObserver(obs)
        onDispose { owner.lifecycle.removeObserver(obs); vm.flush() }
    }

    var selected by remember { mutableStateOf<SettingsCategory?>(null) }
    val section = selected
    if (section == null) {
        SettingsIndex(
            categories = vm.categories,
            modifiedCountOf = { cat -> cat.settings.count { values[it.key]?.modified == true } },
            onOpen = { selected = it },
            onBack = { vm.flush(); onBack() },
        )
    } else {
        BackHandler { selected = null }
        SettingsCategoryDetail(
            category = section,
            values = values,
            vm = vm,
            onBack = { selected = null },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsIndex(
    categories: List<SettingsCategory>,
    modifiedCountOf: (SettingsCategory) -> Int,
    onOpen: (SettingsCategory) -> Unit,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        }
    ) { padding ->
        LazyColumn(Modifier.fillMaxSize().padding(padding)) {
            items(categories, key = { it.title }) { cat ->
                val modified = modifiedCountOf(cat)
                ListItem(
                    headlineContent = { Text(cat.title) },
                    supportingContent = {
                        Text(buildString {
                            append("${cat.settings.size} settings")
                            if (modified > 0) append("  ·  $modified changed")
                        })
                    },
                    trailingContent = {
                        Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, contentDescription = "Open")
                    },
                    modifier = Modifier.clickable { onOpen(cat) },
                )
                HorizontalDivider()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsCategoryDetail(
    category: SettingsCategory,
    values: Map<String, SettingValue>,
    vm: SettingsViewModel,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(category.title) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back to sections")
                    }
                },
            )
        }
    ) { padding ->
        LazyColumn(Modifier.fillMaxSize().padding(padding)) {
            items(category.settings, key = { it.key }) { setting ->
                val sv = values[setting.key]
                SettingRow(vm, setting, modified = sv?.modified == true, raw = sv?.raw)
                HorizontalDivider()
            }
        }
    }
}

package xendroid.compose.settings

import android.util.Log
import java.io.File

/**
 * Pure (JNI-free) decision for the rebuild-on-flush path, extracted for unit testing
 * (mirrors the [modified] extraction in SettingsRepository.kt). Given the in-memory
 * override set, [RebuildPlan.Delete] means "no overrides -> delete the file so
 * LoadGameConfig skips it"; [RebuildPlan.Write] carries the EXACT keys to re-emit (the
 * whole file is rebuilt from these, since there is no native key-erase).
 */
sealed interface RebuildPlan {
    data object Delete : RebuildPlan
    data class Write(val keys: Set<String>) : RebuildPlan
}

fun rebuildPlan(overrides: Map<String, String>): RebuildPlan =
    if (overrides.isEmpty()) RebuildPlan.Delete else RebuildPlan.Write(overrides.keys.toSet())

/**
 * The SPARSE per-game override store for a single title id. Native applies
 * <app_data_dir>/config/<TITLE_ID>.config.toml as an overlay on top of the global
 * config at boot (config.cc LoadGameConfig/ReadGameConfig — only physically present
 * keys override). There is no native key-erase (the Config JNI exposes only
 * open/save/close), so the source of truth is the in-memory [overrides] map and every
 * [flush] REBUILDS the file from it (deleting it when empty). This guarantees a minimal
 * sparse file and lets un-overriding a key actually remove it from disk.
 */
class GameSettingsRepository(private val store: ConfigStore, private val titleId: String) {

    /** key -> override raw String (only the keys the user explicitly set for THIS game). */
    private val overrides = mutableMapOf<String, String>()
    /** key -> effective global value (the inherited value), captured once at open. */
    private var globalValues: Map<String, String?> = emptyMap()
    private var opened = false

    val isCustomDriverSupported: Boolean
        get() = runCatching { File("/dev/kgsl-3d0").exists() }.getOrDefault(false)

    @Synchronized
    fun open() {
        if (opened) return
        // 1) load existing per-game overrides into the in-memory map (sparse). NOTE: only the
        //    124 schema keys are loaded, so a key in the file outside the schema (hand-edited, or
        //    a future xenia cvar) is NOT carried and will be dropped by the next rebuild-on-flush.
        //    Acceptable for the in-app editor; this is the only path that writes the per-game file.
        runCatching {
            val h = store.openGameConfig(titleId)
            try {
                SettingsSchema.allSettings.forEach { s ->
                    h.getString(s.section, s.name)?.let { overrides[s.key] = it }
                }
            } finally {
                h.closeDiscard()  // read-only pass; skip the pointless re-serialize + write
            }
        }.onFailure { Log.w("GameSettingsRepo", "per-game open failed ($titleId)", it) }
        // 2) snapshot the global effective values (the INHERITED display). Must not throw -- if it
        //    did, globalValues would stay empty and every row would show the schema default instead
        //    of the real global value.
        runCatching {
            val snap = store.openLiveSnapshot()
            globalValues = SettingsSchema.allSettings.associate { s ->
                s.key to snap.getString(s.section, s.name)
            }
            snap.closeDiscard()
        }.onFailure { Log.w("GameSettingsRepo", "global snapshot failed", it) }
        opened = true
    }

    @Synchronized
    fun ensureOpen() { if (!opened) open() }

    fun isOverridden(s: Setting): Boolean = overrides.containsKey(s.key)

    /** Raw override value for [s] if overridden, else null. Lets the per-game observable carry the
     *  value so a value-only edit changes the snapshot (containsKey still means "overridden"). */
    fun rawOverride(s: Setting): String? = overrides[s.key]

    private fun inheritedRaw(s: Setting): String =
        globalValues[s.key] ?: schemaDefaultString(s)

    // ---- typed reads: override value if present, else inherited ----
    fun boolOf(s: Setting.Bool): Boolean =
        ConfigValueShape.parseBool(overrides[s.key] ?: inheritedRaw(s), s.default)
    fun intOf(s: Setting.IntRange): Int =
        ConfigValueShape.parseInt(overrides[s.key] ?: inheritedRaw(s), s.default)
    fun listValueOf(s: Setting.ListChoice): String = overrides[s.key] ?: inheritedRaw(s)
    fun driverPathOf(s: Setting.Action): String = overrides[s.key] ?: ""
    fun inheritedLabel(s: Setting): String = labelFor(s, inheritedRaw(s))

    // ---- writes: mutate the in-memory map only; persisted on flush ----
    fun setBool(s: Setting.Bool, v: Boolean) { overrides[s.key] = ConfigValueShape.bool(v) }
    fun setInt(s: Setting.IntRange, v: Int) {
        overrides[s.key] = ConfigValueShape.int(v.coerceIn(s.min, s.max))
    }
    fun setListValue(s: Setting.ListChoice, value: String) { overrides[s.key] = value }
    fun setDriverPath(s: Setting.Action, value: String) { overrides[s.key] = value }

    /** Turn override ON: seed from the current inherited value. OFF: drop the key. */
    fun setOverride(s: Setting, on: Boolean) {
        if (on) {
            if (!overrides.containsKey(s.key)) overrides[s.key] = inheritedRaw(s)
        } else {
            overrides.remove(s.key)
        }
    }

    /** REBUILD the file from the surviving overrides (no native erase exists). Empty -> delete
     *  so LoadGameConfig's existence check skips it and the game falls back fully to global. */
    @Synchronized
    fun flush() {
        if (!opened) return
        val file = store.perGameConfigFile(titleId)
        when (val plan = rebuildPlan(overrides)) {
            is RebuildPlan.Delete -> file.delete()  // LoadGameConfig skips a missing file
            is RebuildPlan.Write -> {
                file.parentFile?.mkdirs()
                file.writeText("")  // start from an EMPTY table
                val h = store.openGameConfig(titleId)
                try {
                    SettingsSchema.allSettings.forEach { s ->
                        if (s.key in plan.keys) overrides[s.key]?.let { raw -> putTyped(h, s, raw) }
                    }
                } finally {
                    h.closeFile()  // the ONLY disk write
                }
            }
        }
    }

    private fun putTyped(h: ConfigHandle, s: Setting, raw: String) = when (s) {
        is Setting.Bool       -> h.putBool(s.section, s.name, ConfigValueShape.parseBool(raw, s.default))
        is Setting.IntRange   -> h.putInt(s.section, s.name, ConfigValueShape.parseInt(raw, s.default))
        is Setting.ListChoice -> h.putString(s.section, s.name, raw)
        is Setting.Action     -> h.putString(s.section, s.name, raw)
    }

    private fun schemaDefaultString(s: Setting): String = when (s) {
        is Setting.Bool       -> if (s.default) "true" else "false"
        is Setting.IntRange   -> s.default.toString()
        is Setting.ListChoice -> s.default
        is Setting.Action     -> s.default
    }

    private fun labelFor(s: Setting, raw: String): String = when (s) {
        is Setting.ListChoice -> s.options.firstOrNull { it.value == raw }?.label
            ?: raw.ifEmpty { "(default)" }
        else -> raw
    }
}

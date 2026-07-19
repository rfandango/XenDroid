package xendroid.compose.settings

import java.io.File

/** Snapshot of one setting's current persisted value (raw String) + modified flag. */
data class SettingValue(val raw: String?, val modified: Boolean)

/**
 * Pure (JNI-free) modified-from-default logic, extracted for unit testing. Both the
 * live and the template raw value fall back to the schema default before comparing,
 * so an absent (null) key never NPEs and never spuriously reads as "modified".
 * Encodes the EmulatorSettings.java:591 NPE fix as a regression guard.
 */
fun modified(effectiveLive: String?, effectiveTemplate: String?, schemaDefault: String): Boolean {
    val live = effectiveLive ?: schemaDefault
    val template = effectiveTemplate ?: schemaDefault
    return live != template
}

class SettingsRepository(private val store: ConfigStore) {

    companion object {
        /** Immutable per process; parse the bundled template once. */
        @Volatile private var cachedTemplateDefaults: Map<String, String?>? = null
    }

    private var live: ConfigHandle? = null
    private var templateDefaults: Map<String, String?> = emptyMap()
    /** True once an edit was made; a clean flush frees the handle without writing. */
    private var dirty = false
    /** Set by close(): after teardown, ensureOpen() must not resurrect the handle. */
    private var terminated = false

    val isCustomDriverSupported: Boolean
        get() = runCatching { File("/dev/kgsl-3d0").exists() }.getOrDefault(false)

    /** Open the live config + read template defaults. Call once per screen entry.
     *  @Synchronized so the off-main load() and the lifecycle pause/resume flush can't
     *  race on `live` (double-open leak / open-during-close). */
    @Synchronized
    fun open() {
        if (live != null || terminated) return
        live = store.openLive()
        templateDefaults = cachedTemplateDefaults ?: run {
            val baseline = store.openTemplateBaseline()
            val defaults = SettingsSchema.allSettings.associate { s ->
                s.key to baseline.getString(s.section, s.name)
            }
            baseline.closeDiscard()
            defaults.also { cachedTemplateDefaults = it }
        }
    }

    /** Re-open after a pause-flush nulled the handle. No-op if already open. */
    @Synchronized
    fun ensureOpen() { if (live == null) open() }

    private fun handle(): ConfigHandle = checkNotNull(live) { "SettingsRepository not open()ed" }

    // Reads are null-safe against an un-open()ed handle: the screen composes rows
    // before the async open() (which waits on ensureLoaded off-main) completes, so a
    // not-yet-open repo returns schema defaults rather than crashing on handle().
    @Synchronized
    fun rawValue(s: Setting): String? = live?.getString(s.section, s.name)

    @Synchronized
    fun valueOf(s: Setting): SettingValue {
        val raw = live?.getString(s.section, s.name)
        return SettingValue(raw, modified = isModified(s, raw))
    }

    /** Null-safe modified check: compare the effective live value (raw ?: schema default)
     *  against the template default (template ?: schema default). Never NPEs on absent keys. */
    private fun isModified(s: Setting, raw: String?): Boolean =
        modified(raw, templateDefaults[s.key], schemaDefaultString(s))

    private fun schemaDefaultString(s: Setting): String = when (s) {
        is Setting.Bool      -> if (s.default) "true" else "false"
        is Setting.IntRange  -> s.default.toString()
        is Setting.ListChoice -> s.default
        is Setting.Action    -> s.default
    }

    // ---- Typed reads with schema-default fallback (null-safe vs un-open()ed handle) ----
    @Synchronized
    fun boolOf(s: Setting.Bool): Boolean = live?.getBool(s.section, s.name, s.default) ?: s.default
    @Synchronized
    fun intOf(s: Setting.IntRange): Int = live?.getInt(s.section, s.name, s.default) ?: s.default
    @Synchronized
    fun listValueOf(s: Setting.ListChoice): String =
        live?.getString(s.section, s.name) ?: s.default
    @Synchronized
    fun stringOf(s: Setting): String = live?.getString(s.section, s.name) ?: ""

    // ---- Writes (in-memory; persisted on close) ----
    @Synchronized
    fun setBool(s: Setting.Bool, v: Boolean) { handle().putBool(s.section, s.name, v); dirty = true }
    @Synchronized
    fun setInt(s: Setting.IntRange, v: Int) {
        handle().putInt(s.section, s.name, v.coerceIn(s.min, s.max)); dirty = true
    }
    @Synchronized
    fun setListValue(s: Setting.ListChoice, value: String) {
        handle().putString(s.section, s.name, value); dirty = true
    }
    @Synchronized
    fun setRawString(s: Setting, value: String) {
        handle().putString(s.section, s.name, value); dirty = true
    }

    /** Free the handle; write to disk only when dirty. Synchronous: the ON_PAUSE
     *  write must be durable before a background kill. */
    @Synchronized
    fun flushAndClose() {
        val h = live ?: return
        if (dirty) h.closeFile() else h.closeDiscard()
        live = null
        dirty = false
    }

    /** Terminal teardown for onCleared: flush then block resurrection. */
    @Synchronized
    fun close() {
        flushAndClose()
        terminated = true
    }

    /** Durably persist the custom Vulkan driver path INDEPENDENTLY of the screen handle.
     *  The SAF .zip picker pauses the settings activity (ON_PAUSE flushes + nulls `live`),
     *  so the install callback can't write through the (null) screen handle -- that was the
     *  Turnip-loader bug. Flush any open screen edits FIRST (so the on-disk file is current
     *  and nothing is clobbered), then do a self-contained open -> put -> closeFile (the only
     *  disk write) so the driver path is durable the instant install completes. The screen
     *  handle re-opens on the next ensureOpen()/onResume(). Pass "" to clear (system driver). */
    @Synchronized
    fun persistDriverPath(value: String) {
        flushAndClose()   // persist+close the screen handle (no-op if already paused-flushed)
        val h = store.openLive()
        try {
            h.putString("Vulkan", "vulkan_lib_path", value)
        } finally {
            h.closeFile()
        }
    }
}

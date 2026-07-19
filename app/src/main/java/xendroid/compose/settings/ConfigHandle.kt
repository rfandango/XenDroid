package xendroid.compose.settings

import xendroid.emulator.Emulator

/** Build a "Section|name" tag (single '|' separator the native side splits on). */
fun tag(section: String, name: String): String = "$section|$name"

/**
 * Single-use wrapper over [Emulator.Config]. A file handle persists to disk on
 * [closeFile]; a string/template handle returns serialized TOML on [closeString].
 * Closing deletes the native table — the handle is dangling afterwards, so this
 * class guards against reuse and double-close.
 */
class ConfigHandle private constructor(
    private val config: Emulator.Config,
    private val isFile: Boolean,
) {
    @Volatile private var closed = false

    private fun checkOpen() = check(!closed) { "ConfigHandle already closed (single-use)" }

    fun getString(section: String, name: String): String? {
        checkOpen()
        return config.load_config_entry(tag(section, name))  // null = unset
    }

    fun getBool(section: String, name: String, def: Boolean): Boolean =
        ConfigValueShape.parseBool(getString(section, name), def)

    /** Native ints come back via std::to_string; tolerate a trailing ".0..0" from
     *  a value that round-tripped as double. */
    fun getInt(section: String, name: String, def: Int): Int =
        ConfigValueShape.parseInt(getString(section, name), def)

    fun getDouble(section: String, name: String, def: Double): Double =
        getString(section, name)?.toDoubleOrNull() ?: def

    // ---- Type-preserving puts (emit the canonical shape the native re-infers) ----

    fun putBool(section: String, name: String, v: Boolean) =
        save(section, name, ConfigValueShape.bool(v))

    /** Caller guarantees [v] fits int32; values beyond that store as a STRING natively. */
    fun putInt(section: String, name: String, v: Int) =
        save(section, name, ConfigValueShape.int(v))

    /** Always include a '.' so it round-trips as a TOML double, never an int. */
    fun putDouble(section: String, name: String, v: Double) =
        save(section, name, ConfigValueShape.double(v))

    /** Enums / paths / list values: stored verbatim as a string. */
    fun putString(section: String, name: String, v: String) = save(section, name, v)

    private fun save(section: String, name: String, value: String) {
        checkOpen()
        config.save_config_entry(tag(section, name), value)
    }

    /** Persist to disk (file handle only). Idempotent: subsequent calls no-op. */
    fun closeFile() {
        if (closed) return
        require(isFile) { "closeFile() on a string handle" }
        closed = true
        config.close_config_file()   // the ONLY disk write
    }

    /** Serialize + free (string/template handle only). Returns TOML text. */
    fun closeString(): String {
        check(!isFile) { "closeString() on a file handle" }
        check(!closed) { "already closed" }
        closed = true
        return config.close_config()
    }

    /** Free without serializing or writing - for read-only use of any handle. */
    fun closeDiscard() {
        if (closed) return
        closed = true
        config.free_config()
    }

    companion object {
        /** @throws Emulator.ConfigFileException on parse error / missing file. */
        @Throws(Emulator.ConfigFileException::class)
        fun openFile(path: String): ConfigHandle =
            ConfigHandle(Emulator.Config.open_config_file(path), isFile = true)

        @Throws(Emulator.ConfigFileException::class)
        fun openString(tomlText: String): ConfigHandle =
            ConfigHandle(Emulator.Config.open_config_from_string(tomlText), isFile = false)
    }
}

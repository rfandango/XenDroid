package xendroid.compose.settings

/** One selectable list option: stored [value] (verbatim) + UI [label]. */
data class ListOption(val value: String, val label: String)

sealed interface Setting {
    val section: String
    val name: String
    val title: String
    /** One-line user-facing description, shown as a long-press tooltip. */
    val desc: String
    val key: String get() = "$section|$name"

    data class Bool(
        override val section: String, override val name: String,
        override val title: String, val default: Boolean,
        override val desc: String = "",
    ) : Setting

    /** SeekBar-backed int. [min]/[max] from the legacy XML (NOT the TOML). */
    data class IntRange(
        override val section: String, override val name: String,
        override val title: String, val default: Int, val min: Int, val max: Int,
        override val desc: String = "",
    ) : Setting

    /** Stored verbatim as a string; options preserve non-contiguous values. */
    data class ListChoice(
        override val section: String, override val name: String,
        override val title: String, val default: String, val options: List<ListOption>,
        override val desc: String = "",
    ) : Setting

    /** Custom Vulkan driver picker (.zip), gated on support_custom_driver. No typed value. */
    data class Action(
        override val section: String, override val name: String,
        override val title: String, val default: String,
        override val desc: String = "",
    ) : Setting
}

data class SettingsCategory(val title: String, val settings: List<Setting>)

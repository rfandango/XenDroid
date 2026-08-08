// SPDX-License-Identifier: WTFPL
#include "xenia/game_quirks.h"

#include <array>
#include <string>
#include <variant>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"

namespace xe {
namespace game_quirks {

using QuirkValue = std::variant<bool, int64_t, double, const char*>;

struct Quirk {
  uint32_t title_id;
  const char* cvar;
  QuirkValue value;
  const char* note;
};

// Entries apply at per-game-config priority, so a user's own per-game config
// still overrides them.
static const Quirk kQuirks[] = {
    // Ninja Gaiden 2: the opening FMV renders as three time-skewed vertical
    // slices when the restore half of the saverest inline is active. Every
    // emitted expansion verifies correct, so the current reading is a latent
    // guest-side race between the slice decode workers that the faster
    // epilogues expose. Keep the save half inlined, call the restores.
    {0x544307D5, "inline_gprlr_saverest_parts", int64_t(1),
     "FMV slice desync with inlined restores"},
};

// Same path/priority as a per-game config file.
static bool ApplyOne(const Quirk& quirk) {
  if (!cvar::ConfigVars) {
    return false;
  }
  auto it = cvar::ConfigVars->find(quirk.cvar);
  if (it == cvar::ConfigVars->end()) {
    XELOGW("Game quirk for {:08X} references unknown cvar '{}'", quirk.title_id,
           quirk.cvar);
    return false;
  }
  auto* config_var = static_cast<cvar::IConfigVar*>(it->second);
  std::visit(
      [config_var](auto&& value) {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, const char*>) {
          toml::value<std::string> node{std::string(value)};
          config_var->LoadGameConfigValue(&node);
        } else {
          toml::value<V> node{value};
          config_var->LoadGameConfigValue(&node);
        }
      },
      quirk.value);
  return true;
}

size_t Apply(uint32_t title_id) {
  size_t applied = 0;
  for (const auto& quirk : kQuirks) {
    if (quirk.title_id != title_id) {
      continue;
    }
    if (ApplyOne(quirk)) {
      XELOGI("Game quirk for {:08X}: {} ({})", title_id, quirk.cvar,
             quirk.note);
      ++applied;
    }
  }
  return applied;
}

}  // namespace game_quirks
}  // namespace xe

/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_GAME_COMPAT_DB_H_
#define XENIA_APP_GAME_COMPAT_DB_H_

#include <cstdint>
#include <string>

namespace xe {
namespace app {

// Order encodes "better compat" — higher value sorts above.
enum class CompatState : uint8_t {
  kUnknown = 0,
  kUnplayable,
  kLoads,
  kGameplay,
  kPlayable,
};

struct CompatEntry {
  CompatState state = CompatState::kUnknown;
  std::string url;
};

CompatState GetCompatState(uint32_t title_id);

// Issue URL from the compatibility database for a title, or empty if absent.
std::string GetCompatUrl(uint32_t title_id);

// GitHub issue-search query fragment for a title. Returns the title id, or all
// sibling ids (x360db alternative ids) OR'd in parentheses when present.
// URL-encoded for use after the "is:open " term.
std::string CompatSearchQuery(uint32_t title_id);

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_GAME_COMPAT_DB_H_

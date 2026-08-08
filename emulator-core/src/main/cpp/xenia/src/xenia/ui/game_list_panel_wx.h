/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_GAME_LIST_PANEL_WX_H_
#define XENIA_UI_GAME_LIST_PANEL_WX_H_

#include <array>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <wx/bmpbndl.h>
#include <wx/dataview.h>
#include <wx/panel.h>
#include <wx/srchctrl.h>

#include "xenia/app/game_compat_db.h"

namespace xe {
namespace app {

class EmulatorWindow;

class GameListPanel : public wxPanel {
 public:
  using LaunchCallback = std::function<void(const std::filesystem::path&)>;
  using SelectionChangedCallback = std::function<void()>;

  GameListPanel(wxWindow* parent, EmulatorWindow* emulator_window);
  ~GameListPanel() override = default;

  void Reload();
  void SetLaunchCallback(LaunchCallback cb) { launch_cb_ = std::move(cb); }
  void SetSelectionChangedCallback(SelectionChangedCallback cb) {
    selection_changed_cb_ = std::move(cb);
  }
  // Returns the selected title's path, or empty if nothing selected.
  std::filesystem::path GetSelectedPath() const;

  // Moves the selection by `delta` rows (positive = down). Wraps and selects
  // the first row if nothing was selected.
  void MoveSelection(int delta);
  // Launches the selected title via the launch callback (if one is wired).
  void ActivateSelected();
  void FocusSearch();

 private:
  struct Disc {
    std::filesystem::path path;
    std::string label;
  };
  struct Entry {
    uint32_t title_id = 0;
    std::string title_name;
    std::filesystem::path path;
    std::vector<Disc> discs;
    time_t last_run_time = 0;
    wxBitmapBundle icon;
    uint32_t achievements_total = 0;
    uint32_t achievements_unlocked = 0;
    uint32_t gamerscore_total = 0;
    uint32_t gamerscore_earned = 0;
  };

  // Selected entry, or nullptr if none.
  const Entry* SelectedEntry() const;

  // Launches `path` if it exists, else offers to browse for a replacement and
  // drops the stale `path` from `title_id`.
  void LaunchOrPrompt(uint32_t title_id, const std::filesystem::path& path);

  // Modal dialog for renaming/removing discs of a multi-disc title.
  void ShowEditDiscsDialog(size_t entry_index);

  void OnSearch(wxCommandEvent& event);
  void OnItemActivated(wxDataViewEvent& event);
  void OnSelectionChanged(wxDataViewEvent& event);
  void OnColumnHeaderClick(wxDataViewEvent& event);
  void OnListMouseMotion(wxMouseEvent& event);
  // Compat DB state for `e`, optimistically upgraded by the user's local
  // achievement progress (10%+ promotes Unknown to Gameplay; 80%+ promotes
  // anything to Playable).
  CompatState GetEntryCompatState(const Entry& e) const;
  void SortEntries();

  int sort_column_ = 5;  // Last Played
  bool sort_descending_ = true;
  // Cached so we only call SetToolTip on change, avoiding flicker.
  wxString last_tooltip_text_;
  void OnItemContextMenu(wxDataViewEvent& event);
  void Repopulate();
  void LoadTimestampsFromProfiles();
  void StartIconLoad();
  void ProcessIconChunk(size_t start, int gen);
  void UpdateSearchPlaceholder();

  EmulatorWindow* emulator_window_;
  wxSearchCtrl* search_ = nullptr;
  wxDataViewListCtrl* list_ = nullptr;
  wxPanel* loading_panel_ = nullptr;
  std::vector<Entry> entries_;
  std::vector<size_t> visible_indices_;
  LaunchCallback launch_cb_;
  SelectionChangedCallback selection_changed_cb_;
  std::string filter_lower_;
  // Bumped on each Reload so in-flight icon-load chunks abort.
  int icon_load_generation_ = 0;
  // Resolved at construction time from the panel's monitor; used for sizing
  // bitmaps, column widths, and the row height so the list grows with the
  // current display's DPI scale.
  int icon_size_px_ = 0;
  double dpi_scale_ = 1.0;
  wxBitmapBundle not_played_placeholder_;
  std::array<wxBitmapBundle, 5> compat_balls_;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_UI_GAME_LIST_PANEL_WX_H_

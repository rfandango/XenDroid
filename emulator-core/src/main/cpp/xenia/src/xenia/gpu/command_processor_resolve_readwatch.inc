/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Resolve-to-guest-RAM copy decisions and read-watch consumption tracking,
// shared by the D3D12 and Vulkan command processors. Like
// command_processor_memexport.inc, this is #included into each backend's class
// body so each gets its own inlinable copy with no inheritance or virtual
// dispatch. The include site must already provide SharedMemory, the base
// memory_ (Memory*), xenia/memory.h (VirtualHeap, IsWritableProtect), and
// <atomic> <deque> <memory> <unordered_set> <algorithm>. No include guard:
// injected into a class body.

// Whether the guest physical range a resolve targets is committed and writable,
// so copying the resolve back into guest RAM is safe. False means skip the
// copy.
bool IsResolveDestinationResident(uint32_t address, uint32_t length) const {
  VirtualHeap* physical_heap = memory_->GetPhysicalHeap();
  if (!physical_heap) {
    return false;
  }
  HeapAllocationInfo alloc_info;
  if (!physical_heap->QueryRegionInfo(address, &alloc_info) ||
      !(alloc_info.state & kMemoryAllocationCommit) ||
      !IsWritableProtect(alloc_info.protect)) {
    return false;
  }
  return uint64_t(address) + length <=
         uint64_t(alloc_info.base_address) + alloc_info.region_size;
}

// Marks the guest pages the CPU has read since a resolve wrote them, from the
// memory read callback on the guest thread. Read on the GPU thread, so atomic.
void MarkResolvePagesRead(uint32_t physical_address, uint32_t length) {
  if (!resolve_read_pages_ || !length) {
    return;
  }
  const uint32_t page_count = SharedMemory::kBufferSize >> 12;
  uint32_t first_page = physical_address >> 12;
  if (first_page >= page_count) {
    return;
  }
  uint32_t last_page =
      std::min((physical_address + length - 1) >> 12, page_count - 1);
  for (uint32_t page = first_page; page <= last_page; ++page) {
    resolve_read_pages_[page >> 6].fetch_or(uint64_t(1) << (page & 63),
                                            std::memory_order_relaxed);
  }
}

// Returns whether the CPU has read any guest page in the range since a resolve
// last wrote it.
bool AnyResolvePageRead(uint32_t base, uint32_t length) const {
  if (!resolve_read_pages_ || !length) {
    return false;
  }
  const uint32_t page_count = SharedMemory::kBufferSize >> 12;
  uint32_t first_page = base >> 12;
  if (first_page >= page_count) {
    return false;
  }
  uint32_t last_page = std::min((base + length - 1) >> 12, page_count - 1);
  for (uint32_t page = first_page; page <= last_page; ++page) {
    if (resolve_read_pages_[page >> 6].load(std::memory_order_relaxed) &
        (uint64_t(1) << (page & 63))) {
      return true;
    }
  }
  return false;
}

// Arms a resolve range for read watching if not already armed, evicting the
// oldest key past the cap. Returns true if this is the first resolve of the
// range.
bool TryArmResolveReadWatch(uint64_t resolve_key, uint32_t address,
                            uint32_t length) {
  if (!resolve_armed_keys_.insert(resolve_key).second) {
    // Already armed by an earlier resolve of this range.
    return false;
  }
  resolve_armed_key_order_.push_back(resolve_key);
  if (resolve_armed_key_order_.size() > kMaxResolveArmedKeys) {
    // Evict the oldest key. Its range will re-seed guest RAM and re-arm on its
    // next resolve, which is safe (just one extra copy).
    resolve_armed_keys_.erase(resolve_armed_key_order_.front());
    resolve_armed_key_order_.pop_front();
  }
  memory_->EnablePhysicalMemoryAccessCallbacks(address, length, false, true);
  return true;
}

// Decides whether a resolve of [address, length) should be copied into guest
// RAM under the readback mode, and whether that copy must stall the GPU
// afterwards. Returns false to skip the copy (fast mode, un-read range). all
// copies every resolve. fast copies only CPU-read ranges, plus each range's
// first resolve to seed it. sync (the readback_resolve_sync cvar) makes the
// copy stall for current-frame coherency. A range's first resolve in fast mode
// always stalls so a prompt CPU reader does not race the deferred seed copy.
bool DecideResolveHostCopy(ReadbackResolveMode mode, uint32_t address,
                           uint32_t length, bool sync, bool& stall_after_copy) {
  stall_after_copy = sync;
  if (mode == ReadbackResolveMode::kFast) {
    uint64_t resolve_key = MakeReadbackResolveKey(address, length);
    bool first_resolve = TryArmResolveReadWatch(resolve_key, address, length);
    if (!first_resolve && !AnyResolvePageRead(address, length)) {
      return false;
    }
    stall_after_copy = sync || first_resolve;
  }
  return true;
}

// Allocates the read-page bitset. Called from each backend's SetupContext,
// which then registers the per-backend read callback thunk.
void InitResolveReadWatch() {
  resolve_read_pages_ = std::make_unique<std::atomic<uint64_t>[]>(
      (SharedMemory::kBufferSize >> 12) / 64);
}

// Unregisters the read callback and drops all tracking. Called from each
// backend's ShutdownContext.
void ResetResolveReadWatch() {
  if (resolve_read_callback_) {
    memory_->UnregisterPhysicalMemoryReadCallback(resolve_read_callback_);
    resolve_read_callback_ = nullptr;
  }
  resolve_read_pages_.reset();
  resolve_armed_keys_.clear();
  resolve_armed_key_order_.clear();
}

// Guest-page bitset of pages the CPU read since a resolve wrote them. Set on
// the guest thread from the read callback, read on the GPU thread, so atomic.
std::unique_ptr<std::atomic<uint64_t>[]> resolve_read_pages_;
// Resolve ranges already armed for read watching, bounded via the FIFO below.
std::unordered_set<uint64_t> resolve_armed_keys_;
// FIFO insertion order for resolve_armed_keys_ so it stays bounded. Evicting
// the oldest key just makes its range re-seed guest RAM once, which is safe.
std::deque<uint64_t> resolve_armed_key_order_;
static constexpr size_t kMaxResolveArmedKeys = 4096;
// Handle from RegisterPhysicalMemoryReadCallback, or null when not armed.
void* resolve_read_callback_ = nullptr;

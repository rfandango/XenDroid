/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XEVENT_H_
#define XENIA_KERNEL_XEVENT_H_

#include "xenia/base/threading.h"
#include "xenia/kernel/xobject.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {

// https://www.nirsoft.net/kernel_struct/vista/KEVENT.html
struct X_KEVENT {
  X_DISPATCH_HEADER header;
};
static_assert_size(X_KEVENT, 0x10);

class XEvent : public XObject {
 public:
  static const XObject::Type kObjectType = XObject::Type::Event;

  explicit XEvent(KernelState* kernel_state);
  ~XEvent() override;

  void Initialize(bool manual_reset, bool initial_state);
  void InitializeNative(void* native_ptr, const X_DISPATCH_HEADER* header);

  int32_t Set(uint32_t priority_increment, bool wait);
  // Diagnostics: who last signaled this event (guest thread handle + lr,
  // guest uptime ms). Read by the long-wait reporter in XObject::Wait.
  uint32_t last_set_thread() const { return last_set_thread_; }
  uint32_t last_set_lr() const { return last_set_lr_; }
  uint32_t last_set_uptime_ms() const { return last_set_uptime_ms_; }
  // Diagnostics: who created this event (guest thread handle + lr). A
  // never-signaled event's creator names the subsystem that owns it.
  uint32_t creator_thread() const { return creator_thread_; }
  uint32_t creator_lr() const { return creator_lr_; }
  // Public so XObject::SignalAndWait (which signals the host handle directly,
  // bypassing Set()) can keep the setter bookkeeping accurate.
  void RecordSetter();
  int32_t Pulse(uint32_t priority_increment, bool wait);
  int32_t Reset();
  void Query(uint32_t* out_type, uint32_t* out_state);
  void Clear();

  bool Save(ByteStream* stream) override;
  static object_ref<XEvent> Restore(KernelState* kernel_state,
                                    ByteStream* stream);

  uint32_t cooperative_pulse_epoch() const override {
    return pulse_epoch_.load();
  }

 protected:
  xe::threading::WaitHandle* GetWaitHandle() override { return event_.get(); }
  void WaitCallback() override;

  void CooperativeWaitBegin(XThread* thread) override;
  void CooperativeWaitEnd(XThread* thread) override;
  bool CooperativeMayAcquire(XThread* thread) override;

 private:
  void RecordCreator();

  bool manual_reset_ = false;
  uint32_t last_set_thread_ = 0;
  uint32_t last_set_lr_ = 0;
  uint32_t last_set_uptime_ms_ = 0;
  uint32_t creator_thread_ = 0;
  uint32_t creator_lr_ = 0;
  std::unique_ptr<xe::threading::Event> event_;
  // Parked cooperative waiters, so Pulse knows one will consume a set.
  CooperativeWaiterFifo waiters_;
  std::atomic<uint32_t> pulse_epoch_{0};
};

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XEVENT_H_

/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/testing/util.h"

#include "xenia/base/byte_order.h"
#include "xenia/cpu/thread_state.h"

using namespace xe;
using namespace xe::cpu;
using namespace xe::cpu::hir;
using namespace xe::cpu::testing;
using xe::cpu::ppc::PPCContext;

namespace {

// Guest memory is big endian, these read and write in host order.
uint32_t GuestRead32(Memory* memory, uint32_t address) {
  return xe::byte_swap(*memory->TranslateVirtual<uint32_t*>(address));
}
void GuestWrite32(Memory* memory, uint32_t address, uint32_t value) {
  *memory->TranslateVirtual<uint32_t*>(address) = xe::byte_swap(value);
}

// A second guest thread on the same processor. Its context has its own
// reservation but shares the granule table, so tests can watch one thread's
// store cancel another's reservation.
class SecondThread {
 public:
  SecondThread(Processor* processor, Memory* memory) : memory_(memory) {
    stack_address_ = memory->SystemHeapAlloc(kStackSize);
    thread_state_ = std::make_unique<ThreadState>(processor, 0x200,
                                                  stack_address_ + kStackSize);
  }
  ~SecondThread() {
    thread_state_.reset();
    memory_->SystemHeapFree(stack_address_);
  }

  PPCContext* context() const { return thread_state_->context(); }

 private:
  static constexpr uint32_t kStackSize = 64 * 1024;
  Memory* memory_;
  uint32_t stack_address_ = 0;
  std::unique_ptr<ThreadState> thread_state_;
};

// 128 byte aligned, so addr and addr + 4 share a granule and addr + 128 does
// not.
uint32_t AllocScratch(Memory* memory) {
  return memory->SystemHeapAlloc(256, 128);
}

void FreeScratch(Memory* memory, PPCContext* ctx) {
  memory->SystemHeapFree(static_cast<uint32_t>(ctx->r[4]));
}

// lwarx only, leaving the reservation live on return.
void EmitGuestReservedLoad(HIRBuilder& b) {
  auto addr = LoadGPR(b, 4);
  StoreGPR(b, 3,
           b.ZeroExtend(b.ByteSwap(b.LoadWithReserve(addr, INT32_TYPE)),
                        INT64_TYPE));
  b.Return();
}

// A full lwarx/stwcx. pair, storing r5 and reporting the result in r6.
void EmitGuestReservedPair(HIRBuilder& b) {
  auto addr = LoadGPR(b, 4);
  StoreGPR(b, 3,
           b.ZeroExtend(b.ByteSwap(b.LoadWithReserve(addr, INT32_TYPE)),
                        INT64_TYPE));
  auto stored = b.StoreWithReserve(
      addr, b.ByteSwap(b.Truncate(LoadGPR(b, 5), INT32_TYPE)), INT32_TYPE);
  StoreGPR(b, 6, b.ZeroExtend(stored, INT64_TYPE));
  b.Return();
}

}  // namespace

// =============================================================================
// Host reservation pair
// =============================================================================
TEST_CASE("RESERVED_HOST_PAIR", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0x11223344u);

        auto* backend = ctx->processor->backend();
        REQUIRE(backend->ReservedLoad32(ctx, addr) == 0x11223344u);
        REQUIRE(backend->ReservedStore32(ctx, addr, 0xAABBCCDDu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0xAABBCCDDu);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

TEST_CASE("RESERVED_HOST_STORE_WITHOUT_LOAD", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0x11223344u);

        auto* backend = ctx->processor->backend();
        REQUIRE_FALSE(backend->ReservedStore32(ctx, addr, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x11223344u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

// A store consumes the reservation, so a second one has nothing to use.
TEST_CASE("RESERVED_HOST_STORE_TWICE", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);

        auto* backend = ctx->processor->backend();
        backend->ReservedLoad32(ctx, addr);
        REQUIRE(backend->ReservedStore32(ctx, addr, 0x12345678u));
        REQUIRE_FALSE(backend->ReservedStore32(ctx, addr, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x12345678u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

// Storing to an address the load didn't reserve fails, same granule or not.
TEST_CASE("RESERVED_HOST_ADDRESS_MISMATCH", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);
        GuestWrite32(test.memory.get(), addr + 4, 0u);

        auto* backend = ctx->processor->backend();
        backend->ReservedLoad32(ctx, addr);
        REQUIRE_FALSE(backend->ReservedStore32(ctx, addr + 4, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr + 4) == 0u);
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

TEST_CASE("RESERVED_HOST_VALUE_CHANGED", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);

        auto* backend = ctx->processor->backend();
        backend->ReservedLoad32(ctx, addr);
        GuestWrite32(test.memory.get(), addr, 0x99999999u);
        REQUIRE_FALSE(backend->ReservedStore32(ctx, addr, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x99999999u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

TEST_CASE("RESERVED_HOST_PAIR_64", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        *test.memory->TranslateVirtual<uint64_t*>(addr) =
            xe::byte_swap(uint64_t(0x1122334455667788ull));

        auto* backend = ctx->processor->backend();
        REQUIRE(backend->ReservedLoad64(ctx, addr) == 0x1122334455667788ull);
        REQUIRE(backend->ReservedStore64(ctx, addr, 0xAABBCCDDEEFF0011ull));
        REQUIRE(xe::byte_swap(*test.memory->TranslateVirtual<uint64_t*>(
                    addr)) == 0xAABBCCDDEEFF0011ull);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

// =============================================================================
// Cross-thread invalidation
// =============================================================================
TEST_CASE("RESERVED_HOST_CROSS_THREAD_INVALIDATE", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  SecondThread other(test.processors[0].get(), test.memory.get());
  test.Run(
      [&test, &other](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);

        auto* backend = ctx->processor->backend();
        backend->ReservedLoad32(other.context(), addr);
        backend->ReservedLoad32(ctx, addr);
        // bumps the granule, so the other thread's reservation is gone
        REQUIRE(backend->ReservedStore32(ctx, addr, 0x12345678u));
        REQUIRE_FALSE(
            backend->ReservedStore32(other.context(), addr, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x12345678u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

// A store anywhere in the granule cancels the reservation.
TEST_CASE("RESERVED_HOST_GRANULE_INVALIDATE", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  SecondThread other(test.processors[0].get(), test.memory.get());
  test.Run(
      [&test, &other](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);
        GuestWrite32(test.memory.get(), addr + 4, 0u);

        auto* backend = ctx->processor->backend();
        backend->ReservedLoad32(other.context(), addr);
        backend->ReservedLoad32(ctx, addr + 4);
        REQUIRE(backend->ReservedStore32(ctx, addr + 4, 0x12345678u));
        REQUIRE_FALSE(
            backend->ReservedStore32(other.context(), addr, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

// A reservation in a different granule survives.
TEST_CASE("RESERVED_HOST_OTHER_GRANULE_SURVIVES", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  SecondThread other(test.processors[0].get(), test.memory.get());
  test.Run(
      [&test, &other](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);
        GuestWrite32(test.memory.get(), addr + 128, 0u);

        auto* backend = ctx->processor->backend();
        backend->ReservedLoad32(other.context(), addr + 128);
        backend->ReservedLoad32(ctx, addr);
        REQUIRE(backend->ReservedStore32(ctx, addr, 0x12345678u));
        REQUIRE(
            backend->ReservedStore32(other.context(), addr + 128, 0xAABBCCDDu));
        REQUIRE(GuestRead32(test.memory.get(), addr + 128) == 0xAABBCCDDu);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

// =============================================================================
// Host and JIT sharing one reservation domain
// =============================================================================
// A host store completes a reservation the JIT took.
TEST_CASE("RESERVED_GUEST_LOAD_HOST_STORE", "[reserve]") {
  TestFunction test(EmitGuestReservedLoad);
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0x11223344u);
      },
      [&test](PPCContext* ctx) {
        uint32_t addr = static_cast<uint32_t>(ctx->r[4]);
        REQUIRE(ctx->r[3] == 0x11223344ull);

        auto* backend = ctx->processor->backend();
        REQUIRE(backend->ReservedStore32(ctx, addr, 0xAABBCCDDu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0xAABBCCDDu);
        FreeScratch(test.memory.get(), ctx);
      });
}

// A host store from another thread cancels a reservation the JIT took.
TEST_CASE("RESERVED_HOST_STORE_INVALIDATES_GUEST", "[reserve]") {
  TestFunction test(EmitGuestReservedLoad);
  SecondThread other(test.processors[0].get(), test.memory.get());
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);
      },
      [&test, &other](PPCContext* ctx) {
        uint32_t addr = static_cast<uint32_t>(ctx->r[4]);
        auto* backend = ctx->processor->backend();
        backend->ReservedLoad32(other.context(), addr);
        REQUIRE(backend->ReservedStore32(other.context(), addr, 0x12345678u));
        // the guest's lwarx reservation is stale now
        REQUIRE_FALSE(backend->ReservedStore32(ctx, addr, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x12345678u);
        FreeScratch(test.memory.get(), ctx);
      });
}

// A JIT stwcx. cancels a reservation host code took.
TEST_CASE("RESERVED_GUEST_STORE_INVALIDATES_HOST", "[reserve]") {
  TestFunction test(EmitGuestReservedPair);
  SecondThread other(test.processors[0].get(), test.memory.get());
  test.Run(
      [&test, &other](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        ctx->r[5] = 0x12345678ull;
        GuestWrite32(test.memory.get(), addr, 0u);
        ctx->processor->backend()->ReservedLoad32(other.context(), addr);
      },
      [&test, &other](PPCContext* ctx) {
        uint32_t addr = static_cast<uint32_t>(ctx->r[4]);
        // the guest's own pair succeeded
        REQUIRE(ctx->r[6] == 1ull);
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x12345678u);

        REQUIRE_FALSE(ctx->processor->backend()->ReservedStore32(
            other.context(), addr, 0xDEADBEEFu));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x12345678u);
        FreeScratch(test.memory.get(), ctx);
      });
}

// =============================================================================
// Processor::GuestAtomic wrappers
// =============================================================================
TEST_CASE("GUEST_ATOMIC_INCREMENT32", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0x11223344u);

        // returns the value from before the update
        REQUIRE(ctx->processor->GuestAtomicIncrement32(ctx, addr) ==
                0x11223344u);
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x11223345u);
        REQUIRE(ctx->processor->GuestAtomicDecrement32(ctx, addr) ==
                0x11223345u);
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x11223344u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

TEST_CASE("GUEST_ATOMIC_BITWISE32", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0x0000FF00u);

        REQUIRE(ctx->processor->GuestAtomicOr32(ctx, addr, 0x000000FFu) ==
                0x0000FF00u);
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x0000FFFFu);
        REQUIRE(ctx->processor->GuestAtomicXor32(ctx, addr, 0x0000FFFFu) ==
                0x0000FFFFu);
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0u);
        GuestWrite32(test.memory.get(), addr, 0x0000FFFFu);
        REQUIRE(ctx->processor->GuestAtomicAnd32(ctx, addr, 0x0000FF00u) ==
                0x0000FFFFu);
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x0000FF00u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

TEST_CASE("GUEST_ATOMIC_CAS32", "[reserve]") {
  TestFunction test([](HIRBuilder& b) { b.Return(); });
  test.Run(
      [&test](PPCContext* ctx) {
        uint32_t addr = AllocScratch(test.memory.get());
        ctx->r[4] = addr;
        GuestWrite32(test.memory.get(), addr, 0u);

        REQUIRE_FALSE(
            ctx->processor->GuestAtomicCAS32(ctx, 1u, 0x12345678u, addr));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0u);
        REQUIRE(ctx->processor->GuestAtomicCAS32(ctx, 0u, 0x12345678u, addr));
        REQUIRE(GuestRead32(test.memory.get(), addr) == 0x12345678u);
      },
      [&test](PPCContext* ctx) { FreeScratch(test.memory.get(), ctx); });
}

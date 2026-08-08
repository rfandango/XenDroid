/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/vulkan_upload_buffer_pool.h"

#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/ui/vulkan/vulkan_util.h"

namespace xe {
namespace ui {
namespace vulkan {

// Host-visible memory sizes are likely to be internally rounded to
// nonCoherentAtomSize (it's the memory mapping granularity, though as the map
// or flush range must be clamped to the actual allocation size as a special
// case, but it's still unlikely that the allocation won't be aligned to it), so
// try not to waste that padding.
VulkanUploadBufferPool::VulkanUploadBufferPool(
    const VulkanDevice* const vulkan_device, const VkBufferUsageFlags usage,
    const size_t page_size)
    : GraphicsUploadBufferPool(size_t(
          xe::round_up(VkDeviceSize(page_size),
                       vulkan_device->properties().nonCoherentAtomSize))),
      vulkan_device_(vulkan_device),
      usage_(usage) {}

uint8_t* VulkanUploadBufferPool::Request(uint64_t submission_index, size_t size,
                                         size_t alignment, VkBuffer& buffer_out,
                                         VkDeviceSize& offset_out) {
  size_t offset;
  const VulkanPage* page =
      static_cast<const VulkanPage*>(GraphicsUploadBufferPool::Request(
          submission_index, size, alignment, offset));
  if (!page) {
    return nullptr;
  }
  if (VkDeviceSize(offset) + VkDeviceSize(size) > page->mapping_size_) {
    XELOGE(
        "Vulkan upload pool handed out {}+{} bytes in a page mapping {} bytes "
        "at {}",
        uint64_t(offset), uint64_t(size), uint64_t(page->mapping_size_),
        page->mapping_);
    return nullptr;
  }
  buffer_out = page->buffer_;
  offset_out = VkDeviceSize(offset);
  return reinterpret_cast<uint8_t*>(page->mapping_) + offset;
}

uint8_t* VulkanUploadBufferPool::RequestPartial(uint64_t submission_index,
                                                size_t size, size_t alignment,
                                                VkBuffer& buffer_out,
                                                VkDeviceSize& offset_out,
                                                VkDeviceSize& size_out) {
  size_t offset, size_obtained;
  const VulkanPage* page =
      static_cast<const VulkanPage*>(GraphicsUploadBufferPool::RequestPartial(
          submission_index, size, alignment, offset, size_obtained));
  if (!page) {
    return nullptr;
  }
  if (VkDeviceSize(offset) + VkDeviceSize(size_obtained) >
      page->mapping_size_) {
    XELOGE(
        "Vulkan upload pool handed out {}+{} bytes in a page mapping {} bytes "
        "at {}",
        uint64_t(offset), uint64_t(size_obtained),
        uint64_t(page->mapping_size_), page->mapping_);
    return nullptr;
  }
  buffer_out = page->buffer_;
  offset_out = VkDeviceSize(offset);
  size_out = VkDeviceSize(size_obtained);
  return reinterpret_cast<uint8_t*>(page->mapping_) + offset;
}

GraphicsUploadBufferPool::Page*
VulkanUploadBufferPool::CreatePageImplementation() {
  if (memory_type_ == kMemoryTypeUnavailable) {
    // Don't try to create everything again and again if totally broken.
    return nullptr;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = 0;
  buffer_create_info.size = VkDeviceSize(page_size_);
  buffer_create_info.usage = usage_;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = nullptr;
  VkBuffer buffer;
  if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer) !=
      VK_SUCCESS) {
    XELOGE("Failed to create a Vulkan upload buffer with {} bytes", page_size_);
    return nullptr;
  }

  if (memory_type_ == kMemoryTypeUnknown) {
    VkMemoryRequirements memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);
    memory_type_ =
        util::ChooseHostMemoryType(vulkan_device_->memory_types(),
                                   memory_requirements.memoryTypeBits, false);
    if (memory_type_ == UINT32_MAX) {
      XELOGE(
          "No host-visible memory types can store an Vulkan upload buffer with "
          "{} bytes",
          page_size_);
      memory_type_ = kMemoryTypeUnavailable;
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      return nullptr;
    }
    // Log whether we're using ReBAR memory for staging buffers.
    if ((1u << memory_type_) &
        vulkan_device_->memory_types().device_local_host_visible) {
      XELOGI("Vulkan upload buffer pool using ReBAR memory (type {})",
             memory_type_);
    } else {
      XELOGI("Vulkan upload buffer pool using system memory (type {})",
             memory_type_);
    }
    allocation_size_ = memory_requirements.size;
    if (allocation_size_ > page_size_) {
      // Try to occupy the allocation padding. If that's going to require even
      // more memory for some reason, don't.
      buffer_create_info.size = allocation_size_;
      VkBuffer buffer_expanded;
      if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr,
                             &buffer_expanded) == VK_SUCCESS) {
        VkMemoryRequirements memory_requirements_expanded;
        dfn.vkGetBufferMemoryRequirements(device, buffer_expanded,
                                          &memory_requirements_expanded);
        uint32_t memory_type_expanded = util::ChooseHostMemoryType(
            vulkan_device_->memory_types(), memory_requirements.memoryTypeBits,
            false);
        if (memory_requirements_expanded.size <= allocation_size_ &&
            memory_type_expanded != UINT32_MAX) {
          page_size_ = size_t(allocation_size_);
          allocation_size_ = memory_requirements_expanded.size;
          memory_type_ = memory_type_expanded;
          dfn.vkDestroyBuffer(device, buffer, nullptr);
          buffer = buffer_expanded;
        } else {
          dfn.vkDestroyBuffer(device, buffer_expanded, nullptr);
        }
      }
    }
  }

  VkMemoryAllocateInfo memory_allocate_info;
  VkMemoryAllocateInfo* memory_allocate_info_last = &memory_allocate_info;
  memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memory_allocate_info.pNext = nullptr;
  memory_allocate_info.allocationSize = allocation_size_;
  memory_allocate_info.memoryTypeIndex = memory_type_;
  VkMemoryDedicatedAllocateInfo memory_dedicated_allocate_info;
  if (vulkan_device_->extensions().ext_1_1_KHR_dedicated_allocation) {
    memory_allocate_info_last->pNext = &memory_dedicated_allocate_info;
    memory_allocate_info_last = reinterpret_cast<VkMemoryAllocateInfo*>(
        &memory_dedicated_allocate_info);
    memory_dedicated_allocate_info.sType =
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    memory_dedicated_allocate_info.pNext = nullptr;
    memory_dedicated_allocate_info.image = VK_NULL_HANDLE;
    memory_dedicated_allocate_info.buffer = buffer;
  }
  VkDeviceMemory memory;
  if (dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &memory) !=
      VK_SUCCESS) {
    XELOGE("Failed to allocate {} bytes of Vulkan upload buffer memory",
           allocation_size_);
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    return nullptr;
  }

  if (dfn.vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
    XELOGE("Failed to bind memory to a Vulkan upload buffer with {} bytes",
           page_size_);
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    dfn.vkFreeMemory(device, memory, nullptr);
    return nullptr;
  }

  void* mapping;
  if (dfn.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapping) !=
      VK_SUCCESS) {
    XELOGE("Failed to map {} bytes of Vulkan upload buffer memory",
           allocation_size_);
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    dfn.vkFreeMemory(device, memory, nullptr);
    return nullptr;
  }

  util::ResetSanitizerTags(mapping, allocation_size_);

  XELOGI("Vulkan upload pool {} (usage {:X}): page mapped at {}, {} bytes",
         static_cast<const void*>(this), uint32_t(usage_), mapping,
         uint64_t(allocation_size_));

  return new VulkanPage(vulkan_device_, buffer, memory, mapping,
                        allocation_size_);
}

void VulkanUploadBufferPool::FlushPageWrites(Page* page, size_t offset,
                                             size_t size) {
  util::FlushMappedMemoryRange(
      vulkan_device_, static_cast<const VulkanPage*>(page)->memory_,
      memory_type_, VkDeviceSize(offset), allocation_size_, VkDeviceSize(size));
}

VulkanUploadBufferPool::VulkanPage::~VulkanPage() {
  XELOGI("Vulkan upload pool: destroying page mapped at {}", mapping_);
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  dfn.vkDestroyBuffer(device, buffer_, nullptr);
  // Unmapping is done implicitly when the memory is freed.
  dfn.vkFreeMemory(device, memory_, nullptr);
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

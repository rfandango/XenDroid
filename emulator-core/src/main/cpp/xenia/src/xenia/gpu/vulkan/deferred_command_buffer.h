/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_DEFERRED_COMMAND_BUFFER_H_
#define XENIA_GPU_VULKAN_DEFERRED_COMMAND_BUFFER_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "xenia/base/assert.h"
#include "xenia/base/math.h"
#include "xenia/ui/vulkan/vulkan_api.h"

namespace xe {
namespace gpu {
namespace vulkan {

class VulkanCommandProcessor;

class DeferredCommandBuffer {
 public:
  DeferredCommandBuffer(const VulkanCommandProcessor& command_processor,
                        size_t initial_size_bytes = 1024 * 1024);

  void Reset();
  void Execute(VkCommandBuffer command_buffer);
  bool empty() const { return command_stream_size_ == 0; }

  // render_pass_begin->pNext of all barriers must be null.
  void CmdVkBeginRenderPass(const VkRenderPassBeginInfo* render_pass_begin,
                            VkSubpassContents contents) {
    assert_null(render_pass_begin->pNext);
    size_t arguments_size = sizeof(ArgsVkBeginRenderPass);
    uint32_t clear_value_count = render_pass_begin->clearValueCount;
    size_t clear_values_offset = 0;
    if (clear_value_count) {
      arguments_size = xe::align(arguments_size, alignof(VkClearValue));
      clear_values_offset = arguments_size;
      arguments_size += sizeof(VkClearValue) * clear_value_count;
    }
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkBeginRenderPass, arguments_size));
    auto& args = *reinterpret_cast<ArgsVkBeginRenderPass*>(args_ptr);
    args.render_pass = render_pass_begin->renderPass;
    args.framebuffer = render_pass_begin->framebuffer;
    args.render_area = render_pass_begin->renderArea;
    args.clear_value_count = clear_value_count;
    args.contents = contents;
    BeginRenderAreaTracking(arguments_size, /*dynamic_rendering=*/false);
    if (clear_value_count) {
      std::memcpy(args_ptr + clear_values_offset,
                  render_pass_begin->pClearValues,
                  sizeof(VkClearValue) * clear_value_count);
    }
  }

  void CmdVkBindDescriptorSets(VkPipelineBindPoint pipeline_bind_point,
                               VkPipelineLayout layout, uint32_t first_set,
                               uint32_t descriptor_set_count,
                               const VkDescriptorSet* descriptor_sets,
                               uint32_t dynamic_offset_count,
                               const uint32_t* dynamic_offsets) {
    size_t arguments_size =
        xe::align(sizeof(ArgsVkBindDescriptorSets), alignof(VkDescriptorSet));
    size_t descriptor_sets_offset = arguments_size;
    arguments_size += sizeof(VkDescriptorSet) * descriptor_set_count;
    size_t dynamic_offsets_offset = 0;
    if (dynamic_offset_count) {
      arguments_size = xe::align(arguments_size, alignof(uint32_t));
      dynamic_offsets_offset = arguments_size;
      arguments_size += sizeof(uint32_t) * dynamic_offset_count;
    }
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkBindDescriptorSets, arguments_size));
    auto& args = *reinterpret_cast<ArgsVkBindDescriptorSets*>(args_ptr);
    args.pipeline_bind_point = pipeline_bind_point;
    args.layout = layout;
    args.first_set = first_set;
    args.descriptor_set_count = descriptor_set_count;
    args.dynamic_offset_count = dynamic_offset_count;
    std::memcpy(args_ptr + descriptor_sets_offset, descriptor_sets,
                sizeof(VkDescriptorSet) * descriptor_set_count);
    if (dynamic_offset_count) {
      std::memcpy(args_ptr + dynamic_offsets_offset, dynamic_offsets,
                  sizeof(uint32_t) * dynamic_offset_count);
    }
  }

  void CmdVkBindIndexBuffer(VkBuffer buffer, VkDeviceSize offset,
                            VkIndexType index_type) {
    auto& args = *reinterpret_cast<ArgsVkBindIndexBuffer*>(WriteCommand(
        Command::kVkBindIndexBuffer, sizeof(ArgsVkBindIndexBuffer)));
    args.buffer = buffer;
    args.offset = offset;
    args.index_type = index_type;
  }

  void CmdVkBindPipeline(VkPipelineBindPoint pipeline_bind_point,
                         VkPipeline pipeline) {
    auto& args = *reinterpret_cast<ArgsVkBindPipeline*>(
        WriteCommand(Command::kVkBindPipeline, sizeof(ArgsVkBindPipeline)));
    args.pipeline_bind_point = pipeline_bind_point;
    args.pipeline = pipeline;
  }

  // Deferred-creation variant: records a STABLE pointer to a pipeline cache
  // slot (a pipelines_ map node's Pipeline::pipeline field) instead of a
  // handle value, dereferenced at Execute() time. The slot may still be
  // VK_NULL_HANDLE at replay if asynchronous creation hasn't finished (or
  // failed); in that case the bind is skipped and all graphics draws until the
  // next successful graphics pipeline bind are dropped. Used only for guest
  // graphics pipelines; the external / compute bind paths keep using the
  // by-value CmdVkBindPipeline.
  void CmdVkBindPipelineDeferred(VkPipelineBindPoint pipeline_bind_point,
                                 const std::atomic<VkPipeline>* pipeline) {
    auto& args = *reinterpret_cast<ArgsVkBindPipelineDeferred*>(WriteCommand(
        Command::kVkBindPipelineDeferred, sizeof(ArgsVkBindPipelineDeferred)));
    args.pipeline_bind_point = pipeline_bind_point;
    args.pipeline = pipeline;
  }

  void CmdVkBindVertexBuffers(uint32_t first_binding, uint32_t binding_count,
                              const VkBuffer* buffers,
                              const VkDeviceSize* offsets) {
    size_t arguments_size =
        xe::align(sizeof(ArgsVkBindVertexBuffers), alignof(VkBuffer));
    size_t buffers_offset = arguments_size;
    arguments_size =
        xe::align(arguments_size + sizeof(VkBuffer) * binding_count,
                  alignof(VkDeviceSize));
    size_t offsets_offset = arguments_size;
    arguments_size += sizeof(VkDeviceSize) * binding_count;
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkBindVertexBuffers, arguments_size));
    auto& args = *reinterpret_cast<ArgsVkBindVertexBuffers*>(args_ptr);
    args.first_binding = first_binding;
    args.binding_count = binding_count;
    std::memcpy(args_ptr + buffers_offset, buffers,
                sizeof(VkBuffer) * binding_count);
    std::memcpy(args_ptr + offsets_offset, offsets,
                sizeof(VkDeviceSize) * binding_count);
  }

  void CmdVkBeginQuery(VkQueryPool query_pool, uint32_t query,
                       VkQueryControlFlags flags) {
    auto& args = *reinterpret_cast<ArgsVkBeginQuery*>(
        WriteCommand(Command::kVkBeginQuery, sizeof(ArgsVkBeginQuery)));
    args.query_pool = query_pool;
    args.query = query;
    args.flags = flags;
  }

  void CmdVkEndQuery(VkQueryPool query_pool, uint32_t query) {
    auto& args = *reinterpret_cast<ArgsVkEndQuery*>(
        WriteCommand(Command::kVkEndQuery, sizeof(ArgsVkEndQuery)));
    args.query_pool = query_pool;
    args.query = query;
  }

  void CmdVkCopyQueryPoolResults(VkQueryPool query_pool, uint32_t first_query,
                                 uint32_t query_count, VkBuffer dst_buffer,
                                 VkDeviceSize dst_offset, VkDeviceSize stride,
                                 VkQueryResultFlags flags) {
    auto& args = *reinterpret_cast<ArgsVkCopyQueryPoolResults*>(WriteCommand(
        Command::kVkCopyQueryPoolResults, sizeof(ArgsVkCopyQueryPoolResults)));
    args.query_pool = query_pool;
    args.first_query = first_query;
    args.query_count = query_count;
    args.dst_buffer = dst_buffer;
    args.dst_offset = dst_offset;
    args.stride = stride;
    args.flags = flags;
  }

  void CmdVkResetQueryPool(VkQueryPool query_pool, uint32_t first_query,
                           uint32_t query_count) {
    auto& args = *reinterpret_cast<ArgsVkResetQueryPool*>(
        WriteCommand(Command::kVkResetQueryPool, sizeof(ArgsVkResetQueryPool)));
    args.query_pool = query_pool;
    args.first_query = first_query;
    args.query_count = query_count;
  }

  void CmdVkWriteTimestamp(VkPipelineStageFlagBits pipeline_stage,
                           VkQueryPool query_pool, uint32_t query) {
    auto& args = *reinterpret_cast<ArgsVkWriteTimestamp*>(WriteCommand(
        Command::kVkWriteTimestamp, sizeof(ArgsVkWriteTimestamp)));
    args.pipeline_stage = pipeline_stage;
    args.query_pool = query_pool;
    args.query = query;
  }

  void CmdClearAttachmentsEmplace(uint32_t attachment_count,
                                  VkClearAttachment*& attachments_out,
                                  uint32_t rect_count,
                                  VkClearRect*& rects_out) {
    size_t arguments_size =
        xe::align(sizeof(ArgsVkClearAttachments), alignof(VkClearAttachment));
    size_t attachments_offset = arguments_size;
    arguments_size =
        xe::align(arguments_size + sizeof(VkClearAttachment) * attachment_count,
                  alignof(VkClearRect));
    size_t rects_offset = arguments_size;
    arguments_size += sizeof(VkClearRect) * rect_count;
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkClearAttachments, arguments_size));
    auto& args = *reinterpret_cast<ArgsVkClearAttachments*>(args_ptr);
    args.attachment_count = attachment_count;
    args.rect_count = rect_count;
    attachments_out =
        reinterpret_cast<VkClearAttachment*>(args_ptr + attachments_offset);
    rects_out = reinterpret_cast<VkClearRect*>(args_ptr + rects_offset);
  }
  void CmdVkClearAttachments(uint32_t attachment_count,
                             const VkClearAttachment* attachments,
                             uint32_t rect_count, const VkClearRect* rects) {
    VkClearAttachment* attachments_arg;
    VkClearRect* rects_arg;
    CmdClearAttachmentsEmplace(attachment_count, attachments_arg, rect_count,
                               rects_arg);
    std::memcpy(attachments_arg, attachments,
                sizeof(VkClearAttachment) * attachment_count);
    std::memcpy(rects_arg, rects, sizeof(VkClearRect) * rect_count);
    for (uint32_t i = 0; i < rect_count; ++i) {
      AccumulateDrawnRect(rects[i].rect);
    }
  }

  VkImageSubresourceRange* CmdClearColorImageEmplace(
      VkImage image, VkImageLayout image_layout, const VkClearColorValue* color,
      uint32_t range_count) {
    const size_t header_size = xe::align(sizeof(ArgsVkClearColorImage),
                                         alignof(VkImageSubresourceRange));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkClearColorImage,
        header_size + sizeof(VkImageSubresourceRange) * range_count));
    auto& args = *reinterpret_cast<ArgsVkClearColorImage*>(args_ptr);
    args.image = image;
    args.image_layout = image_layout;
    args.color = *color;
    args.range_count = range_count;
    return reinterpret_cast<VkImageSubresourceRange*>(args_ptr + header_size);
  }
  void CmdVkClearColorImage(VkImage image, VkImageLayout image_layout,
                            const VkClearColorValue* color,
                            uint32_t range_count,
                            const VkImageSubresourceRange* ranges) {
    std::memcpy(
        CmdClearColorImageEmplace(image, image_layout, color, range_count),
        ranges, sizeof(VkImageSubresourceRange) * range_count);
  }

  VkBufferCopy* CmdCopyBufferEmplace(VkBuffer src_buffer, VkBuffer dst_buffer,
                                     uint32_t region_count) {
    const size_t header_size =
        xe::align(sizeof(ArgsVkCopyBuffer), alignof(VkBufferCopy));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkCopyBuffer,
                     header_size + sizeof(VkBufferCopy) * region_count));
    auto& args = *reinterpret_cast<ArgsVkCopyBuffer*>(args_ptr);
    args.src_buffer = src_buffer;
    args.dst_buffer = dst_buffer;
    args.region_count = region_count;
    return reinterpret_cast<VkBufferCopy*>(args_ptr + header_size);
  }
  void CmdVkCopyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer,
                       uint32_t region_count, const VkBufferCopy* regions) {
    std::memcpy(CmdCopyBufferEmplace(src_buffer, dst_buffer, region_count),
                regions, sizeof(VkBufferCopy) * region_count);
  }

  VkBufferImageCopy* CmdCopyBufferToImageEmplace(VkBuffer src_buffer,
                                                 VkImage dst_image,
                                                 VkImageLayout dst_image_layout,
                                                 uint32_t region_count) {
    const size_t header_size =
        xe::align(sizeof(ArgsVkCopyBufferToImage), alignof(VkBufferImageCopy));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkCopyBufferToImage,
                     header_size + sizeof(VkBufferImageCopy) * region_count));
    auto& args = *reinterpret_cast<ArgsVkCopyBufferToImage*>(args_ptr);
    args.src_buffer = src_buffer;
    args.dst_image = dst_image;
    args.dst_image_layout = dst_image_layout;
    args.region_count = region_count;
    return reinterpret_cast<VkBufferImageCopy*>(args_ptr + header_size);
  }
  void CmdVkCopyBufferToImage(VkBuffer src_buffer, VkImage dst_image,
                              VkImageLayout dst_image_layout,
                              uint32_t region_count,
                              const VkBufferImageCopy* regions) {
    std::memcpy(CmdCopyBufferToImageEmplace(src_buffer, dst_image,
                                            dst_image_layout, region_count),
                regions, sizeof(VkBufferImageCopy) * region_count);
  }

  void CmdVkFillBuffer(VkBuffer dst_buffer, VkDeviceSize dst_offset,
                       VkDeviceSize size, uint32_t data) {
    auto& args = *reinterpret_cast<ArgsVkFillBuffer*>(
        WriteCommand(Command::kVkFillBuffer, sizeof(ArgsVkFillBuffer)));
    args.dst_buffer = dst_buffer;
    args.dst_offset = dst_offset;
    args.size = size;
    args.data = data;
  }

  VkImageBlit* CmdBlitImageEmplace(VkImage src_image,
                                   VkImageLayout src_image_layout,
                                   VkImage dst_image,
                                   VkImageLayout dst_image_layout,
                                   uint32_t region_count, VkFilter filter) {
    const size_t header_size =
        xe::align(sizeof(ArgsVkBlitImage), alignof(VkImageBlit));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkBlitImage,
                     header_size + sizeof(VkImageBlit) * region_count));
    auto& args = *reinterpret_cast<ArgsVkBlitImage*>(args_ptr);
    args.src_image = src_image;
    args.src_image_layout = src_image_layout;
    args.dst_image = dst_image;
    args.dst_image_layout = dst_image_layout;
    args.region_count = region_count;
    args.filter = filter;
    return reinterpret_cast<VkImageBlit*>(args_ptr + header_size);
  }
  void CmdVkBlitImage(VkImage src_image, VkImageLayout src_image_layout,
                      VkImage dst_image, VkImageLayout dst_image_layout,
                      uint32_t region_count, const VkImageBlit* regions,
                      VkFilter filter) {
    std::memcpy(CmdBlitImageEmplace(src_image, src_image_layout, dst_image,
                                    dst_image_layout, region_count, filter),
                regions, sizeof(VkImageBlit) * region_count);
  }

  void CmdVkDispatch(uint32_t group_count_x, uint32_t group_count_y,
                     uint32_t group_count_z) {
    auto& args = *reinterpret_cast<ArgsVkDispatch*>(
        WriteCommand(Command::kVkDispatch, sizeof(ArgsVkDispatch)));
    args.group_count_x = group_count_x;
    args.group_count_y = group_count_y;
    args.group_count_z = group_count_z;
  }

  void CmdVkDraw(uint32_t vertex_count, uint32_t instance_count,
                 uint32_t first_vertex, uint32_t first_instance) {
    AccumulateDrawnScissor();
    auto& args = *reinterpret_cast<ArgsVkDraw*>(
        WriteCommand(Command::kVkDraw, sizeof(ArgsVkDraw)));
    args.vertex_count = vertex_count;
    args.instance_count = instance_count;
    args.first_vertex = first_vertex;
    args.first_instance = first_instance;
  }

  void CmdVkDrawIndexed(uint32_t index_count, uint32_t instance_count,
                        uint32_t first_index, int32_t vertex_offset,
                        uint32_t first_instance) {
    AccumulateDrawnScissor();
    auto& args = *reinterpret_cast<ArgsVkDrawIndexed*>(
        WriteCommand(Command::kVkDrawIndexed, sizeof(ArgsVkDrawIndexed)));
    args.index_count = index_count;
    args.instance_count = instance_count;
    args.first_index = first_index;
    args.vertex_offset = vertex_offset;
    args.first_instance = first_instance;
  }

  void CmdVkEndRenderPass() { WriteCommand(Command::kVkEndRenderPass, 0); }

  // Dynamic rendering (VK_KHR_dynamic_rendering / Vulkan 1.3)
  // Simpler than CmdVkBeginRenderPass - doesn't need
  // VkRenderPass/VkFramebuffer.
  void CmdVkBeginRendering(const VkRenderingInfo* rendering_info);
  void CmdVkEndRendering() { WriteCommand(Command::kVkEndRendering, 0); }

  // pNext of all barriers must be null.
  void CmdVkPipelineBarrier(VkPipelineStageFlags src_stage_mask,
                            VkPipelineStageFlags dst_stage_mask,
                            VkDependencyFlags dependency_flags,
                            uint32_t memory_barrier_count,
                            const VkMemoryBarrier* memory_barriers,
                            uint32_t buffer_memory_barrier_count,
                            const VkBufferMemoryBarrier* buffer_memory_barriers,
                            uint32_t image_memory_barrier_count,
                            const VkImageMemoryBarrier* image_memory_barriers);

  void CmdVkPushConstants(VkPipelineLayout layout,
                          VkShaderStageFlags stage_flags, uint32_t offset,
                          uint32_t size, const void* values) {
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkPushConstants, sizeof(ArgsVkPushConstants) + size));
    auto& args = *reinterpret_cast<ArgsVkPushConstants*>(args_ptr);
    args.layout = layout;
    args.stage_flags = stage_flags;
    args.offset = offset;
    args.size = size;
    std::memcpy(args_ptr + sizeof(ArgsVkPushConstants), values, size);
  }

  void CmdVkSetRenderingInputAttachmentIndices(
      uint32_t color_attachment_count, const uint32_t* color_attachment_input_indices) {
    auto& args = *reinterpret_cast<ArgsVkSetRenderingInputAttachmentIndices*>(
        WriteCommand(Command::kVkSetRenderingInputAttachmentIndices,
                     sizeof(ArgsVkSetRenderingInputAttachmentIndices)));
    args.color_attachment_count = color_attachment_count;
    for (uint32_t i = 0; i < 4; ++i) {
      args.color_attachment_input_indices[i] =
          i < color_attachment_count ? color_attachment_input_indices[i]
                                     : VK_ATTACHMENT_UNUSED;
    }
  }

  void CmdVkSetBlendConstants(const float* blend_constants) {
    auto& args = *reinterpret_cast<ArgsVkSetBlendConstants*>(WriteCommand(
        Command::kVkSetBlendConstants, sizeof(ArgsVkSetBlendConstants)));
    std::memcpy(args.blend_constants, blend_constants, sizeof(float) * 4);
  }

  void CmdVkSetDepthBias(float depth_bias_constant_factor,
                         float depth_bias_clamp,
                         float depth_bias_slope_factor) {
    auto& args = *reinterpret_cast<ArgsVkSetDepthBias*>(
        WriteCommand(Command::kVkSetDepthBias, sizeof(ArgsVkSetDepthBias)));
    args.depth_bias_constant_factor = depth_bias_constant_factor;
    args.depth_bias_clamp = depth_bias_clamp;
    args.depth_bias_slope_factor = depth_bias_slope_factor;
  }

  void CmdVkSetScissor(uint32_t first_scissor, uint32_t scissor_count,
                       const VkRect2D* scissors) {
    constexpr size_t header_size =
        xe::align(sizeof(ArgsVkSetScissor), alignof(VkRect2D));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkSetScissor,
                     header_size + sizeof(VkRect2D) * scissor_count));
    auto& args = *reinterpret_cast<ArgsVkSetScissor*>(args_ptr);
    args.first_scissor = first_scissor;
    args.scissor_count = scissor_count;
    std::memcpy(args_ptr + header_size, scissors,
                sizeof(VkRect2D) * scissor_count);
    // Dynamic state persists across render pass boundaries, so this is kept
    // outside the per-pass tracking.
    if (first_scissor == 0 && scissor_count) {
      current_scissor_ = scissors[0];
    }
  }

  void CmdVkSetStencilCompareMask(VkStencilFaceFlags face_mask,
                                  uint32_t compare_mask) {
    auto& args = *reinterpret_cast<ArgsSetStencilMaskReference*>(
        WriteCommand(Command::kVkSetStencilCompareMask,
                     sizeof(ArgsSetStencilMaskReference)));
    args.face_mask = face_mask;
    args.mask_reference = compare_mask;
  }

  void CmdVkSetStencilReference(VkStencilFaceFlags face_mask,
                                uint32_t reference) {
    auto& args = *reinterpret_cast<ArgsSetStencilMaskReference*>(WriteCommand(
        Command::kVkSetStencilReference, sizeof(ArgsSetStencilMaskReference)));
    args.face_mask = face_mask;
    args.mask_reference = reference;
  }

  void CmdVkSetStencilWriteMask(VkStencilFaceFlags face_mask,
                                uint32_t write_mask) {
    auto& args = *reinterpret_cast<ArgsSetStencilMaskReference*>(WriteCommand(
        Command::kVkSetStencilWriteMask, sizeof(ArgsSetStencilMaskReference)));
    args.face_mask = face_mask;
    args.mask_reference = write_mask;
  }

  void CmdVkSetViewport(uint32_t first_viewport, uint32_t viewport_count,
                        const VkViewport* viewports) {
    constexpr size_t header_size =
        xe::align(sizeof(ArgsVkSetViewport), alignof(VkViewport));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkSetViewport,
                     header_size + sizeof(VkViewport) * viewport_count));
    auto& args = *reinterpret_cast<ArgsVkSetViewport*>(args_ptr);
    args.first_viewport = first_viewport;
    args.viewport_count = viewport_count;
    std::memcpy(args_ptr + header_size, viewports,
                sizeof(VkViewport) * viewport_count);
  }

  // Extended dynamic state (EDS1/EDS2, core in Vulkan 1.3). Each setter mirrors
  // the value that the static pipeline would otherwise bake.
  void CmdVkSetCullMode(VkCullModeFlags cull_mode) {
    auto& args = *reinterpret_cast<ArgsVkSetCullMode*>(
        WriteCommand(Command::kVkSetCullMode, sizeof(ArgsVkSetCullMode)));
    args.cull_mode = cull_mode;
  }

  void CmdVkSetFrontFace(VkFrontFace front_face) {
    auto& args = *reinterpret_cast<ArgsVkSetFrontFace*>(
        WriteCommand(Command::kVkSetFrontFace, sizeof(ArgsVkSetFrontFace)));
    args.front_face = front_face;
  }

  void CmdVkSetPrimitiveTopology(VkPrimitiveTopology primitive_topology) {
    auto& args = *reinterpret_cast<ArgsVkSetPrimitiveTopology*>(WriteCommand(
        Command::kVkSetPrimitiveTopology, sizeof(ArgsVkSetPrimitiveTopology)));
    args.primitive_topology = primitive_topology;
  }

  void CmdVkSetPrimitiveRestartEnable(VkBool32 primitive_restart_enable) {
    auto& args = *reinterpret_cast<ArgsVkSetBool*>(
        WriteCommand(Command::kVkSetPrimitiveRestartEnable, sizeof(ArgsVkSetBool)));
    args.value = primitive_restart_enable;
  }

  void CmdVkSetDepthTestEnable(VkBool32 depth_test_enable) {
    auto& args = *reinterpret_cast<ArgsVkSetBool*>(
        WriteCommand(Command::kVkSetDepthTestEnable, sizeof(ArgsVkSetBool)));
    args.value = depth_test_enable;
  }

  void CmdVkSetDepthWriteEnable(VkBool32 depth_write_enable) {
    auto& args = *reinterpret_cast<ArgsVkSetBool*>(
        WriteCommand(Command::kVkSetDepthWriteEnable, sizeof(ArgsVkSetBool)));
    args.value = depth_write_enable;
  }

  void CmdVkSetDepthCompareOp(VkCompareOp depth_compare_op) {
    auto& args = *reinterpret_cast<ArgsVkSetDepthCompareOp*>(WriteCommand(
        Command::kVkSetDepthCompareOp, sizeof(ArgsVkSetDepthCompareOp)));
    args.depth_compare_op = depth_compare_op;
  }

  void CmdVkSetStencilTestEnable(VkBool32 stencil_test_enable) {
    auto& args = *reinterpret_cast<ArgsVkSetBool*>(
        WriteCommand(Command::kVkSetStencilTestEnable, sizeof(ArgsVkSetBool)));
    args.value = stencil_test_enable;
  }

  void CmdVkSetStencilOp(VkStencilFaceFlags face_mask, VkStencilOp fail_op,
                         VkStencilOp pass_op, VkStencilOp depth_fail_op,
                         VkCompareOp compare_op) {
    auto& args = *reinterpret_cast<ArgsVkSetStencilOp*>(
        WriteCommand(Command::kVkSetStencilOp, sizeof(ArgsVkSetStencilOp)));
    args.face_mask = face_mask;
    args.fail_op = fail_op;
    args.pass_op = pass_op;
    args.depth_fail_op = depth_fail_op;
    args.compare_op = compare_op;
  }

  // Extended dynamic state 3 (EDS3, VK_EXT_extended_dynamic_state3). Only
  // recorded when the corresponding sub-feature is supported.
  void CmdVkSetDepthClampEnableEXT(VkBool32 depth_clamp_enable) {
    auto& args = *reinterpret_cast<ArgsVkSetBool*>(WriteCommand(
        Command::kVkSetDepthClampEnableEXT, sizeof(ArgsVkSetBool)));
    args.value = depth_clamp_enable;
  }

  void CmdVkSetPolygonModeEXT(VkPolygonMode polygon_mode) {
    auto& args = *reinterpret_cast<ArgsVkSetPolygonModeEXT*>(WriteCommand(
        Command::kVkSetPolygonModeEXT, sizeof(ArgsVkSetPolygonModeEXT)));
    args.polygon_mode = polygon_mode;
  }

  void CmdVkSetColorBlendEnableEXT(uint32_t first_attachment,
                                   uint32_t attachment_count,
                                   const VkBool32* color_blend_enables) {
    constexpr size_t header_size =
        xe::align(sizeof(ArgsVkSetColorBlendEnableEXT), alignof(VkBool32));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkSetColorBlendEnableEXT,
        header_size + sizeof(VkBool32) * attachment_count));
    auto& args = *reinterpret_cast<ArgsVkSetColorBlendEnableEXT*>(args_ptr);
    args.first_attachment = first_attachment;
    args.attachment_count = attachment_count;
    std::memcpy(args_ptr + header_size, color_blend_enables,
                sizeof(VkBool32) * attachment_count);
  }

  void CmdVkSetColorBlendEquationEXT(
      uint32_t first_attachment, uint32_t attachment_count,
      const VkColorBlendEquationEXT* color_blend_equations) {
    constexpr size_t header_size = xe::align(
        sizeof(ArgsVkSetColorBlendEquationEXT), alignof(VkColorBlendEquationEXT));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkSetColorBlendEquationEXT,
        header_size + sizeof(VkColorBlendEquationEXT) * attachment_count));
    auto& args = *reinterpret_cast<ArgsVkSetColorBlendEquationEXT*>(args_ptr);
    args.first_attachment = first_attachment;
    args.attachment_count = attachment_count;
    std::memcpy(args_ptr + header_size, color_blend_equations,
                sizeof(VkColorBlendEquationEXT) * attachment_count);
  }

  void CmdVkSetColorWriteMaskEXT(uint32_t first_attachment,
                                 uint32_t attachment_count,
                                 const VkColorComponentFlags* color_write_masks) {
    constexpr size_t header_size = xe::align(
        sizeof(ArgsVkSetColorWriteMaskEXT), alignof(VkColorComponentFlags));
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kVkSetColorWriteMaskEXT,
        header_size + sizeof(VkColorComponentFlags) * attachment_count));
    auto& args = *reinterpret_cast<ArgsVkSetColorWriteMaskEXT*>(args_ptr);
    args.first_attachment = first_attachment;
    args.attachment_count = attachment_count;
    std::memcpy(args_ptr + header_size, color_write_masks,
                sizeof(VkColorComponentFlags) * attachment_count);
  }

  // Debug marker support for RenderDoc/debug tools annotation.
  void CmdVkBeginDebugUtilsLabelEXT(const char* label_name) {
    size_t label_len = std::strlen(label_name);
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkBeginDebugUtilsLabelEXT,
                     sizeof(ArgsVkDebugUtilsLabel) + label_len + 1));
    auto& args = *reinterpret_cast<ArgsVkDebugUtilsLabel*>(args_ptr);
    args.label_length = static_cast<uint32_t>(label_len);
    std::memcpy(args_ptr + sizeof(ArgsVkDebugUtilsLabel), label_name,
                label_len + 1);
  }

  void CmdVkEndDebugUtilsLabelEXT() {
    WriteCommand(Command::kVkEndDebugUtilsLabelEXT, 0);
  }

  void CmdVkInsertDebugUtilsLabelEXT(const char* label_name) {
    size_t label_len = std::strlen(label_name);
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kVkInsertDebugUtilsLabelEXT,
                     sizeof(ArgsVkDebugUtilsLabel) + label_len + 1));
    auto& args = *reinterpret_cast<ArgsVkDebugUtilsLabel*>(args_ptr);
    args.label_length = static_cast<uint32_t>(label_len);
    std::memcpy(args_ptr + sizeof(ArgsVkDebugUtilsLabel), label_name,
                label_len + 1);
  }

 private:
  enum class Command {
    kVkBeginRenderPass,
    kVkBindDescriptorSets,
    kVkBindIndexBuffer,
    kVkBindPipeline,
    kVkBindPipelineDeferred,
    kVkBindVertexBuffers,
    kVkBeginQuery,
    kVkEndQuery,
    kVkCopyQueryPoolResults,
    kVkResetQueryPool,
    kVkWriteTimestamp,
    kVkClearAttachments,
    kVkClearColorImage,
    kVkCopyBuffer,
    kVkCopyBufferToImage,
    kVkFillBuffer,
    kVkBlitImage,
    kVkDispatch,
    kVkDraw,
    kVkDrawIndexed,
    kVkEndRenderPass,
    kVkBeginRendering,
    kVkEndRendering,
    kVkPipelineBarrier,
    kVkPushConstants,
    kVkSetBlendConstants,
    kVkSetRenderingInputAttachmentIndices,
    kVkSetDepthBias,
    kVkSetScissor,
    kVkSetStencilCompareMask,
    kVkSetStencilReference,
    kVkSetStencilWriteMask,
    kVkSetViewport,
    kVkSetCullMode,
    kVkSetFrontFace,
    kVkSetPrimitiveTopology,
    kVkSetPrimitiveRestartEnable,
    kVkSetDepthTestEnable,
    kVkSetDepthWriteEnable,
    kVkSetDepthCompareOp,
    kVkSetStencilTestEnable,
    kVkSetStencilOp,
    kVkSetDepthClampEnableEXT,
    kVkSetPolygonModeEXT,
    kVkSetColorBlendEnableEXT,
    kVkSetColorBlendEquationEXT,
    kVkSetColorWriteMaskEXT,
    kVkBeginDebugUtilsLabelEXT,
    kVkEndDebugUtilsLabelEXT,
    kVkInsertDebugUtilsLabelEXT,
  };

  struct CommandHeader {
    Command command;
    uint32_t arguments_size_elements;
  };
  static constexpr size_t kCommandHeaderSizeElements =
      (sizeof(CommandHeader) + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);

  struct ArgsVkBeginRenderPass {
    VkRenderPass render_pass;
    VkFramebuffer framebuffer;
    VkRect2D render_area;
    uint32_t clear_value_count;
    VkSubpassContents contents;
    // Followed by aligned optional VkClearValue[].
    static_assert(alignof(VkClearValue) <= alignof(uintmax_t));
  };

  // For VK_KHR_dynamic_rendering / Vulkan 1.3.
  struct ArgsVkBeginRendering {
    VkRenderingFlags flags;
    VkRect2D render_area;
    uint32_t layer_count;
    uint32_t view_mask;
    uint32_t color_attachment_count;
    bool has_depth_attachment;
    bool has_stencil_attachment;
    // Followed by:
    // - VkRenderingAttachmentInfo[color_attachment_count] for color attachments
    // - VkRenderingAttachmentInfo for depth (if has_depth_attachment)
    // - VkRenderingAttachmentInfo for stencil (if has_stencil_attachment)
    static_assert(alignof(VkRenderingAttachmentInfo) <= alignof(uintmax_t));
  };

  struct ArgsVkBindDescriptorSets {
    VkPipelineBindPoint pipeline_bind_point;
    VkPipelineLayout layout;
    uint32_t first_set;
    uint32_t descriptor_set_count;
    uint32_t dynamic_offset_count;
    // Followed by aligned VkDescriptorSet[], optional uint32_t[].
    static_assert(alignof(VkDescriptorSet) <= alignof(uintmax_t));
  };

  struct ArgsVkBindIndexBuffer {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkIndexType index_type;
  };

  struct ArgsVkBindPipeline {
    VkPipelineBindPoint pipeline_bind_point;
    VkPipeline pipeline;
  };

  struct ArgsVkBindPipelineDeferred {
    VkPipelineBindPoint pipeline_bind_point;
    // Stable pointer to a pipeline slot; loaded (acquire) at Execute() time.
    const std::atomic<VkPipeline>* pipeline;
  };

  struct ArgsVkBindVertexBuffers {
    uint32_t first_binding;
    uint32_t binding_count;
    // Followed by aligned VkBuffer[], VkDeviceSize[].
    static_assert(alignof(VkBuffer) <= alignof(uintmax_t));
    static_assert(alignof(VkDeviceSize) <= alignof(uintmax_t));
  };

  struct ArgsVkBeginQuery {
    VkQueryPool query_pool;
    uint32_t query;
    VkQueryControlFlags flags;
  };

  struct ArgsVkEndQuery {
    VkQueryPool query_pool;
    uint32_t query;
  };

  struct ArgsVkCopyQueryPoolResults {
    VkQueryPool query_pool;
    uint32_t first_query;
    uint32_t query_count;
    VkBuffer dst_buffer;
    VkDeviceSize dst_offset;
    VkDeviceSize stride;
    VkQueryResultFlags flags;
  };

  struct ArgsVkResetQueryPool {
    VkQueryPool query_pool;
    uint32_t first_query;
    uint32_t query_count;
  };

  struct ArgsVkWriteTimestamp {
    VkPipelineStageFlagBits pipeline_stage;
    VkQueryPool query_pool;
    uint32_t query;
  };

  struct ArgsVkClearAttachments {
    uint32_t attachment_count;
    uint32_t rect_count;
    // Followed by aligned VkClearAttachment[], VkClearRect[].
    static_assert(alignof(VkClearAttachment) <= alignof(uintmax_t));
    static_assert(alignof(VkClearRect) <= alignof(uintmax_t));
  };

  struct ArgsVkClearColorImage {
    VkImage image;
    VkImageLayout image_layout;
    VkClearColorValue color;
    uint32_t range_count;
    // Followed by aligned VkImageSubresourceRange[].
    static_assert(alignof(VkImageSubresourceRange) <= alignof(uintmax_t));
  };

  struct ArgsVkCopyBuffer {
    VkBuffer src_buffer;
    VkBuffer dst_buffer;
    uint32_t region_count;
    // Followed by aligned VkBufferCopy[].
    static_assert(alignof(VkBufferCopy) <= alignof(uintmax_t));
  };

  struct ArgsVkCopyBufferToImage {
    VkBuffer src_buffer;
    VkImage dst_image;
    VkImageLayout dst_image_layout;
    uint32_t region_count;
    // Followed by aligned VkBufferImageCopy[].
    static_assert(alignof(VkBufferImageCopy) <= alignof(uintmax_t));
  };

  struct ArgsVkFillBuffer {
    VkBuffer dst_buffer;
    VkDeviceSize dst_offset;
    VkDeviceSize size;
    uint32_t data;
  };

  struct ArgsVkBlitImage {
    VkImage src_image;
    VkImageLayout src_image_layout;
    VkImage dst_image;
    VkImageLayout dst_image_layout;
    uint32_t region_count;
    VkFilter filter;
    // Followed by aligned VkImageBlit[].
    static_assert(alignof(VkImageBlit) <= alignof(uintmax_t));
  };

  struct ArgsVkDispatch {
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
  };

  struct ArgsVkDraw {
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
  };

  struct ArgsVkDrawIndexed {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t first_instance;
  };

  struct ArgsVkPipelineBarrier {
    VkPipelineStageFlags src_stage_mask;
    VkPipelineStageFlags dst_stage_mask;
    VkDependencyFlags dependency_flags;
    uint32_t memory_barrier_count;
    uint32_t buffer_memory_barrier_count;
    uint32_t image_memory_barrier_count;
    // Followed by aligned optional VkMemoryBarrier[],
    // optional VkBufferMemoryBarrier[], optional VkImageMemoryBarrier[].
    static_assert(alignof(VkMemoryBarrier) <= alignof(uintmax_t));
    static_assert(alignof(VkBufferMemoryBarrier) <= alignof(uintmax_t));
    static_assert(alignof(VkImageMemoryBarrier) <= alignof(uintmax_t));
  };

  struct ArgsVkPushConstants {
    VkPipelineLayout layout;
    VkShaderStageFlags stage_flags;
    uint32_t offset;
    uint32_t size;
    // Followed by `size` bytes of values.
  };

  struct ArgsVkSetBlendConstants {
    float blend_constants[4];
  };

  struct ArgsVkSetRenderingInputAttachmentIndices {
    uint32_t color_attachment_count;
    uint32_t color_attachment_input_indices[4];
  };

  struct ArgsVkSetDepthBias {
    float depth_bias_constant_factor;
    float depth_bias_clamp;
    float depth_bias_slope_factor;
  };

  struct ArgsVkSetScissor {
    uint32_t first_scissor;
    uint32_t scissor_count;
    // Followed by aligned VkRect2D[].
    static_assert(alignof(VkRect2D) <= alignof(uintmax_t));
  };

  struct ArgsSetStencilMaskReference {
    VkStencilFaceFlags face_mask;
    uint32_t mask_reference;
  };

  struct ArgsVkSetViewport {
    uint32_t first_viewport;
    uint32_t viewport_count;
    // Followed by aligned VkViewport[].
    static_assert(alignof(VkViewport) <= alignof(uintmax_t));
  };

  // Extended dynamic state argument structures.
  struct ArgsVkSetBool {
    VkBool32 value;
  };

  struct ArgsVkSetCullMode {
    VkCullModeFlags cull_mode;
  };

  struct ArgsVkSetFrontFace {
    VkFrontFace front_face;
  };

  struct ArgsVkSetPrimitiveTopology {
    VkPrimitiveTopology primitive_topology;
  };

  struct ArgsVkSetDepthCompareOp {
    VkCompareOp depth_compare_op;
  };

  struct ArgsVkSetStencilOp {
    VkStencilFaceFlags face_mask;
    VkStencilOp fail_op;
    VkStencilOp pass_op;
    VkStencilOp depth_fail_op;
    VkCompareOp compare_op;
  };

  struct ArgsVkSetPolygonModeEXT {
    VkPolygonMode polygon_mode;
  };

  struct ArgsVkSetColorBlendEnableEXT {
    uint32_t first_attachment;
    uint32_t attachment_count;
    // Followed by aligned VkBool32[].
    static_assert(alignof(VkBool32) <= alignof(uintmax_t));
  };

  struct ArgsVkSetColorBlendEquationEXT {
    uint32_t first_attachment;
    uint32_t attachment_count;
    // Followed by aligned VkColorBlendEquationEXT[].
    static_assert(alignof(VkColorBlendEquationEXT) <= alignof(uintmax_t));
  };

  struct ArgsVkSetColorWriteMaskEXT {
    uint32_t first_attachment;
    uint32_t attachment_count;
    // Followed by aligned VkColorComponentFlags[].
    static_assert(alignof(VkColorComponentFlags) <= alignof(uintmax_t));
  };

  struct ArgsVkDebugUtilsLabel {
    uint32_t label_length;
    // Followed by null-terminated label string.
  };

  void* WriteCommand(Command command, size_t arguments_size_bytes);

 public:
  // Shrinks the last recorded begin-pass render area to the union of the
  // pass's draw scissors and clear rects, rounded up to the granularity. Only
  // ever shrinks; a pass with no recorded draws keeps the full extent. Must be
  // called before the stream is executed, while the args are still patchable.
  void ShrinkRenderAreaToDrawn(uint32_t granularity_width,
                               uint32_t granularity_height);

 private:
  // Offset of the last recorded begin-pass argument struct, in stream elements
  // (not a pointer - the stream vector reallocates as it grows).
  static constexpr size_t kNoRenderAreaPatch = SIZE_MAX;
  size_t render_area_patch_offset_ = kNoRenderAreaPatch;
  bool render_area_patch_dynamic_ = false;
  VkRect2D current_scissor_ = {};
  uint32_t pass_drawn_width_ = 0;
  uint32_t pass_drawn_height_ = 0;
  uint32_t pass_recorded_draws_ = 0;

  void BeginRenderAreaTracking(size_t arguments_size_bytes,
                               bool dynamic_rendering) {
    const size_t arguments_size_elements =
        (arguments_size_bytes + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);
    render_area_patch_offset_ = command_stream_size_ - arguments_size_elements;
    render_area_patch_dynamic_ = dynamic_rendering;
    pass_drawn_width_ = 0;
    pass_drawn_height_ = 0;
    pass_recorded_draws_ = 0;
  }

  void AccumulateDrawnRect(const VkRect2D& rect) {
    if (render_area_patch_offset_ == kNoRenderAreaPatch) {
      return;
    }
    pass_drawn_width_ =
        std::max(pass_drawn_width_,
                 uint32_t(std::max(0, rect.offset.x)) + rect.extent.width);
    pass_drawn_height_ =
        std::max(pass_drawn_height_,
                 uint32_t(std::max(0, rect.offset.y)) + rect.extent.height);
  }

  void AccumulateDrawnScissor() {
    if (render_area_patch_offset_ == kNoRenderAreaPatch) {
      return;
    }
    ++pass_recorded_draws_;
    AccumulateDrawnRect(current_scissor_);
  }


  const VulkanCommandProcessor& command_processor_;

  // uintmax_t to ensure uint64_t and pointer alignment of all structures.
  std::vector<uintmax_t> command_stream_;
  // Logical end of the recorded stream. With the size-cursor optimization the
  // vector is grown geometrically and never shrunk, so its size() is the
  // capacity of the arena, not the recorded length — vector::resize()'s
  // zero-fill on every WriteCommand was ~2% of the GPU command processor
  // thread. Maintained correctly in both modes.
  size_t command_stream_size_ = 0;
  // Snapshot of cvars::vulkan_deferred_cmd_size_cursor at construction.
  bool use_size_cursor_;
};

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_DEFERRED_COMMAND_BUFFER_H_

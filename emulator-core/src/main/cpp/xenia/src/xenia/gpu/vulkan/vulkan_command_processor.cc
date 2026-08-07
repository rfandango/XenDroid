/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/vulkan/vulkan_command_processor.h"

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstring>

#include "xenia/base/assert.h"
#include "xenia/base/byte_order.h"
#include "xenia/base/frame_stats.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/base/profiling.h"
#include "xenia/base/xxhash.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/spirv_shader_translator.h"
#include "xenia/gpu/vulkan/vulkan_pipeline_cache.h"
#include "xenia/gpu/vulkan/vulkan_render_target_cache.h"
#include "xenia/gpu/vulkan/vulkan_shader.h"
#include "xenia/gpu/vulkan/vulkan_shared_memory.h"
#include "xenia/gpu/vulkan/vulkan_zpd_query_pool.h"
#include "xenia/gpu/xenos.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/user_module.h"
#include "xenia/ui/vulkan/vulkan_presenter.h"
#include "xenia/ui/vulkan/vulkan_util.h"

DECLARE_bool(clear_memory_page_state);
DECLARE_bool(vulkan_in_pass_resolve_debug_read_usage);
DECLARE_bool(log_gpu_frame_time_breakdown);

DEFINE_int32(
    vulkan_mid_frame_submission_draws, 1300,
    "If greater than 0, end and submit the current command buffer after this "
    "many draws instead of only at the swap, so the GPU overlaps the frame's "
    "rendering with the building of the rest of its command stream. 0 keeps "
    "one submission per frame. Splitting closes the current render pass, so "
    "values that are too small hurt tiled GPUs; try roughly half the title's "
    "per-frame draw count.",
    "GPU");
UPDATE_from_int32(vulkan_mid_frame_submission_draws, 2026, 7, 24, 12, 0);

DEFINE_int32(
    vulkan_vrs_blended, 0,
    "Shade selected draws at a coarse fragment rate: 0 off, 1 = 2x1, "
    "2 = 2x2, 3 = 4x2, 4 = 4x4. Rates above 2x2 are clamped to what the "
    "device reports as supported, which on Adreno means 2x2 on any "
    "multisampled target.",
    "Vulkan");
DEFINE_int32(
    vulkan_vrs_min_shader_dwords, 48,
    "Skip coarse shading for draws whose pixel shader is smaller than this "
    "many guest microcode dwords, roughly three per instruction. Trivial "
    "shaders such as text and UI blits gain nothing from it. 0 disables the "
    "check.",
    "Vulkan");
DEFINE_int32(
    vulkan_vrs_scope, 0,
    "Which draws coarse shading applies to: 0 every draw, 1 depth-tested "
    "draws only, 2 blended draws only, 3 both. UI and text are kept sharp by "
    "vulkan_vrs_min_shader_dwords, not by this; set both to 0 to coarsen "
    "literally every draw.",
    "Vulkan");
DEFINE_bool(
    vulkan_cache_texture_descriptors, true,
    "Skip re-writing and re-binding the texture/sampler descriptor sets on "
    "draws whose resolved image views and samplers have not changed since the "
    "last write. Disable to force a descriptor-set write and bind every draw "
    "(pre-optimization behavior) for debugging texture corruption.",
    "Vulkan");

// A/B selector for the cross-draw texture/sampler descriptor-set reuse gate.
// Default (false) keeps XenDroid's content-hash gate; true switches to upstream
// edge's bitmask+shader-pointer+sampler-vector gate. Only consulted when
// vulkan_cache_texture_descriptors is on. Both schemes coexist so the cvar can
// be flipped live for comparison (see the gate and write-back in UpdateBindings
// for the switch-safety invariants).
DEFINE_bool(
    vulkan_texture_descriptor_reuse_edge, false,
    "When vulkan_cache_texture_descriptors is on, use upstream edge's "
    "bitmask+shader-pointer+sampler-vector descriptor-set reuse gate instead of "
    "XenDroid's content-hash gate. For A/B comparison; default false = the hash "
    "gate.",
    "Vulkan");

DEFINE_bool(
    vulkan_skip_redundant_fetch_constant_writes, true,
    "Don't invalidate texture bindings and the fetch/bool-loop constant "
    "buffers when a register write doesn't change the register's value. "
    "Games commonly re-emit identical fetch constants every draw, and the "
    "derived state is a pure function of the register contents.",
    "Vulkan");

DEFINE_bool(
    vulkan_cache_sampler_parameters, true,
    "Reuse sampler parameters and VkSampler handles across draws, re-deriving "
    "them only for fetch constants written since the previous draw (or on "
    "shader sampler-layout change, new submission, or sampler destruction). "
    "Disable to derive and look up every sampler on every draw "
    "(pre-optimization behavior).",
    "Vulkan");

DEFINE_bool(
    vulkan_fast_register_ranges, true,
    "Process PM4 register range writes with bulk byte-swapped copies and "
    "range-level constant dirty tracking (port of the D3D12 backend's "
    "register-write fast path) instead of a virtual per-register call. "
    "Disable to route every register through WriteRegister "
    "(pre-optimization behavior).",
    "Vulkan");

DEFINE_bool(
    vulkan_allow_reverse_z, true,
    "Pass viewports with minDepth > maxDepth (reverse depth ranges, used by "
    "many games for inverse-Z) directly to the driver. Vulkan allows this, "
    "but some drivers mishandle it (the Direct3D 12 backend always "
    "normalizes, citing drivers where it works and drivers where it "
    "doesn't). Disable to normalize the range and compensate in the NDC "
    "transform like the D3D12 backend does.",
    "Vulkan");

DEFINE_bool(
    vulkan_dynamic_constant_buffers, false,
    "Bind the guest constant buffers as VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC "
    "so only the per-draw dynamic offsets vary; the constants descriptor set is "
    "re-allocated and re-written only when an upload-pool page rolls over. "
    "DEFAULT OFF: on Qualcomm proprietary Vulkan drivers (Adreno 650/740/830 "
    "tested, 2026-06) shaders read ZEROS through the dynamic-offset uniform "
    "descriptors, breaking all rendering (black output / garbage), proven by "
    "RenderDoc pixel-history bisection; Mesa/Turnip reads them correctly. "
    "Safe to enable on Turnip for a small command-processor win. Has no "
    "effect on devices that report maxDescriptorSetUniformBuffersDynamic < 7 "
    "(the guest constant-buffer count).",
    "Vulkan");

DECLARE_bool(gpu_debug_markers);
DECLARE_bool(readback_memexport_fast);
DECLARE_bool(submit_on_primary_buffer_end);

DEFINE_bool(
    vulkan_dynamic_rendering, true,
    "Use VK_KHR_dynamic_rendering instead of traditional render passes. "
    "May improve or worsen performance depending on driver. Requires Vulkan "
    "1.3 or VK_KHR_dynamic_rendering extension support.",
    "Vulkan");

namespace xe {
namespace gpu {
namespace vulkan {

// Generated with `xb buildshaders`.
namespace shaders {
#include "xenia/gpu/shaders/bytecode/vulkan_spirv/apply_gamma_pwl_cs.h"
#include "xenia/gpu/shaders/bytecode/vulkan_spirv/apply_gamma_pwl_fxaa_luma_cs.h"
#include "xenia/gpu/shaders/bytecode/vulkan_spirv/apply_gamma_table_cs.h"
#include "xenia/gpu/shaders/bytecode/vulkan_spirv/apply_gamma_table_fxaa_luma_cs.h"
#include "xenia/gpu/shaders/bytecode/vulkan_spirv/fxaa_cs.h"
#include "xenia/gpu/shaders/bytecode/vulkan_spirv/fxaa_extreme_cs.h"
#include "xenia/gpu/shaders/bytecode/vulkan_spirv/resolve_downscale_cs.h"
}  // namespace shaders

constexpr VkDescriptorPoolSize
    VulkanCommandProcessor::kDescriptorPoolSizeUniformBuffer = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        SpirvShaderTranslator::kConstantBufferCount*
            kLinkedTypeDescriptorPoolSetCount};

constexpr VkDescriptorPoolSize
    VulkanCommandProcessor::kDescriptorPoolSizeStorageBuffer = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kLinkedTypeDescriptorPoolSetCount};

// 2x descriptors for texture images because of unsigned and signed bindings.
constexpr VkDescriptorPoolSize
    VulkanCommandProcessor::kDescriptorPoolSizeTextures[2] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         2 * kLinkedTypeDescriptorPoolSetCount},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kLinkedTypeDescriptorPoolSetCount},
};

VulkanCommandProcessor::VulkanCommandProcessor(
    VulkanGraphicsSystem* graphics_system, kernel::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state),
      completion_timeline_(static_cast<const ui::vulkan::VulkanProvider*>(
                               graphics_system->provider())
                               ->vulkan_device(),
                           "cp"),
      transfer_completion_timeline_(
          static_cast<const ui::vulkan::VulkanProvider*>(
              graphics_system->provider())
              ->vulkan_device(),
          "cp-transfer"),
      deferred_command_buffer_(*this),
      deferred_setup_command_buffer_(*this, 64 * 1024),
      transient_descriptor_allocator_uniform_buffer_(
          static_cast<const ui::vulkan::VulkanProvider*>(
              graphics_system->provider())
              ->vulkan_device(),
          &kDescriptorPoolSizeUniformBuffer, 1,
          kLinkedTypeDescriptorPoolSetCount),
      transient_descriptor_allocator_storage_buffer_(
          static_cast<const ui::vulkan::VulkanProvider*>(
              graphics_system->provider())
              ->vulkan_device(),
          &kDescriptorPoolSizeStorageBuffer, 1,
          kLinkedTypeDescriptorPoolSetCount),
      transient_descriptor_allocator_textures_(
          static_cast<const ui::vulkan::VulkanProvider*>(
              graphics_system->provider())
              ->vulkan_device(),
          kDescriptorPoolSizeTextures,
          uint32_t(xe::countof(kDescriptorPoolSizeTextures)),
          kLinkedTypeDescriptorPoolSetCount) {}

VulkanCommandProcessor::~VulkanCommandProcessor() = default;

void VulkanCommandProcessor::UpdateDebugMarkersEnabled() {
  // Enable debug markers if the CVAR is set or RenderDoc is detected.
  debug_markers_enabled_ = IsGpuDebugMarkersEnabled();
}

void VulkanCommandProcessor::PushDebugMarker(const char* format, ...) {
  if (!debug_markers_enabled_) {
    return;
  }
  char label[256];
  va_list args;
  va_start(args, format);
  vsnprintf(label, sizeof(label), format, args);
  va_end(args);
  deferred_command_buffer_.CmdVkBeginDebugUtilsLabelEXT(label);
}

void VulkanCommandProcessor::PopDebugMarker() {
  if (!debug_markers_enabled_) {
    return;
  }
  deferred_command_buffer_.CmdVkEndDebugUtilsLabelEXT();
}

void VulkanCommandProcessor::InsertDebugMarker(const char* format, ...) {
  if (!debug_markers_enabled_) {
    return;
  }
  char label[256];
  va_list args;
  va_start(args, format);
  vsnprintf(label, sizeof(label), format, args);
  va_end(args);
  deferred_command_buffer_.CmdVkInsertDebugUtilsLabelEXT(label);
}

void VulkanCommandProcessor::ClearCaches() {
  CommandProcessor::ClearCaches();
  cache_clear_requested_ = true;
}

void VulkanCommandProcessor::InvalidateGpuMemory() {
  shared_memory_->InvalidateAllPages();
}

void VulkanCommandProcessor::ClearReadbackBuffers() {
  // Pending copies hold pointers into readback_buffers_, and in-flight transfer
  // copies target its buffers, so drain and drop them before clearing.
  transfer_completion_timeline_.AwaitAllSubmissions();
  pending_readback_copies_.clear();
  readback_buffers_.clear();
  memexport_readback_buffers_.clear();
}

void VulkanCommandProcessor::TracePlaybackWroteMemory(uint32_t base_ptr,
                                                      uint32_t length) {
  shared_memory_->MemoryInvalidationCallback(base_ptr, length, true);
  primitive_processor_->MemoryInvalidationCallback(base_ptr, length, true);
}

void VulkanCommandProcessor::InitializeShaderStorage(
    const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
    std::function<void()> completion_callback) {
  CommandProcessor::InitializeShaderStorage(cache_root, title_id, blocking,
                                            nullptr);
  pipeline_cache_->InitializeShaderStorage(cache_root, title_id, blocking,
                                           std::move(completion_callback));
}

void VulkanCommandProcessor::RestoreEdramSnapshot(const void* snapshot) {}

void VulkanCommandProcessor::PollCompletedSubmission() {
  // Strict ZPD can skip unnecessary work here that can wait for the next full
  // CheckSubmissionCompletionAndDeviceLoss and it's not needed for retirement.
  if (device_lost_) {
    return;
  }
  completion_timeline_.AwaitSubmissionAndUpdateCompleted(
      GetCompletedSubmission());
  PumpQueryResolves();
}

std::string VulkanCommandProcessor::GetTitleStateSuffix() const {
  if (!render_target_cache_) {
    return {};
  }
  std::ostringstream suffix;
  switch (render_target_cache_->GetPath()) {
    case RenderTargetCache::Path::kHostRenderTargets:
      suffix << " - FBO";
      break;
    case RenderTargetCache::Path::kPixelShaderInterlock:
      suffix << " - FSI";
      break;
    default:
      break;
  }
  uint32_t draw_resolution_scale_x =
      texture_cache_ ? texture_cache_->draw_resolution_scale_x() : 1;
  uint32_t draw_resolution_scale_y =
      texture_cache_ ? texture_cache_->draw_resolution_scale_y() : 1;
  if (draw_resolution_scale_x > 1 || draw_resolution_scale_y > 1) {
    suffix << ' ' << draw_resolution_scale_x << 'x' << draw_resolution_scale_y;
  }
  return suffix.str();
}

bool VulkanCommandProcessor::SetupContext() {
  if (!CommandProcessor::SetupContext()) {
    XELOGE("Failed to initialize base command processor context");
    return false;
  }

  // Check if debug markers should be enabled (CVAR or RenderDoc detection).
  UpdateDebugMarkersEnabled();
  if (debug_markers_enabled_) {
    XELOGI("GPU debug markers enabled for RenderDoc/debug tools");
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const ui::vulkan::VulkanDevice::Properties& device_properties =
      vulkan_device->properties();

  if (device_properties.pipelineFragmentShadingRate) {
    vk_cmd_set_fragment_shading_rate_ =
        reinterpret_cast<PFN_vkCmdSetFragmentShadingRateKHR>(
            vulkan_device->vulkan_instance()->functions().vkGetDeviceProcAddr(
                device, "vkCmdSetFragmentShadingRateKHR"));
    if (!vk_cmd_set_fragment_shading_rate_) {
      XELOGW(
          "Device reports pipelineFragmentShadingRate but provides no "
          "vkCmdSetFragmentShadingRateKHR; coarse shading disabled");
    }
    auto get_rates = reinterpret_cast<
        PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR>(
        vulkan_device->vulkan_instance()
            ->functions()
            .vkGetInstanceProcAddr(
                vulkan_device->vulkan_instance()->instance(),
                "vkGetPhysicalDeviceFragmentShadingRatesKHR"));
    if (get_rates) {
      uint32_t rate_count = 0;
      get_rates(vulkan_device->physical_device(), &rate_count, nullptr);
      std::vector<VkPhysicalDeviceFragmentShadingRateKHR> rates(
          rate_count, {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR});
      get_rates(vulkan_device->physical_device(), &rate_count, rates.data());
      static const VkExtent2D kRateTable[] = {
          {1, 1}, {2, 1}, {2, 2}, {4, 2}, {4, 4}};
      for (const auto& r : rates) {
        for (uint32_t i = 0; i < xe::countof(kRateTable); ++i) {
          if (r.fragmentSize.width == kRateTable[i].width &&
              r.fragmentSize.height == kRateTable[i].height) {
            supported_shading_rate_samples_[i] = r.sampleCounts;
          }
        }
      }
      XELOGI(
          "VRS rates by sample-count mask: 2x1={:#x} 2x2={:#x} 4x2={:#x} "
          "4x4={:#x}",
          supported_shading_rate_samples_[1], supported_shading_rate_samples_[2],
          supported_shading_rate_samples_[3],
          supported_shading_rate_samples_[4]);
    }
  }

  // The unconditional inclusion of the vertex shader stage also covers the case
  // of manual index / factor buffer fetch (the system constants and the shared
  // memory are needed for that) in the tessellation vertex shader when
  // fullDrawIndexUint32 is not supported.
  guest_shader_pipeline_stages_ = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  guest_shader_vertex_stages_ = VK_SHADER_STAGE_VERTEX_BIT;
  if (device_properties.tessellationShader) {
    guest_shader_pipeline_stages_ |=
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    guest_shader_vertex_stages_ |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
  }
  if (!device_properties.vertexPipelineStoresAndAtomics) {
    // For memory export from vertex shaders converted to compute shaders.
    guest_shader_pipeline_stages_ |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    guest_shader_vertex_stages_ |= VK_SHADER_STAGE_COMPUTE_BIT;
  }

  // 16384 is bigger than any single uniform buffer that Xenia needs, but is the
  // minimum maxUniformBufferRange, thus the safe minimum amount.
  uniform_buffer_pool_ = std::make_unique<ui::vulkan::VulkanUploadBufferPool>(
      vulkan_device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      xe::align(std::max(ui::GraphicsUploadBufferPool::kDefaultPageSize,
                         size_t(16384)),
                size_t(device_properties.minUniformBufferOffsetAlignment)));

  // Descriptor set layouts that don't depend on the setup of other subsystems.
  VkShaderStageFlags guest_shader_stages =
      guest_shader_vertex_stages_ | VK_SHADER_STAGE_FRAGMENT_BIT;
  // Empty.
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = nullptr;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = 0;
  descriptor_set_layout_create_info.pBindings = nullptr;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layout_empty_) != VK_SUCCESS) {
    XELOGE("Failed to create an empty Vulkan descriptor set layout");
    return false;
  }
  // Guest draw constants.
  // Dynamic constant buffers capability gate, resolved once before the constants layout is
  // created and held constant for the device lifetime. The Vulkan-guaranteed
  // minimum for maxDescriptorSetUniformBuffersDynamic is 8, so this passes on
  // all conformant hardware; if it ever fails, the binding stays plain
  // UNIFORM_BUFFER and the original per-draw write+bind path runs unchanged.
  use_dynamic_constants_ =
      cvars::vulkan_dynamic_constant_buffers &&
      device_properties.maxDescriptorSetUniformBuffersDynamic >=
          SpirvShaderTranslator::kConstantBufferCount;
  const VkDescriptorType constants_descriptor_type =
      use_dynamic_constants_ ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                             : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  VkDescriptorSetLayoutBinding descriptor_set_layout_bindings_constants
      [SpirvShaderTranslator::kConstantBufferCount] = {};
  for (uint32_t i = 0; i < SpirvShaderTranslator::kConstantBufferCount; ++i) {
    VkDescriptorSetLayoutBinding& constants_binding =
        descriptor_set_layout_bindings_constants[i];
    constants_binding.binding = i;
    constants_binding.descriptorType = constants_descriptor_type;
    constants_binding.descriptorCount = 1;
    constants_binding.pImmutableSamplers = nullptr;
  }
  descriptor_set_layout_bindings_constants
      [SpirvShaderTranslator::kConstantBufferSystem]
          .stageFlags =
      // Visible to the vertex/domain stages for user clip planes and to the
      // control/eval/vertex stages for tessellation, plus the usual guest
      // stages.
      guest_shader_stages | guest_shader_vertex_stages_ |
      (device_properties.tessellationShader
           ? (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
              VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
           : 0) |
      (device_properties.geometryShader ? VK_SHADER_STAGE_GEOMETRY_BIT : 0);
  descriptor_set_layout_bindings_constants
      [SpirvShaderTranslator::kConstantBufferFloatVertex]
          .stageFlags = guest_shader_vertex_stages_;
  descriptor_set_layout_bindings_constants
      [SpirvShaderTranslator::kConstantBufferFloatPixel]
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  descriptor_set_layout_bindings_constants
      [SpirvShaderTranslator::kConstantBufferBoolLoop]
          .stageFlags = guest_shader_stages;
  descriptor_set_layout_bindings_constants
      [SpirvShaderTranslator::kConstantBufferFetch]
          .stageFlags = guest_shader_stages;
  descriptor_set_layout_create_info.bindingCount =
      uint32_t(xe::countof(descriptor_set_layout_bindings_constants));
  descriptor_set_layout_create_info.pBindings =
      descriptor_set_layout_bindings_constants;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layout_constants_) != VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan descriptor set layout for guest draw "
        "constant buffers");
    return false;
  }
  // Transient: storage buffer for compute shaders.
  VkDescriptorSetLayoutBinding descriptor_set_layout_binding_transient;
  descriptor_set_layout_binding_transient.binding = 0;
  descriptor_set_layout_binding_transient.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_set_layout_binding_transient.descriptorCount = 1;
  descriptor_set_layout_binding_transient.stageFlags =
      VK_SHADER_STAGE_COMPUTE_BIT;
  descriptor_set_layout_binding_transient.pImmutableSamplers = nullptr;
  descriptor_set_layout_create_info.bindingCount = 1;
  descriptor_set_layout_create_info.pBindings =
      &descriptor_set_layout_binding_transient;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layouts_single_transient_[size_t(
              SingleTransientDescriptorLayout::kStorageBufferCompute)]) !=
      VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan descriptor set layout for a storage buffer "
        "bound to the compute shader");
    return false;
  }
  descriptor_set_layout_binding_transient.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layouts_single_transient_[size_t(
              SingleTransientDescriptorLayout::kStorageBufferFragment)]) !=
      VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan descriptor set layout for a storage buffer "
        "bound to the fragment shader");
    return false;
  }
  // Transient: storage image for the resolve-to-texture fragment variant.
  descriptor_set_layout_binding_transient.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layouts_single_transient_[size_t(
              SingleTransientDescriptorLayout::kStorageImageFragment)]) !=
      VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan descriptor set layout for a storage image "
        "bound to the fragment shader");
    return false;
  }
  descriptor_set_layout_binding_transient.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_set_layout_binding_transient.stageFlags =
      VK_SHADER_STAGE_COMPUTE_BIT;
  descriptor_set_layout_binding_transient.binding = 1;
  descriptor_set_layout_binding_transient.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layouts_single_transient_[size_t(
              SingleTransientDescriptorLayout::kUniformBufferComputeB1)]) !=
      VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan descriptor set layout for a uniform buffer "
        "bound to binding 1 in the compute shader");
    return false;
  }

  shared_memory_ = std::make_unique<VulkanSharedMemory>(
      *this, *memory_, trace_writer_, guest_shader_pipeline_stages_);
  if (!shared_memory_->Initialize()) {
    XELOGE("Failed to initialize shared memory");
    return false;
  }

  primitive_processor_ = std::make_unique<VulkanPrimitiveProcessor>(
      *register_file_, *memory_, trace_writer_, *shared_memory_, *this);
  if (!primitive_processor_->Initialize()) {
    XELOGE("Failed to initialize the geometric primitive processor");
    return false;
  }

  uint32_t shared_memory_binding_count_log2 =
      SpirvShaderTranslator::GetSharedMemoryStorageBufferCountLog2(
          device_properties.maxStorageBufferRange);
  uint32_t shared_memory_binding_count = UINT32_C(1)
                                         << shared_memory_binding_count_log2;

  // Requires the transient descriptor set layouts.
  // Get draw resolution scale and clamp based on device capabilities
  uint32_t draw_resolution_scale_x, draw_resolution_scale_y;
  bool draw_resolution_scale_not_clamped =
      TextureCache::GetConfigDrawResolutionScale(draw_resolution_scale_x,
                                                 draw_resolution_scale_y);
  // Check if sparse binding is supported for resolution scaling
  bool has_sparse_binding = device_properties.sparseBinding &&
                            device_properties.sparseResidencyBuffer;
  if (!TextureCache::ClampDrawResolutionScaleToMaxSupported(
          draw_resolution_scale_x, draw_resolution_scale_y, has_sparse_binding,
          0)) {
    draw_resolution_scale_not_clamped = false;
  }
  if (!draw_resolution_scale_not_clamped) {
    XELOGW(
        "The requested draw resolution scale is not supported by the device or "
        "the emulator, reducing to {}x{}",
        draw_resolution_scale_x, draw_resolution_scale_y);
  }

  render_target_cache_ = std::make_unique<VulkanRenderTargetCache>(
      *register_file_, *memory_, trace_writer_, draw_resolution_scale_x,
      draw_resolution_scale_y, *this);
  if (!render_target_cache_->Initialize(shared_memory_binding_count)) {
    XELOGE("Failed to initialize the render target cache");
    return false;
  }

  // Shared memory, EDRAM, and ZPD FSI counter descriptor set layout.
  bool edram_fragment_shader_interlock =
      render_target_cache_->GetPath() ==
      RenderTargetCache::Path::kPixelShaderInterlock;
  VkDescriptorSetLayoutBinding
      shared_memory_and_edram_descriptor_set_layout_bindings[3];
  shared_memory_and_edram_descriptor_set_layout_bindings[0].binding = 0;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].descriptorCount =
      shared_memory_binding_count;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].stageFlags =
      guest_shader_stages;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].pImmutableSamplers =
      nullptr;
  VkDescriptorSetLayoutCreateInfo
      shared_memory_and_edram_descriptor_set_layout_create_info;
  shared_memory_and_edram_descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  shared_memory_and_edram_descriptor_set_layout_create_info.pNext = nullptr;
  shared_memory_and_edram_descriptor_set_layout_create_info.flags = 0;
  shared_memory_and_edram_descriptor_set_layout_create_info.pBindings =
      shared_memory_and_edram_descriptor_set_layout_bindings;
  if (edram_fragment_shader_interlock) {
    // EDRAM.
    shared_memory_and_edram_descriptor_set_layout_bindings[1].binding = 1;
    shared_memory_and_edram_descriptor_set_layout_bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    shared_memory_and_edram_descriptor_set_layout_bindings[1].descriptorCount =
        1;
    shared_memory_and_edram_descriptor_set_layout_bindings[1].stageFlags =
        VK_SHADER_STAGE_FRAGMENT_BIT;
    shared_memory_and_edram_descriptor_set_layout_bindings[1]
        .pImmutableSamplers = nullptr;
    shared_memory_and_edram_descriptor_set_layout_create_info.bindingCount = 2;
    // ZPD FSI counter.
    shared_memory_and_edram_descriptor_set_layout_bindings[2].binding = 2;
    shared_memory_and_edram_descriptor_set_layout_bindings[2].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    shared_memory_and_edram_descriptor_set_layout_bindings[2].descriptorCount =
        1;
    shared_memory_and_edram_descriptor_set_layout_bindings[2].stageFlags =
        VK_SHADER_STAGE_FRAGMENT_BIT;
    shared_memory_and_edram_descriptor_set_layout_bindings[2]
        .pImmutableSamplers = nullptr;
    shared_memory_and_edram_descriptor_set_layout_create_info.bindingCount = 3;
  } else {
    shared_memory_and_edram_descriptor_set_layout_create_info.bindingCount = 1;
  }
  if (dfn.vkCreateDescriptorSetLayout(
          device, &shared_memory_and_edram_descriptor_set_layout_create_info,
          nullptr,
          &descriptor_set_layout_shared_memory_and_edram_) != VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan descriptor set layout for the shared memory "
        "and the EDRAM");
    return false;
  }

  pipeline_cache_ = std::make_unique<VulkanPipelineCache>(
      *this, *register_file_, *render_target_cache_,
      guest_shader_vertex_stages_);
  if (!pipeline_cache_->Initialize()) {
    XELOGE("Failed to initialize the graphics pipeline cache");
    return false;
  }

  // Requires the transient descriptor set layouts.
  // Use the same draw resolution scale as render target cache
  texture_cache_ = VulkanTextureCache::Create(
      *register_file_, *shared_memory_, draw_resolution_scale_x,
      draw_resolution_scale_y, *this, guest_shader_pipeline_stages_);
  if (!texture_cache_) {
    XELOGE("Failed to initialize the texture cache");
    return false;
  }

  // Needed by NormalizeSampleCount.
  zpd_draw_resolution_scale_x_ = draw_resolution_scale_x;
  zpd_draw_resolution_scale_y_ = draw_resolution_scale_y;

  const VkDeviceSize zpd_fsi_counter_sink_range =
      sizeof(uint32_t) * kZPDQueryPoolCapacity;
  if (edram_fragment_shader_interlock) {
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, zpd_fsi_counter_sink_range,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            ui::vulkan::util::MemoryPurpose::kDeviceLocal,
            zpd_fsi_counter_sink_buffer_,
            zpd_fsi_counter_sink_buffer_memory_)) {
      XELOGE("Failed to create the ZPD FSI counter sink buffer");
      return false;
    }
  }

  // Shared memory, EDRAM, and ZPD FSI counter common bindings.
  VkDescriptorPoolSize descriptor_pool_sizes[1];
  descriptor_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_pool_sizes[0].descriptorCount =
      shared_memory_binding_count +
      2u * uint32_t(edram_fragment_shader_interlock);
  VkDescriptorPoolCreateInfo descriptor_pool_create_info;
  descriptor_pool_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptor_pool_create_info.pNext = nullptr;
  descriptor_pool_create_info.flags = 0;
  descriptor_pool_create_info.maxSets = 1;
  descriptor_pool_create_info.poolSizeCount = 1;
  descriptor_pool_create_info.pPoolSizes = descriptor_pool_sizes;
  if (dfn.vkCreateDescriptorPool(device, &descriptor_pool_create_info, nullptr,
                                 &shared_memory_and_edram_descriptor_pool_) !=
      VK_SUCCESS) {
    XELOGE(
        "Failed to create the Vulkan descriptor pool for shared memory and "
        "EDRAM");
    return false;
  }
  VkDescriptorSetAllocateInfo descriptor_set_allocate_info;
  descriptor_set_allocate_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_set_allocate_info.pNext = nullptr;
  descriptor_set_allocate_info.descriptorPool =
      shared_memory_and_edram_descriptor_pool_;
  descriptor_set_allocate_info.descriptorSetCount = 1;
  descriptor_set_allocate_info.pSetLayouts =
      &descriptor_set_layout_shared_memory_and_edram_;
  if (dfn.vkAllocateDescriptorSets(device, &descriptor_set_allocate_info,
                                   &shared_memory_and_edram_descriptor_set_) !=
      VK_SUCCESS) {
    XELOGE(
        "Failed to allocate the Vulkan descriptor set for shared memory and "
        "EDRAM");
    return false;
  }
  VkDescriptorBufferInfo
      shared_memory_descriptor_buffers_info[SharedMemory::kBufferSize /
                                            (128 << 20)];
  uint32_t shared_memory_binding_range =
      SharedMemory::kBufferSize >> shared_memory_binding_count_log2;
  for (uint32_t i = 0; i < shared_memory_binding_count; ++i) {
    VkDescriptorBufferInfo& shared_memory_descriptor_buffer_info =
        shared_memory_descriptor_buffers_info[i];
    shared_memory_descriptor_buffer_info.buffer = shared_memory_->buffer();
    shared_memory_descriptor_buffer_info.offset =
        shared_memory_binding_range * i;
    shared_memory_descriptor_buffer_info.range = shared_memory_binding_range;
  }
  VkWriteDescriptorSet write_descriptor_sets[3];
  VkWriteDescriptorSet& write_descriptor_set_shared_memory =
      write_descriptor_sets[0];
  write_descriptor_set_shared_memory.sType =
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set_shared_memory.pNext = nullptr;
  write_descriptor_set_shared_memory.dstSet =
      shared_memory_and_edram_descriptor_set_;
  write_descriptor_set_shared_memory.dstBinding = 0;
  write_descriptor_set_shared_memory.dstArrayElement = 0;
  write_descriptor_set_shared_memory.descriptorCount =
      shared_memory_binding_count;
  write_descriptor_set_shared_memory.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write_descriptor_set_shared_memory.pImageInfo = nullptr;
  write_descriptor_set_shared_memory.pBufferInfo =
      shared_memory_descriptor_buffers_info;
  write_descriptor_set_shared_memory.pTexelBufferView = nullptr;
  VkDescriptorBufferInfo edram_descriptor_buffer_info;
  if (edram_fragment_shader_interlock) {
    edram_descriptor_buffer_info.buffer = render_target_cache_->edram_buffer();
    edram_descriptor_buffer_info.offset = 0;
    edram_descriptor_buffer_info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet& write_descriptor_set_edram = write_descriptor_sets[1];
    write_descriptor_set_edram.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_edram.pNext = nullptr;
    write_descriptor_set_edram.dstSet = shared_memory_and_edram_descriptor_set_;
    write_descriptor_set_edram.dstBinding = 1;
    write_descriptor_set_edram.dstArrayElement = 0;
    write_descriptor_set_edram.descriptorCount = 1;
    write_descriptor_set_edram.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_edram.pImageInfo = nullptr;
    write_descriptor_set_edram.pBufferInfo = &edram_descriptor_buffer_info;
    write_descriptor_set_edram.pTexelBufferView = nullptr;
    // ZPD FSI counter.
    VkDescriptorBufferInfo zpd_fsi_counter_descriptor_buffer_info;
    zpd_fsi_counter_descriptor_buffer_info.buffer =
        zpd_fsi_counter_sink_buffer_;
    zpd_fsi_counter_descriptor_buffer_info.offset = 0;
    zpd_fsi_counter_descriptor_buffer_info.range = zpd_fsi_counter_sink_range;
    // Keep binding 2 valid until the real counter buffer is ready.
    VkWriteDescriptorSet& write_descriptor_set_zpd_fsi_counter_init =
        write_descriptor_sets[2];
    write_descriptor_set_zpd_fsi_counter_init.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_zpd_fsi_counter_init.pNext = nullptr;
    write_descriptor_set_zpd_fsi_counter_init.dstSet =
        shared_memory_and_edram_descriptor_set_;
    write_descriptor_set_zpd_fsi_counter_init.dstBinding = 2;
    write_descriptor_set_zpd_fsi_counter_init.dstArrayElement = 0;
    write_descriptor_set_zpd_fsi_counter_init.descriptorCount = 1;
    write_descriptor_set_zpd_fsi_counter_init.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_zpd_fsi_counter_init.pImageInfo = nullptr;
    write_descriptor_set_zpd_fsi_counter_init.pBufferInfo =
        &zpd_fsi_counter_descriptor_buffer_info;
    write_descriptor_set_zpd_fsi_counter_init.pTexelBufferView = nullptr;
  }
  dfn.vkUpdateDescriptorSets(device,
                             1 + 2 * uint32_t(edram_fragment_shader_interlock),
                             write_descriptor_sets, 0, nullptr);
  if (edram_fragment_shader_interlock) {
    zpd_fsi_counter_descriptor_buffer_ = zpd_fsi_counter_sink_buffer_;
    zpd_fsi_counter_descriptor_range_ = zpd_fsi_counter_sink_range;
  }

  // Swap objects.

  // Gamma ramp, either device-local and host-visible at once, or separate
  // device-local texel buffer and host-visible upload buffer.
  gamma_ramp_256_entry_table_current_frame_ = UINT32_MAX;
  gamma_ramp_pwl_current_frame_ = UINT32_MAX;
  // Try to create a device-local host-visible buffer first, to skip copying.
  constexpr uint32_t kGammaRampSize256EntryTable = sizeof(uint32_t) * 256;
  constexpr uint32_t kGammaRampSizePWL = sizeof(uint16_t) * 2 * 3 * 128;
  constexpr uint32_t kGammaRampSize =
      kGammaRampSize256EntryTable + kGammaRampSizePWL;
  VkBufferCreateInfo gamma_ramp_host_visible_buffer_create_info;
  gamma_ramp_host_visible_buffer_create_info.sType =
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  gamma_ramp_host_visible_buffer_create_info.pNext = nullptr;
  gamma_ramp_host_visible_buffer_create_info.flags = 0;
  gamma_ramp_host_visible_buffer_create_info.size =
      kGammaRampSize * kMaxFramesInFlight;
  gamma_ramp_host_visible_buffer_create_info.usage =
      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
  gamma_ramp_host_visible_buffer_create_info.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE;
  gamma_ramp_host_visible_buffer_create_info.queueFamilyIndexCount = 0;
  gamma_ramp_host_visible_buffer_create_info.pQueueFamilyIndices = nullptr;
  if (dfn.vkCreateBuffer(device, &gamma_ramp_host_visible_buffer_create_info,
                         nullptr, &gamma_ramp_buffer_) == VK_SUCCESS) {
    bool use_gamma_ramp_host_visible_buffer = false;
    VkMemoryRequirements gamma_ramp_host_visible_buffer_memory_requirements;
    dfn.vkGetBufferMemoryRequirements(
        device, gamma_ramp_buffer_,
        &gamma_ramp_host_visible_buffer_memory_requirements);
    uint32_t gamma_ramp_host_visible_buffer_memory_types =
        gamma_ramp_host_visible_buffer_memory_requirements.memoryTypeBits &
        (vulkan_device->memory_types().device_local &
         vulkan_device->memory_types().host_visible);
    VkMemoryAllocateInfo gamma_ramp_host_visible_buffer_memory_allocate_info;
    // Prefer a host-uncached (because it's write-only) memory type, but try a
    // host-cached host-visible device-local one as well.
    if (xe::bit_scan_forward(
            gamma_ramp_host_visible_buffer_memory_types &
                ~vulkan_device->memory_types().host_cached,
            &(gamma_ramp_host_visible_buffer_memory_allocate_info
                  .memoryTypeIndex)) ||
        xe::bit_scan_forward(
            gamma_ramp_host_visible_buffer_memory_types,
            &(gamma_ramp_host_visible_buffer_memory_allocate_info
                  .memoryTypeIndex))) {
      VkMemoryAllocateInfo*
          gamma_ramp_host_visible_buffer_memory_allocate_info_last =
              &gamma_ramp_host_visible_buffer_memory_allocate_info;
      gamma_ramp_host_visible_buffer_memory_allocate_info.sType =
          VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      gamma_ramp_host_visible_buffer_memory_allocate_info.pNext = nullptr;
      gamma_ramp_host_visible_buffer_memory_allocate_info.allocationSize =
          gamma_ramp_host_visible_buffer_memory_requirements.size;
      VkMemoryDedicatedAllocateInfo
          gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info;
      if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
        gamma_ramp_host_visible_buffer_memory_allocate_info_last->pNext =
            &gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info;
        gamma_ramp_host_visible_buffer_memory_allocate_info_last =
            reinterpret_cast<VkMemoryAllocateInfo*>(
                &gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info);
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.sType =
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.pNext =
            nullptr;
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.image =
            VK_NULL_HANDLE;
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.buffer =
            gamma_ramp_buffer_;
      }
      if (dfn.vkAllocateMemory(
              device, &gamma_ramp_host_visible_buffer_memory_allocate_info,
              nullptr, &gamma_ramp_buffer_memory_) == VK_SUCCESS) {
        if (dfn.vkBindBufferMemory(device, gamma_ramp_buffer_,
                                   gamma_ramp_buffer_memory_,
                                   0) == VK_SUCCESS) {
          if (dfn.vkMapMemory(device, gamma_ramp_buffer_memory_, 0,
                              VK_WHOLE_SIZE, 0,
                              &gamma_ramp_upload_mapping_) == VK_SUCCESS) {
            use_gamma_ramp_host_visible_buffer = true;
            gamma_ramp_upload_memory_size_ =
                gamma_ramp_host_visible_buffer_memory_allocate_info
                    .allocationSize;
            gamma_ramp_upload_memory_type_ =
                gamma_ramp_host_visible_buffer_memory_allocate_info
                    .memoryTypeIndex;
          }
        }
        if (!use_gamma_ramp_host_visible_buffer) {
          dfn.vkFreeMemory(device, gamma_ramp_buffer_memory_, nullptr);
          gamma_ramp_buffer_memory_ = VK_NULL_HANDLE;
        }
      }
    }
    if (!use_gamma_ramp_host_visible_buffer) {
      dfn.vkDestroyBuffer(device, gamma_ramp_buffer_, nullptr);
      gamma_ramp_buffer_ = VK_NULL_HANDLE;
    }
  }
  if (gamma_ramp_buffer_ == VK_NULL_HANDLE) {
    // Create separate buffers for the shader and uploading.
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, kGammaRampSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
            ui::vulkan::util::MemoryPurpose::kDeviceLocal, gamma_ramp_buffer_,
            gamma_ramp_buffer_memory_)) {
      XELOGE("Failed to create the gamma ramp buffer");
      return false;
    }
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, kGammaRampSize * kMaxFramesInFlight,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            ui::vulkan::util::MemoryPurpose::kUpload, gamma_ramp_upload_buffer_,
            gamma_ramp_upload_buffer_memory_, &gamma_ramp_upload_memory_type_,
            &gamma_ramp_upload_memory_size_)) {
      XELOGE("Failed to create the gamma ramp upload buffer");
      return false;
    }
    if (dfn.vkMapMemory(device, gamma_ramp_upload_buffer_memory_, 0,
                        VK_WHOLE_SIZE, 0,
                        &gamma_ramp_upload_mapping_) != VK_SUCCESS) {
      XELOGE("Failed to map the gamma ramp upload buffer");
      return false;
    }
  }

  // Gamma ramp buffer views.
  uint32_t gamma_ramp_frame_count =
      gamma_ramp_upload_buffer_ == VK_NULL_HANDLE ? kMaxFramesInFlight : 1;
  VkBufferViewCreateInfo gamma_ramp_buffer_view_create_info;
  gamma_ramp_buffer_view_create_info.sType =
      VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
  gamma_ramp_buffer_view_create_info.pNext = nullptr;
  gamma_ramp_buffer_view_create_info.flags = 0;
  gamma_ramp_buffer_view_create_info.buffer = gamma_ramp_buffer_;
  // 256-entry table.
  gamma_ramp_buffer_view_create_info.format =
      VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  gamma_ramp_buffer_view_create_info.range = kGammaRampSize256EntryTable;
  for (uint32_t i = 0; i < gamma_ramp_frame_count; ++i) {
    gamma_ramp_buffer_view_create_info.offset = kGammaRampSize * i;
    if (dfn.vkCreateBufferView(device, &gamma_ramp_buffer_view_create_info,
                               nullptr, &gamma_ramp_buffer_views_[i * 2]) !=
        VK_SUCCESS) {
      XELOGE("Failed to create a 256-entry table gamma ramp buffer view");
      return false;
    }
  }
  // Piecewise linear.
  gamma_ramp_buffer_view_create_info.format = VK_FORMAT_R16G16_UINT;
  gamma_ramp_buffer_view_create_info.range = kGammaRampSizePWL;
  for (uint32_t i = 0; i < gamma_ramp_frame_count; ++i) {
    gamma_ramp_buffer_view_create_info.offset =
        kGammaRampSize * i + kGammaRampSize256EntryTable;
    if (dfn.vkCreateBufferView(device, &gamma_ramp_buffer_view_create_info,
                               nullptr, &gamma_ramp_buffer_views_[i * 2 + 1]) !=
        VK_SUCCESS) {
      XELOGE("Failed to create a PWL gamma ramp buffer view");
      return false;
    }
  }

  // Swap descriptor set layouts.
  VkDescriptorSetLayoutBinding swap_descriptor_set_layout_binding;
  swap_descriptor_set_layout_binding.binding = 0;
  swap_descriptor_set_layout_binding.descriptorCount = 1;
  swap_descriptor_set_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  swap_descriptor_set_layout_binding.pImmutableSamplers = nullptr;
  VkDescriptorSetLayoutCreateInfo swap_descriptor_set_layout_create_info;
  swap_descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  swap_descriptor_set_layout_create_info.pNext = nullptr;
  swap_descriptor_set_layout_create_info.flags = 0;
  swap_descriptor_set_layout_create_info.bindingCount = 1;
  swap_descriptor_set_layout_create_info.pBindings =
      &swap_descriptor_set_layout_binding;
  swap_descriptor_set_layout_binding.descriptorType =
      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &swap_descriptor_set_layout_create_info, nullptr,
          &swap_descriptor_set_layout_sampled_image_) != VK_SUCCESS) {
    XELOGE(
        "Failed to create the presentation sampled image descriptor set "
        "layout");
    return false;
  }
  swap_descriptor_set_layout_binding.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &swap_descriptor_set_layout_create_info, nullptr,
          &swap_descriptor_set_layout_uniform_texel_buffer_) != VK_SUCCESS) {
    XELOGE(
        "Failed to create the presentation uniform texel buffer descriptor set "
        "layout");
    return false;
  }
  swap_descriptor_set_layout_binding.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &swap_descriptor_set_layout_create_info, nullptr,
          &swap_descriptor_set_layout_storage_image_) != VK_SUCCESS) {
    XELOGE(
        "Failed to create the presentation storage image descriptor set "
        "layout");
    return false;
  }
  // FXAA source descriptor set layout (combined image sampler for linear
  // filtering).
  swap_descriptor_set_layout_binding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &swap_descriptor_set_layout_create_info, nullptr,
          &fxaa_source_descriptor_set_layout_) != VK_SUCCESS) {
    XELOGE("Failed to create the FXAA source descriptor set layout");
    return false;
  }

  // Swap descriptor pool.
  std::array<VkDescriptorPoolSize, 4> swap_descriptor_pool_sizes;
  VkDescriptorPoolCreateInfo swap_descriptor_pool_create_info;
  swap_descriptor_pool_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  swap_descriptor_pool_create_info.pNext = nullptr;
  swap_descriptor_pool_create_info.flags = 0;
  swap_descriptor_pool_create_info.maxSets = 0;
  swap_descriptor_pool_create_info.poolSizeCount = 0;
  swap_descriptor_pool_create_info.pPoolSizes =
      swap_descriptor_pool_sizes.data();
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_sampled_image =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info
                                       .poolSizeCount++];
    swap_descriptor_pool_size_sampled_image.type =
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    // Source images.
    swap_descriptor_pool_size_sampled_image.descriptorCount =
        kMaxFramesInFlight;
    swap_descriptor_pool_create_info.maxSets += kMaxFramesInFlight;
  }
  // 256-entry table and PWL gamma ramps. If the gamma ramp buffer is
  // host-visible, for multiple frames.
  uint32_t gamma_ramp_buffer_view_count = 2 * gamma_ramp_frame_count;
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_uniform_texel_buffer =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info
                                       .poolSizeCount++];
    swap_descriptor_pool_size_uniform_texel_buffer.type =
        VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    swap_descriptor_pool_size_uniform_texel_buffer.descriptorCount =
        gamma_ramp_buffer_view_count;
    swap_descriptor_pool_create_info.maxSets += gamma_ramp_buffer_view_count;
  }
  // Destination storage images for compute shader output.
  // Also includes storage image for writing to FXAA source (gamma+luma output).
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_storage_image =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info
                                       .poolSizeCount++];
    swap_descriptor_pool_size_storage_image.type =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swap_descriptor_pool_size_storage_image.descriptorCount =
        kMaxFramesInFlight * 2;  // dest + FXAA source storage
    swap_descriptor_pool_create_info.maxSets += kMaxFramesInFlight * 2;
  }
  // FXAA source combined image samplers.
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_combined_image_sampler =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info
                                       .poolSizeCount++];
    swap_descriptor_pool_size_combined_image_sampler.type =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    swap_descriptor_pool_size_combined_image_sampler.descriptorCount =
        kMaxFramesInFlight;
    swap_descriptor_pool_create_info.maxSets += kMaxFramesInFlight;
  }
  if (dfn.vkCreateDescriptorPool(device, &swap_descriptor_pool_create_info,
                                 nullptr,
                                 &swap_descriptor_pool_) != VK_SUCCESS) {
    XELOGE("Failed to create the presentation descriptor pool");
    return false;
  }

  // Swap descriptor set allocation.
  VkDescriptorSetAllocateInfo swap_descriptor_set_allocate_info;
  swap_descriptor_set_allocate_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  swap_descriptor_set_allocate_info.pNext = nullptr;
  swap_descriptor_set_allocate_info.descriptorPool = swap_descriptor_pool_;
  swap_descriptor_set_allocate_info.descriptorSetCount = 1;
  swap_descriptor_set_allocate_info.pSetLayouts =
      &swap_descriptor_set_layout_uniform_texel_buffer_;
  for (uint32_t i = 0; i < gamma_ramp_buffer_view_count; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_gamma_ramp_[i]) !=
        VK_SUCCESS) {
      XELOGE("Failed to allocate the gamma ramp descriptor sets");
      return false;
    }
  }
  swap_descriptor_set_allocate_info.pSetLayouts =
      &swap_descriptor_set_layout_sampled_image_;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_source_[i]) !=
        VK_SUCCESS) {
      XELOGE(
          "Failed to allocate the presentation source image descriptor sets");
      return false;
    }
  }
  swap_descriptor_set_allocate_info.pSetLayouts =
      &swap_descriptor_set_layout_storage_image_;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_dest_[i]) !=
        VK_SUCCESS) {
      XELOGE(
          "Failed to allocate the presentation destination image descriptor "
          "sets");
      return false;
    }
  }
  swap_descriptor_set_allocate_info.pSetLayouts =
      &fxaa_source_descriptor_set_layout_;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &fxaa_source_descriptors_[i]) !=
        VK_SUCCESS) {
      XELOGE("Failed to allocate the FXAA source image descriptor sets");
      return false;
    }
  }
  // Allocate storage image descriptor sets for writing to FXAA source.
  swap_descriptor_set_allocate_info.pSetLayouts =
      &swap_descriptor_set_layout_storage_image_;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &fxaa_source_storage_descriptors_[i]) !=
        VK_SUCCESS) {
      XELOGE("Failed to allocate the FXAA source storage descriptor sets");
      return false;
    }
  }

  // Gamma ramp descriptor sets.
  VkWriteDescriptorSet gamma_ramp_write_descriptor_set;
  gamma_ramp_write_descriptor_set.sType =
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  gamma_ramp_write_descriptor_set.pNext = nullptr;
  gamma_ramp_write_descriptor_set.dstBinding = 0;
  gamma_ramp_write_descriptor_set.dstArrayElement = 0;
  gamma_ramp_write_descriptor_set.descriptorCount = 1;
  gamma_ramp_write_descriptor_set.descriptorType =
      VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
  gamma_ramp_write_descriptor_set.pImageInfo = nullptr;
  gamma_ramp_write_descriptor_set.pBufferInfo = nullptr;
  for (uint32_t i = 0; i < gamma_ramp_buffer_view_count; ++i) {
    gamma_ramp_write_descriptor_set.dstSet = swap_descriptors_gamma_ramp_[i];
    gamma_ramp_write_descriptor_set.pTexelBufferView =
        &gamma_ramp_buffer_views_[i];
    dfn.vkUpdateDescriptorSets(device, 1, &gamma_ramp_write_descriptor_set, 0,
                               nullptr);
  }

  // Gamma ramp application compute pipeline layout.
  std::array<VkDescriptorSetLayout, kSwapApplyGammaDescriptorSetCount>
      swap_apply_gamma_descriptor_set_layouts{};
  swap_apply_gamma_descriptor_set_layouts[kSwapApplyGammaDescriptorSetRamp] =
      swap_descriptor_set_layout_uniform_texel_buffer_;
  swap_apply_gamma_descriptor_set_layouts[kSwapApplyGammaDescriptorSetSource] =
      swap_descriptor_set_layout_sampled_image_;
  swap_apply_gamma_descriptor_set_layouts[kSwapApplyGammaDescriptorSetDest] =
      swap_descriptor_set_layout_storage_image_;
  VkPushConstantRange swap_apply_gamma_push_constant_range;
  swap_apply_gamma_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  swap_apply_gamma_push_constant_range.offset = 0;
  swap_apply_gamma_push_constant_range.size = sizeof(ApplyGammaConstants);
  VkPipelineLayoutCreateInfo swap_apply_gamma_pipeline_layout_create_info;
  swap_apply_gamma_pipeline_layout_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  swap_apply_gamma_pipeline_layout_create_info.pNext = nullptr;
  swap_apply_gamma_pipeline_layout_create_info.flags = 0;
  swap_apply_gamma_pipeline_layout_create_info.setLayoutCount =
      uint32_t(swap_apply_gamma_descriptor_set_layouts.size());
  swap_apply_gamma_pipeline_layout_create_info.pSetLayouts =
      swap_apply_gamma_descriptor_set_layouts.data();
  swap_apply_gamma_pipeline_layout_create_info.pushConstantRangeCount = 1;
  swap_apply_gamma_pipeline_layout_create_info.pPushConstantRanges =
      &swap_apply_gamma_push_constant_range;
  if (dfn.vkCreatePipelineLayout(
          device, &swap_apply_gamma_pipeline_layout_create_info, nullptr,
          &swap_apply_gamma_pipeline_layout_) != VK_SUCCESS) {
    XELOGE("Failed to create the gamma ramp application pipeline layout");
    return false;
  }

  // Gamma ramp application compute pipelines.
  VkShaderModule swap_apply_gamma_table_shader_module =
      ui::vulkan::util::CreateShaderModule(
          vulkan_device, shaders::apply_gamma_table_cs,
          sizeof(shaders::apply_gamma_table_cs));
  VkShaderModule swap_apply_gamma_pwl_shader_module =
      ui::vulkan::util::CreateShaderModule(vulkan_device,
                                           shaders::apply_gamma_pwl_cs,
                                           sizeof(shaders::apply_gamma_pwl_cs));
  VkShaderModule swap_apply_gamma_table_fxaa_luma_shader_module =
      ui::vulkan::util::CreateShaderModule(
          vulkan_device, shaders::apply_gamma_table_fxaa_luma_cs,
          sizeof(shaders::apply_gamma_table_fxaa_luma_cs));
  VkShaderModule swap_apply_gamma_pwl_fxaa_luma_shader_module =
      ui::vulkan::util::CreateShaderModule(
          vulkan_device, shaders::apply_gamma_pwl_fxaa_luma_cs,
          sizeof(shaders::apply_gamma_pwl_fxaa_luma_cs));
  if (swap_apply_gamma_table_shader_module == VK_NULL_HANDLE ||
      swap_apply_gamma_pwl_shader_module == VK_NULL_HANDLE ||
      swap_apply_gamma_table_fxaa_luma_shader_module == VK_NULL_HANDLE ||
      swap_apply_gamma_pwl_fxaa_luma_shader_module == VK_NULL_HANDLE) {
    XELOGE(
        "Failed to create the gamma ramp application compute shader modules");
    if (swap_apply_gamma_table_shader_module != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(device, swap_apply_gamma_table_shader_module,
                                nullptr);
    }
    if (swap_apply_gamma_pwl_shader_module != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(device, swap_apply_gamma_pwl_shader_module,
                                nullptr);
    }
    if (swap_apply_gamma_table_fxaa_luma_shader_module != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(
          device, swap_apply_gamma_table_fxaa_luma_shader_module, nullptr);
    }
    if (swap_apply_gamma_pwl_fxaa_luma_shader_module != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(
          device, swap_apply_gamma_pwl_fxaa_luma_shader_module, nullptr);
    }
    return false;
  }

  VkComputePipelineCreateInfo swap_apply_gamma_pipeline_create_info;
  swap_apply_gamma_pipeline_create_info.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  swap_apply_gamma_pipeline_create_info.pNext = nullptr;
  swap_apply_gamma_pipeline_create_info.flags = 0;
  swap_apply_gamma_pipeline_create_info.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  swap_apply_gamma_pipeline_create_info.stage.pNext = nullptr;
  swap_apply_gamma_pipeline_create_info.stage.flags = 0;
  swap_apply_gamma_pipeline_create_info.stage.stage =
      VK_SHADER_STAGE_COMPUTE_BIT;
  swap_apply_gamma_pipeline_create_info.stage.pName = "main";
  swap_apply_gamma_pipeline_create_info.stage.pSpecializationInfo = nullptr;
  swap_apply_gamma_pipeline_create_info.layout =
      swap_apply_gamma_pipeline_layout_;
  swap_apply_gamma_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  swap_apply_gamma_pipeline_create_info.basePipelineIndex = -1;

  swap_apply_gamma_pipeline_create_info.stage.module =
      swap_apply_gamma_table_shader_module;
  VkResult swap_apply_gamma_pipeline_256_entry_table_create_result =
      dfn.vkCreateComputePipelines(
          device, VK_NULL_HANDLE, 1, &swap_apply_gamma_pipeline_create_info,
          nullptr, &swap_apply_gamma_256_entry_table_pipeline_);
  swap_apply_gamma_pipeline_create_info.stage.module =
      swap_apply_gamma_pwl_shader_module;
  VkResult swap_apply_gamma_pipeline_pwl_create_result =
      dfn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                   &swap_apply_gamma_pipeline_create_info,
                                   nullptr, &swap_apply_gamma_pwl_pipeline_);
  swap_apply_gamma_pipeline_create_info.stage.module =
      swap_apply_gamma_table_fxaa_luma_shader_module;
  VkResult swap_apply_gamma_pipeline_256_entry_table_fxaa_luma_create_result =
      dfn.vkCreateComputePipelines(
          device, VK_NULL_HANDLE, 1, &swap_apply_gamma_pipeline_create_info,
          nullptr, &swap_apply_gamma_256_entry_table_fxaa_luma_pipeline_);
  swap_apply_gamma_pipeline_create_info.stage.module =
      swap_apply_gamma_pwl_fxaa_luma_shader_module;
  VkResult swap_apply_gamma_pipeline_pwl_fxaa_luma_create_result =
      dfn.vkCreateComputePipelines(
          device, VK_NULL_HANDLE, 1, &swap_apply_gamma_pipeline_create_info,
          nullptr, &swap_apply_gamma_pwl_fxaa_luma_pipeline_);
  dfn.vkDestroyShaderModule(device, swap_apply_gamma_table_shader_module,
                            nullptr);
  dfn.vkDestroyShaderModule(device, swap_apply_gamma_pwl_shader_module,
                            nullptr);
  dfn.vkDestroyShaderModule(
      device, swap_apply_gamma_table_fxaa_luma_shader_module, nullptr);
  dfn.vkDestroyShaderModule(
      device, swap_apply_gamma_pwl_fxaa_luma_shader_module, nullptr);
  if (swap_apply_gamma_pipeline_256_entry_table_create_result != VK_SUCCESS ||
      swap_apply_gamma_pipeline_pwl_create_result != VK_SUCCESS ||
      swap_apply_gamma_pipeline_256_entry_table_fxaa_luma_create_result !=
          VK_SUCCESS ||
      swap_apply_gamma_pipeline_pwl_fxaa_luma_create_result != VK_SUCCESS) {
    XELOGE("Failed to create the gamma ramp application compute pipelines");
    return false;
  }

  // FXAA sampler (linear filtering, clamp to edge).
  VkSamplerCreateInfo fxaa_sampler_create_info;
  fxaa_sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  fxaa_sampler_create_info.pNext = nullptr;
  fxaa_sampler_create_info.flags = 0;
  fxaa_sampler_create_info.magFilter = VK_FILTER_LINEAR;
  fxaa_sampler_create_info.minFilter = VK_FILTER_LINEAR;
  fxaa_sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  fxaa_sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  fxaa_sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  fxaa_sampler_create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  fxaa_sampler_create_info.mipLodBias = 0.0f;
  fxaa_sampler_create_info.anisotropyEnable = VK_FALSE;
  fxaa_sampler_create_info.maxAnisotropy = 1.0f;
  fxaa_sampler_create_info.compareEnable = VK_FALSE;
  fxaa_sampler_create_info.compareOp = VK_COMPARE_OP_NEVER;
  fxaa_sampler_create_info.minLod = 0.0f;
  fxaa_sampler_create_info.maxLod = 0.0f;
  fxaa_sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  fxaa_sampler_create_info.unnormalizedCoordinates = VK_FALSE;
  if (dfn.vkCreateSampler(device, &fxaa_sampler_create_info, nullptr,
                          &fxaa_sampler_) != VK_SUCCESS) {
    XELOGE("Failed to create the FXAA sampler");
    return false;
  }

  // FXAA pipeline layout.
  // set=0: destination storage image (rgb10_a2)
  // set=1: source combined image sampler (FXAA source with luma in alpha)
  std::array<VkDescriptorSetLayout, 2> fxaa_descriptor_set_layouts = {
      swap_descriptor_set_layout_storage_image_,
      fxaa_source_descriptor_set_layout_,
  };
  VkPushConstantRange fxaa_push_constant_range;
  fxaa_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  fxaa_push_constant_range.offset = 0;
  fxaa_push_constant_range.size = sizeof(FxaaConstants);
  VkPipelineLayoutCreateInfo fxaa_pipeline_layout_create_info;
  fxaa_pipeline_layout_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  fxaa_pipeline_layout_create_info.pNext = nullptr;
  fxaa_pipeline_layout_create_info.flags = 0;
  fxaa_pipeline_layout_create_info.setLayoutCount =
      uint32_t(fxaa_descriptor_set_layouts.size());
  fxaa_pipeline_layout_create_info.pSetLayouts =
      fxaa_descriptor_set_layouts.data();
  fxaa_pipeline_layout_create_info.pushConstantRangeCount = 1;
  fxaa_pipeline_layout_create_info.pPushConstantRanges =
      &fxaa_push_constant_range;
  if (dfn.vkCreatePipelineLayout(device, &fxaa_pipeline_layout_create_info,
                                 nullptr,
                                 &fxaa_pipeline_layout_) != VK_SUCCESS) {
    XELOGE("Failed to create the FXAA pipeline layout");
    return false;
  }

  // FXAA compute pipelines.
  VkShaderModule fxaa_shader_module = ui::vulkan::util::CreateShaderModule(
      vulkan_device, shaders::fxaa_cs, sizeof(shaders::fxaa_cs));
  VkShaderModule fxaa_extreme_shader_module =
      ui::vulkan::util::CreateShaderModule(vulkan_device,
                                           shaders::fxaa_extreme_cs,
                                           sizeof(shaders::fxaa_extreme_cs));
  if (fxaa_shader_module == VK_NULL_HANDLE ||
      fxaa_extreme_shader_module == VK_NULL_HANDLE) {
    XELOGE("Failed to create the FXAA compute shader modules");
    if (fxaa_shader_module != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(device, fxaa_shader_module, nullptr);
    }
    if (fxaa_extreme_shader_module != VK_NULL_HANDLE) {
      dfn.vkDestroyShaderModule(device, fxaa_extreme_shader_module, nullptr);
    }
    return false;
  }

  VkComputePipelineCreateInfo fxaa_pipeline_create_info;
  fxaa_pipeline_create_info.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  fxaa_pipeline_create_info.pNext = nullptr;
  fxaa_pipeline_create_info.flags = 0;
  fxaa_pipeline_create_info.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fxaa_pipeline_create_info.stage.pNext = nullptr;
  fxaa_pipeline_create_info.stage.flags = 0;
  fxaa_pipeline_create_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  fxaa_pipeline_create_info.stage.pName = "main";
  fxaa_pipeline_create_info.stage.pSpecializationInfo = nullptr;
  fxaa_pipeline_create_info.layout = fxaa_pipeline_layout_;
  fxaa_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  fxaa_pipeline_create_info.basePipelineIndex = -1;

  fxaa_pipeline_create_info.stage.module = fxaa_shader_module;
  VkResult fxaa_pipeline_create_result = dfn.vkCreateComputePipelines(
      device, VK_NULL_HANDLE, 1, &fxaa_pipeline_create_info, nullptr,
      &fxaa_pipeline_);
  fxaa_pipeline_create_info.stage.module = fxaa_extreme_shader_module;
  VkResult fxaa_extreme_pipeline_create_result = dfn.vkCreateComputePipelines(
      device, VK_NULL_HANDLE, 1, &fxaa_pipeline_create_info, nullptr,
      &fxaa_extreme_pipeline_);
  dfn.vkDestroyShaderModule(device, fxaa_shader_module, nullptr);
  dfn.vkDestroyShaderModule(device, fxaa_extreme_shader_module, nullptr);
  if (fxaa_pipeline_create_result != VK_SUCCESS ||
      fxaa_extreme_pipeline_create_result != VK_SUCCESS) {
    XELOGE("Failed to create the FXAA compute pipelines");
    return false;
  }

  // Resolve downscale compute shader for scaled resolution readback.
  // Descriptor set layout: binding 0 = source storage buffer,
  //                        binding 1 = destination storage buffer.
  {
    std::array<VkDescriptorSetLayoutBinding, 2> resolve_downscale_bindings;
    // Source buffer (readonly).
    resolve_downscale_bindings[0].binding = 0;
    resolve_downscale_bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resolve_downscale_bindings[0].descriptorCount = 1;
    resolve_downscale_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    resolve_downscale_bindings[0].pImmutableSamplers = nullptr;
    // Destination buffer (writeonly).
    resolve_downscale_bindings[1].binding = 1;
    resolve_downscale_bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resolve_downscale_bindings[1].descriptorCount = 1;
    resolve_downscale_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    resolve_downscale_bindings[1].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo resolve_downscale_set_layout_create_info;
    resolve_downscale_set_layout_create_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    resolve_downscale_set_layout_create_info.pNext = nullptr;
    resolve_downscale_set_layout_create_info.flags = 0;
    resolve_downscale_set_layout_create_info.bindingCount =
        uint32_t(resolve_downscale_bindings.size());
    resolve_downscale_set_layout_create_info.pBindings =
        resolve_downscale_bindings.data();
    if (dfn.vkCreateDescriptorSetLayout(
            device, &resolve_downscale_set_layout_create_info, nullptr,
            &resolve_downscale_descriptor_set_layout_) != VK_SUCCESS) {
      XELOGE("Failed to create the resolve downscale descriptor set layout");
      return false;
    }

    // Pipeline layout with push constants.
    VkPushConstantRange resolve_downscale_push_constant_range;
    resolve_downscale_push_constant_range.stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    resolve_downscale_push_constant_range.offset = 0;
    resolve_downscale_push_constant_range.size =
        sizeof(ResolveDownscaleConstants);

    VkPipelineLayoutCreateInfo resolve_downscale_pipeline_layout_create_info;
    resolve_downscale_pipeline_layout_create_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    resolve_downscale_pipeline_layout_create_info.pNext = nullptr;
    resolve_downscale_pipeline_layout_create_info.flags = 0;
    resolve_downscale_pipeline_layout_create_info.setLayoutCount = 1;
    resolve_downscale_pipeline_layout_create_info.pSetLayouts =
        &resolve_downscale_descriptor_set_layout_;
    resolve_downscale_pipeline_layout_create_info.pushConstantRangeCount = 1;
    resolve_downscale_pipeline_layout_create_info.pPushConstantRanges =
        &resolve_downscale_push_constant_range;
    if (dfn.vkCreatePipelineLayout(
            device, &resolve_downscale_pipeline_layout_create_info, nullptr,
            &resolve_downscale_pipeline_layout_) != VK_SUCCESS) {
      XELOGE("Failed to create the resolve downscale pipeline layout");
      return false;
    }

    // Compute pipeline.
    VkShaderModule resolve_downscale_shader_module =
        ui::vulkan::util::CreateShaderModule(
            vulkan_device, shaders::resolve_downscale_cs,
            sizeof(shaders::resolve_downscale_cs));
    if (resolve_downscale_shader_module == VK_NULL_HANDLE) {
      XELOGE("Failed to create the resolve downscale shader module");
      return false;
    }

    VkComputePipelineCreateInfo resolve_downscale_pipeline_create_info;
    resolve_downscale_pipeline_create_info.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    resolve_downscale_pipeline_create_info.pNext = nullptr;
    resolve_downscale_pipeline_create_info.flags = 0;
    resolve_downscale_pipeline_create_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    resolve_downscale_pipeline_create_info.stage.pNext = nullptr;
    resolve_downscale_pipeline_create_info.stage.flags = 0;
    resolve_downscale_pipeline_create_info.stage.stage =
        VK_SHADER_STAGE_COMPUTE_BIT;
    resolve_downscale_pipeline_create_info.stage.module =
        resolve_downscale_shader_module;
    resolve_downscale_pipeline_create_info.stage.pName = "main";
    resolve_downscale_pipeline_create_info.stage.pSpecializationInfo = nullptr;
    resolve_downscale_pipeline_create_info.layout =
        resolve_downscale_pipeline_layout_;
    resolve_downscale_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
    resolve_downscale_pipeline_create_info.basePipelineIndex = -1;

    VkResult resolve_downscale_pipeline_result = dfn.vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &resolve_downscale_pipeline_create_info,
        nullptr, &resolve_downscale_pipeline_);
    dfn.vkDestroyShaderModule(device, resolve_downscale_shader_module, nullptr);
    if (resolve_downscale_pipeline_result != VK_SUCCESS) {
      XELOGE("Failed to create the resolve downscale compute pipeline");
      return false;
    }

    // Descriptor pool chain for resolve downscale shader.
    // Uses pool chain to avoid mid-frame GPU stalls on pool exhaustion.
    // Each pool has 64 sets with 2 storage buffers each.
    VkDescriptorPoolSize resolve_downscale_pool_size;
    resolve_downscale_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resolve_downscale_pool_size.descriptorCount = 128;  // 64 sets * 2 buffers
    resolve_downscale_descriptor_pool_chain_ =
        std::make_unique<ui::vulkan::VulkanDescriptorPoolChain>(
            vulkan_device, 0, 64, &resolve_downscale_pool_size, 1,
            resolve_downscale_descriptor_set_layout_);
  }

  // Initialize the ZPD occlusion query pool and resources.
  zpd_host_query_pool_ = std::make_unique<VulkanZPDQueryPool>();
  EnsureZPDQueryResources();

  // Per-submission GPU timestamps for the frame time breakdown diagnostics.
  if (cvars::log_gpu_frame_time_breakdown &&
      vulkan_device->properties().timestampPeriod > 0.0f) {
    VkQueryPoolCreateInfo timestamp_pool_create_info = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    timestamp_pool_create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    timestamp_pool_create_info.queryCount = kFrameTimestampSlots * 2;
    if (dfn.vkCreateQueryPool(device, &timestamp_pool_create_info, nullptr,
                              &frame_timestamp_pool_) == VK_SUCCESS) {
      if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
              vulkan_device, kFrameTimestampSlots * 2 * sizeof(uint64_t),
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              ui::vulkan::util::MemoryPurpose::kReadback,
              frame_timestamp_buffer_, frame_timestamp_buffer_memory_, nullptr,
              &frame_timestamp_buffer_size_) ||
          dfn.vkMapMemory(device, frame_timestamp_buffer_memory_, 0,
                          VK_WHOLE_SIZE, 0,
                          reinterpret_cast<void**>(
                              &frame_timestamp_mapping_)) != VK_SUCCESS) {
        XELOGW(
            "VulkanCommandProcessor: No readback buffer for frame timestamps; "
            "GPU execution time reporting disabled");
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                               frame_timestamp_buffer_);
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                               frame_timestamp_buffer_memory_);
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                                               frame_timestamp_pool_);
        frame_timestamp_mapping_ = nullptr;
      }
    }
  }
  // Per-resolve GPU timestamps, ring-buffered per submission; only meaningful
  // alongside the per-submission pair above.
  if (frame_timestamp_mapping_) {
    VkQueryPoolCreateInfo resolve_ts_pool_info = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    resolve_ts_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    resolve_ts_pool_info.queryCount = kResolveTimestampPairsPerSubmission *
                                      kResolveTimestampRingSubmissions * 2;
    if (dfn.vkCreateQueryPool(device, &resolve_ts_pool_info, nullptr,
                              &resolve_timestamp_pool_) == VK_SUCCESS) {
      if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
              vulkan_device, resolve_ts_pool_info.queryCount * sizeof(uint64_t),
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              ui::vulkan::util::MemoryPurpose::kReadback,
              resolve_timestamp_buffer_, resolve_timestamp_buffer_memory_,
              nullptr, &resolve_timestamp_buffer_size_) ||
          dfn.vkMapMemory(device, resolve_timestamp_buffer_memory_, 0,
                          VK_WHOLE_SIZE, 0,
                          reinterpret_cast<void**>(
                              &resolve_timestamp_mapping_)) != VK_SUCCESS) {
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                               resolve_timestamp_buffer_);
        ui::vulkan::util::DestroyAndNullHandle(
            dfn.vkFreeMemory, device, resolve_timestamp_buffer_memory_);
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                                               resolve_timestamp_pool_);
        resolve_timestamp_mapping_ = nullptr;
      }
    }
  }
  // Per-render-pass timestamps, bucketed by framebuffer extent.
  if (frame_timestamp_mapping_) {
    VkQueryPoolCreateInfo pass_ts_pool_info = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    pass_ts_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    pass_ts_pool_info.queryCount =
        kPassTimestampPairsPerSubmission * kPassTimestampRingSubmissions * 2;
    if (dfn.vkCreateQueryPool(device, &pass_ts_pool_info, nullptr,
                              &pass_timestamp_pool_) == VK_SUCCESS) {
      if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
              vulkan_device, pass_ts_pool_info.queryCount * sizeof(uint64_t),
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              ui::vulkan::util::MemoryPurpose::kReadback,
              pass_timestamp_buffer_, pass_timestamp_buffer_memory_, nullptr,
              &pass_timestamp_buffer_size_) ||
          dfn.vkMapMemory(
              device, pass_timestamp_buffer_memory_, 0, VK_WHOLE_SIZE, 0,
              reinterpret_cast<void**>(&pass_timestamp_mapping_)) !=
              VK_SUCCESS) {
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                               pass_timestamp_buffer_);
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                               pass_timestamp_buffer_memory_);
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                                               pass_timestamp_pool_);
        pass_timestamp_mapping_ = nullptr;
      }
    }
  }

  // Just not to expose uninitialized memory.
  std::memset(&system_constants_, 0, sizeof(system_constants_));
  // ZPD FSI counter uses UINT32_MAX as its skip sentinel outside query draws.
  system_constants_.zpd_fsi_counter_index = UINT32_MAX;
  zpd_fsi_counter_index_force_update_ = true;

  return true;
}

void VulkanCommandProcessor::ShutdownContext() {
  AwaitAllQueueOperationsCompletion();
  transfer_completion_timeline_.AwaitAllSubmissions();

  ShutdownZPDQueryResources();
  zpd_host_query_pool_.reset();

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  if (frame_timestamp_mapping_) {
    dfn.vkUnmapMemory(device, frame_timestamp_buffer_memory_);
    frame_timestamp_mapping_ = nullptr;
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         frame_timestamp_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         frame_timestamp_buffer_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                                         frame_timestamp_pool_);
  vk_submit_times_.clear();
  frame_timestamp_prev_end_ = 0;
  if (resolve_timestamp_mapping_) {
    dfn.vkUnmapMemory(device, resolve_timestamp_buffer_memory_);
    resolve_timestamp_mapping_ = nullptr;
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         resolve_timestamp_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         resolve_timestamp_buffer_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                                         resolve_timestamp_pool_);
  resolve_ts_count_ = 0;
  if (pass_timestamp_mapping_) {
    dfn.vkUnmapMemory(device, pass_timestamp_buffer_memory_);
    pass_timestamp_mapping_ = nullptr;
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         pass_timestamp_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         pass_timestamp_buffer_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                                         pass_timestamp_pool_);
  pass_ts_count_ = 0;
  pass_ts_open_pair_ = UINT32_MAX;
  pass_bucket_stats_.clear();

  DestroyScratchBuffer();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_pwl_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyPipeline, device,
      swap_apply_gamma_256_entry_table_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyPipeline, device, swap_apply_gamma_pwl_fxaa_luma_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyPipeline, device,
      swap_apply_gamma_256_entry_table_fxaa_luma_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         swap_apply_gamma_pipeline_layout_);

  // FXAA cleanup.
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         fxaa_extreme_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         fxaa_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         fxaa_pipeline_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroySampler, device,
                                         fxaa_sampler_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device,
                                         fxaa_source_image_view_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device,
                                         fxaa_source_image_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         fxaa_source_memory_);
  fxaa_source_width_ = 0;
  fxaa_source_height_ = 0;
  fxaa_source_last_submission_ = 0;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout,
                                         device,
                                         fxaa_source_descriptor_set_layout_);

  // Resolve downscale cleanup.
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         resolve_downscale_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         resolve_downscale_buffer_memory_);
  resolve_downscale_buffer_size_ = 0;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         resolve_downscale_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         resolve_downscale_pipeline_layout_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyDescriptorSetLayout, device,
      resolve_downscale_descriptor_set_layout_);
  resolve_downscale_descriptor_pool_chain_.reset();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device,
                                         swap_descriptor_pool_);

  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyDescriptorSetLayout, device,
      swap_descriptor_set_layout_storage_image_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyDescriptorSetLayout, device,
      swap_descriptor_set_layout_uniform_texel_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyDescriptorSetLayout, device,
      swap_descriptor_set_layout_sampled_image_);
  for (VkBufferView& gamma_ramp_buffer_view : gamma_ramp_buffer_views_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBufferView, device,
                                           gamma_ramp_buffer_view);
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         gamma_ramp_upload_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         gamma_ramp_upload_buffer_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         gamma_ramp_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         gamma_ramp_buffer_memory_);

  // Clean up all readback buffers.
  for (auto& pair : readback_buffers_) {
    for (int i = 0; i < 2; i++) {
      if (pair.second.mapped_data[i] != nullptr) {
        dfn.vkUnmapMemory(device, pair.second.memories[i]);
      }
    }
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                           pair.second.buffers[0]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                           pair.second.memories[0]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                           pair.second.buffers[1]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                           pair.second.memories[1]);
  }
  readback_buffers_.clear();

  // Clean up all memexport readback buffers.
  for (auto& pair : memexport_readback_buffers_) {
    for (int i = 0; i < 2; i++) {
      if (pair.second.mapped_data[i] != nullptr) {
        dfn.vkUnmapMemory(device, pair.second.memories[i]);
      }
    }
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                           pair.second.buffers[0]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                           pair.second.memories[0]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                           pair.second.buffers[1]);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                           pair.second.memories[1]);
  }
  memexport_readback_buffers_.clear();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         memexport_readback_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         memexport_readback_buffer_memory_);
  memexport_readback_buffer_size_ = 0;

  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyDescriptorPool, device,
      shared_memory_and_edram_descriptor_pool_);
  zpd_fsi_counter_descriptor_buffer_ = VK_NULL_HANDLE;
  zpd_fsi_counter_descriptor_range_ = 0;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         zpd_fsi_counter_sink_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         zpd_fsi_counter_sink_buffer_memory_);

  texture_cache_.reset();

  pipeline_cache_.reset();

  render_target_cache_.reset();

  primitive_processor_.reset();

  shared_memory_.reset();

  ClearTransientDescriptorPools();

  for (const auto& pipeline_layout_pair : pipeline_layouts_) {
    dfn.vkDestroyPipelineLayout(
        device, pipeline_layout_pair.second.GetPipelineLayout(), nullptr);
  }
  pipeline_layouts_.clear();
  for (const auto& descriptor_set_layout_pair :
       descriptor_set_layouts_textures_) {
    dfn.vkDestroyDescriptorSetLayout(device, descriptor_set_layout_pair.second,
                                     nullptr);
  }
  descriptor_set_layouts_textures_.clear();

  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyDescriptorSetLayout, device,
      descriptor_set_layout_shared_memory_and_edram_);
  for (VkDescriptorSetLayout& descriptor_set_layout_single_transient :
       descriptor_set_layouts_single_transient_) {
    ui::vulkan::util::DestroyAndNullHandle(
        dfn.vkDestroyDescriptorSetLayout, device,
        descriptor_set_layout_single_transient);
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout,
                                         device,
                                         descriptor_set_layout_constants_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout,
                                         device, descriptor_set_layout_empty_);

  uniform_buffer_pool_.reset();

  sparse_bind_wait_stage_mask_ = 0;
  sparse_buffer_binds_.clear();
  sparse_memory_binds_.clear();

  deferred_command_buffer_.Reset();
  deferred_setup_command_buffer_.Reset();
  for (const auto& command_buffer_pair : command_buffers_submitted_) {
    dfn.vkDestroyCommandPool(device, command_buffer_pair.second.pool, nullptr);
  }
  command_buffers_submitted_.clear();
  for (const CommandBuffer& command_buffer : command_buffers_writable_) {
    dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
  }
  command_buffers_writable_.clear();

  pending_readback_copies_.clear();
  for (const auto& command_buffer_pair : transfer_command_buffers_submitted_) {
    dfn.vkDestroyCommandPool(device, command_buffer_pair.second.pool, nullptr);
  }
  transfer_command_buffers_submitted_.clear();
  for (const CommandBuffer& command_buffer :
       transfer_command_buffers_writable_) {
    dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
  }
  transfer_command_buffers_writable_.clear();

  for (const auto& destroy_pair : destroy_framebuffers_) {
    dfn.vkDestroyFramebuffer(device, destroy_pair.second, nullptr);
  }
  destroy_framebuffers_.clear();
  for (const auto& destroy_pair : destroy_buffers_) {
    dfn.vkDestroyBuffer(device, destroy_pair.second, nullptr);
  }
  destroy_buffers_.clear();
  for (const auto& destroy_pair : destroy_image_views_) {
    dfn.vkDestroyImageView(device, destroy_pair.second, nullptr);
  }
  destroy_image_views_.clear();
  for (const auto& destroy_pair : destroy_images_) {
    dfn.vkDestroyImage(device, destroy_pair.second, nullptr);
  }
  destroy_images_.clear();
  for (const auto& destroy_pair : destroy_memory_) {
    dfn.vkFreeMemory(device, destroy_pair.second, nullptr);
  }
  destroy_memory_.clear();

  std::memset(closed_frame_submissions_, 0, sizeof(closed_frame_submissions_));
  frame_completed_ = 0;
  frame_current_ = 1;
  frame_open_ = false;

  for (const auto& semaphore : submissions_in_flight_semaphores_) {
    dfn.vkDestroySemaphore(device, semaphore.second, nullptr);
  }
  submissions_in_flight_semaphores_.clear();
  current_submission_wait_stage_masks_.clear();
  for (VkSemaphore semaphore : current_submission_wait_semaphores_) {
    dfn.vkDestroySemaphore(device, semaphore, nullptr);
  }
  current_submission_wait_semaphores_.clear();
  submission_open_ = false;

  for (VkSemaphore semaphore : semaphores_free_) {
    dfn.vkDestroySemaphore(device, semaphore, nullptr);
  }
  semaphores_free_.clear();

  for (const auto& semaphore : transfer_wait_semaphores_in_flight_) {
    dfn.vkDestroySemaphore(device, semaphore.second, nullptr);
  }
  transfer_wait_semaphores_in_flight_.clear();
  for (VkSemaphore semaphore : transfer_wait_semaphores_free_) {
    dfn.vkDestroySemaphore(device, semaphore, nullptr);
  }
  transfer_wait_semaphores_free_.clear();

  device_lost_ = false;

  CommandProcessor::ShutdownContext();
}

void VulkanCommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  // Fetch (0x4800-0x48BF) and bool/loop (0x4900-0x4927) constants: the
  // derived state (texture bindings, fetch/bool-loop UBO contents) is a pure
  // function of the register value, and games commonly re-emit identical
  // constants every draw — a same-value write doesn't need to dirty
  // anything. The previous value must be read before the base class stores
  // the new one. The two ranges are checked as one span; the gap registers
  // in between don't consume the flag.
  bool constant_value_unchanged = false;
  if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
      index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31 &&
      cvars::vulkan_skip_redundant_fetch_constant_writes) {
    constant_value_unchanged = register_file_->values[index] == value;
  }

  CommandProcessor::WriteRegister(index, value);

  if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    if (frame_open_) {
      uint32_t float_constant_index =
          (index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (float_constant_index >= 256) {
        float_constant_index -= 256;
        if (current_float_constant_map_pixel_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          current_constant_buffers_up_to_date_ &= ~(
              UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel);
        }
      } else {
        if (current_float_constant_map_vertex_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          current_constant_buffers_up_to_date_ &= ~(
              UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
        }
      }
    }
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    if (!constant_value_unchanged) {
      current_constant_buffers_up_to_date_ &=
          ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferBoolLoop);
    }
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    if (!constant_value_unchanged) {
      current_constant_buffers_up_to_date_ &=
          ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch);
      uint32_t fetch_slot =
          (index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6;
      uint32_t fetch_slot_bit_clear = ~(uint32_t(1) << fetch_slot);
      current_samplers_fetch_up_to_date_vertex_ &= fetch_slot_bit_clear;
      current_samplers_fetch_up_to_date_pixel_ &= fetch_slot_bit_clear;
      if (texture_cache_) {
        texture_cache_->TextureFetchConstantWritten(fetch_slot);
      }
    }
  }
}
void VulkanCommandProcessor::WriteShaderConstantsFromMem(
    uint32_t start_index, uint32_t* base, uint32_t num_registers) {
  if (frame_open_) {
    // One usage-map scan for the whole range instead of a map test per
    // register (the float constant maps are indexed by vec4 constant).
    constexpr uint32_t kFloatVertexBit =
        UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex;
    constexpr uint32_t kFloatPixelBit =
        UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel;
    uint32_t map_index = (start_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
    uint32_t end_map_index =
        (start_index + num_registers + 3 - XE_GPU_REG_SHADER_CONSTANT_000_X) >>
        2;
    if (current_constant_buffers_up_to_date_ & kFloatVertexBit) {
      uint32_t vertex_end = std::min(end_map_index, UINT32_C(256));
      for (uint32_t i = map_index; i < vertex_end; ++i) {
        if (current_float_constant_map_vertex_[i >> 6] &
            (UINT64_C(1) << (i & 63))) {
          current_constant_buffers_up_to_date_ &= ~kFloatVertexBit;
          break;
        }
      }
    }
    if (end_map_index > 256 &&
        (current_constant_buffers_up_to_date_ & kFloatPixelBit)) {
      for (uint32_t i = std::max(map_index, UINT32_C(256)); i < end_map_index;
           ++i) {
        uint32_t pixel_index = i - 256;
        if (current_float_constant_map_pixel_[pixel_index >> 6] &
            (UINT64_C(1) << (pixel_index & 63))) {
          current_constant_buffers_up_to_date_ &= ~kFloatPixelBit;
          break;
        }
      }
    }
  }
  xe::copy_and_swap_32_unaligned(&register_file_->values[start_index], base,
                                 num_registers);
}

void VulkanCommandProcessor::WriteBoolLoopFromMem(uint32_t start_index,
                                                  uint32_t* base,
                                                  uint32_t num_registers) {
  bool changed;
  if (cvars::vulkan_skip_redundant_fetch_constant_writes) {
    // Same value-compare semantics as the per-register path.
    changed = false;
    for (uint32_t i = 0; i < num_registers; ++i) {
      uint32_t value = xe::load_and_swap<uint32_t>(base + i);
      changed |= register_file_->values[start_index + i] != value;
      register_file_->values[start_index + i] = value;
    }
  } else {
    changed = true;
    xe::copy_and_swap_32_unaligned(&register_file_->values[start_index], base,
                                   num_registers);
  }
  if (changed) {
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferBoolLoop);
  }
}

void VulkanCommandProcessor::WriteFetchFromMem(uint32_t start_index,
                                               uint32_t* base,
                                               uint32_t num_registers) {
  if (!cvars::vulkan_skip_redundant_fetch_constant_writes) {
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch);
    uint32_t first_slot =
        (start_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6;
    uint32_t last_slot = (start_index + num_registers - 1 -
                          XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) /
                         6;
    for (uint32_t slot = first_slot; slot <= last_slot; ++slot) {
      uint32_t slot_bit_clear = ~(UINT32_C(1) << slot);
      current_samplers_fetch_up_to_date_vertex_ &= slot_bit_clear;
      current_samplers_fetch_up_to_date_pixel_ &= slot_bit_clear;
    }
    if (texture_cache_) {
      texture_cache_->TextureFetchConstantsWritten(first_slot, last_slot);
    }
    xe::copy_and_swap_32_unaligned(&register_file_->values[start_index], base,
                                   num_registers);
    return;
  }
  // Per-fetch-slot value compare, same semantics as the per-register path:
  // the identical fetch constants games re-emit every draw must not dirty
  // texture bindings, the fetch constant buffer or the sampler caches.
  bool any_changed = false;
  uint32_t index = start_index;
  uint32_t* src = base;
  uint32_t remaining = num_registers;
  while (remaining) {
    uint32_t slot = (index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 6;
    uint32_t slot_end_index =
        XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + (slot + 1) * 6;
    uint32_t count = std::min(remaining, slot_end_index - index);
    bool slot_changed = false;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t value = xe::load_and_swap<uint32_t>(src + i);
      slot_changed |= register_file_->values[index + i] != value;
      register_file_->values[index + i] = value;
    }
    if (slot_changed) {
      any_changed = true;
      uint32_t slot_bit_clear = ~(UINT32_C(1) << slot);
      current_samplers_fetch_up_to_date_vertex_ &= slot_bit_clear;
      current_samplers_fetch_up_to_date_pixel_ &= slot_bit_clear;
      if (texture_cache_) {
        texture_cache_->TextureFetchConstantWritten(slot);
      }
    }
    index += count;
    src += count;
    remaining -= count;
  }
  if (any_changed) {
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch);
  }
}

void VulkanCommandProcessor::WritePossiblySpecialRegistersFromMem(
    uint32_t start_index, uint32_t* base, uint32_t num_registers) {
  uint32_t end = start_index + num_registers;
  for (uint32_t index = start_index; index < end; ++index, ++base) {
    uint32_t value = xe::load_and_swap<uint32_t>(base);
    register_file_->values[index] = value;
    unsigned expr =
        (index - XE_GPU_REG_SCRATCH_REG0 < 8) |
        (index == XE_GPU_REG_COHER_STATUS_HOST) |
        ((index - XE_GPU_REG_DC_LUT_RW_INDEX) <=
         (XE_GPU_REG_DC_LUT_30_COLOR - XE_GPU_REG_DC_LUT_RW_INDEX));
    if (expr != 0) {
      HandleSpecialRegisterWrite(index, value);
    } else if (index == XE_GPU_REG_VGT_MAX_VTX_INDX ||
             index == XE_GPU_REG_VGT_MIN_VTX_INDX ||
             index == XE_GPU_REG_VGT_INDX_OFFSET ||
             index == XE_GPU_REG_VGT_DMA_SIZE ||
             index == XE_GPU_REG_VGT_HOS_MAX_TESS_LEVEL ||
             index == XE_GPU_REG_VGT_HOS_MIN_TESS_LEVEL) {
    // Tessellation factor range and index parameters are in the system
    // constants buffer.
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
  } else if ((index >= XE_GPU_REG_PA_CL_UCP_0_X &&
              index <= XE_GPU_REG_PA_CL_UCP_5_W) ||
             index == XE_GPU_REG_PA_CL_CLIP_CNTL) {
    // User clip planes are in the system constants buffer.
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
    }
  }
}

void VulkanCommandProcessor::WriteRegistersFromMem(uint32_t start_index,
                                                   uint32_t* base,
                                                   uint32_t num_registers) {
  if (!cvars::vulkan_fast_register_ranges) {
    for (uint32_t i = 0; i < num_registers; ++i) {
      uint32_t data = xe::load_and_swap<uint32_t>(base + i);
      VulkanCommandProcessor::WriteRegister(start_index + i, data);
    }
    return;
  }
  // Clamp to the register file (the per-register path warns and drops
  // out-of-bounds indices; reaching here with one means a corrupt packet
  // either way - Type0 base indices are 15-bit and can exceed the file).
  if (start_index + num_registers > RegisterFile::kRegisterCount) {
    XELOGW(
        "WriteRegistersFromMem: out-of-bounds register range [{:04X}, {:04X}) "
        "clamped",
        start_index, start_index + num_registers);
    if (start_index >= RegisterFile::kRegisterCount) {
      return;
    }
    num_registers = uint32_t(RegisterFile::kRegisterCount) - start_index;
  }
  uint32_t current_index = start_index;
  uint32_t end = start_index + num_registers;
  // Split the range into segments by side-effect family; everything outside
  // the special families is a pure store done as one bulk byte-swapped copy.
  auto do_range = [&](uint32_t range_end, auto&& callback) -> bool {
    if (current_index < range_end) {
      uint32_t count = std::min(range_end, end) - current_index;
      callback(current_index, base, count);
      current_index += count;
      base += count;
    }
    return current_index >= end;
  };
  auto regular = [this](uint32_t index, uint32_t* src, uint32_t count) {
    xe::copy_and_swap_32_unaligned(&register_file_->values[index], src, count);
  };
  if (do_range(XE_GPU_REG_SCRATCH_REG0, regular)) {
    return;
  }
  if (do_range(XE_GPU_REG_DC_LUT_30_COLOR + 1,
               [this](uint32_t index, uint32_t* src, uint32_t count) {
                 WritePossiblySpecialRegistersFromMem(index, src, count);
               })) {
    return;
  }
  if (do_range(XE_GPU_REG_SHADER_CONSTANT_000_X, regular)) {
    return;
  }
  if (do_range(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0,
               [this](uint32_t index, uint32_t* src, uint32_t count) {
                 WriteShaderConstantsFromMem(index, src, count);
               })) {
    return;
  }
  if (do_range(XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5 + 1,
               [this](uint32_t index, uint32_t* src, uint32_t count) {
                 WriteFetchFromMem(index, src, count);
               })) {
    return;
  }
  if (do_range(XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031, regular)) {
    return;
  }
  if (do_range(XE_GPU_REG_SHADER_CONSTANT_LOOP_31 + 1,
               [this](uint32_t index, uint32_t* src, uint32_t count) {
                 WriteBoolLoopFromMem(index, src, count);
               })) {
    return;
  }
  do_range(RegisterFile::kRegisterCount, regular);
}

void VulkanCommandProcessor::WriteRegisterRangeFromRing(
    xe::RingBuffer* ring, uint32_t base, uint32_t num_registers) {
  if (!cvars::vulkan_fast_register_ranges) {
    CommandProcessor::WriteRegisterRangeFromRing(ring, base, num_registers);
    return;
  }
  RingBuffer::ReadRange range =
      ring->BeginRead(num_registers * sizeof(uint32_t));
  if (XE_LIKELY(!range.second)) {
    WriteRegistersFromMem(
        base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.first)),
        num_registers);
    ring->EndRead(range);
  } else {
    WriteRegisterRangeFromRing_WraparoundCase(ring, base, num_registers);
  }
}

XE_NOINLINE
void VulkanCommandProcessor::WriteRegisterRangeFromRing_WraparoundCase(
    xe::RingBuffer* ring, uint32_t base, uint32_t num_registers) {
  RingBuffer::ReadRange range =
      ring->BeginRead(num_registers * sizeof(uint32_t));
  // Callers (ExecutePacketType0/3) guarantee the ring holds the whole range;
  // BeginRead clamps to capacity, which would make the second-half copy read
  // out of bounds if this were ever violated.
  assert_true(size_t(range.first_length) + size_t(range.second_length) ==
              num_registers * sizeof(uint32_t));
  uint32_t num_first = uint32_t(range.first_length / sizeof(uint32_t));
  WriteRegistersFromMem(
      base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.first)),
      num_first);
  WriteRegistersFromMem(
      base + num_first,
      reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.second)),
      num_registers - num_first);
  ring->EndRead(range);
}
void VulkanCommandProcessor::SparseBindBuffer(
    VkBuffer buffer, uint32_t bind_count, const VkSparseMemoryBind* binds,
    VkPipelineStageFlags wait_stage_mask) {
  if (!bind_count) {
    return;
  }
  SparseBufferBind& buffer_bind = sparse_buffer_binds_.emplace_back();
  buffer_bind.buffer = buffer;
  buffer_bind.bind_offset = sparse_memory_binds_.size();
  buffer_bind.bind_count = bind_count;
  sparse_memory_binds_.reserve(sparse_memory_binds_.size() + bind_count);
  sparse_memory_binds_.insert(sparse_memory_binds_.end(), binds,
                              binds + bind_count);
  sparse_bind_wait_stage_mask_ |= wait_stage_mask;
}

void VulkanCommandProcessor::OnGammaRamp256EntryTableValueWritten() {
  gamma_ramp_256_entry_table_current_frame_ = UINT32_MAX;
}

void VulkanCommandProcessor::OnGammaRampPWLValueWritten() {
  gamma_ramp_pwl_current_frame_ = UINT32_MAX;
}

void VulkanCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr,
                                       uint32_t frontbuffer_width,
                                       uint32_t frontbuffer_height) {
  SCOPE_profile_cpu_f("gpu");

  // Count one presented guest frame for the debug overlay's FPS / frame-time
  // stats (the game's real frame rate, independent of the host present path).
  xe::RecordGuestPresent();

  if (render_target_cache_) {
    render_target_cache_->LogResolveDetailsOnFrameEnd();
  }

  if (cvars::log_gpu_frame_time_breakdown) {
    auto& s = vk_frame_sync_stats_;
    s.frames++;
    const uint64_t now = FrameStatsNow();
    if (!s.last_report_ns) {
      s.last_report_ns = now;
    } else if (now - s.last_report_ns >= 1000000000ull) {
      const double f = static_cast<double>(s.frames);
      XELOGI(
          "VkFrameSync: {} frames | per frame: awaits={:.1f} await={:.1f}ms "
          "submissions={:.1f} resolves={:.1f} memexport_awaits={:.1f} "
          "readback_awaits={:.1f} "
          "| submit->fence avg={:.1f}ms max={:.1f}ms | blocking={:.1f} "
          "delta avg={:.2f} | gpu exec avg={:.1f}ms max={:.1f}ms "
          "gap avg={:.1f}ms | resolve_ms={:.2f} avg={:.3f} max={:.2f} "
          "dropped={} | draws={:.0f} rp_begins={:.0f} splits={:.1f}",
          s.frames, s.awaits / f, s.await_ns / f / 1e6, s.submissions / f,
          s.resolves / f, s.memexport_awaits / f, s.readback_awaits / f,
          s.sub_completions
              ? s.sub_latency_ns / static_cast<double>(s.sub_completions) / 1e6
              : 0.0,
          s.sub_latency_max_ns / 1e6, s.blocking_awaits / f,
          s.blocking_awaits
              ? s.await_delta / static_cast<double>(s.blocking_awaits)
              : 0.0,
          s.gpu_samples
              ? s.gpu_exec_ns / static_cast<double>(s.gpu_samples) / 1e6
              : 0.0,
          s.gpu_exec_max_ns / 1e6,
          s.gpu_samples
              ? s.gpu_gap_ns / static_cast<double>(s.gpu_samples) / 1e6
              : 0.0,
          s.resolve_gpu_ns / f / 1e6,
          s.resolve_gpu_samples
              ? s.resolve_gpu_ns /
                    static_cast<double>(s.resolve_gpu_samples) / 1e6
              : 0.0,
          s.resolve_gpu_max_ns / 1e6, s.resolve_ts_dropped, s.draws / f,
          s.render_pass_begins / f, s.primary_buffer_splits / f);
      // Per-render-pass-bucket GPU time (key: WxH, bit31 = ownership transfer).
      if (!pass_bucket_stats_.empty()) {
        for (const auto& kv : pass_bucket_stats_) {
          const uint32_t key = kv.first;
          const bool transfer = (key & 0x80000000u) != 0;
          XELOGI(
              "VkPassTime: {}{}x{} : {:.2f}ms/fr ({:.1f}pass {:.0f}draw/fr, "
              "{:.3f}ms ea)",
              transfer ? "xfer " : "", (key >> 16) & 0x7FFF, key & 0xFFFF,
              kv.second.ns / f / 1e6, kv.second.passes / f, kv.second.draws / f,
              kv.second.passes
                  ? kv.second.ns / static_cast<double>(kv.second.passes) / 1e6
                  : 0.0);
        }
        pass_bucket_stats_.clear();
      }
      if (pass_ts_dropped_) {
        XELOGI("VkPassTime: {} pass pairs dropped (raise pool)",
               pass_ts_dropped_);
        pass_ts_dropped_ = 0;
      }
      s = VkFrameSyncStats();
      s.last_report_ns = now;
    }
  }

  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    return;
  }

  // In case the swap command is the only one in the frame.
  if (!BeginSubmission(true)) {
    return;
  }

  // Obtaining the actual front buffer size to pass to RefreshGuestOutput,
  // resolution-scaled if it's a resolve destination, or not otherwise.
  uint32_t frontbuffer_width_scaled, frontbuffer_height_scaled;
  xenos::TextureFormat frontbuffer_format;
  VkImageView swap_texture_view = texture_cache_->RequestSwapTexture(
      frontbuffer_width_scaled, frontbuffer_height_scaled, frontbuffer_format);
  if (swap_texture_view == VK_NULL_HANDLE) {
    return;
  }

  auto aspect = graphics_system_->GetScaledAspectRatio();

  presenter->RefreshGuestOutput(
      frontbuffer_width_scaled, frontbuffer_height_scaled, aspect.first,
      aspect.second,
      [this, frontbuffer_width_scaled, frontbuffer_height_scaled,
       frontbuffer_format, swap_texture_view](
          ui::Presenter::GuestOutputRefreshContext& context) -> bool {
        // In case the swap command is the only one in the frame.
        if (!BeginSubmission(true)) {
          return false;
        }

        auto& vulkan_context = static_cast<
            ui::vulkan::VulkanPresenter::VulkanGuestOutputRefreshContext&>(
            context);
        uint64_t guest_output_image_version = vulkan_context.image_version();

        const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
        const ui::vulkan::VulkanDevice::Functions& dfn =
            vulkan_device->functions();
        const VkDevice device = vulkan_device->device();

        uint32_t swap_frame_index =
            uint32_t(frame_current_ % kMaxFramesInFlight);

        // This is according to D3D::InitializePresentationParameters from a
        // game executable, which initializes the 256-entry table gamma ramp for
        // 8_8_8_8 output and the PWL gamma ramp for 2_10_10_10.
        // TODO(Triang3l): Choose between the table and PWL based on
        // DC_LUTA_CONTROL, support both for all formats (and also different
        // increments for PWL).
        bool use_pwl_gamma_ramp =
            frontbuffer_format == xenos::TextureFormat::k_2_10_10_10 ||
            frontbuffer_format ==
                xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16;

        SwapPostEffect swap_post_effect = GetActualSwapPostEffect();
        bool use_fxaa = swap_post_effect == SwapPostEffect::kFxaa ||
                        swap_post_effect == SwapPostEffect::kFxaaExtreme;

        if (use_fxaa) {
          // Make sure the texture of the correct size is available for FXAA.
          if (fxaa_source_image_ != VK_NULL_HANDLE &&
              (fxaa_source_width_ != frontbuffer_width_scaled ||
               fxaa_source_height_ != frontbuffer_height_scaled)) {
            // Need to resize the FXAA source texture.
            if (GetCompletedSubmission() < fxaa_source_last_submission_) {
              // Still in use - defer destruction.
              destroy_memory_.emplace_back(fxaa_source_last_submission_,
                                           fxaa_source_memory_);
              destroy_images_.emplace_back(fxaa_source_last_submission_,
                                           fxaa_source_image_);
              destroy_image_views_.emplace_back(fxaa_source_last_submission_,
                                                fxaa_source_image_view_);
            } else {
              dfn.vkDestroyImageView(device, fxaa_source_image_view_, nullptr);
              dfn.vkDestroyImage(device, fxaa_source_image_, nullptr);
              dfn.vkFreeMemory(device, fxaa_source_memory_, nullptr);
            }
            fxaa_source_image_ = VK_NULL_HANDLE;
            fxaa_source_image_view_ = VK_NULL_HANDLE;
            fxaa_source_memory_ = VK_NULL_HANDLE;
            fxaa_source_width_ = 0;
            fxaa_source_height_ = 0;
            fxaa_source_last_submission_ = 0;
          }
          if (fxaa_source_image_ == VK_NULL_HANDLE) {
            // Create the FXAA source texture.
            VkImageCreateInfo fxaa_source_image_create_info;
            fxaa_source_image_create_info.sType =
                VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            fxaa_source_image_create_info.pNext = nullptr;
            fxaa_source_image_create_info.flags = 0;
            fxaa_source_image_create_info.imageType = VK_IMAGE_TYPE_2D;
            fxaa_source_image_create_info.format = kFxaaSourceFormat;
            fxaa_source_image_create_info.extent.width =
                frontbuffer_width_scaled;
            fxaa_source_image_create_info.extent.height =
                frontbuffer_height_scaled;
            fxaa_source_image_create_info.extent.depth = 1;
            fxaa_source_image_create_info.mipLevels = 1;
            fxaa_source_image_create_info.arrayLayers = 1;
            fxaa_source_image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
            fxaa_source_image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            fxaa_source_image_create_info.usage =
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            fxaa_source_image_create_info.sharingMode =
                VK_SHARING_MODE_EXCLUSIVE;
            fxaa_source_image_create_info.queueFamilyIndexCount = 0;
            fxaa_source_image_create_info.pQueueFamilyIndices = nullptr;
            fxaa_source_image_create_info.initialLayout =
                VK_IMAGE_LAYOUT_UNDEFINED;
            if (dfn.vkCreateImage(device, &fxaa_source_image_create_info,
                                  nullptr, &fxaa_source_image_) != VK_SUCCESS) {
              XELOGE("Failed to create the FXAA source image");
              use_fxaa = false;
            } else {
              VkMemoryRequirements fxaa_source_memory_requirements;
              dfn.vkGetImageMemoryRequirements(
                  device, fxaa_source_image_, &fxaa_source_memory_requirements);
              VkMemoryAllocateInfo fxaa_source_memory_allocate_info;
              fxaa_source_memory_allocate_info.sType =
                  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
              fxaa_source_memory_allocate_info.pNext = nullptr;
              fxaa_source_memory_allocate_info.allocationSize =
                  fxaa_source_memory_requirements.size;
              fxaa_source_memory_allocate_info.memoryTypeIndex =
                  ui::vulkan::util::ChooseMemoryType(
                      vulkan_device->memory_types(),
                      fxaa_source_memory_requirements.memoryTypeBits,
                      ui::vulkan::util::MemoryPurpose::kDeviceLocal);
              if (fxaa_source_memory_allocate_info.memoryTypeIndex ==
                      UINT32_MAX ||
                  dfn.vkAllocateMemory(
                      device, &fxaa_source_memory_allocate_info, nullptr,
                      &fxaa_source_memory_) != VK_SUCCESS) {
                XELOGE("Failed to allocate FXAA source image memory");
                dfn.vkDestroyImage(device, fxaa_source_image_, nullptr);
                fxaa_source_image_ = VK_NULL_HANDLE;
                use_fxaa = false;
              } else if (dfn.vkBindImageMemory(device, fxaa_source_image_,
                                               fxaa_source_memory_,
                                               0) != VK_SUCCESS) {
                XELOGE("Failed to bind FXAA source image memory");
                dfn.vkFreeMemory(device, fxaa_source_memory_, nullptr);
                fxaa_source_memory_ = VK_NULL_HANDLE;
                dfn.vkDestroyImage(device, fxaa_source_image_, nullptr);
                fxaa_source_image_ = VK_NULL_HANDLE;
                use_fxaa = false;
              } else {
                VkImageViewCreateInfo fxaa_source_image_view_create_info;
                fxaa_source_image_view_create_info.sType =
                    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                fxaa_source_image_view_create_info.pNext = nullptr;
                fxaa_source_image_view_create_info.flags = 0;
                fxaa_source_image_view_create_info.image = fxaa_source_image_;
                fxaa_source_image_view_create_info.viewType =
                    VK_IMAGE_VIEW_TYPE_2D;
                fxaa_source_image_view_create_info.format = kFxaaSourceFormat;
                fxaa_source_image_view_create_info.components.r =
                    VK_COMPONENT_SWIZZLE_IDENTITY;
                fxaa_source_image_view_create_info.components.g =
                    VK_COMPONENT_SWIZZLE_IDENTITY;
                fxaa_source_image_view_create_info.components.b =
                    VK_COMPONENT_SWIZZLE_IDENTITY;
                fxaa_source_image_view_create_info.components.a =
                    VK_COMPONENT_SWIZZLE_IDENTITY;
                fxaa_source_image_view_create_info.subresourceRange.aspectMask =
                    VK_IMAGE_ASPECT_COLOR_BIT;
                fxaa_source_image_view_create_info.subresourceRange
                    .baseMipLevel = 0;
                fxaa_source_image_view_create_info.subresourceRange.levelCount =
                    1;
                fxaa_source_image_view_create_info.subresourceRange
                    .baseArrayLayer = 0;
                fxaa_source_image_view_create_info.subresourceRange.layerCount =
                    1;
                if (dfn.vkCreateImageView(
                        device, &fxaa_source_image_view_create_info, nullptr,
                        &fxaa_source_image_view_) != VK_SUCCESS) {
                  XELOGE("Failed to create the FXAA source image view");
                  dfn.vkFreeMemory(device, fxaa_source_memory_, nullptr);
                  fxaa_source_memory_ = VK_NULL_HANDLE;
                  dfn.vkDestroyImage(device, fxaa_source_image_, nullptr);
                  fxaa_source_image_ = VK_NULL_HANDLE;
                  use_fxaa = false;
                } else {
                  fxaa_source_width_ = frontbuffer_width_scaled;
                  fxaa_source_height_ = frontbuffer_height_scaled;
                }
              }
            }
          }
        }

        // FXAA can result in more than 8 bits of precision.
        context.SetIs8bpc(!use_pwl_gamma_ramp && !use_fxaa);

        // Update the gamma ramp if it's out of date.
        uint32_t& gamma_ramp_frame_index_ref =
            use_pwl_gamma_ramp ? gamma_ramp_pwl_current_frame_
                               : gamma_ramp_256_entry_table_current_frame_;
        if (gamma_ramp_frame_index_ref == UINT32_MAX) {
          constexpr uint32_t kGammaRampSize256EntryTable =
              sizeof(uint32_t) * 256;
          constexpr uint32_t kGammaRampSizePWL = sizeof(uint16_t) * 2 * 3 * 128;
          constexpr uint32_t kGammaRampSize =
              kGammaRampSize256EntryTable + kGammaRampSizePWL;
          uint32_t gamma_ramp_offset_in_frame =
              use_pwl_gamma_ramp ? kGammaRampSize256EntryTable : 0;
          uint32_t gamma_ramp_upload_offset =
              kGammaRampSize * swap_frame_index + gamma_ramp_offset_in_frame;
          uint32_t gamma_ramp_size = use_pwl_gamma_ramp
                                         ? kGammaRampSizePWL
                                         : kGammaRampSize256EntryTable;
          void* gamma_ramp_frame_upload =
              reinterpret_cast<uint8_t*>(gamma_ramp_upload_mapping_) +
              gamma_ramp_upload_offset;
          if (std::endian::native != std::endian::little &&
              use_pwl_gamma_ramp) {
            // R16G16 is first R16, where the shader expects the base, and
            // second G16, where the delta should be, but gamma_ramp_pwl_rgb()
            // is an array of 32-bit DC_LUT_PWL_DATA registers - swap 16 bits in
            // each 32.
            auto gamma_ramp_pwl_upload =
                reinterpret_cast<reg::DC_LUT_PWL_DATA*>(
                    gamma_ramp_frame_upload);
            const reg::DC_LUT_PWL_DATA* gamma_ramp_pwl = gamma_ramp_pwl_rgb();
            for (size_t i = 0; i < 128 * 3; ++i) {
              reg::DC_LUT_PWL_DATA& gamma_ramp_pwl_upload_entry =
                  gamma_ramp_pwl_upload[i];
              reg::DC_LUT_PWL_DATA gamma_ramp_pwl_entry = gamma_ramp_pwl[i];
              gamma_ramp_pwl_upload_entry.base = gamma_ramp_pwl_entry.delta;
              gamma_ramp_pwl_upload_entry.delta = gamma_ramp_pwl_entry.base;
            }
          } else {
            std::memcpy(
                gamma_ramp_frame_upload,
                use_pwl_gamma_ramp
                    ? static_cast<const void*>(gamma_ramp_pwl_rgb())
                    : static_cast<const void*>(gamma_ramp_256_entry_table()),
                gamma_ramp_size);
          }
          bool gamma_ramp_has_upload_buffer =
              gamma_ramp_upload_buffer_memory_ != VK_NULL_HANDLE;
          ui::vulkan::util::FlushMappedMemoryRange(
              vulkan_device,
              gamma_ramp_has_upload_buffer ? gamma_ramp_upload_buffer_memory_
                                           : gamma_ramp_buffer_memory_,
              gamma_ramp_upload_memory_type_, gamma_ramp_upload_offset,
              gamma_ramp_upload_memory_size_, gamma_ramp_size);
          if (gamma_ramp_has_upload_buffer) {
            // Copy from the host-visible buffer to the device-local one.
            PushBufferMemoryBarrier(
                gamma_ramp_buffer_, gamma_ramp_offset_in_frame, gamma_ramp_size,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED, false);
            SubmitBarriers(true);
            InsertDebugMarker("Gamma Ramp Upload: %u bytes",
                              static_cast<uint32_t>(gamma_ramp_size));
            VkBufferCopy gamma_ramp_buffer_copy;
            gamma_ramp_buffer_copy.srcOffset = gamma_ramp_upload_offset;
            gamma_ramp_buffer_copy.dstOffset = gamma_ramp_offset_in_frame;
            gamma_ramp_buffer_copy.size = gamma_ramp_size;
            deferred_command_buffer_.CmdVkCopyBuffer(gamma_ramp_upload_buffer_,
                                                     gamma_ramp_buffer_, 1,
                                                     &gamma_ramp_buffer_copy);
            PushBufferMemoryBarrier(
                gamma_ramp_buffer_, gamma_ramp_offset_in_frame, gamma_ramp_size,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
          }
          // The device-local, but not host-visible, gamma ramp buffer doesn't
          // have per-frame sets of gamma ramps.
          gamma_ramp_frame_index_ref =
              gamma_ramp_has_upload_buffer ? 0 : swap_frame_index;
        }

        // Track FXAA source texture submission.
        if (use_fxaa) {
          fxaa_source_last_submission_ = GetCurrentSubmission();
        }

        // Determine the destination image for gamma application.
        // If FXAA is enabled, gamma writes to the FXAA source texture.
        // Otherwise, it writes directly to the guest output.
        VkImage apply_gamma_dest_image =
            use_fxaa ? fxaa_source_image_ : vulkan_context.image();
        VkImageView apply_gamma_dest_image_view =
            use_fxaa ? fxaa_source_image_view_ : vulkan_context.image_view();

        // Transition destination image for compute shader storage image write.
        if (use_fxaa) {
          // FXAA source - transition from undefined (we'll be overwriting).
          PushImageMemoryBarrier(apply_gamma_dest_image,
                                 ui::vulkan::util::InitializeSubresourceRange(),
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL);
        } else if (vulkan_context.image_ever_written_previously()) {
          // Guest output - insert a barrier after the last presenter's usage.
          // Will be overwriting all the contents, so oldLayout can be
          // UNDEFINED.
          PushImageMemoryBarrier(
              apply_gamma_dest_image,
              ui::vulkan::util::InitializeSubresourceRange(),
              ui::vulkan::VulkanPresenter::kGuestOutputInternalStageMask,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              ui::vulkan::VulkanPresenter::kGuestOutputInternalAccessMask,
              VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_GENERAL);
        } else {
          // Guest output - first write to the image, just transition.
          PushImageMemoryBarrier(apply_gamma_dest_image,
                                 ui::vulkan::util::InitializeSubresourceRange(),
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL);
        }

        // End the current render pass before inserting barriers.
        SubmitBarriers(true);

        PushDebugMarker("Apply Gamma Ramp: %s%s",
                        use_pwl_gamma_ramp ? "PWL" : "256-entry table",
                        use_fxaa ? " (with FXAA luma)" : "");

        // Select the appropriate gamma pipeline.
        VkPipeline gamma_pipeline;
        if (use_pwl_gamma_ramp) {
          gamma_pipeline = use_fxaa ? swap_apply_gamma_pwl_fxaa_luma_pipeline_
                                    : swap_apply_gamma_pwl_pipeline_;
        } else {
          gamma_pipeline =
              use_fxaa ? swap_apply_gamma_256_entry_table_fxaa_luma_pipeline_
                       : swap_apply_gamma_256_entry_table_pipeline_;
        }
        deferred_command_buffer_.CmdVkBindPipeline(
            VK_PIPELINE_BIND_POINT_COMPUTE, gamma_pipeline);

        // Update the source descriptor set with the swap texture.
        VkDescriptorSet swap_descriptor_source =
            swap_descriptors_source_[swap_frame_index];
        VkDescriptorImageInfo swap_descriptor_source_image_info;
        swap_descriptor_source_image_info.sampler = VK_NULL_HANDLE;
        swap_descriptor_source_image_info.imageView = swap_texture_view;
        swap_descriptor_source_image_info.imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet swap_descriptor_source_write;
        swap_descriptor_source_write.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        swap_descriptor_source_write.pNext = nullptr;
        swap_descriptor_source_write.dstSet = swap_descriptor_source;
        swap_descriptor_source_write.dstBinding = 0;
        swap_descriptor_source_write.dstArrayElement = 0;
        swap_descriptor_source_write.descriptorCount = 1;
        swap_descriptor_source_write.descriptorType =
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        swap_descriptor_source_write.pImageInfo =
            &swap_descriptor_source_image_info;
        swap_descriptor_source_write.pBufferInfo = nullptr;
        swap_descriptor_source_write.pTexelBufferView = nullptr;

        // Update the destination descriptor set with the destination image.
        // When FXAA is enabled, use a separate descriptor set for writing to
        // FXAA source to avoid the issue where both passes would use the same
        // descriptor set (which gets updated twice before submission).
        VkDescriptorSet swap_descriptor_dest =
            use_fxaa ? fxaa_source_storage_descriptors_[swap_frame_index]
                     : swap_descriptors_dest_[swap_frame_index];
        VkDescriptorImageInfo swap_descriptor_dest_image_info;
        swap_descriptor_dest_image_info.sampler = VK_NULL_HANDLE;
        swap_descriptor_dest_image_info.imageView = apply_gamma_dest_image_view;
        swap_descriptor_dest_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet swap_descriptor_dest_write;
        swap_descriptor_dest_write.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        swap_descriptor_dest_write.pNext = nullptr;
        swap_descriptor_dest_write.dstSet = swap_descriptor_dest;
        swap_descriptor_dest_write.dstBinding = 0;
        swap_descriptor_dest_write.dstArrayElement = 0;
        swap_descriptor_dest_write.descriptorCount = 1;
        swap_descriptor_dest_write.descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        swap_descriptor_dest_write.pImageInfo =
            &swap_descriptor_dest_image_info;
        swap_descriptor_dest_write.pBufferInfo = nullptr;
        swap_descriptor_dest_write.pTexelBufferView = nullptr;

        std::array<VkWriteDescriptorSet, 2> swap_descriptor_writes = {
            swap_descriptor_source_write, swap_descriptor_dest_write};
        dfn.vkUpdateDescriptorSets(device,
                                   uint32_t(swap_descriptor_writes.size()),
                                   swap_descriptor_writes.data(), 0, nullptr);

        // Set push constants.
        ApplyGammaConstants apply_gamma_constants;
        apply_gamma_constants.size[0] = frontbuffer_width_scaled;
        apply_gamma_constants.size[1] = frontbuffer_height_scaled;
        deferred_command_buffer_.CmdVkPushConstants(
            swap_apply_gamma_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(apply_gamma_constants), &apply_gamma_constants);

        // Bind descriptor sets.
        std::array<VkDescriptorSet, kSwapApplyGammaDescriptorSetCount>
            swap_descriptor_sets{};
        swap_descriptor_sets[kSwapApplyGammaDescriptorSetRamp] =
            swap_descriptors_gamma_ramp_[2 * gamma_ramp_frame_index_ref +
                                         uint32_t(use_pwl_gamma_ramp)];
        swap_descriptor_sets[kSwapApplyGammaDescriptorSetSource] =
            swap_descriptor_source;
        swap_descriptor_sets[kSwapApplyGammaDescriptorSetDest] =
            swap_descriptor_dest;
        // TODO(Triang3l): Red / blue swap without imageViewFormatSwizzle.
        deferred_command_buffer_.CmdVkBindDescriptorSets(
            VK_PIPELINE_BIND_POINT_COMPUTE, swap_apply_gamma_pipeline_layout_,
            0, uint32_t(swap_descriptor_sets.size()),
            swap_descriptor_sets.data(), 0, nullptr);

        // Dispatch compute shader. Local size is 16x8.
        uint32_t group_count_x = (frontbuffer_width_scaled + 15) / 16;
        uint32_t group_count_y = (frontbuffer_height_scaled + 7) / 8;
        ++submission_in_progress_.dispatch_count;
        deferred_command_buffer_.CmdVkDispatch(group_count_x, group_count_y, 1);

        PopDebugMarker();

        // Apply FXAA if enabled.
        if (use_fxaa) {
          PushDebugMarker("FXAA: %s",
                          swap_post_effect == SwapPostEffect::kFxaaExtreme
                              ? "Extreme"
                              : "Standard");

          // Transition FXAA source from storage image write to sampled read.
          // Transition guest output to storage image write.
          PushImageMemoryBarrier(
              fxaa_source_image_,
              ui::vulkan::util::InitializeSubresourceRange(),
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
              VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
          if (vulkan_context.image_ever_written_previously()) {
            PushImageMemoryBarrier(
                vulkan_context.image(),
                ui::vulkan::util::InitializeSubresourceRange(),
                ui::vulkan::VulkanPresenter::kGuestOutputInternalStageMask,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                ui::vulkan::VulkanPresenter::kGuestOutputInternalAccessMask,
                VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL);
          } else {
            PushImageMemoryBarrier(
                vulkan_context.image(),
                ui::vulkan::util::InitializeSubresourceRange(),
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL);
          }
          SubmitBarriers(true);

          // Bind FXAA pipeline.
          deferred_command_buffer_.CmdVkBindPipeline(
              VK_PIPELINE_BIND_POINT_COMPUTE,
              swap_post_effect == SwapPostEffect::kFxaaExtreme
                  ? fxaa_extreme_pipeline_
                  : fxaa_pipeline_);

          // Update FXAA source descriptor (combined image sampler).
          VkDescriptorSet fxaa_source_descriptor =
              fxaa_source_descriptors_[swap_frame_index];
          VkDescriptorImageInfo fxaa_source_descriptor_image_info;
          fxaa_source_descriptor_image_info.sampler = fxaa_sampler_;
          fxaa_source_descriptor_image_info.imageView = fxaa_source_image_view_;
          fxaa_source_descriptor_image_info.imageLayout =
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          VkWriteDescriptorSet fxaa_source_descriptor_write;
          fxaa_source_descriptor_write.sType =
              VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          fxaa_source_descriptor_write.pNext = nullptr;
          fxaa_source_descriptor_write.dstSet = fxaa_source_descriptor;
          fxaa_source_descriptor_write.dstBinding = 0;
          fxaa_source_descriptor_write.dstArrayElement = 0;
          fxaa_source_descriptor_write.descriptorCount = 1;
          fxaa_source_descriptor_write.descriptorType =
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
          fxaa_source_descriptor_write.pImageInfo =
              &fxaa_source_descriptor_image_info;
          fxaa_source_descriptor_write.pBufferInfo = nullptr;
          fxaa_source_descriptor_write.pTexelBufferView = nullptr;

          // Update FXAA destination descriptor (guest output image).
          VkDescriptorSet fxaa_dest_descriptor =
              swap_descriptors_dest_[swap_frame_index];
          VkDescriptorImageInfo fxaa_dest_descriptor_image_info;
          fxaa_dest_descriptor_image_info.sampler = VK_NULL_HANDLE;
          fxaa_dest_descriptor_image_info.imageView =
              vulkan_context.image_view();
          fxaa_dest_descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
          VkWriteDescriptorSet fxaa_dest_descriptor_write;
          fxaa_dest_descriptor_write.sType =
              VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          fxaa_dest_descriptor_write.pNext = nullptr;
          fxaa_dest_descriptor_write.dstSet = fxaa_dest_descriptor;
          fxaa_dest_descriptor_write.dstBinding = 0;
          fxaa_dest_descriptor_write.dstArrayElement = 0;
          fxaa_dest_descriptor_write.descriptorCount = 1;
          fxaa_dest_descriptor_write.descriptorType =
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
          fxaa_dest_descriptor_write.pImageInfo =
              &fxaa_dest_descriptor_image_info;
          fxaa_dest_descriptor_write.pBufferInfo = nullptr;
          fxaa_dest_descriptor_write.pTexelBufferView = nullptr;

          std::array<VkWriteDescriptorSet, 2> fxaa_descriptor_writes = {
              fxaa_source_descriptor_write, fxaa_dest_descriptor_write};
          dfn.vkUpdateDescriptorSets(device,
                                     uint32_t(fxaa_descriptor_writes.size()),
                                     fxaa_descriptor_writes.data(), 0, nullptr);

          // Set FXAA push constants.
          FxaaConstants fxaa_constants;
          fxaa_constants.size[0] = frontbuffer_width_scaled;
          fxaa_constants.size[1] = frontbuffer_height_scaled;
          fxaa_constants.size_inv[0] =
              1.0f / static_cast<float>(frontbuffer_width_scaled);
          fxaa_constants.size_inv[1] =
              1.0f / static_cast<float>(frontbuffer_height_scaled);
          deferred_command_buffer_.CmdVkPushConstants(
              fxaa_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
              sizeof(fxaa_constants), &fxaa_constants);

          // Bind FXAA descriptor sets.
          std::array<VkDescriptorSet, 2> fxaa_descriptor_sets = {
              fxaa_dest_descriptor, fxaa_source_descriptor};
          deferred_command_buffer_.CmdVkBindDescriptorSets(
              VK_PIPELINE_BIND_POINT_COMPUTE, fxaa_pipeline_layout_, 0,
              uint32_t(fxaa_descriptor_sets.size()),
              fxaa_descriptor_sets.data(), 0, nullptr);

          // Dispatch FXAA compute shader.
          ++submission_in_progress_.dispatch_count;
          deferred_command_buffer_.CmdVkDispatch(group_count_x, group_count_y,
                                                 1);

          PopDebugMarker();
        }

        // Insert the release barrier - transition from GENERAL to the
        // presenter's expected layout.
        PushImageMemoryBarrier(
            vulkan_context.image(),
            ui::vulkan::util::InitializeSubresourceRange(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            ui::vulkan::VulkanPresenter::kGuestOutputInternalStageMask,
            VK_ACCESS_SHADER_WRITE_BIT,
            ui::vulkan::VulkanPresenter::kGuestOutputInternalAccessMask,
            VK_IMAGE_LAYOUT_GENERAL,
            ui::vulkan::VulkanPresenter::kGuestOutputInternalLayout);

        // Need to submit all the commands before giving the image back to the
        // presenter so it can submit its own commands for displaying it to the
        // queue, and also need to submit the release barrier.
        EndSubmission(true);
        return true;
      });

  // End the frame even if did not present for any reason (the image refresher
  // was not called), to prevent leaking per-frame resources.
  EndSubmission(true);
}

void VulkanCommandProcessor::OnPrimaryBufferEnd() {
  // Pump any completed resolves now since the guest is likely about to poll.
  PumpQueryResolves();
  PumpPendingRetire();

  if (cvars::submit_on_primary_buffer_end && submission_open_ &&
      !scratch_buffer_used_ && CanEndSubmissionImmediately()) {
    ++vk_frame_sync_stats_.primary_buffer_splits;
    EndSubmission(false);
  }
}

bool VulkanCommandProcessor::PushBufferMemoryBarrier(
    VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
    VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
    VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask,
    uint32_t src_queue_family_index, uint32_t dst_queue_family_index,
    bool skip_if_equal) {
  if (skip_if_equal && src_stage_mask == dst_stage_mask &&
      src_access_mask == dst_access_mask &&
      src_queue_family_index == dst_queue_family_index) {
    return false;
  }

  // Separate different barriers for overlapping buffer ranges into different
  // pipeline barrier commands.
  for (const VkBufferMemoryBarrier& other_buffer_memory_barrier :
       pending_barriers_buffer_memory_barriers_) {
    if (other_buffer_memory_barrier.buffer != buffer ||
        (size != VK_WHOLE_SIZE &&
         offset + size <= other_buffer_memory_barrier.offset) ||
        (other_buffer_memory_barrier.size != VK_WHOLE_SIZE &&
         other_buffer_memory_barrier.offset +
                 other_buffer_memory_barrier.size <=
             offset)) {
      continue;
    }
    if (other_buffer_memory_barrier.offset == offset &&
        other_buffer_memory_barrier.size == size &&
        other_buffer_memory_barrier.srcAccessMask == src_access_mask &&
        other_buffer_memory_barrier.dstAccessMask == dst_access_mask &&
        other_buffer_memory_barrier.srcQueueFamilyIndex ==
            src_queue_family_index &&
        other_buffer_memory_barrier.dstQueueFamilyIndex ==
            dst_queue_family_index) {
      // The barrier is already pending.
      current_pending_barrier_.src_stage_mask |= src_stage_mask;
      current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
      return true;
    }
    SplitPendingBarrier();
    break;
  }

  current_pending_barrier_.src_stage_mask |= src_stage_mask;
  current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
  VkBufferMemoryBarrier& buffer_memory_barrier =
      pending_barriers_buffer_memory_barriers_.emplace_back();
  buffer_memory_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  buffer_memory_barrier.pNext = nullptr;
  buffer_memory_barrier.srcAccessMask = src_access_mask;
  buffer_memory_barrier.dstAccessMask = dst_access_mask;
  buffer_memory_barrier.srcQueueFamilyIndex = src_queue_family_index;
  buffer_memory_barrier.dstQueueFamilyIndex = dst_queue_family_index;
  buffer_memory_barrier.buffer = buffer;
  buffer_memory_barrier.offset = offset;
  buffer_memory_barrier.size = size;
  return true;
}

bool VulkanCommandProcessor::PushImageMemoryBarrier(
    VkImage image, const VkImageSubresourceRange& subresource_range,
    VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
    VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask,
    VkImageLayout old_layout, VkImageLayout new_layout,
    uint32_t src_queue_family_index, uint32_t dst_queue_family_index,
    bool skip_if_equal) {
  if (skip_if_equal && src_stage_mask == dst_stage_mask &&
      src_access_mask == dst_access_mask && old_layout == new_layout &&
      src_queue_family_index == dst_queue_family_index) {
    return false;
  }

  // Separate different barriers for overlapping image subresource ranges into
  // different pipeline barrier commands.
  for (const VkImageMemoryBarrier& other_image_memory_barrier :
       pending_barriers_image_memory_barriers_) {
    if (other_image_memory_barrier.image != image ||
        !(other_image_memory_barrier.subresourceRange.aspectMask &
          subresource_range.aspectMask) ||
        (subresource_range.levelCount != VK_REMAINING_MIP_LEVELS &&
         subresource_range.baseMipLevel + subresource_range.levelCount <=
             other_image_memory_barrier.subresourceRange.baseMipLevel) ||
        (other_image_memory_barrier.subresourceRange.levelCount !=
             VK_REMAINING_MIP_LEVELS &&
         other_image_memory_barrier.subresourceRange.baseMipLevel +
                 other_image_memory_barrier.subresourceRange.levelCount <=
             subresource_range.baseMipLevel) ||
        (subresource_range.layerCount != VK_REMAINING_ARRAY_LAYERS &&
         subresource_range.baseArrayLayer + subresource_range.layerCount <=
             other_image_memory_barrier.subresourceRange.baseArrayLayer) ||
        (other_image_memory_barrier.subresourceRange.layerCount !=
             VK_REMAINING_ARRAY_LAYERS &&
         other_image_memory_barrier.subresourceRange.baseArrayLayer +
                 other_image_memory_barrier.subresourceRange.layerCount <=
             subresource_range.baseArrayLayer)) {
      continue;
    }
    if (other_image_memory_barrier.subresourceRange.aspectMask ==
            subresource_range.aspectMask &&
        other_image_memory_barrier.subresourceRange.baseMipLevel ==
            subresource_range.baseMipLevel &&
        other_image_memory_barrier.subresourceRange.levelCount ==
            subresource_range.levelCount &&
        other_image_memory_barrier.subresourceRange.baseArrayLayer ==
            subresource_range.baseArrayLayer &&
        other_image_memory_barrier.subresourceRange.layerCount ==
            subresource_range.layerCount &&
        other_image_memory_barrier.srcAccessMask == src_access_mask &&
        other_image_memory_barrier.dstAccessMask == dst_access_mask &&
        other_image_memory_barrier.oldLayout == old_layout &&
        other_image_memory_barrier.newLayout == new_layout &&
        other_image_memory_barrier.srcQueueFamilyIndex ==
            src_queue_family_index &&
        other_image_memory_barrier.dstQueueFamilyIndex ==
            dst_queue_family_index) {
      // The barrier is already pending.
      current_pending_barrier_.src_stage_mask |= src_stage_mask;
      current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
      return true;
    }
    SplitPendingBarrier();
    break;
  }

  current_pending_barrier_.src_stage_mask |= src_stage_mask;
  current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
  VkImageMemoryBarrier& image_memory_barrier =
      pending_barriers_image_memory_barriers_.emplace_back();
  image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  image_memory_barrier.pNext = nullptr;
  image_memory_barrier.srcAccessMask = src_access_mask;
  image_memory_barrier.dstAccessMask = dst_access_mask;
  image_memory_barrier.oldLayout = old_layout;
  image_memory_barrier.newLayout = new_layout;
  image_memory_barrier.srcQueueFamilyIndex = src_queue_family_index;
  image_memory_barrier.dstQueueFamilyIndex = dst_queue_family_index;
  image_memory_barrier.image = image;
  image_memory_barrier.subresourceRange = subresource_range;
  return true;
}

bool VulkanCommandProcessor::SubmitBarriers(bool force_end_render_pass) {
  assert_true(submission_open_);
  SplitPendingBarrier();
  if (pending_barriers_.empty()) {
    if (force_end_render_pass) {
      EndRenderPass();
    }
    return false;
  }
  EndRenderPass();
  for (auto it = pending_barriers_.cbegin(); it != pending_barriers_.cend();
       ++it) {
    auto it_next = std::next(it);
    bool is_last = it_next == pending_barriers_.cend();
    // .data() + offset, not &[offset], for buffer and image barriers, because
    // if there are no buffer or image memory barriers in the last pipeline
    // barriers, the offsets may be equal to the sizes of the vectors.
    deferred_command_buffer_.CmdVkPipelineBarrier(
        it->src_stage_mask ? it->src_stage_mask
                           : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        it->dst_stage_mask ? it->dst_stage_mask
                           : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr,
        uint32_t((is_last ? pending_barriers_buffer_memory_barriers_.size()
                          : it_next->buffer_memory_barriers_offset) -
                 it->buffer_memory_barriers_offset),
        pending_barriers_buffer_memory_barriers_.data() +
            it->buffer_memory_barriers_offset,
        uint32_t((is_last ? pending_barriers_image_memory_barriers_.size()
                          : it_next->image_memory_barriers_offset) -
                 it->image_memory_barriers_offset),
        pending_barriers_image_memory_barriers_.data() +
            it->image_memory_barriers_offset);
  }
  pending_barriers_.clear();
  pending_barriers_buffer_memory_barriers_.clear();
  pending_barriers_image_memory_barriers_.clear();
  current_pending_barrier_.buffer_memory_barriers_offset = 0;
  current_pending_barrier_.image_memory_barriers_offset = 0;
  return true;
}

void VulkanCommandProcessor::OpenPassTimestamp(uint32_t bucket_key) {
  if (!pass_timestamp_mapping_) {
    return;
  }
  if (pass_ts_submission_ != GetCurrentSubmission()) {
    pass_ts_submission_ = GetCurrentSubmission();
    pass_ts_count_ = 0;
  }
  if (pass_ts_count_ >= kPassTimestampPairsPerSubmission) {
    ++pass_ts_dropped_;
    return;
  }
  const uint32_t pair =
      uint32_t(pass_ts_submission_ % kPassTimestampRingSubmissions) *
          kPassTimestampPairsPerSubmission +
      pass_ts_count_;
  pass_ts_keys_[pair] = bucket_key;
  deferred_command_buffer_.CmdVkWriteTimestamp(
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pass_timestamp_pool_, pair * 2);
  pass_ts_open_pair_ = pair;
  pass_ts_open_submission_ = pass_ts_submission_;
  pass_open_draws_ = 0;
}

void VulkanCommandProcessor::ClosePassTimestamp() {
  if (pass_ts_open_pair_ == UINT32_MAX) {
    return;
  }
  // A pass that spanned a submission split has its begin in a prior
  // submission's already-copied range; abandon it rather than corrupt this
  // submission's pair count (its begin query is simply never copied).
  if (pass_ts_open_submission_ != GetCurrentSubmission()) {
    pass_ts_open_pair_ = UINT32_MAX;
    ++pass_ts_dropped_;
    return;
  }
  deferred_command_buffer_.CmdVkWriteTimestamp(
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pass_timestamp_pool_,
      pass_ts_open_pair_ * 2 + 1);
  pass_ts_draws_[pass_ts_open_pair_] = pass_open_draws_;
  ++pass_ts_count_;
  pass_ts_open_pair_ = UINT32_MAX;
}

void VulkanCommandProcessor::SubmitBarriersAndEnterRenderTargetCacheRenderPass(
    VkRenderPass render_pass,
    const VulkanRenderTargetCache::Framebuffer* framebuffer) {
  SubmitBarriers(false);

  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  bool use_dynamic_rendering = cvars::vulkan_dynamic_rendering &&
                               vulkan_device->properties().dynamicRendering;

  // Check if we can stay in the current render pass.
  if (use_dynamic_rendering) {
    // For dynamic rendering, compare framebuffer directly.
    if (in_render_pass_ && current_framebuffer_ == framebuffer &&
        current_render_pass_ == VK_NULL_HANDLE) {
      return;
    }
  } else {
    if (current_render_pass_ == render_pass &&
        current_framebuffer_ == framebuffer) {
      return;
    }
  }

  // End current render pass/rendering if active.
  if (in_render_pass_) {
    if (use_dynamic_rendering) {
      deferred_command_buffer_.CmdVkEndRendering();
    } else {
      deferred_command_buffer_.CmdVkEndRenderPass();
    }
    in_render_pass_ = false;
    ClosePassTimestamp();
  }

  current_render_pass_ = use_dynamic_rendering ? VK_NULL_HANDLE : render_pass;
  current_framebuffer_ = framebuffer;
  ++vk_frame_sync_stats_.render_pass_begins;
  OpenPassTimestamp((uint32_t(framebuffer->host_extent.width) << 16) |
                    framebuffer->host_extent.height);

  if (use_dynamic_rendering) {
    // Use dynamic rendering - construct VkRenderingInfo from render targets.
    // The PSI ("accuracy") path emulates EDRAM via SSBOs and uses no real
    // attachments (mirrors the zero-attachment fsi_render_pass_).
    VkRenderingAttachmentInfo color_attachments[xenos::kMaxColorRenderTargets];
    VkRenderingAttachmentInfo depth_attachment = {};
    VkRenderingAttachmentInfo stencil_attachment = {};
    uint32_t color_attachment_count = 0;

    bool has_depth = false;
    bool has_stencil = false;
    if (render_target_cache_->GetPath() ==
        RenderTargetCache::Path::kHostRenderTargets) {
      render_target_cache_->GetLastUpdateRenderingAttachments(
          color_attachments, &color_attachment_count, &depth_attachment,
          &stencil_attachment);
      has_depth =
          depth_attachment.sType == VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      has_stencil = stencil_attachment.sType ==
                    VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    }

    VkRenderingInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.renderArea.offset.x = 0;
    rendering_info.renderArea.offset.y = 0;
    rendering_info.renderArea.extent = framebuffer->host_extent;
    rendering_info.layerCount = 1;
    rendering_info.colorAttachmentCount = color_attachment_count;
    rendering_info.pColorAttachments =
        color_attachment_count ? color_attachments : nullptr;
    rendering_info.pDepthAttachment = has_depth ? &depth_attachment : nullptr;
    rendering_info.pStencilAttachment =
        has_stencil ? &stencil_attachment : nullptr;

    deferred_command_buffer_.CmdVkBeginRendering(&rendering_info);
  } else {
    // Use traditional render pass.
    VkRenderPassBeginInfo render_pass_begin_info;
    render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin_info.pNext = nullptr;
    render_pass_begin_info.renderPass = render_pass;
    render_pass_begin_info.framebuffer = framebuffer->framebuffer;
    render_pass_begin_info.renderArea.offset.x = 0;
    render_pass_begin_info.renderArea.offset.y = 0;
    // TODO(Triang3l): Actual dirty width / height in the deferred command
    // buffer.
    render_pass_begin_info.renderArea.extent = framebuffer->host_extent;
    render_pass_begin_info.clearValueCount = 0;
    render_pass_begin_info.pClearValues = nullptr;
    deferred_command_buffer_.CmdVkBeginRenderPass(&render_pass_begin_info,
                                                  VK_SUBPASS_CONTENTS_INLINE);
  }
  in_render_pass_ = true;

  // Resume any pending ZPD segment now that the pass is open.
  OpenQuerySegment(false);
}

void VulkanCommandProcessor::SubmitBarriersAndEnterRenderTargetCacheRenderPass(
    VkRenderPass render_pass,
    const VulkanRenderTargetCache::Framebuffer* framebuffer,
    VkImageView transfer_dest_view, bool transfer_dest_is_depth) {
  SubmitBarriers(false);

  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  bool use_dynamic_rendering = cvars::vulkan_dynamic_rendering &&
                               vulkan_device->properties().dynamicRendering;

  // Check if we can stay in the current render pass.
  if (use_dynamic_rendering) {
    // For dynamic rendering, compare framebuffer directly.
    if (in_render_pass_ && current_framebuffer_ == framebuffer &&
        current_render_pass_ == VK_NULL_HANDLE) {
      return;
    }
  } else {
    if (current_render_pass_ == render_pass &&
        current_framebuffer_ == framebuffer) {
      return;
    }
  }

  // End current render pass/rendering if active.
  if (in_render_pass_) {
    if (use_dynamic_rendering) {
      deferred_command_buffer_.CmdVkEndRendering();
    } else {
      deferred_command_buffer_.CmdVkEndRenderPass();
    }
    in_render_pass_ = false;
    ClosePassTimestamp();
  }

  current_render_pass_ = use_dynamic_rendering ? VK_NULL_HANDLE : render_pass;
  current_framebuffer_ = framebuffer;
  ++vk_frame_sync_stats_.render_pass_begins;
  // Bit 31 tags EDRAM ownership-transfer passes.
  OpenPassTimestamp(0x80000000u |
                    (uint32_t(framebuffer->host_extent.width) << 16) |
                    framebuffer->host_extent.height);

  if (use_dynamic_rendering) {
    // Use dynamic rendering for transfers - construct VkRenderingInfo from
    // the transfer destination.
    VkRenderingAttachmentInfo color_attachment = {};
    VkRenderingAttachmentInfo depth_attachment = {};
    VkRenderingAttachmentInfo stencil_attachment = {};

    if (transfer_dest_is_depth) {
      depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      depth_attachment.pNext = nullptr;
      depth_attachment.imageView = transfer_dest_view;
      depth_attachment.imageLayout =
          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth_attachment.resolveMode = VK_RESOLVE_MODE_NONE;
      depth_attachment.resolveImageView = VK_NULL_HANDLE;
      depth_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      depth_attachment.clearValue = {};
      // Stencil uses the same attachment.
      stencil_attachment = depth_attachment;
    } else {
      color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      color_attachment.pNext = nullptr;
      color_attachment.imageView = transfer_dest_view;
      // Must agree with the layout the usage barrier actually recorded, which
      // is RENDERING_LOCAL_READ when local-read attachments are enabled.
      color_attachment.imageLayout = render_target_cache_->color_draw_layout();
      color_attachment.resolveMode = VK_RESOLVE_MODE_NONE;
      color_attachment.resolveImageView = VK_NULL_HANDLE;
      color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color_attachment.clearValue = {};
    }

    VkRenderingInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.renderArea.offset.x = 0;
    rendering_info.renderArea.offset.y = 0;
    rendering_info.renderArea.extent = framebuffer->host_extent;
    rendering_info.layerCount = 1;
    rendering_info.colorAttachmentCount = transfer_dest_is_depth ? 0 : 1;
    rendering_info.pColorAttachments =
        transfer_dest_is_depth ? nullptr : &color_attachment;
    rendering_info.pDepthAttachment =
        transfer_dest_is_depth ? &depth_attachment : nullptr;
    rendering_info.pStencilAttachment =
        transfer_dest_is_depth ? &stencil_attachment : nullptr;

    deferred_command_buffer_.CmdVkBeginRendering(&rendering_info);
  } else {
    // Use traditional render pass.
    VkRenderPassBeginInfo render_pass_begin_info;
    render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin_info.pNext = nullptr;
    render_pass_begin_info.renderPass = render_pass;
    render_pass_begin_info.framebuffer = framebuffer->framebuffer;
    render_pass_begin_info.renderArea.offset.x = 0;
    render_pass_begin_info.renderArea.offset.y = 0;
    render_pass_begin_info.renderArea.extent = framebuffer->host_extent;
    render_pass_begin_info.clearValueCount = 0;
    render_pass_begin_info.pClearValues = nullptr;
    deferred_command_buffer_.CmdVkBeginRenderPass(&render_pass_begin_info,
                                                  VK_SUBPASS_CONTENTS_INLINE);
  }
  in_render_pass_ = true;

  OpenQuerySegment(false);
}

void VulkanCommandProcessor::EndRenderPass() {
  assert_true(submission_open_);
  // Sentinel, not 1x1: the rate is undefined at pass start, so the first
  // draw of every pass must emit one.
  current_shading_rate_ = UINT32_MAX;
  if (!in_render_pass_) {
    return;
  }
  // Close native Vulkan occlusion queries before ending the pass. FSI counter
  // segments don't use vkCmdBeginQuery / vkCmdEndQuery and can stay logically
  // open across render passes.
  if (GetZPDMode() != ZPDMode::kFake && zpd_active_segment_.segment_active &&
      !zpd_active_query_is_fsi_) {
    CloseQuerySegment();
    if (zpd_active_segment_.logical_active) {
      zpd_active_segment_.segment_pending_begin = true;
    }
  }
  // Use current_render_pass_ to determine which end command to use.
  // VK_NULL_HANDLE means we used dynamic rendering, otherwise traditional.
  if (current_render_pass_ == VK_NULL_HANDLE) {
    deferred_command_buffer_.CmdVkEndRendering();
  } else {
    deferred_command_buffer_.CmdVkEndRenderPass();
  }
  current_render_pass_ = VK_NULL_HANDLE;
  current_framebuffer_ = nullptr;
  in_render_pass_ = false;
  ClosePassTimestamp();
}

VkDescriptorSet VulkanCommandProcessor::AllocateSingleTransientDescriptor(
    SingleTransientDescriptorLayout transient_descriptor_layout) {
  assert_true(frame_open_);
  VkDescriptorSet descriptor_set;
  std::vector<VkDescriptorSet>& transient_descriptors_free =
      single_transient_descriptors_free_[size_t(transient_descriptor_layout)];
  if (!transient_descriptors_free.empty()) {
    descriptor_set = transient_descriptors_free.back();
    transient_descriptors_free.pop_back();
  } else {
    const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    bool is_storage_buffer =
        transient_descriptor_layout ==
            SingleTransientDescriptorLayout::kStorageBufferCompute ||
        transient_descriptor_layout ==
            SingleTransientDescriptorLayout::kStorageBufferFragment;
    ui::vulkan::LinkedTypeDescriptorSetAllocator&
        transient_descriptor_allocator =
            is_storage_buffer ? transient_descriptor_allocator_storage_buffer_
                              : transient_descriptor_allocator_uniform_buffer_;
    VkDescriptorPoolSize descriptor_count;
    descriptor_count.type = is_storage_buffer
                                ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_count.descriptorCount = 1;
    descriptor_set = transient_descriptor_allocator.Allocate(
        GetSingleTransientDescriptorLayout(transient_descriptor_layout),
        &descriptor_count, 1);
    if (descriptor_set == VK_NULL_HANDLE) {
      return VK_NULL_HANDLE;
    }
  }
  UsedSingleTransientDescriptor used_descriptor;
  used_descriptor.frame = frame_current_;
  used_descriptor.layout = transient_descriptor_layout;
  used_descriptor.set = descriptor_set;
  single_transient_descriptors_used_.emplace_back(used_descriptor);
  return descriptor_set;
}

VkDescriptorSetLayout VulkanCommandProcessor::GetTextureDescriptorSetLayout(
    bool is_vertex, size_t texture_count, size_t sampler_count) {
  size_t binding_count = texture_count + sampler_count;
  if (!binding_count) {
    return descriptor_set_layout_empty_;
  }

  TextureDescriptorSetLayoutKey texture_descriptor_set_layout_key;
  texture_descriptor_set_layout_key.texture_count = uint32_t(texture_count);
  texture_descriptor_set_layout_key.sampler_count = uint32_t(sampler_count);
  texture_descriptor_set_layout_key.is_vertex = uint32_t(is_vertex);
  auto it_existing =
      descriptor_set_layouts_textures_.find(texture_descriptor_set_layout_key);
  if (it_existing != descriptor_set_layouts_textures_.end()) {
    return it_existing->second;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  descriptor_set_layout_bindings_.clear();
  descriptor_set_layout_bindings_.reserve(binding_count);
  VkShaderStageFlags stage_flags =
      is_vertex ? guest_shader_vertex_stages_ : VK_SHADER_STAGE_FRAGMENT_BIT;
  for (size_t i = 0; i < texture_count; ++i) {
    VkDescriptorSetLayoutBinding& descriptor_set_layout_binding =
        descriptor_set_layout_bindings_.emplace_back();
    descriptor_set_layout_binding.binding = uint32_t(i);
    descriptor_set_layout_binding.descriptorType =
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_set_layout_binding.descriptorCount = 1;
    descriptor_set_layout_binding.stageFlags = stage_flags;
  }
  for (size_t i = 0; i < sampler_count; ++i) {
    VkDescriptorSetLayoutBinding& descriptor_set_layout_binding =
        descriptor_set_layout_bindings_.emplace_back();
    descriptor_set_layout_binding.binding = uint32_t(texture_count + i);
    descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptor_set_layout_binding.descriptorCount = 1;
    descriptor_set_layout_binding.stageFlags = stage_flags;
  }
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = nullptr;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = uint32_t(binding_count);
  descriptor_set_layout_create_info.pBindings =
      descriptor_set_layout_bindings_.data();
  VkDescriptorSetLayout texture_descriptor_set_layout;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &texture_descriptor_set_layout) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  descriptor_set_layouts_textures_.emplace(texture_descriptor_set_layout_key,
                                           texture_descriptor_set_layout);
  return texture_descriptor_set_layout;
}

const VulkanPipelineCache::PipelineLayoutProvider*
VulkanCommandProcessor::GetPipelineLayout(size_t texture_count_pixel,
                                          size_t sampler_count_pixel,
                                          size_t texture_count_vertex,
                                          size_t sampler_count_vertex) {
  PipelineLayoutKey pipeline_layout_key;
  pipeline_layout_key.texture_count_pixel = uint16_t(texture_count_pixel);
  pipeline_layout_key.sampler_count_pixel = uint16_t(sampler_count_pixel);
  pipeline_layout_key.texture_count_vertex = uint16_t(texture_count_vertex);
  pipeline_layout_key.sampler_count_vertex = uint16_t(sampler_count_vertex);
  {
    auto it = pipeline_layouts_.find(pipeline_layout_key);
    if (it != pipeline_layouts_.end()) {
      return &it->second;
    }
  }

  VkDescriptorSetLayout descriptor_set_layout_textures_vertex =
      GetTextureDescriptorSetLayout(true, texture_count_vertex,
                                    sampler_count_vertex);
  if (descriptor_set_layout_textures_vertex == VK_NULL_HANDLE) {
    XELOGE(
        "Failed to obtain a Vulkan descriptor set layout for {} sampled images "
        "and {} samplers for guest vertex shaders",
        texture_count_vertex, sampler_count_vertex);
    return nullptr;
  }
  VkDescriptorSetLayout descriptor_set_layout_textures_pixel =
      GetTextureDescriptorSetLayout(false, texture_count_pixel,
                                    sampler_count_pixel);
  if (descriptor_set_layout_textures_pixel == VK_NULL_HANDLE) {
    XELOGE(
        "Failed to obtain a Vulkan descriptor set layout for {} sampled images "
        "and {} samplers for guest pixel shaders",
        texture_count_pixel, sampler_count_pixel);
    return nullptr;
  }

  VkDescriptorSetLayout
      descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetCount];
  // Immutable layouts.
  descriptor_set_layouts
      [SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram] =
          descriptor_set_layout_shared_memory_and_edram_;
  descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetConstants] =
      descriptor_set_layout_constants_;
  // Mutable layouts.
  descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetTexturesVertex] =
      descriptor_set_layout_textures_vertex;
  descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetTexturesPixel] =
      descriptor_set_layout_textures_pixel;

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  VkPipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = nullptr;
  pipeline_layout_create_info.flags = 0;
  pipeline_layout_create_info.setLayoutCount =
      uint32_t(xe::countof(descriptor_set_layouts));
  pipeline_layout_create_info.pSetLayouts = descriptor_set_layouts;
  pipeline_layout_create_info.pushConstantRangeCount = 0;
  pipeline_layout_create_info.pPushConstantRanges = nullptr;
  VkPipelineLayout pipeline_layout;
  if (dfn.vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr,
                                 &pipeline_layout) != VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan pipeline layout for guest drawing with {} "
        "pixel shader and {} vertex shader textures",
        texture_count_pixel, texture_count_vertex);
    return nullptr;
  }
  auto emplaced_pair = pipeline_layouts_.emplace(
      std::piecewise_construct, std::forward_as_tuple(pipeline_layout_key),
      std::forward_as_tuple(pipeline_layout,
                            descriptor_set_layout_textures_vertex,
                            descriptor_set_layout_textures_pixel));
  // unordered_map insertion doesn't invalidate element references.
  return &emplaced_pair.first->second;
}

VulkanCommandProcessor::ScratchBufferAcquisition
VulkanCommandProcessor::AcquireScratchGpuBuffer(
    VkDeviceSize size, VkPipelineStageFlags initial_stage_mask,
    VkAccessFlags initial_access_mask) {
  assert_true(submission_open_);
  assert_false(scratch_buffer_used_);
  if (!submission_open_ || scratch_buffer_used_ || !size) {
    return ScratchBufferAcquisition();
  }

  uint64_t submission_current = GetCurrentSubmission();

  if (scratch_buffer_ != VK_NULL_HANDLE && size <= scratch_buffer_size_) {
    // Already used previously - transition.
    PushBufferMemoryBarrier(scratch_buffer_, 0, VK_WHOLE_SIZE,
                            scratch_buffer_last_stage_mask_, initial_stage_mask,
                            scratch_buffer_last_access_mask_,
                            initial_access_mask);
    scratch_buffer_last_stage_mask_ = initial_stage_mask;
    scratch_buffer_last_access_mask_ = initial_access_mask;
    scratch_buffer_last_usage_submission_ = submission_current;
    scratch_buffer_used_ = true;
    return ScratchBufferAcquisition(*this, scratch_buffer_, initial_stage_mask,
                                    initial_access_mask);
  }

  size = xe::align(size, kScratchBufferSizeIncrement);

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();

  VkDeviceMemory new_scratch_buffer_memory;
  VkBuffer new_scratch_buffer;
  // VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT for
  // texture loading.
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, size,
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal, new_scratch_buffer,
          new_scratch_buffer_memory)) {
    XELOGE(
        "VulkanCommandProcessor: Failed to create a {} MB scratch GPU buffer",
        size >> 20);
    return ScratchBufferAcquisition();
  }

  if (GetCompletedSubmission() >= scratch_buffer_last_usage_submission_) {
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    if (scratch_buffer_ != VK_NULL_HANDLE) {
      dfn.vkDestroyBuffer(device, scratch_buffer_, nullptr);
    }
    if (scratch_buffer_memory_ != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, scratch_buffer_memory_, nullptr);
    }
  } else {
    if (scratch_buffer_ != VK_NULL_HANDLE) {
      destroy_buffers_.emplace_back(scratch_buffer_last_usage_submission_,
                                    scratch_buffer_);
    }
    if (scratch_buffer_memory_ != VK_NULL_HANDLE) {
      destroy_memory_.emplace_back(scratch_buffer_last_usage_submission_,
                                   scratch_buffer_memory_);
    }
  }

  scratch_buffer_memory_ = new_scratch_buffer_memory;
  scratch_buffer_ = new_scratch_buffer;
  scratch_buffer_size_ = size;
  // Not used yet, no need for a barrier.
  scratch_buffer_last_stage_mask_ = initial_access_mask;
  scratch_buffer_last_access_mask_ = initial_stage_mask;
  scratch_buffer_last_usage_submission_ = submission_current;
  scratch_buffer_used_ = true;
  return ScratchBufferAcquisition(*this, new_scratch_buffer, initial_stage_mask,
                                  initial_access_mask);
}

void VulkanCommandProcessor::BindExternalGraphicsPipeline(
    VkPipeline pipeline, bool keep_dynamic_depth_bias,
    bool keep_dynamic_blend_constants, bool keep_dynamic_stencil_mask_ref) {
  if (!keep_dynamic_depth_bias) {
    dynamic_depth_bias_update_needed_ = true;
  }
  if (!keep_dynamic_blend_constants) {
    dynamic_blend_constants_update_needed_ = true;
  }
  if (!keep_dynamic_stencil_mask_ref) {
    dynamic_stencil_compare_mask_front_update_needed_ = true;
    dynamic_stencil_compare_mask_back_update_needed_ = true;
    dynamic_stencil_write_mask_front_update_needed_ = true;
    dynamic_stencil_write_mask_back_update_needed_ = true;
    dynamic_stencil_reference_front_update_needed_ = true;
    dynamic_stencil_reference_back_update_needed_ = true;
  }
  // External (transfer/resolve) pipelines bake all extended dynamic state
  // statically, invalidating the guest draw's dynamic EDS state. Re-dirty every
  // EDS flag so the next guest draw re-emits it (UpdateDynamicState only emits
  // when the matching capability is active, so this is inert when EDS is off).
  dynamic_cull_mode_update_needed_ = true;
  dynamic_front_face_update_needed_ = true;
  dynamic_primitive_topology_update_needed_ = true;
  dynamic_primitive_restart_enable_update_needed_ = true;
  dynamic_depth_test_enable_update_needed_ = true;
  dynamic_depth_write_enable_update_needed_ = true;
  dynamic_depth_compare_op_update_needed_ = true;
  dynamic_stencil_test_enable_update_needed_ = true;
  dynamic_stencil_op_front_update_needed_ = true;
  dynamic_stencil_op_back_update_needed_ = true;
  dynamic_depth_clamp_enable_update_needed_ = true;
  dynamic_polygon_mode_update_needed_ = true;
  dynamic_color_blend_enable_update_needed_ = true;
  dynamic_color_blend_equation_update_needed_ = true;
  dynamic_color_write_mask_update_needed_ = true;
  if (current_external_graphics_pipeline_ == pipeline) {
    return;
  }
  deferred_command_buffer_.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS,
                                             pipeline);
  current_external_graphics_pipeline_ = pipeline;
  current_guest_graphics_pipeline_ = nullptr;
  current_guest_graphics_pipeline_layout_ = VK_NULL_HANDLE;
}

void VulkanCommandProcessor::BindExternalComputePipeline(VkPipeline pipeline) {
  if (current_external_compute_pipeline_ == pipeline) {
    return;
  }
  deferred_command_buffer_.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE,
                                             pipeline);
  current_external_compute_pipeline_ = pipeline;
}

void VulkanCommandProcessor::SetViewport(const VkViewport& viewport) {
  if (!dynamic_viewport_update_needed_) {
    dynamic_viewport_update_needed_ |= dynamic_viewport_.x != viewport.x;
    dynamic_viewport_update_needed_ |= dynamic_viewport_.y != viewport.y;
    dynamic_viewport_update_needed_ |=
        dynamic_viewport_.width != viewport.width;
    dynamic_viewport_update_needed_ |=
        dynamic_viewport_.height != viewport.height;
    dynamic_viewport_update_needed_ |=
        dynamic_viewport_.minDepth != viewport.minDepth;
    dynamic_viewport_update_needed_ |=
        dynamic_viewport_.maxDepth != viewport.maxDepth;
  }
  if (dynamic_viewport_update_needed_) {
    dynamic_viewport_ = viewport;
    deferred_command_buffer_.CmdVkSetViewport(0, 1, &dynamic_viewport_);
    dynamic_viewport_update_needed_ = false;
  }
}

void VulkanCommandProcessor::SetScissor(const VkRect2D& scissor) {
  if (!dynamic_scissor_update_needed_) {
    dynamic_scissor_update_needed_ |=
        dynamic_scissor_.offset.x != scissor.offset.x;
    dynamic_scissor_update_needed_ |=
        dynamic_scissor_.offset.y != scissor.offset.y;
    dynamic_scissor_update_needed_ |=
        dynamic_scissor_.extent.width != scissor.extent.width;
    dynamic_scissor_update_needed_ |=
        dynamic_scissor_.extent.height != scissor.extent.height;
  }
  if (dynamic_scissor_update_needed_) {
    dynamic_scissor_ = scissor;
    deferred_command_buffer_.CmdVkSetScissor(0, 1, &dynamic_scissor_);
    dynamic_scissor_update_needed_ = false;
  }
}

Shader* VulkanCommandProcessor::LoadShader(xenos::ShaderType shader_type,
                                           const uint32_t* host_address,
                                           uint32_t dword_count) {
  return pipeline_cache_->LoadShader(shader_type, host_address, dword_count);
}

bool VulkanCommandProcessor::IssueDraw(xenos::PrimitiveType prim_type,
                                       uint32_t index_count,
                                       IndexBufferInfo* index_buffer_info,
                                       bool major_mode_explicit) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;

  // One-time confirmation of the effective guest depth path for this title
  // (after per-game config has loaded) - lets a setting like the per-game
  // vulkan_depth_unorm24 override be verified from the log.
  static std::atomic<bool> depth_path_logged{false};
  {
    bool expected = false;
    if (depth_path_logged.compare_exchange_strong(expected, true)) {
      bool unorm24 = render_target_cache_->depth_unorm24_vulkan_format_supported();
      XELOGI(
          "Depth path: vulkan_depth_unorm24(effective)={} -> guest 24-bit depth "
          "uses {}",
          cvars::vulkan_depth_unorm24, unorm24 ? "native D24_UNORM_S8" : "float32 emulation");
    }
  }

  xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode == xenos::EdramMode::kCopy) {
    // Special copy handling.
    return IssueCopy();
  }

  const ui::vulkan::VulkanDevice::Properties& device_properties =
      GetVulkanDevice()->properties();

  memexport_ranges_.clear();

  // Vertex shader analysis.
  auto vertex_shader = static_cast<VulkanShader*>(active_vertex_shader());
  if (!vertex_shader) {
    // Always need a vertex shader.
    return false;
  }
  pipeline_cache_->AnalyzeShaderUcode(*vertex_shader);
  // TODO(Triang3l): If the shader uses memory export, but
  // vertexPipelineStoresAndAtomics is not supported, convert the vertex shader
  // to a compute shader and dispatch it after the draw if the draw doesn't use
  // tessellation.
  if (vertex_shader->memexport_eM_written() != 0 &&
      device_properties.vertexPipelineStoresAndAtomics) {
    draw_util::AddMemExportRanges(regs, *vertex_shader, memexport_ranges_);
  }

  // Pixel shader analysis.
  bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  bool is_rasterization_done =
      draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  VulkanShader* pixel_shader = nullptr;
  if (is_rasterization_done) {
    // See xenos::EdramMode for explanation why the pixel shader is only used
    // when it's kColorDepth here.
    if (edram_mode == xenos::EdramMode::kColorDepth) {
      pixel_shader = static_cast<VulkanShader*>(active_pixel_shader());
      if (pixel_shader) {
        pipeline_cache_->AnalyzeShaderUcode(*pixel_shader);
        if (!draw_util::IsPixelShaderNeededWithRasterization(*pixel_shader,
                                                             regs)) {
          pixel_shader = nullptr;
        }
      }
    }
  } else {
    // Disabling pixel shader for this case is also required by the pipeline
    // cache.
    if (memexport_ranges_.empty()) {
      // This draw has no effect.
      return true;
    }
  }
  if (pixel_shader && pixel_shader->memexport_eM_written() != 0 &&
      device_properties.fragmentStoresAndAtomics) {
    draw_util::AddMemExportRanges(regs, *pixel_shader, memexport_ranges_);
  }

  uint32_t ps_param_gen_pos = UINT32_MAX;
  uint32_t interpolator_mask =
      pixel_shader ? (vertex_shader->writes_interpolators() &
                      pixel_shader->GetInterpolatorInputMask(
                          regs.Get<reg::SQ_PROGRAM_CNTL>(),
                          regs.Get<reg::SQ_CONTEXT_MISC>(), ps_param_gen_pos))
                   : 0;

  PrimitiveProcessor::ProcessingResult primitive_processing_result;
  SpirvShaderTranslator::Modification vertex_shader_modification;
  SpirvShaderTranslator::Modification pixel_shader_modification;
  VulkanShader::VulkanTranslation* vertex_shader_translation;
  VulkanShader::VulkanTranslation* pixel_shader_translation;
  uint32_t normalized_color_mask;
  reg::RB_DEPTHCONTROL normalized_depth_control;
  draw_util::HostDepthPolygonOffset host_depth_polygon_offset;
  bool apply_host_depth_polygon_offset = false;

  // Two iterations because a submission (even the current one - in which case
  // it needs to be ended, and a new one must be started) may need to be awaited
  // in case of a sampler count overflow, and if that happens, all subsystem
  // updates done previously must be performed again because the updates done
  // before the awaiting may be referencing objects destroyed by
  // CompletedSubmissionUpdated.
  for (uint32_t i = 0; i < 2; ++i) {
    if (!BeginSubmission(true)) {
      return false;
    }

    // Process primitives.
    if (!primitive_processor_->Process(primitive_processing_result)) {
      return false;
    }
    if (!primitive_processing_result.host_draw_vertex_count) {
      // Nothing to draw.
      return true;
    }
    // TODO(Triang3l): Geometry-type-specific vertex shader, vertex shader as
    // compute.
    // Skip unsupported host vertex shader types (but allow tessellation types
    // through - they will be handled in pipeline creation or rejected there if
    // not fully supported yet). Both AsTriangleStrip fallbacks are needed on
    // hosts without geometry shader support (MoltenVK / MoltenVK-like); the
    // SPIR-V translator and pipeline cache both handle them end-to-end.
    if (primitive_processing_result.host_vertex_shader_type !=
            Shader::HostVertexShaderType::kVertex &&
        primitive_processing_result.host_vertex_shader_type !=
            Shader::HostVertexShaderType::kPointListAsTriangleStrip &&
        primitive_processing_result.host_vertex_shader_type !=
            Shader::HostVertexShaderType::kRectangleListAsTriangleStrip &&
        !Shader::IsHostVertexShaderTypeDomain(
            primitive_processing_result.host_vertex_shader_type)) {
      return false;
    }

    normalized_depth_control = draw_util::GetNormalizedDepthControl(regs);

    // Compute which color render targets are used.
    normalized_color_mask =
        pixel_shader ? draw_util::GetNormalizedColorMask(
                           regs, pixel_shader->writes_color_targets())
                     : 0;
    apply_host_depth_polygon_offset =
        pixel_shader && !pixel_shader->writes_depth() &&
        render_target_cache_->GetPath() ==
            RenderTargetCache::Path::kHostRenderTargets &&
        draw_util::GetHostDepthPolygonOffsetIfNeeded(
            regs, primitive_polygonal, normalized_depth_control,
            normalized_color_mask, host_depth_polygon_offset);

    // Shader modifications.
    vertex_shader_modification =
        pipeline_cache_->GetCurrentVertexShaderModification(
            *vertex_shader, primitive_processing_result.host_vertex_shader_type,
            interpolator_mask, ps_param_gen_pos != UINT32_MAX);
    pixel_shader_modification =
        pixel_shader ? pipeline_cache_->GetCurrentPixelShaderModification(
                           *pixel_shader, interpolator_mask, ps_param_gen_pos,
                           normalized_depth_control, normalized_color_mask,
                           apply_host_depth_polygon_offset)
                     : SpirvShaderTranslator::Modification(0);

    // Translate the shaders now to obtain the sampler bindings.
    vertex_shader_translation = static_cast<VulkanShader::VulkanTranslation*>(
        vertex_shader->GetOrCreateTranslation(
            vertex_shader_modification.value));
    pixel_shader_translation =
        pixel_shader ? static_cast<VulkanShader::VulkanTranslation*>(
                           pixel_shader->GetOrCreateTranslation(
                               pixel_shader_modification.value))
                     : nullptr;
    if (!pipeline_cache_->EnsureShadersTranslated(vertex_shader_translation,
                                                  pixel_shader_translation)) {
      return false;
    }

    // Obtain the samplers. Note that the bindings don't depend on the shader
    // modification, so if on the second iteration of this loop it becomes
    // different for some reason (like a race condition with the guest in index
    // buffer processing in the primitive processor resulting in different host
    // vertex shader types), the bindings will stay the same.
    // Cross-draw cache (cvar vulkan_cache_sampler_parameters): a stage's
    // previous parameters and handles are reused when the binding list (the
    // shader) is the same and no sampler was destroyed since; parameters are
    // re-derived only for fetch slots written since the last draw, and
    // UseSampler (whose at-least-once-per-submission contract keeps handles
    // safe from the LRU eviction) is skipped when the stage's last full
    // UseSampler pass already happened in the current submission.
    uint32_t samplers_overflowed_count = 0;
    const bool sampler_cache_handles_valid =
        cvars::vulkan_cache_sampler_parameters &&
        current_samplers_destroy_generation_ ==
            texture_cache_->sampler_destroy_generation();
    bool sampler_stage_validated[2] = {false, false};
    for (uint32_t j = 0; j < 2; ++j) {
      std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>>&
          shader_samplers =
              j ? current_samplers_pixel_ : current_samplers_vertex_;
      const VulkanShader*& cached_samplers_shader =
          j ? current_samplers_shader_pixel_ : current_samplers_shader_vertex_;
      const VulkanShader* shader = j ? pixel_shader : vertex_shader;
      if (!shader) {
        // Downstream consumers (texture set hash, descriptor writes) iterate
        // the whole vector - it must match this draw's (empty) binding list.
        shader_samplers.clear();
        cached_samplers_shader = nullptr;
        continue;
      }
      const std::vector<VulkanShader::SamplerBinding>& shader_sampler_bindings =
          shader->GetSamplerBindingsAfterTranslation();
      if (!i && sampler_cache_handles_valid &&
          cached_samplers_shader == shader &&
          shader_samplers.size() == shader_sampler_bindings.size()) {
        // Cache hit for this stage.
        const uint32_t fetch_up_to_date =
            j ? current_samplers_fetch_up_to_date_pixel_
              : current_samplers_fetch_up_to_date_vertex_;
        const bool submission_changed =
            (j ? current_samplers_submission_pixel_
               : current_samplers_submission_vertex_) !=
            GetCurrentSubmission();
        for (size_t k = 0; k < shader_sampler_bindings.size(); ++k) {
          std::pair<VulkanTextureCache::SamplerParameters, VkSampler>&
              shader_sampler_pair = shader_samplers[k];
          bool need_use_sampler =
              submission_changed ||
              shader_sampler_pair.second == VK_NULL_HANDLE;
          if (!(fetch_up_to_date &
                (uint32_t(1) << shader_sampler_bindings[k].fetch_constant))) {
            VulkanTextureCache::SamplerParameters parameters =
                texture_cache_->GetSamplerParameters(
                    shader_sampler_bindings[k]);
            if (parameters != shader_sampler_pair.first) {
              shader_sampler_pair.first = parameters;
              shader_sampler_pair.second = VK_NULL_HANDLE;
              need_use_sampler = true;
            }
          }
          if (need_use_sampler) {
            bool sampler_overflowed;
            VkSampler shader_sampler = texture_cache_->UseSampler(
                shader_sampler_pair.first, sampler_overflowed);
            shader_sampler_pair.second = shader_sampler;
            if (shader_sampler == VK_NULL_HANDLE) {
              if (!sampler_overflowed) {
                return false;
              }
              ++samplers_overflowed_count;
            }
          }
        }
        sampler_stage_validated[j] = true;
        continue;
      }
      // Full pass: cache disabled or missed, or the i == 1 overflow retry
      // (where the parameters derived at i == 0 are kept, but every sampler
      // must be revalidated with UseSampler in the possibly new submission).
      if (!i) {
        shader_samplers.clear();
        shader_samplers.reserve(shader_sampler_bindings.size());
        for (const VulkanShader::SamplerBinding& shader_sampler_binding :
             shader_sampler_bindings) {
          shader_samplers.emplace_back(
              texture_cache_->GetSamplerParameters(shader_sampler_binding),
              VK_NULL_HANDLE);
        }
      }
      // Record the binding-list identity before the UseSampler loop so a
      // mid-loop failure return can't leave the vector holding this shader's
      // parameters while the cached key still names the previous shader
      // (the stamps below are only committed on full success, so the next
      // draw re-derives dirty slots and UseSamplers null handles as needed).
      cached_samplers_shader = shader;
      for (std::pair<VulkanTextureCache::SamplerParameters, VkSampler>&
               shader_sampler_pair : shader_samplers) {
        // UseSampler calls are needed even on the second iteration in case the
        // submission was broken (and thus the last usage submission indices for
        // the used samplers need to be updated) due to an overflow within one
        // submission. Though sampler overflow is a very rare situation overall.
        bool sampler_overflowed;
        VkSampler shader_sampler = texture_cache_->UseSampler(
            shader_sampler_pair.first, sampler_overflowed);
        shader_sampler_pair.second = shader_sampler;
        if (shader_sampler == VK_NULL_HANDLE) {
          if (!sampler_overflowed || i) {
            // If !sampler_overflowed, just failed to create a sampler for some
            // reason.
            // If i == 1, an overflow has happened twice, can't recover from it
            // anymore (would enter an infinite loop otherwise if the number of
            // attempts was not limited to 2). Possibly too many unique samplers
            // in one draw, or failed to await submission completion.
            return false;
          }
          ++samplers_overflowed_count;
        }
      }
      sampler_stage_validated[j] = true;
    }
    if (!samplers_overflowed_count) {
      // The sampler pass completed with every needed UseSampler call made in
      // the current submission - commit the cache stamps. Only stages
      // actually validated this draw get their dirty masks cleared; a skipped
      // stage's cached parameters may still reference fetch slots that were
      // written and not re-derived.
      if (cvars::vulkan_cache_sampler_parameters) {
        const uint64_t submission_now = GetCurrentSubmission();
        if (sampler_stage_validated[0]) {
          current_samplers_fetch_up_to_date_vertex_ = ~uint32_t(0);
          current_samplers_submission_vertex_ = submission_now;
        }
        if (sampler_stage_validated[1]) {
          current_samplers_fetch_up_to_date_pixel_ = ~uint32_t(0);
          current_samplers_submission_pixel_ = submission_now;
        }
        current_samplers_destroy_generation_ =
            texture_cache_->sampler_destroy_generation();
      }
      break;
    }
    assert_zero(i);
    // Free space for as many samplers as how many haven't been allocated
    // successfully - obtain the submission index that needs to be awaited to
    // reuse `samplers_overflowed_count` slots. This must be done after all the
    // UseSampler calls, not inside the loop calling UseSampler, because earlier
    // UseSampler calls may "mark for deletion" some samplers that later
    // UseSampler calls in the loop may actually demand.
    uint64_t sampler_overflow_await_submission =
        texture_cache_->GetSubmissionToAwaitOnSamplerOverflow(
            samplers_overflowed_count);
    assert_true(sampler_overflow_await_submission <= GetCurrentSubmission());
    CheckSubmissionCompletionAndDeviceLoss(sampler_overflow_await_submission);
  }

  // Set up the render targets - this may perform dispatches and draws.
  if (!render_target_cache_->Update(is_rasterization_done,
                                    normalized_depth_control,
                                    normalized_color_mask, *vertex_shader)) {
    return false;
  }

  // Create the pipeline (for this, need the render pass from the render target
  // cache), translating the shaders - doing this now to obtain the used
  // textures.
  VulkanPipelineCache::Pipeline* pipeline;
  if (!pipeline_cache_->ConfigurePipeline(
          vertex_shader_translation, pixel_shader_translation,
          primitive_processing_result, normalized_depth_control,
          normalized_color_mask,
          render_target_cache_->last_update_render_pass_key(), &pipeline)) {
    XELOGE("IssueDraw: ConfigurePipeline failed for VS={:016X} PS={:016X}",
           vertex_shader->ucode_data_hash(),
           pixel_shader ? pixel_shader->ucode_data_hash() : 0);
    return false;
  }

  VkPipeline current_pipeline =
      pipeline->pipeline.load(std::memory_order_acquire);
  if (current_pipeline == VK_NULL_HANDLE) {
    // Pipeline is not ready yet - wait for it to be created.
    pipeline_cache_->EndSubmission();
    current_pipeline = pipeline->pipeline.load(std::memory_order_acquire);
    if (current_pipeline == VK_NULL_HANDLE) {
      // Still not ready - something is wrong.
      return false;
    }
  }
  // If async mode is active, this may be a placeholder pipeline. The real
  // pipeline will be swapped in by the creation thread when ready.
  // We re-load the handle to pick up any swap that may have happened.
  current_pipeline = pipeline->pipeline.load(std::memory_order_acquire);

  // Push debug marker with Xbox 360 draw context for RenderDoc annotation.
  // Done early so texture loads appear nested under the draw that uses them.
  if (debug_markers_enabled_) {
    char label[draw_util::kDebugMarkerLabelMaxLength];
    draw_util::FormatDrawDebugMarker(
        label, sizeof(label), prim_type, primitive_processing_result,
        vertex_shader ? vertex_shader->ucode_data_hash() : 0,
        pixel_shader ? pixel_shader->ucode_data_hash() : 0);
    PushDebugMarker("%s", label);
  }

  // Update the textures before most other work in the submission because
  // samplers depend on this (and in case of sampler overflow in a submission,
  // submissions must be split) - may perform dispatches and copying.
  uint32_t used_texture_mask =
      vertex_shader->GetUsedTextureMaskAfterTranslation() |
      (pixel_shader != nullptr
           ? pixel_shader->GetUsedTextureMaskAfterTranslation()
           : 0);
  texture_cache_->RequestTextures(used_texture_mask);

  auto pipeline_layout =
      static_cast<const PipelineLayout*>(pipeline->pipeline_layout);
  // Update the graphics pipeline, and if the new graphics pipeline has a
  // different layout, invalidate incompatible descriptor sets before updating
  // current_guest_graphics_pipeline_layout_.
  // With asynchronous creation the handle may still be VK_NULL_HANDLE here -
  // a deferred bind of the stable slot pointer is recorded instead of
  // stalling. The pipeline cache's EndSubmission at the submission boundary
  // either waits for creation to complete (default) or lets the replay drop
  // the draws of still-unready pipelines (vulkan_async_skip_draws). The
  // de-dup compares slot POINTERS, not handle values.
  // Wrapped in a lambda so the same (EDS-aware) bind and descriptor-set
  // invalidation can be re-run after in-pass transfers replace the bound
  // pipeline (see below, after entering the render pass).
  auto bind_guest_graphics_pipeline = [&]() {
  if (current_guest_graphics_pipeline_ != &pipeline->pipeline) {
    deferred_command_buffer_.CmdVkBindPipelineDeferred(
        VK_PIPELINE_BIND_POINT_GRAPHICS, &pipeline->pipeline);
    current_guest_graphics_pipeline_ = &pipeline->pipeline;
    current_external_graphics_pipeline_ = VK_NULL_HANDLE;
  }
  if (current_guest_graphics_pipeline_layout_ != pipeline_layout) {
    if (current_guest_graphics_pipeline_layout_) {
      // Keep descriptor set layouts for which the new pipeline layout is
      // compatible with the previous one (pipeline layouts are compatible for
      // set N if set layouts 0 through N are compatible). The two texture sets
      // are the highest enum indices, so a change of the vertex texture set
      // layout breaks compatibility for the pixel texture set too; a pixel-only
      // change breaks only the pixel set.
      //
      // Previously the unconditional clear of the texture values-up-to-date
      // bits in UpdateBindings made this case safe even though the computed
      // descriptor_sets_kept was discarded. Now that the value cache may keep
      // those bits set, this block must actively invalidate the values bit,
      // the bound bit, and the value-cache hash for every texture set whose
      // layout is no longer compatible - otherwise a set written under layout A
      // could be skipped and bound under an incompatible layout B.
      uint32_t descriptor_sets_kept =
          uint32_t(SpirvShaderTranslator::kDescriptorSetCount);
      if (current_guest_graphics_pipeline_layout_
              ->descriptor_set_layout_textures_vertex_ref() !=
          pipeline_layout->descriptor_set_layout_textures_vertex_ref()) {
        descriptor_sets_kept = std::min(
            descriptor_sets_kept,
            uint32_t(SpirvShaderTranslator::kDescriptorSetTexturesVertex));
      }
      if (current_guest_graphics_pipeline_layout_
              ->descriptor_set_layout_textures_pixel_ref() !=
          pipeline_layout->descriptor_set_layout_textures_pixel_ref()) {
        descriptor_sets_kept = std::min(
            descriptor_sets_kept,
            uint32_t(SpirvShaderTranslator::kDescriptorSetTexturesPixel));
      }
      // Invalidate every texture set at or above the lowest changed index
      // (the cascade: incompatibility for set N breaks all sets >= N). This
      // supersedes upstream's `bound_up_to_date_ &= (1<<descriptor_sets_kept)-1`
      // masking: in this block descriptor_sets_kept never drops below
      // kDescriptorSetTexturesVertex, so the per-set clears below cover the same
      // bound bits AND add the value/hash invalidation the texture-set cache needs.
      if (descriptor_sets_kept <=
          uint32_t(SpirvShaderTranslator::kDescriptorSetTexturesVertex)) {
        current_graphics_descriptor_set_values_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
        current_graphics_descriptor_sets_bound_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
        current_texture_descriptor_set_hash_valid_vertex_ = false;
      }
      if (descriptor_sets_kept <=
          uint32_t(SpirvShaderTranslator::kDescriptorSetTexturesPixel)) {
        current_graphics_descriptor_set_values_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
        current_graphics_descriptor_sets_bound_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
        current_texture_descriptor_set_hash_valid_pixel_ = false;
      }
    } else {
      // No or unknown pipeline layout previously bound - all bindings are in an
      // indeterminate state. The transient pool has NOT reset within the frame
      // (that path resets the hash validity at frame open), so the cached
      // texture sets are still valid memory; clearing the bound bits forces a
      // re-bind of the still-valid sets without a re-write. Do not invalidate
      // the hash here - the value cache stays usable across an intra-frame
      // submission/command-buffer restart.
      current_graphics_descriptor_sets_bound_up_to_date_ = 0;
    }
    current_guest_graphics_pipeline_layout_ = pipeline_layout;
  }
  };
  bind_guest_graphics_pipeline();

  bool host_render_targets_used = render_target_cache_->GetPath() ==
                                  RenderTargetCache::Path::kHostRenderTargets;

  // Get dynamic rasterizer state.
  draw_util::ViewportInfo viewport_info;

  // Just handling maxViewportDimensions is enough - viewportBoundsRange[1] must
  // be at least 2 * max(maxViewportDimensions[0...1]) - 1, and
  // maxViewportDimensions must be greater than or equal to the size of the
  // largest possible framebuffer attachment (if the viewport has positive
  // offset and is between maxViewportDimensions and viewportBoundsRange[1],
  // GetHostViewportInfo will adjust ndc_scale/ndc_offset to clamp it, and the
  // clamped range will be outside the largest possible framebuffer anyway.
  // FIXME(Triang3l): Possibly handle maxViewportDimensions and
  // viewportBoundsRange separately because when using fragment shader
  // interlocks, framebuffers are not used, while the range may be wider than
  // dimensions? Though viewport bigger than 4096 - the smallest possible
  // maximum dimension (which is below the 8192 texture size limit on the Xbox
  // 360) - and with offset, is probably a situation that never happens in real
  // life. Or even disregard the viewport bounds range in the fragment shader
  // interlocks case completely - apply the viewport and the scissor offset
  // directly to pixel address and to things like ps_param_gen.
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();
  draw_util::GetViewportInfoArgs gviargs{};
  gviargs.Setup(
      draw_resolution_scale_x, draw_resolution_scale_y,
      texture_cache_->draw_resolution_scale_x_divisor(),
      texture_cache_->draw_resolution_scale_y_divisor(), false,
      device_properties.maxViewportDimensions[0],
      device_properties.maxViewportDimensions[1],
      cvars::vulkan_allow_reverse_z, normalized_depth_control,
      host_render_targets_used &&
          render_target_cache_->depth_float24_convert_in_pixel_shader(),
      host_render_targets_used, pixel_shader && pixel_shader->writes_depth());
  gviargs.SetupRegisterValues(regs);

  draw_util::GetHostViewportInfo(&gviargs, viewport_info);
  // Update dynamic graphics pipeline state.
  UpdateDynamicState(viewport_info, primitive_polygonal,
                     normalized_depth_control, draw_resolution_scale_x,
                     draw_resolution_scale_y, apply_host_depth_polygon_offset,
                     pipeline->dynamic_state);

  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();

  // Whether to load the guest 32-bit (usually big-endian) vertex index
  // indirectly in the vertex shader if full 32-bit indices are not supported by
  // the host.
  bool shader_32bit_index_dma =
      !device_properties.fullDrawIndexUint32 &&
      primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
      vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32 &&
      primitive_processing_result.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kVertex;

  // Update system constants before uploading them.
  UpdateSystemConstantValues(
      primitive_polygonal, primitive_processing_result, shader_32bit_index_dma,
      viewport_info, used_texture_mask, normalized_depth_control,
      normalized_color_mask,
      apply_host_depth_polygon_offset ? &host_depth_polygon_offset : nullptr);

  // Update uniform buffers and descriptor sets after binding the pipeline with
  // the new layout.
  if (!UpdateBindings(vertex_shader, pixel_shader)) {
    return false;
  }

  // Ensure vertex buffers are resident.
  //
  // Use the vertex_fetch_bitmap instead of vertex_bindings() to avoid using
  // cached/stale vertex binding indices. The bitmap is populated during shader
  // translation and represents which fetch constant indices the shader actually
  // references, allowing us to check the current register values at draw time.
  const Shader::ConstantRegisterMap& constant_map_vertex =
      vertex_shader->constant_register_map();
  {
    uint32_t vfetch_addresses[96];
    uint32_t vfetch_sizes[96];
    uint32_t vfetch_current_queued = 0;
    for (uint32_t i = 0;
         i < xe::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
      uint32_t vfetch_bits_remaining =
          constant_map_vertex.vertex_fetch_bitmap[i];
      uint32_t j;
      while (xe::bit_scan_forward(vfetch_bits_remaining, &j)) {
        vfetch_bits_remaining = xe::clear_lowest_bit(vfetch_bits_remaining);
        uint32_t vfetch_index = i * 32 + j;
        xenos::xe_gpu_vertex_fetch_t vfetch_constant =
            regs.GetVertexFetch(vfetch_index);
        switch (vfetch_constant.type) {
          case xenos::FetchConstantType::kVertex:
            break;
          case xenos::FetchConstantType::kInvalidVertex:
            if (cvars::gpu_allow_invalid_fetch_constants) {
              break;
            }
            XELOGW(
                "Vertex fetch constant {} ({:08X} {:08X}) has \"invalid\" "
                "type! "
                "This is incorrect behavior, but you can try bypassing this by "
                "launching Xenia with "
                "--gpu_allow_invalid_fetch_constants=true.",
                vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1);
            return false;
          default:
            // Type is kTexture (2) or kInvalidTexture (3) - completely wrong
            // for vertex data
            if (cvars::gpu_allow_invalid_fetch_constants) {
              XELOGW(
                  "Vertex fetch constant {} ({:08X} {:08X}) has wrong type {} "
                  "(texture fetch constant in vertex slot) - allowing due to "
                  "--gpu_allow_invalid_fetch_constants=true. This will likely "
                  "crash or produce garbage!",
                  vfetch_index, vfetch_constant.dword_0,
                  vfetch_constant.dword_1,
                  static_cast<uint32_t>(vfetch_constant.type));
              break;
            }
            XELOGW(
                "Vertex fetch constant {} ({:08X} {:08X}) is completely "
                "invalid! Type={} - this slot contains a texture fetch "
                "constant (type 2), not a vertex fetch constant (type 0). "
                "This may indicate the shader is reading from the wrong fetch "
                "constant index, or the game has a bug.",
                vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1,
                static_cast<uint32_t>(vfetch_constant.type));
            return false;
        }
        vfetch_addresses[vfetch_current_queued] = vfetch_constant.address;
        vfetch_sizes[vfetch_current_queued++] = vfetch_constant.size;
      }
    }

    if (vfetch_current_queued) {
      // Pre-acquire the critical region so we're not repeatedly re-acquiring
      // it in RequestRange - SharedMemory tracks dirty pages and only uploads
      // what actually changed, making redundant calls cheap under a hoisted
      // lock.
      auto shared_memory_request_range_hoisted =
          global_critical_region::Acquire();

      for (uint32_t i = 0; i < vfetch_current_queued; ++i) {
        if (!shared_memory_->RequestRange(vfetch_addresses[i] << 2,
                                          vfetch_sizes[i] << 2)) {
          XELOGE(
              "Failed to request vertex buffer at 0x{:08X} (size {}) in the "
              "shared memory",
              vfetch_addresses[i] << 2, vfetch_sizes[i] << 2);
          return false;
        }
      }
    }
  }

  // Synchronize the memory pages backing memory scatter export streams, and
  // calculate the range that includes the streams for the buffer barrier.
  uint32_t memexport_extent_start = UINT32_MAX, memexport_extent_end = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    uint32_t memexport_range_base_bytes = memexport_range.base_address_dwords
                                          << 2;
    if (!shared_memory_->RequestRange(memexport_range_base_bytes,
                                      memexport_range.size_bytes)) {
      XELOGE(
          "Failed to request memexport stream at 0x{:08X} (size {}) in the "
          "shared memory",
          memexport_range_base_bytes, memexport_range.size_bytes);
      return false;
    }
    memexport_extent_start =
        std::min(memexport_extent_start, memexport_range_base_bytes);
    memexport_extent_end =
        std::max(memexport_extent_end,
                 memexport_range_base_bytes + memexport_range.size_bytes);
  }

  // Insert the shared memory barrier if needed.
  // TODO(Triang3l): Find some PM4 command that can be used for indication of
  // when memexports should be awaited instead of inserting the barrier in Use
  // every time if memory export was done in the previous draw?
  if (memexport_extent_start < memexport_extent_end) {
    shared_memory_->Use(
        VulkanSharedMemory::Usage::kGuestDrawReadWrite,
        std::make_pair(memexport_extent_start,
                       memexport_extent_end - memexport_extent_start));
  } else {
    // With in-pass resolves, fragment shaders may write shared memory inside
    // any guest pass - declare the write usage up front so no barrier is
    // needed at the resolve point.
    shared_memory_->Use(
        (render_target_cache_->local_read_attachments() &&
         !cvars::vulkan_in_pass_resolve_debug_read_usage)
            ? VulkanSharedMemory::Usage::kGuestDrawReadWrite
            : VulkanSharedMemory::Usage::kRead);
  }

  // After all commands that may dispatch, copy or insert barriers, submit the
  // barriers (may end the render pass), and (re)enter the render pass before
  // drawing.
  SubmitBarriersAndEnterRenderTargetCacheRenderPass(
      render_target_cache_->last_update_render_pass(),
      render_target_cache_->last_update_framebuffer());

  // Runs even at rate 0: pipelines declare the rate dynamic whenever the
  // device supports it, so the state must always be set.
  if (fragment_shading_rate_available() && in_render_pass()) {
    // Depth testing separates world geometry from HUD, text and fullscreen
    // post; an always-passing test is UI drawn with it on in name only.
    const bool depth_tested =
        pipeline->dynamic_state.depth_test_enable != VK_FALSE &&
        pipeline->dynamic_state.depth_compare_op != VK_COMPARE_OP_ALWAYS;
    const bool blended =
        pipeline->dynamic_state.color_blend_enable[0] != VK_FALSE;
    bool in_scope = true;
    switch (cvars::vulkan_vrs_scope) {
      case 1:
        in_scope = depth_tested;
        break;
      case 2:
        in_scope = blended;
        break;
      case 3:
        in_scope = depth_tested && blended;
        break;
      default:
        break;
    }
    // Coarse shading only saves pixel shader work, so a trivial shader pays
    // the blockiness for nothing. Render states alone could not separate these.
    if (in_scope && cvars::vulkan_vrs_min_shader_dwords > 0) {
      in_scope = pixel_shader != nullptr &&
                 pixel_shader->ucode_dword_count() >=
                     size_t(cvars::vulkan_vrs_min_shader_dwords);
    }
    const uint32_t requested =
        in_scope ? uint32_t(std::min(cvars::vulkan_vrs_blended, 4)) : 0u;
    // On Adreno the wide rates are single-sample only.
    const uint32_t sample_count = uint32_t(1)
        << uint32_t(render_target_cache_->last_update_render_pass_key()
                        .msaa_samples);
    const uint32_t rate = ClampShadingRate(requested, sample_count);
    ++vrs_hist_[0];
    vrs_hist_[1] += uint32_t(rate > 0);
    vrs_hist_[2] += uint32_t(rate < requested && requested > 0);
    vrs_hist_[3] += uint32_t(!depth_tested);
    vrs_hist_[4] += uint32_t(depth_tested && !in_scope);
    if (rate < requested && !vrs_clamp_reported_) {
      vrs_clamp_reported_ = true;
      XELOGW("VRS: requested rate {} unsupported at {} samples, using {}",
             requested, sample_count, rate);
    }
    if ((vrs_hist_[0] % 20000) == 0) {
      XELOGI(
          "VkVrs: {} draws | COARSE {} | clamped {} | not-depth-tested {} "
          "| shader-too-small {}",
          vrs_hist_[0], vrs_hist_[1], vrs_hist_[2], vrs_hist_[3],
          vrs_hist_[4]);
    }
    if (rate != current_shading_rate_) {
      static const VkExtent2D kRates[] = {
          {1, 1}, {2, 1}, {2, 2}, {4, 2}, {4, 4}};
      deferred_command_buffer_.CmdVkSetFragmentShadingRate(
          kRates[rate].width, kRates[rate].height);
      current_shading_rate_ = rate;
    }
  }

  // Encode render-target transfers that the render target cache queued for
  // execution inside this draw's render pass (avoids breaking the pass on
  // tile-based GPUs). If they can't be merged into the active pass, fall back
  // to performing them in a separate pass and re-enter the guest pass. Either
  // way the transfers change pipeline / dynamic / binding state, so re-emit it
  // before the actual guest draw below.
  if (render_target_cache_->HasPendingDrawPassTransfers()) {
    if (!render_target_cache_->EncodePendingDrawPassTransfers()) {
      if (!render_target_cache_->FlushPendingDrawPassTransfers()) {
        return false;
      }
      SubmitBarriersAndEnterRenderTargetCacheRenderPass(
          render_target_cache_->last_update_render_pass(),
          render_target_cache_->last_update_framebuffer());
    }
    // Re-bind the guest pipeline (deferred, EDS-aware, with descriptor-set
    // invalidation) - the transfer draws bound their own external pipelines and
    // cleared the guest pipeline/layout state.
    bind_guest_graphics_pipeline();
    UpdateDynamicState(viewport_info, primitive_polygonal,
                       normalized_depth_control, draw_resolution_scale_x,
                       draw_resolution_scale_y, apply_host_depth_polygon_offset,
                       pipeline->dynamic_state);
    if (!UpdateBindings(vertex_shader, pixel_shader)) {
      return false;
    }
  }

  // Track for device-lost diagnostics.
  ++submission_in_progress_.draw_count;
  submission_in_progress_.last_vs_hash = vertex_shader->ucode_data_hash();
  submission_in_progress_.last_ps_hash =
      pixel_shader ? pixel_shader->ucode_data_hash() : 0;
  submission_in_progress_.last_render_pass_key =
      render_target_cache_->last_update_render_pass_key().key;

  // Draw.
  if (primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kNone ||
      shader_32bit_index_dma) {
    deferred_command_buffer_.CmdVkDraw(
        primitive_processing_result.host_draw_vertex_count, 1, 0, 0);
  } else {
    std::pair<VkBuffer, VkDeviceSize> index_buffer;
    switch (primitive_processing_result.index_buffer_type) {
      case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA:
        index_buffer.first = shared_memory_->buffer();
        index_buffer.second = primitive_processing_result.guest_index_base;
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted:
        index_buffer = primitive_processor_->GetConvertedIndexBuffer(
            primitive_processing_result.host_index_buffer_handle);
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA:
        index_buffer = primitive_processor_->GetBuiltinIndexBuffer(
            primitive_processing_result.host_index_buffer_handle);
        break;
      default:
        assert_unhandled_case(primitive_processing_result.index_buffer_type);
        return false;
    }
    deferred_command_buffer_.CmdVkBindIndexBuffer(
        index_buffer.first, index_buffer.second,
        primitive_processing_result.host_index_format ==
                xenos::IndexFormat::kInt16
            ? VK_INDEX_TYPE_UINT16
            : VK_INDEX_TYPE_UINT32);
    deferred_command_buffer_.CmdVkDrawIndexed(
        primitive_processing_result.host_draw_vertex_count, 1, 0, 0, 0);
  }

  // Pop debug marker for draw call.
  PopDebugMarker();

  // Invalidate textures in memexported memory and watch for changes.
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    shared_memory_->RangeWrittenByGpu(memexport_range.base_address_dwords << 2,
                                      memexport_range.size_bytes);
  }

  // CPU readback for memexport data (if enabled).
  if (GetGPUSetting(GPUSetting::ReadbackMemexport) &&
      !memexport_ranges_.empty()) {
    // Calculate total size of all memexport ranges.
    uint32_t memexport_total_size = 0;
    for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
      memexport_total_size += memexport_range.size_bytes;
    }

    if (memexport_total_size > 0) {
      if (cvars::readback_memexport_fast) {
        // Fast mode: use double-buffered readback (delayed sync)
        IssueDraw_MemexportReadbackFastPath(memexport_total_size);
      } else {
        // Full mode: immediate sync (original behavior)
        IssueDraw_MemexportReadbackFullPath(memexport_total_size);
      }
    }
  }

  // Optionally split the frame into multiple submissions (every
  // vulkan_mid_frame_submission_draws real draws) so the GPU starts rendering
  // while the rest of the frame's command stream is still being built, instead
  // of receiving the whole frame at swap time and idling until then. Counted
  // here, at the tail of IssueDraw, so only genuine rasterizing draws are
  // counted (kCopy/empty/no-VS paths return earlier). EndSubmission(false)
  // closes the render pass and submits without closing the frame; the next
  // IssueDraw's BeginSubmission(true) reopens the submission, and the same call
  // resets draws_since_submission_ to 0. Guard mirrors OnPrimaryBufferEnd().
  ++draws_since_submission_;
  ++vk_frame_sync_stats_.draws;
  if (pass_ts_open_pair_ != UINT32_MAX) {
    ++pass_open_draws_;
  }
  if (cvars::vulkan_mid_frame_submission_draws > 0 &&
      draws_since_submission_ >=
          uint32_t(cvars::vulkan_mid_frame_submission_draws) &&
      submission_open_ && !scratch_buffer_used_ &&
      CanEndSubmissionImmediately()) {
    EndSubmission(false);
  }

  return true;
}

void VulkanCommandProcessor::IssueDraw_MemexportReadbackFullPath(
    uint32_t memexport_total_size) {
  // Full mode: immediate sync (original behavior)
  VkBuffer readback_buffer = RequestReadbackBuffer(memexport_total_size);
  if (readback_buffer != VK_NULL_HANDLE) {
    const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();

    VkBuffer shared_memory_buffer = shared_memory_->buffer();

    // Ensure shared memory is ready for transfer and end any active render
    // pass.
    shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
    SubmitBarriers(true);

    InsertDebugMarker("Memexport Readback (sync): %u bytes, %zu ranges",
                      memexport_total_size, memexport_ranges_.size());

    // Copy each memexport range to the readback buffer.
    uint32_t readback_buffer_offset = 0;
    for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
      VkBufferCopy copy_region = {};
      copy_region.srcOffset = memexport_range.base_address_dwords << 2;
      copy_region.dstOffset = readback_buffer_offset;
      copy_region.size = memexport_range.size_bytes;

      deferred_command_buffer_.CmdVkCopyBuffer(
          shared_memory_buffer, readback_buffer, 1, &copy_region);

      readback_buffer_offset += memexport_range.size_bytes;
    }

    // Wait for GPU to finish (SYNCHRONIZATION STALL)
    vk_frame_sync_stats_.memexport_awaits++;
    if (AwaitAllQueueOperationsCompletion()) {
      // Map staging buffer and copy to guest memory.
      void* mapped_data;
      if (dfn.vkMapMemory(device, memexport_readback_buffer_memory_, 0,
                          memexport_total_size, 0,
                          &mapped_data) == VK_SUCCESS) {
        if (mapped_data) {
          const uint8_t* readback_bytes =
              static_cast<const uint8_t*>(mapped_data);
          for (const draw_util::MemExportRange& memexport_range :
               memexport_ranges_) {
            memory::vastcpy(memory_->TranslatePhysical(
                                memexport_range.base_address_dwords << 2),
                            const_cast<uint8_t*>(readback_bytes),
                            memexport_range.size_bytes);
            readback_bytes += memexport_range.size_bytes;
          }
        } else {
          XELOGE(
              "VulkanCommandProcessor: Failed to map readback buffer "
              "(mapped_data is null)");
        }
        dfn.vkUnmapMemory(device, memexport_readback_buffer_memory_);
      } else {
        XELOGE(
            "VulkanCommandProcessor: Failed to map readback buffer memory "
            "for memexport");
      }
    } else {
      XELOGE(
          "VulkanCommandProcessor: Failed to complete queue operations for "
          "memexport readback");
    }
  }
}

void VulkanCommandProcessor::IssueDraw_MemexportReadbackFastPath(
    uint32_t memexport_total_size) {
  // Fast mode: double-buffered readback (similar to resolve readback)
  // Create a key based on first range address and total size
  // This should be stable across frames for the same memexport operation
  if (memexport_ranges_.empty()) {
    return;
  }

  uint64_t memexport_key = MakeReadbackResolveKey(
      memexport_ranges_[0].base_address_dwords, memexport_total_size);

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  ReadbackBuffer& rb = memexport_readback_buffers_[memexport_key];
  rb.last_used_frame = frame_current_;

  uint32_t write_index = rb.current_index;
  uint32_t size = AlignReadbackBufferSize(memexport_total_size);

  // Allocate/resize write buffer if needed
  if (size > rb.sizes[write_index]) {
    // Create buffer with TRANSFER_DST usage for copying from GPU.
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer new_buffer;
    if (dfn.vkCreateBuffer(device, &buffer_info, nullptr, &new_buffer) !=
        VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to create memexport readback buffer "
          "of {} MB",
          size >> 20);
      return;
    }

    // Get memory requirements.
    VkMemoryRequirements memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, new_buffer, &memory_requirements);

    // Allocate HOST_VISIBLE | HOST_CACHED | HOST_COHERENT memory for readback.
    const uint32_t memory_type_index = ui::vulkan::util::ChooseMemoryType(
        vulkan_device->memory_types(), memory_requirements.memoryTypeBits,
        ui::vulkan::util::MemoryPurpose::kReadback);

    if (memory_type_index == UINT32_MAX) {
      XELOGE(
          "VulkanCommandProcessor: Failed to find memory type for memexport "
          "readback buffer");
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      return;
    }

    VkMemoryAllocateInfo memory_info = {};
    memory_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_info.allocationSize = memory_requirements.size;
    memory_info.memoryTypeIndex = memory_type_index;

    VkDeviceMemory new_memory;
    if (dfn.vkAllocateMemory(device, &memory_info, nullptr, &new_memory) !=
        VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to allocate memexport readback "
          "buffer memory");
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      return;
    }

    // Bind memory to buffer.
    if (dfn.vkBindBufferMemory(device, new_buffer, new_memory, 0) !=
        VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to bind memexport readback buffer "
          "memory");
      dfn.vkFreeMemory(device, new_memory, nullptr);
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      return;
    }

    // Clean up old buffer if exists
    // Must wait for GPU to finish using the buffer before destroying it
    if (rb.buffers[write_index] != VK_NULL_HANDLE) {
      if (!AwaitAllQueueOperationsCompletion()) {
        XELOGE(
            "VulkanCommandProcessor: Failed to wait for GPU before "
            "destroying old memexport readback buffer");
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        dfn.vkFreeMemory(device, new_memory, nullptr);
        return;
      }
    }
    if (rb.mapped_data[write_index] != nullptr) {
      dfn.vkUnmapMemory(device, rb.memories[write_index]);
      rb.mapped_data[write_index] = nullptr;
    }
    if (rb.buffers[write_index] != VK_NULL_HANDLE) {
      dfn.vkDestroyBuffer(device, rb.buffers[write_index], nullptr);
    }
    if (rb.memories[write_index] != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, rb.memories[write_index], nullptr);
    }

    rb.buffers[write_index] = new_buffer;
    rb.memories[write_index] = new_memory;
    rb.sizes[write_index] = size;

    // Map the new buffer persistently
    if (dfn.vkMapMemory(device, new_memory, 0, size, 0,
                        &rb.mapped_data[write_index]) != VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to persistently map memexport "
          "readback buffer");
      rb.mapped_data[write_index] = nullptr;
    }
  }

  VkBuffer shared_memory_buffer = shared_memory_->buffer();

  // Ensure shared memory is ready for transfer and end any active render pass.
  shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
  // Sync against any prior write to this readback buffer (across submissions
  // or earlier in the frame). The host read at `read_index` (other slot) is
  // unaffected.
  PushBufferMemoryBarrier(
      rb.buffers[write_index], 0, VK_WHOLE_SIZE, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, false);
  SubmitBarriers(true);

  InsertDebugMarker("Memexport Readback (async): %u bytes, %zu ranges",
                    memexport_total_size, memexport_ranges_.size());

  // Copy exported data to current frame's buffer
  uint32_t readback_buffer_offset = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = memexport_range.base_address_dwords << 2;
    copy_region.dstOffset = readback_buffer_offset;
    copy_region.size = memexport_range.size_bytes;

    deferred_command_buffer_.CmdVkCopyBuffer(
        shared_memory_buffer, rb.buffers[write_index], 1, &copy_region);

    readback_buffer_offset += memexport_range.size_bytes;
  }

  // Use delayed sync (read from previous frame's buffer)
  uint32_t read_index = 1 - write_index;

  bool is_cache_miss = false;
  // If previous buffer doesn't exist or is too small, fall back to sync
  // This happens on first use or buffer resize - subsequent frames will be fast
  if (rb.buffers[read_index] == VK_NULL_HANDLE ||
      memexport_total_size > rb.sizes[read_index]) {
    is_cache_miss = true;
    read_index = write_index;
    if (!AwaitAllQueueOperationsCompletion()) {
      return;
    }
  }

  // TODO(has207): figure out why not copying only on cache hit
  // doesn't work on vulkan but works in d3d12.
  // DISABLED:// Only copy on cache miss (when we have fresh data from GPU sync)
  // DISABLED:// On cache hit, we'd be copying stale data from previous frame
  // DISABLED://if (is_cache_miss && rb.buffers[read_index] != VK_NULL_HANDLE &&
  if (rb.buffers[read_index] != VK_NULL_HANDLE &&
      memexport_total_size <= rb.sizes[read_index] &&
      rb.mapped_data[read_index] != nullptr) {
    const uint8_t* readback_bytes =
        static_cast<const uint8_t*>(rb.mapped_data[read_index]);
    for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
      uint8_t* dest_ptr =
          memory_->TranslatePhysical(memexport_range.base_address_dwords << 2);
      // vastcpy requires 64-byte alignment for non-temporal stores.
      // If addresses aren't aligned, fall back to memcpy.
      if ((reinterpret_cast<uintptr_t>(dest_ptr) & 63) == 0 &&
          (reinterpret_cast<uintptr_t>(readback_bytes) & 63) == 0) {
        memory::vastcpy(dest_ptr, const_cast<uint8_t*>(readback_bytes),
                        memexport_range.size_bytes);
      } else {
        std::memcpy(dest_ptr, readback_bytes, memexport_range.size_bytes);
      }
      readback_bytes += memexport_range.size_bytes;
    }
  }

  // Swap buffer index for next time this specific memexport address is used
  // This way next time we write to the other buffer and read from this one
  rb.current_index = 1 - rb.current_index;
}

void VulkanCommandProcessor::EvictOldReadbackBuffers(
    std::unordered_map<uint64_t, ReadbackBuffer>& buffer_map) {
  if (frame_current_ <= kReadbackBufferEvictionAgeFrames) {
    return;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  for (auto it = buffer_map.begin(); it != buffer_map.end();) {
    if (it->second.last_used_frame <
        frame_current_ - kReadbackBufferEvictionAgeFrames) {
      // Unmap and release both buffers
      for (int i = 0; i < 2; i++) {
        if (it->second.mapped_data[i] != nullptr) {
          dfn.vkUnmapMemory(device, it->second.memories[i]);
        }
        if (it->second.buffers[i] != VK_NULL_HANDLE) {
          dfn.vkDestroyBuffer(device, it->second.buffers[i], nullptr);
        }
        if (it->second.memories[i] != VK_NULL_HANDLE) {
          dfn.vkFreeMemory(device, it->second.memories[i], nullptr);
        }
      }
      it = buffer_map.erase(it);
    } else {
      ++it;
    }
  }
}

void VulkanCommandProcessor::ReclaimCompletedTransferResources() {
  const uint64_t completed =
      transfer_completion_timeline_.UpdateAndGetCompletedSubmission();
  while (!transfer_command_buffers_submitted_.empty() &&
         transfer_command_buffers_submitted_.front().first <= completed) {
    transfer_command_buffers_writable_.push_back(
        transfer_command_buffers_submitted_.front().second);
    transfer_command_buffers_submitted_.pop_front();
  }
  while (!transfer_wait_semaphores_in_flight_.empty() &&
         transfer_wait_semaphores_in_flight_.front().first <= completed) {
    transfer_wait_semaphores_free_.push_back(
        transfer_wait_semaphores_in_flight_.front().second);
    transfer_wait_semaphores_in_flight_.pop_front();
  }
}

bool VulkanCommandProcessor::SubmitReadbackCopiesToTransferQueue(
    VkSemaphore graphics_signal_semaphore) {
  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  ReclaimCompletedTransferResources();

  // Acquire a transfer command buffer.
  if (transfer_command_buffers_writable_.empty()) {
    CommandBuffer command_buffer;
    VkCommandPoolCreateInfo command_pool_create_info;
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.pNext = nullptr;
    command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    command_pool_create_info.queueFamilyIndex =
        vulkan_device->queue_family_transfer();
    if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr,
                                &command_buffer.pool) != VK_SUCCESS) {
      XELOGE("Failed to create a Vulkan transfer command pool");
      return false;
    }
    VkCommandBufferAllocateInfo command_buffer_allocate_info;
    command_buffer_allocate_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.pNext = nullptr;
    command_buffer_allocate_info.commandPool = command_buffer.pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1;
    if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info,
                                     &command_buffer.buffer) != VK_SUCCESS) {
      XELOGE("Failed to allocate a Vulkan transfer command buffer");
      dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
      return false;
    }
    transfer_command_buffers_writable_.push_back(command_buffer);
  }
  CommandBuffer command_buffer = transfer_command_buffers_writable_.back();

  if (dfn.vkResetCommandPool(device, command_buffer.pool, 0) != VK_SUCCESS) {
    XELOGE("Failed to reset a Vulkan transfer command pool");
    return false;
  }
  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = nullptr;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  command_buffer_begin_info.pInheritanceInfo = nullptr;
  if (dfn.vkBeginCommandBuffer(command_buffer.buffer,
                               &command_buffer_begin_info) != VK_SUCCESS) {
    XELOGE("Failed to begin a Vulkan transfer command buffer");
    return false;
  }

  VkBuffer shared_memory_buffer = shared_memory_->buffer();
  for (const PendingReadbackCopy& copy : pending_readback_copies_) {
    VkBufferCopy region = {};
    region.srcOffset = copy.src_offset;
    region.dstOffset = 0;
    region.size = copy.size;
    dfn.vkCmdCopyBuffer(command_buffer.buffer, shared_memory_buffer,
                        copy.readback_buffer->buffers[copy.buffer_index], 1,
                        &region);
  }

  if (dfn.vkEndCommandBuffer(command_buffer.buffer) != VK_SUCCESS) {
    XELOGE("Failed to end a Vulkan transfer command buffer");
    return false;
  }

  // Wait for the graphics submission that wrote the resolved data into shared
  // memory (the semaphore also makes those writes visible to the transfer
  // queue). Shared memory is created with concurrent sharing for this.
  VkPipelineStageFlags wait_stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &graphics_signal_semaphore;
  submit_info.pWaitDstStageMask = &wait_stage_mask;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer.buffer;

  const uint64_t transfer_submission =
      transfer_completion_timeline_.GetUpcomingSubmission();
  if (transfer_completion_timeline_.AcquireFenceAndSubmit(
          vulkan_device->queue_family_transfer(), 0, 1, &submit_info) !=
      VK_SUCCESS) {
    XELOGE("VulkanCommandProcessor: Failed to submit resolve readback copies");
    return false;
  }

  transfer_command_buffers_submitted_.emplace_back(transfer_submission,
                                                   command_buffer);
  transfer_command_buffers_writable_.pop_back();
  transfer_wait_semaphores_in_flight_.emplace_back(transfer_submission,
                                                   graphics_signal_semaphore);
  pending_readback_copies_.clear();
  return true;
}

bool VulkanCommandProcessor::IssueCopy() {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  if (!BeginSubmission(true)) {
    return false;
  }

  // Push debug marker for resolve operation.
  if (debug_markers_enabled_) {
    PushDebugMarker("IssueCopy (Resolve)");
  }

  // Bracket the resolve emission region with GPU timestamps (BOTTOM_OF_PIPE:
  // the region begins and ends at full barriers already, so the wait-for-idle
  // Turnip implies is ~free here).
  uint32_t resolve_ts_pair = UINT32_MAX;
  if (resolve_timestamp_mapping_) {
    if (resolve_ts_submission_ != GetCurrentSubmission()) {
      resolve_ts_submission_ = GetCurrentSubmission();
      resolve_ts_count_ = 0;
    }
    if (resolve_ts_count_ < kResolveTimestampPairsPerSubmission) {
      resolve_ts_pair =
          uint32_t(resolve_ts_submission_ % kResolveTimestampRingSubmissions) *
              kResolveTimestampPairsPerSubmission +
          resolve_ts_count_;
      deferred_command_buffer_.CmdVkWriteTimestamp(
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, resolve_timestamp_pool_,
          resolve_ts_pair * 2);
    } else {
      ++vk_frame_sync_stats_.resolve_ts_dropped;
    }
  }

  uint32_t written_address, written_length;
  const bool resolve_succeeded = render_target_cache_->Resolve(
      *memory_, *shared_memory_, *texture_cache_, written_address,
      written_length);
  if (resolve_ts_pair != UINT32_MAX) {
    // Always close an opened pair - a WAIT_BIT results copy over a written
    // begin with no end would hang the GPU.
    deferred_command_buffer_.CmdVkWriteTimestamp(
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, resolve_timestamp_pool_,
        resolve_ts_pair * 2 + 1);
    ++resolve_ts_count_;
  }
  if (!resolve_succeeded) {
    if (debug_markers_enabled_) {
      PopDebugMarker();
    }
    return false;
  }
  ++submission_in_progress_.resolve_count;
  ++vk_frame_sync_stats_.resolves;

  // CPU readback resolve path (if not disabled).
  ReadbackResolveMode readback_mode = GetReadbackResolveMode();
  if (readback_mode != ReadbackResolveMode::kDisabled &&
      !texture_cache_->IsDrawResolutionScaled() && written_length > 0) {
    // Early check: if destination memory is not accessible, skip all the
    // expensive GPU readback work.
    VirtualHeap* physical_heap = memory_->GetPhysicalHeap();
    bool memory_accessible = false;
    if (physical_heap) {
      HeapAllocationInfo alloc_info;
      if (physical_heap->QueryRegionInfo(written_address, &alloc_info) &&
          (alloc_info.state & kMemoryAllocationCommit) &&
          IsWritableProtect(alloc_info.protect)) {
        uint32_t end_address = written_address + written_length;
        uint32_t region_end = alloc_info.base_address + alloc_info.region_size;
        if (end_address <= region_end) {
          memory_accessible = true;
        }
      }
    }

    if (!memory_accessible) {
      // Destination memory not accessible, skip readback entirely
      PopDebugMarker();
      return true;
    }

    // Create a key for this specific resolve operation
    uint64_t resolve_key =
        MakeReadbackResolveKey(written_address, written_length);

    // "uma" mode: on a unified-memory GPU the shared-memory buffer is mapped, so
    // the resolved bytes can be read straight out of it into guest RAM with no
    // GPU staging copy, no barriers and no readback buffers.
    if (readback_mode == ReadbackResolveMode::kUma &&
        shared_memory_->IsHostMapped()) {
      const uint64_t resolve_submission = GetCurrentSubmission();
      ReadbackBuffer& uma_rb = readback_buffers_[resolve_key];
      uma_rb.last_used_frame = frame_current_;

      bool do_read;
      if (uma_rb.last_write_submission == 0) {
        // First time this destination is resolved: shared memory does not hold a
        // resolve result yet, so make the just-recorded resolve visible to the
        // host and drain to read this frame's data (also serves on-demand
        // consumers such as screenshots).
        PushBufferMemoryBarrier(
            shared_memory_->buffer(), 0, VK_WHOLE_SIZE,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, false);
        SubmitBarriers(true);
        vk_frame_sync_stats_.readback_awaits++;
        if (!AwaitAllQueueOperationsCompletion()) {
          XELOGE(
              "VulkanCommandProcessor: Failed to complete queue operations for "
              "uma resolve readback");
          PopDebugMarker();
          return true;
        }
        do_read = true;
      } else {
        // Already resolved before: read the previous resolve's bytes directly,
        // but only once that submission has retired, so the CPU never races the
        // GPU still writing the same region (tear-free) and the write is
        // host-visible. Uses the cached completed-submission value - no blocking
        // poll (vkGetFenceStatus stalls on Turnip/kgsl). If it has not retired,
        // skip: guest RAM keeps the last consistent readback. Worst-case
        // staleness is bounded by the frames-in-flight depth.
        do_read = GetCompletedSubmission() >= uma_rb.last_write_submission;
      }

      if (do_read) {
        InsertDebugMarker("Resolve Readback (uma): 0x%08X, %u bytes",
                          written_address, written_length);
        uint8_t* dest_ptr = memory_->TranslatePhysical(written_address);
        shared_memory_->ReadHostMapped(written_address, written_length,
                                       dest_ptr);
      }
      uma_rb.last_write_submission = resolve_submission;
      PopDebugMarker();
      return true;
    }

    const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();

    ReadbackBuffer& rb = readback_buffers_[resolve_key];
    rb.last_used_frame = frame_current_;

    uint32_t write_index = rb.current_index;
    uint32_t size = AlignReadbackBufferSize(written_length);

    // "uma" reaches here only as a fallback when the shared-memory buffer is
    // not host-mapped; behave like "fast" then.
    const bool use_delayed_sync =
        (readback_mode == ReadbackResolveMode::kFast ||
         readback_mode == ReadbackResolveMode::kSome ||
         readback_mode == ReadbackResolveMode::kUma);
    // fast/some can offload the copy to the dedicated transfer queue (DMA
    // engine) so it runs at full bandwidth in parallel with rendering instead
    // of serializing on the graphics queue.
    const bool use_transfer_queue =
        use_delayed_sync &&
        vulkan_device->queue_family_transfer() != UINT32_MAX;

    // Allocate/resize write buffer if needed
    if (size > rb.sizes[write_index]) {
      // Create buffer with TRANSFER_DST usage for copying from GPU. When a
      // transfer queue exists it is written by both the graphics queue (sync
      // fallback) and the transfer queue, so share it concurrently between
      // them.
      const uint32_t readback_queue_families[2] = {
          vulkan_device->queue_family_graphics_compute(),
          vulkan_device->queue_family_transfer()};
      VkBufferCreateInfo buffer_info = {};
      buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      buffer_info.size = size;
      buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      if (vulkan_device->queue_family_transfer() != UINT32_MAX) {
        buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        buffer_info.queueFamilyIndexCount = 2;
        buffer_info.pQueueFamilyIndices = readback_queue_families;
      } else {
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      }

      VkBuffer new_buffer;
      if (dfn.vkCreateBuffer(device, &buffer_info, nullptr, &new_buffer) !=
          VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to create readback buffer of {} MB",
            size >> 20);
        PopDebugMarker();
        return true;
      }

      // Get memory requirements.
      VkMemoryRequirements memory_requirements;
      dfn.vkGetBufferMemoryRequirements(device, new_buffer,
                                        &memory_requirements);

      // Allocate HOST_VISIBLE | HOST_CACHED | HOST_COHERENT memory for
      // readback.
      const uint32_t memory_type_index = ui::vulkan::util::ChooseMemoryType(
          vulkan_device->memory_types(), memory_requirements.memoryTypeBits,
          ui::vulkan::util::MemoryPurpose::kReadback);

      if (memory_type_index == UINT32_MAX) {
        XELOGE(
            "VulkanCommandProcessor: Failed to find memory type for readback "
            "buffer");
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        PopDebugMarker();
        return true;
      }

      VkMemoryAllocateInfo memory_info = {};
      memory_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      memory_info.allocationSize = memory_requirements.size;
      memory_info.memoryTypeIndex = memory_type_index;

      VkDeviceMemory new_memory;
      if (dfn.vkAllocateMemory(device, &memory_info, nullptr, &new_memory) !=
          VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to allocate readback buffer "
            "memory");
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        PopDebugMarker();
        return true;
      }

      // Bind memory to buffer.
      if (dfn.vkBindBufferMemory(device, new_buffer, new_memory, 0) !=
          VK_SUCCESS) {
        XELOGE("VulkanCommandProcessor: Failed to bind readback buffer memory");
        dfn.vkFreeMemory(device, new_memory, nullptr);
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        PopDebugMarker();
        return true;
      }

      // Clean up old buffer if exists
      // Must wait for GPU to finish using the buffer before destroying it -
      // both the graphics queue and any in-flight transfer-queue copy to it.
      if (rb.buffers[write_index] != VK_NULL_HANDLE) {
        if (!AwaitAllQueueOperationsCompletion()) {
          XELOGE(
              "VulkanCommandProcessor: Failed to wait for GPU before "
              "destroying old readback buffer");
          dfn.vkDestroyBuffer(device, new_buffer, nullptr);
          dfn.vkFreeMemory(device, new_memory, nullptr);
          PopDebugMarker();
          return true;
        }
        transfer_completion_timeline_.AwaitAllSubmissions();
      }
      if (rb.mapped_data[write_index] != nullptr) {
        dfn.vkUnmapMemory(device, rb.memories[write_index]);
        rb.mapped_data[write_index] = nullptr;
      }
      if (rb.buffers[write_index] != VK_NULL_HANDLE) {
        dfn.vkDestroyBuffer(device, rb.buffers[write_index], nullptr);
      }
      if (rb.memories[write_index] != VK_NULL_HANDLE) {
        dfn.vkFreeMemory(device, rb.memories[write_index], nullptr);
      }

      rb.buffers[write_index] = new_buffer;
      rb.memories[write_index] = new_memory;
      rb.sizes[write_index] = size;

      // Map the new buffer persistently
      if (dfn.vkMapMemory(device, new_memory, 0, size, 0,
                          &rb.mapped_data[write_index]) != VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to persistently map readback "
            "buffer");
        rb.mapped_data[write_index] = nullptr;
      }
    }

    if (use_transfer_queue) {
      uint32_t read_index = 1 - write_index;
      bool read_available = rb.buffers[read_index] != VK_NULL_HANDLE &&
                            written_length <= rb.sizes[read_index] &&
                            rb.mapped_data[read_index] != nullptr;
      if (read_available) {
        // "some" only reads on a cache miss, so a cache hit does nothing.
        if (readback_mode != ReadbackResolveMode::kSome) {
          // "fast" defers the copy to the transfer queue and reads last frame's
          // buffer without waiting. A later resolve may overwrite shared memory
          // while the copy reads it. Accepted as the documented approximate
          // tradeoff for fast/some (full stays synchronous on graphics).
          pending_readback_copies_.push_back(
              {&rb, write_index, written_address, written_length});
          uint8_t* dest_ptr = memory_->TranslatePhysical(written_address);
          memory::vastcpy(dest_ptr,
                          static_cast<uint8_t*>(rb.mapped_data[read_index]),
                          written_length);
        }
      } else {
        // No previous buffer yet (first use or resize). Copy and read
        // synchronously on the graphics queue so on-demand readback
        // (screenshots) still gets data this frame.
        VkBuffer shared_memory_buffer = shared_memory_->buffer();
        shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
        PushBufferMemoryBarrier(
            rb.buffers[write_index], 0, VK_WHOLE_SIZE,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
        SubmitBarriers(true);
        InsertDebugMarker("Resolve Readback (sync): 0x%08X, %u bytes",
                          written_address, written_length);
        VkBufferCopy copy_region = {};
        copy_region.srcOffset = written_address;
        copy_region.dstOffset = 0;
        copy_region.size = written_length;
        deferred_command_buffer_.CmdVkCopyBuffer(
            shared_memory_buffer, rb.buffers[write_index], 1, &copy_region);
        vk_frame_sync_stats_.readback_awaits++;
        if (!AwaitAllQueueOperationsCompletion()) {
          XELOGE(
              "VulkanCommandProcessor: Failed to complete queue operations for "
              "resolve readback fallback");
          PopDebugMarker();
          return true;
        }
        if (rb.mapped_data[write_index] != nullptr) {
          uint8_t* dest_ptr = memory_->TranslatePhysical(written_address);
          memory::vastcpy(dest_ptr,
                          static_cast<uint8_t*>(rb.mapped_data[write_index]),
                          written_length);
        }
      }
    } else {
      VkBuffer shared_memory_buffer = shared_memory_->buffer();

      // Ensure shared memory is ready for transfer and end any active render
      // pass.
      shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
      // Sync against any prior write to this readback buffer.
      PushBufferMemoryBarrier(
          rb.buffers[write_index], 0, VK_WHOLE_SIZE,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
      SubmitBarriers(true);

      InsertDebugMarker("Resolve Readback: 0x%08X, %u bytes", written_address,
                        written_length);

      // Copy GPU buffer → staging buffer.
      VkBufferCopy copy_region = {};
      copy_region.srcOffset = written_address;
      copy_region.dstOffset = 0;
      copy_region.size = written_length;

      deferred_command_buffer_.CmdVkCopyBuffer(
          shared_memory_buffer, rb.buffers[write_index], 1, &copy_region);

      uint32_t read_index = write_index;
      if (use_delayed_sync) {
        // Use previous frame's data (avoid stall)
        read_index = 1 - write_index;
      } else {
        // Wait for GPU to finish (accurate but slow)
        vk_frame_sync_stats_.readback_awaits++;
        if (!AwaitAllQueueOperationsCompletion()) {
          XELOGE(
              "VulkanCommandProcessor: Failed to complete queue operations for "
              "resolve readback");
          PopDebugMarker();
          return true;
        }
      }

      bool is_cache_miss = false;
      // If using delayed sync but previous buffer doesn't exist, use current
      // buffer with sync as fallback
      if (use_delayed_sync && (rb.buffers[read_index] == VK_NULL_HANDLE ||
                               written_length > rb.sizes[read_index])) {
        is_cache_miss = true;
        read_index = write_index;
        vk_frame_sync_stats_.readback_awaits++;
        if (!AwaitAllQueueOperationsCompletion()) {
          XELOGE(
              "VulkanCommandProcessor: Failed to complete queue operations for "
              "resolve readback fallback");
          PopDebugMarker();
          return true;
        }
      }

      bool should_copy =
          (readback_mode == ReadbackResolveMode::kSome) ? is_cache_miss : true;
      if (should_copy && rb.buffers[read_index] != VK_NULL_HANDLE &&
          written_length <= rb.sizes[read_index] &&
          rb.mapped_data[read_index] != nullptr) {
        uint8_t* dest_ptr = memory_->TranslatePhysical(written_address);
        memory::vastcpy(dest_ptr,
                        static_cast<uint8_t*>(rb.mapped_data[read_index]),
                        written_length);
      }
    }

    // Swap buffer index for next time this specific resolve address is used
    rb.current_index = 1 - rb.current_index;
  } else if (readback_mode != ReadbackResolveMode::kDisabled &&
             texture_cache_->IsDrawResolutionScaled() && written_length > 0) {
    /* Scaled resolution readback path - GPU compute shader downscaling */

    // Early check: if destination memory is not accessible, skip all the
    // expensive GPU readback work.
    VirtualHeap* physical_heap = memory_->GetPhysicalHeap();
    bool memory_accessible = false;
    if (physical_heap) {
      HeapAllocationInfo alloc_info;
      if (physical_heap->QueryRegionInfo(written_address, &alloc_info) &&
          (alloc_info.state & kMemoryAllocationCommit) &&
          IsWritableProtect(alloc_info.protect)) {
        uint32_t end_address = written_address + written_length;
        uint32_t region_end = alloc_info.base_address + alloc_info.region_size;
        if (end_address <= region_end) {
          memory_accessible = true;
        }
      }
    }

    if (!memory_accessible) {
      // Destination memory not accessible, skip readback entirely
      if (debug_markers_enabled_) {
        PopDebugMarker();
      }
      return true;
    }

    // Get scaled resolve buffer (works for both sparse and simple buffer modes)
    VkBuffer scaled_buffer = texture_cache_->GetCurrentScaledResolveBuffer();
    if (scaled_buffer == VK_NULL_HANDLE) {
      XELOGE("VulkanCommandProcessor: No scaled resolve buffer available");
      if (debug_markers_enabled_) {
        PopDebugMarker();
      }
      return true;
    }

    // Get the current scaled resolve range (not the buffer's total range)
    uint32_t scaled_length = static_cast<uint32_t>(
        texture_cache_->GetCurrentScaledResolveRangeLengthScaled());
    uint64_t scaled_address =
        texture_cache_->GetCurrentScaledResolveRangeStartScaled();

    // Calculate offset within the buffer using the buffer's base address.
    // GetCurrentScaledResolveBufferBaseOffset() returns:
    // - For sparse buffers: buffer_index << 30 (same as D3D12's 1GB chunks)
    // - For simple buffers: the buffer's range_start_scaled
    uint64_t buffer_base =
        texture_cache_->GetCurrentScaledResolveBufferBaseOffset();
    if (scaled_address < buffer_base) {
      XELOGE(
          "VulkanCommandProcessor: Scaled address {} is before buffer start {}",
          scaled_address, buffer_base);
      if (debug_markers_enabled_) {
        PopDebugMarker();
      }
      return true;
    }
    uint64_t source_offset = scaled_address - buffer_base;

    // Get format info for downscaling
    auto copy_dest_info = register_file_->Get<reg::RB_COPY_DEST_INFO>();
    const FormatInfo* format_info =
        FormatInfo::Get(static_cast<uint32_t>(copy_dest_info.copy_dest_format));
    uint32_t bits_per_pixel = format_info->bits_per_pixel;

    uint32_t scale_x = texture_cache_->draw_resolution_scale_x();
    uint32_t scale_y = texture_cache_->draw_resolution_scale_y();

    assert_true(scale_x >= 1 &&
                scale_x <= TextureCache::kMaxDrawResolutionScaleAlongAxis);
    assert_true(scale_y >= 1 &&
                scale_y <= TextureCache::kMaxDrawResolutionScaleAlongAxis);
    assert_true(scale_x > 1 || scale_y > 1);
    assert_true(bits_per_pixel == 8 || bits_per_pixel == 16 ||
                bits_per_pixel == 32 || bits_per_pixel == 64);

    // Use the same keying as non-scaled path
    uint64_t resolve_key =
        MakeReadbackResolveKey(written_address, written_length);

    ReadbackBuffer& rb = readback_buffers_[resolve_key];
    rb.last_used_frame = frame_current_;

    uint32_t write_index = rb.current_index;
    // Readback buffer is now 1x size (downscaled), not scaled_length
    uint32_t size = AlignReadbackBufferSize(written_length);

    const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();

    // Allocate/resize write buffer if needed (1x size now)
    if (size > rb.sizes[write_index]) {
      VkBufferCreateInfo buffer_info = {};
      buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      buffer_info.size = size;
      buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      VkBuffer new_buffer;
      if (dfn.vkCreateBuffer(device, &buffer_info, nullptr, &new_buffer) !=
          VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to create scaled readback buffer "
            "of {} MB",
            size >> 20);
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      VkMemoryRequirements memory_requirements;
      dfn.vkGetBufferMemoryRequirements(device, new_buffer,
                                        &memory_requirements);

      const uint32_t memory_type_index = ui::vulkan::util::ChooseMemoryType(
          vulkan_device->memory_types(), memory_requirements.memoryTypeBits,
          ui::vulkan::util::MemoryPurpose::kReadback);

      if (memory_type_index == UINT32_MAX) {
        XELOGE(
            "VulkanCommandProcessor: Failed to find memory type for scaled "
            "readback buffer");
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      VkMemoryAllocateInfo memory_info = {};
      memory_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      memory_info.allocationSize = memory_requirements.size;
      memory_info.memoryTypeIndex = memory_type_index;

      VkDeviceMemory new_memory;
      if (dfn.vkAllocateMemory(device, &memory_info, nullptr, &new_memory) !=
          VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to allocate scaled readback "
            "buffer memory");
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      if (dfn.vkBindBufferMemory(device, new_buffer, new_memory, 0) !=
          VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to bind scaled readback buffer "
            "memory");
        dfn.vkFreeMemory(device, new_memory, nullptr);
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      // Clean up old buffer if exists
      if (rb.buffers[write_index] != VK_NULL_HANDLE) {
        if (!AwaitAllQueueOperationsCompletion()) {
          XELOGE(
              "VulkanCommandProcessor: Failed to wait for GPU before "
              "destroying old scaled readback buffer");
          dfn.vkDestroyBuffer(device, new_buffer, nullptr);
          dfn.vkFreeMemory(device, new_memory, nullptr);
          if (debug_markers_enabled_) {
            PopDebugMarker();
          }
          return true;
        }
      }
      if (rb.mapped_data[write_index] != nullptr) {
        dfn.vkUnmapMemory(device, rb.memories[write_index]);
        rb.mapped_data[write_index] = nullptr;
      }
      if (rb.buffers[write_index] != VK_NULL_HANDLE) {
        dfn.vkDestroyBuffer(device, rb.buffers[write_index], nullptr);
      }
      if (rb.memories[write_index] != VK_NULL_HANDLE) {
        dfn.vkFreeMemory(device, rb.memories[write_index], nullptr);
      }

      rb.buffers[write_index] = new_buffer;
      rb.memories[write_index] = new_memory;
      rb.sizes[write_index] = size;

      // Map the new buffer persistently
      if (dfn.vkMapMemory(device, new_memory, 0, size, 0,
                          &rb.mapped_data[write_index]) != VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to persistently map scaled "
            "readback buffer");
        rb.mapped_data[write_index] = nullptr;
      }
    }

    // Ensure intermediate buffer for GPU downscaling is large enough
    uint32_t downscale_buffer_size = AlignReadbackBufferSize(written_length);
    if (downscale_buffer_size > resolve_downscale_buffer_size_) {
      // Clean up old buffer
      if (resolve_downscale_buffer_ != VK_NULL_HANDLE) {
        if (!AwaitAllQueueOperationsCompletion()) {
          XELOGE(
              "VulkanCommandProcessor: Failed to wait for GPU before "
              "destroying old downscale buffer");
          if (debug_markers_enabled_) {
            PopDebugMarker();
          }
          return true;
        }
        dfn.vkDestroyBuffer(device, resolve_downscale_buffer_, nullptr);
        dfn.vkFreeMemory(device, resolve_downscale_buffer_memory_, nullptr);
        resolve_downscale_buffer_ = VK_NULL_HANDLE;
        resolve_downscale_buffer_memory_ = VK_NULL_HANDLE;
        resolve_downscale_buffer_size_ = 0;
      }

      VkBufferCreateInfo buffer_info = {};
      buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      buffer_info.size = downscale_buffer_size;
      buffer_info.usage =
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      if (dfn.vkCreateBuffer(device, &buffer_info, nullptr,
                             &resolve_downscale_buffer_) != VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to create {} MB downscale buffer",
            downscale_buffer_size >> 20);
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      VkMemoryRequirements memory_requirements;
      dfn.vkGetBufferMemoryRequirements(device, resolve_downscale_buffer_,
                                        &memory_requirements);

      const uint32_t memory_type_index = ui::vulkan::util::ChooseMemoryType(
          vulkan_device->memory_types(), memory_requirements.memoryTypeBits,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal);

      if (memory_type_index == UINT32_MAX) {
        XELOGE(
            "VulkanCommandProcessor: Failed to find memory type for downscale "
            "buffer");
        dfn.vkDestroyBuffer(device, resolve_downscale_buffer_, nullptr);
        resolve_downscale_buffer_ = VK_NULL_HANDLE;
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      VkMemoryAllocateInfo memory_info = {};
      memory_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      memory_info.allocationSize = memory_requirements.size;
      memory_info.memoryTypeIndex = memory_type_index;

      if (dfn.vkAllocateMemory(device, &memory_info, nullptr,
                               &resolve_downscale_buffer_memory_) !=
          VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to allocate downscale buffer "
            "memory");
        dfn.vkDestroyBuffer(device, resolve_downscale_buffer_, nullptr);
        resolve_downscale_buffer_ = VK_NULL_HANDLE;
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      if (dfn.vkBindBufferMemory(device, resolve_downscale_buffer_,
                                 resolve_downscale_buffer_memory_,
                                 0) != VK_SUCCESS) {
        XELOGE(
            "VulkanCommandProcessor: Failed to bind downscale buffer memory");
        dfn.vkFreeMemory(device, resolve_downscale_buffer_memory_, nullptr);
        dfn.vkDestroyBuffer(device, resolve_downscale_buffer_, nullptr);
        resolve_downscale_buffer_ = VK_NULL_HANDLE;
        resolve_downscale_buffer_memory_ = VK_NULL_HANDLE;
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }

      resolve_downscale_buffer_size_ = downscale_buffer_size;
    }

    // Allocate descriptor set for source and destination buffers.
    // Uses pool chain to avoid mid-frame GPU stalls on pool exhaustion.
    VkDescriptorSet descriptor_set =
        resolve_downscale_descriptor_pool_chain_->Allocate(
            GetCurrentSubmission());
    if (descriptor_set == VK_NULL_HANDLE) {
      XELOGE(
          "VulkanCommandProcessor: Failed to allocate resolve downscale "
          "descriptor set from pool chain");
      if (debug_markers_enabled_) {
        PopDebugMarker();
      }
      return true;
    }

    // Ensure submission is open
    if (!BeginSubmission(true)) {
      XELOGE(
          "VulkanCommandProcessor: Failed to begin submission for scaled "
          "resolve readback");
      if (debug_markers_enabled_) {
        PopDebugMarker();
      }
      return true;
    }

    // Update descriptor set with buffer bindings
    // Bind source buffer at offset 0 to avoid alignment issues - offset is
    // passed via push constants and applied in the shader
    std::array<VkDescriptorBufferInfo, 2> buffer_infos;
    buffer_infos[0].buffer = scaled_buffer;
    buffer_infos[0].offset = 0;
    buffer_infos[0].range = VK_WHOLE_SIZE;
    // Destination buffer (intermediate device-local buffer)
    buffer_infos[1].buffer = resolve_downscale_buffer_;
    buffer_infos[1].offset = 0;
    buffer_infos[1].range = written_length;

    std::array<VkWriteDescriptorSet, 2> descriptor_writes;
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].pNext = nullptr;
    descriptor_writes[0].dstSet = descriptor_set;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[0].pImageInfo = nullptr;
    descriptor_writes[0].pBufferInfo = &buffer_infos[0];
    descriptor_writes[0].pTexelBufferView = nullptr;

    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].pNext = nullptr;
    descriptor_writes[1].dstSet = descriptor_set;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[1].pImageInfo = nullptr;
    descriptor_writes[1].pBufferInfo = &buffer_infos[1];
    descriptor_writes[1].pTexelBufferView = nullptr;

    dfn.vkUpdateDescriptorSets(device, uint32_t(descriptor_writes.size()),
                               descriptor_writes.data(), 0, nullptr);

    // End any active render pass and submit barriers
    SubmitBarriers(true);

    // Barrier for source buffer - ensure resolve copy compute shader writes
    // are complete before readback compute shader reads.
    VkBufferMemoryBarrier source_barrier = {};
    source_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    source_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    source_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    source_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_barrier.buffer = scaled_buffer;
    source_barrier.offset = 0;
    source_barrier.size = VK_WHOLE_SIZE;
    deferred_command_buffer_.CmdVkPipelineBarrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &source_barrier,
        0, nullptr);

    PushDebugMarker("Resolve Downscale: 0x%08X, %u bytes -> %u bytes",
                    written_address, scaled_length, written_length);

    // Bind compute pipeline
    BindExternalComputePipeline(resolve_downscale_pipeline_);

    // Push constants
    uint32_t pixel_size_log2;
    xe::bit_scan_forward(bits_per_pixel >> 3, &pixel_size_log2);
    uint32_t bytes_per_pixel = 1u << pixel_size_log2;
    uint32_t tile_size_1x = 32 * 32 * bytes_per_pixel;
    uint32_t tile_count = written_length / tile_size_1x;

    // Skip if no complete tiles to process
    if (tile_count == 0) {
      XELOGW(
          "VulkanCommandProcessor: Scaled resolve has no complete tiles "
          "(written_length=%u, tile_size=%u)",
          written_length, tile_size_1x);
      if (debug_markers_enabled_) {
        PopDebugMarker();  // Pop "Resolve Downscale"
        PopDebugMarker();  // Pop "IssueCopy (Resolve)"
      }
      return true;
    }

    ResolveDownscaleConstants constants;
    constants.scale_x = scale_x;
    constants.scale_y = scale_y;
    constants.pixel_size_log2 = pixel_size_log2;
    constants.tile_count = tile_count;
    constants.source_offset_bytes = static_cast<uint32_t>(source_offset);
    // Optionally sample from center of scaled block instead of top-left.
    constants.half_pixel_offset = (cvars::readback_resolve_half_pixel_offset &&
                                   (scale_x > 1 || scale_y > 1))
                                      ? 1
                                      : 0;
    deferred_command_buffer_.CmdVkPushConstants(
        resolve_downscale_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(constants), &constants);

    // Bind descriptor set
    deferred_command_buffer_.CmdVkBindDescriptorSets(
        VK_PIPELINE_BIND_POINT_COMPUTE, resolve_downscale_pipeline_layout_, 0,
        1, &descriptor_set, 0, nullptr);

    // Dispatch compute shader - one thread group per 32x32 tile
    ++submission_in_progress_.dispatch_count;
    deferred_command_buffer_.CmdVkDispatch(tile_count, 1, 1);

    // Barriers before copy: compute-write -> transfer-read on the source, and
    // transfer-write -> transfer-write on the readback destination (to sync
    // against any prior write to this slot, across submissions or within the
    // frame).
    VkBufferMemoryBarrier pre_copy_barriers[2] = {};
    pre_copy_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    pre_copy_barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre_copy_barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre_copy_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_copy_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_copy_barriers[0].buffer = resolve_downscale_buffer_;
    pre_copy_barriers[0].offset = 0;
    pre_copy_barriers[0].size = written_length;
    pre_copy_barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    pre_copy_barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre_copy_barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre_copy_barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_copy_barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_copy_barriers[1].buffer = rb.buffers[write_index];
    pre_copy_barriers[1].offset = 0;
    pre_copy_barriers[1].size = VK_WHOLE_SIZE;
    deferred_command_buffer_.CmdVkPipelineBarrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, pre_copy_barriers, 0,
        nullptr);

    // Copy downscaled data to readback buffer
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = written_length;
    deferred_command_buffer_.CmdVkCopyBuffer(
        resolve_downscale_buffer_, rb.buffers[write_index], 1, &copy_region);

    PopDebugMarker();

    // Scaled resolution downscales into a separate buffer, so "uma" cannot read
    // shared memory directly here; treat it like "fast".
    bool use_delayed_sync = (readback_mode == ReadbackResolveMode::kFast ||
                             readback_mode == ReadbackResolveMode::kSome ||
                             readback_mode == ReadbackResolveMode::kUma);
    uint32_t read_index = write_index;

    if (use_delayed_sync) {
      // Use previous frame's data (avoid stall)
      read_index = 1 - write_index;
    } else {
      // Wait for GPU to finish (accurate but slow)
      vk_frame_sync_stats_.readback_awaits++;
      if (!AwaitAllQueueOperationsCompletion()) {
        XELOGE(
            "VulkanCommandProcessor: Failed to complete queue operations for "
            "scaled resolve readback");
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }
    }

    // Check if we have valid data to read from
    bool is_cache_miss = false;
    if (use_delayed_sync && (rb.buffers[read_index] == VK_NULL_HANDLE ||
                             written_length > rb.sizes[read_index] ||
                             rb.mapped_data[read_index] == nullptr)) {
      // Cache miss - need to sync and use current buffer
      is_cache_miss = true;
      read_index = write_index;
      vk_frame_sync_stats_.readback_awaits++;
      if (!AwaitAllQueueOperationsCompletion()) {
        XELOGE(
            "VulkanCommandProcessor: Failed to complete queue operations for "
            "scaled resolve readback fallback");
        if (debug_markers_enabled_) {
          PopDebugMarker();
        }
        return true;
      }
    }

    // Copy to guest memory
    // "some" mode: only copy on cache miss (saves CPU)
    // "fast" mode: always copy (1 frame behind, no GPU stall)
    // "full" mode: always copy (GPU sync already done above)
    bool should_copy =
        (readback_mode == ReadbackResolveMode::kSome) ? is_cache_miss : true;

    // Simple memcpy - data is already downscaled by GPU
    if (should_copy && rb.mapped_data[read_index] != nullptr &&
        written_length <= rb.sizes[read_index]) {
      uint8_t* physaddr = memory_->TranslatePhysical(written_address);
      memory::vastcpy(physaddr,
                      static_cast<uint8_t*>(rb.mapped_data[read_index]),
                      written_length);
    }

    // Swap buffer index for next time
    rb.current_index = 1 - rb.current_index;
  }

  // Pop debug marker for resolve operation.
  if (debug_markers_enabled_) {
    PopDebugMarker();
  }

  return true;
}

VkBuffer VulkanCommandProcessor::RequestReadbackBuffer(uint32_t size) {
  if (size == 0) {
    return VK_NULL_HANDLE;
  }

  size = AlignReadbackBufferSize(size);

  if (size > memexport_readback_buffer_size_) {
    const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();

    // Create buffer with TRANSFER_DST usage for copying from GPU.
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer new_buffer;
    if (dfn.vkCreateBuffer(device, &buffer_info, nullptr, &new_buffer) !=
        VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to create readback buffer of {} MB",
          size >> 20);
      return VK_NULL_HANDLE;
    }

    // Get memory requirements.
    VkMemoryRequirements memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, new_buffer, &memory_requirements);

    // Allocate HOST_VISIBLE | HOST_CACHED | HOST_COHERENT memory for readback.
    const uint32_t memory_type_index = ui::vulkan::util::ChooseMemoryType(
        vulkan_device->memory_types(), memory_requirements.memoryTypeBits,
        ui::vulkan::util::MemoryPurpose::kReadback);
    if (memory_type_index == UINT32_MAX) {
      XELOGE(
          "VulkanCommandProcessor: Failed to find suitable memory type for "
          "readback buffer");
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      return VK_NULL_HANDLE;
    }

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = memory_requirements.size;
    alloc_info.memoryTypeIndex = memory_type_index;

    VkDeviceMemory new_memory;
    if (dfn.vkAllocateMemory(device, &alloc_info, nullptr, &new_memory) !=
        VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to allocate memory for readback "
          "buffer");
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      return VK_NULL_HANDLE;
    }

    // Bind memory to buffer.
    if (dfn.vkBindBufferMemory(device, new_buffer, new_memory, 0) !=
        VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to bind memory to readback buffer");
      dfn.vkFreeMemory(device, new_memory, nullptr);
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      return VK_NULL_HANDLE;
    }

    // Destroy old buffer if it exists.
    if (memexport_readback_buffer_ != VK_NULL_HANDLE) {
      dfn.vkDestroyBuffer(device, memexport_readback_buffer_, nullptr);
      dfn.vkFreeMemory(device, memexport_readback_buffer_memory_, nullptr);
    }

    memexport_readback_buffer_ = new_buffer;
    memexport_readback_buffer_memory_ = new_memory;
    memexport_readback_buffer_size_ = size;
  }

  return memexport_readback_buffer_;
}

void VulkanCommandProcessor::EnsureZPDQueryResources() {
  if (GetZPDMode() == ZPDMode::kFake || !zpd_host_query_pool_) {
    return;
  }

  bool can_recreate =
      !zpd_active_segment_.logical_active &&
      !zpd_active_segment_.segment_active &&
      zpd_active_query_index_ == UINT32_MAX && !zpd_active_query_is_fsi_ &&
      !zpd_host_query_pool_->has_pending_resolve_batch() &&
      zpd_resolves_in_flight_.empty() && zpd_deferred_releases_.empty();

  bool initialize_fsi_counter = render_target_cache_->GetPath() ==
                                RenderTargetCache::Path::kPixelShaderInterlock;

  zpd_host_query_pool_->EnsureInitialized(GetVulkanDevice(),
                                          kZPDQueryPoolCapacity, can_recreate,
                                          initialize_fsi_counter);

  if (initialize_fsi_counter &&
      zpd_host_query_pool_->fsi_counter_initialized()) {
    VkBuffer fsi_counter_buffer = zpd_host_query_pool_->fsi_counter_buffer();
    VkDeviceSize fsi_counter_range =
        sizeof(uint32_t) * zpd_host_query_pool_->capacity();
    if (zpd_fsi_counter_descriptor_buffer_ != fsi_counter_buffer ||
        zpd_fsi_counter_descriptor_range_ != fsi_counter_range) {
      VkDescriptorBufferInfo fsi_counter_descriptor_buffer_info;
      fsi_counter_descriptor_buffer_info.buffer = fsi_counter_buffer;
      fsi_counter_descriptor_buffer_info.offset = 0;
      fsi_counter_descriptor_buffer_info.range = fsi_counter_range;

      VkWriteDescriptorSet fsi_counter_descriptor_write;
      fsi_counter_descriptor_write.sType =
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      fsi_counter_descriptor_write.pNext = nullptr;
      fsi_counter_descriptor_write.dstSet =
          shared_memory_and_edram_descriptor_set_;
      fsi_counter_descriptor_write.dstBinding = 2;
      fsi_counter_descriptor_write.dstArrayElement = 0;
      fsi_counter_descriptor_write.descriptorCount = 1;
      fsi_counter_descriptor_write.descriptorType =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      fsi_counter_descriptor_write.pImageInfo = nullptr;
      fsi_counter_descriptor_write.pBufferInfo =
          &fsi_counter_descriptor_buffer_info;
      fsi_counter_descriptor_write.pTexelBufferView = nullptr;

      const ui::vulkan::VulkanDevice::Functions& dfn =
          GetVulkanDevice()->functions();
      dfn.vkUpdateDescriptorSets(GetVulkanDevice()->device(), 1,
                                 &fsi_counter_descriptor_write, 0, nullptr);

      zpd_fsi_counter_descriptor_buffer_ = fsi_counter_buffer;
      zpd_fsi_counter_descriptor_range_ = fsi_counter_range;
    }
  } else if (!IsZPDQueryPoolReady() && cvars::occlusion_query_log) {
    XELOGI(
        "ZPD/Vulkan: FSI counter resources unavailable; keeping counter index "
        "sentinel active");
  }
  zpd_fsi_counter_index_force_update_ = true;
}

bool VulkanCommandProcessor::IsZPDQueryPoolReady() const {
  if (!zpd_host_query_pool_) {
    return false;
  }
  if (!render_target_cache_ ||
      render_target_cache_->GetPath() !=
          RenderTargetCache::Path::kPixelShaderInterlock) {
    return zpd_host_query_pool_->is_initialized();
  }
  VkDeviceSize fsi_counter_range =
      sizeof(uint32_t) * zpd_host_query_pool_->capacity();
  return zpd_host_query_pool_->fsi_counter_initialized() &&
         zpd_fsi_counter_descriptor_buffer_ ==
             zpd_host_query_pool_->fsi_counter_buffer() &&
         zpd_fsi_counter_descriptor_range_ == fsi_counter_range;
}

bool VulkanCommandProcessor::CanOpenZPDQuery() const {
  if (!submission_open_) {
    return false;
  }
  bool use_fsi_counter_path =
      render_target_cache_ &&
      render_target_cache_->GetPath() ==
          RenderTargetCache::Path::kPixelShaderInterlock;
  return use_fsi_counter_path || in_render_pass_;
}

CommandProcessor::QueryOpenResult VulkanCommandProcessor::OpenZPDQuery(
    ReportHandle report_handle, bool can_close_submission) {
  bool use_fsi_counter_path =
      zpd_host_query_pool_->fsi_counter_initialized() &&
      render_target_cache_->GetPath() ==
          RenderTargetCache::Path::kPixelShaderInterlock;

  if (!BeginSubmission(true)) {
    return QueryOpenResult::kFailed;
  }

  if (!use_fsi_counter_path && !in_render_pass_) {
    return QueryOpenResult::kDeferred;
  }

  bool retried_after_submission_flip = false;
  while (true) {
    bool is_pool_exhausted = !zpd_host_query_pool_->has_free_indices();

    if (is_pool_exhausted) {
      PumpQueryResolves();
      is_pool_exhausted = !zpd_host_query_pool_->has_free_indices();
    }

    if (is_pool_exhausted &&
        (GetZPDMode() == ZPDMode::kFast || GetZPDMode() == ZPDMode::kFastAlt)) {
      return QueryOpenResult::kPoolExhausted;
    }

    uint64_t wait_for = 0;
    if (is_pool_exhausted && !zpd_resolves_in_flight_.empty()) {
      wait_for = zpd_resolves_in_flight_.front().submission;
    }

    if (wait_for == 0) {
      break;
    }

    if (submission_open_ && wait_for == GetCurrentSubmission()) {
      if (retried_after_submission_flip || !can_close_submission ||
          !CanEndSubmissionImmediately()) {
        return QueryOpenResult::kDeferred;
      }

      VkRenderPass saved_render_pass = VK_NULL_HANDLE;
      const VulkanRenderTargetCache::Framebuffer* saved_framebuffer = nullptr;
      if (in_render_pass_) {
        saved_render_pass = current_render_pass_;
        saved_framebuffer = current_framebuffer_;
        EndRenderPass();
      }
      if (!EndSubmission(false)) {
        return QueryOpenResult::kFailed;
      }
      if (!BeginSubmission(true)) {
        return QueryOpenResult::kFailed;
      }

      if (saved_framebuffer) {
        bool saved_pending_begin = zpd_active_segment_.segment_pending_begin;
        zpd_active_segment_.segment_pending_begin = false;
        SubmitBarriersAndEnterRenderTargetCacheRenderPass(saved_render_pass,
                                                          saved_framebuffer);
        zpd_active_segment_.segment_pending_begin = saved_pending_begin;
        if (!in_render_pass_) {
          return QueryOpenResult::kDeferred;
        }
      }

      retried_after_submission_flip = true;
      continue;
    }

    uint64_t completed_submission = GetCompletedSubmission();
    if (wait_for > completed_submission) {
      if (cvars::occlusion_query_log) {
        XELOGI("ZPD: Stall awaiting submission={} completed_before={}",
               wait_for, completed_submission);
      }
      completion_timeline_.AwaitSubmissionAndUpdateCompleted(wait_for);
      PumpQueryResolves();
    }

    break;
  }

  if (!use_fsi_counter_path && !in_render_pass_) {
    return QueryOpenResult::kDeferred;
  }

  if (!zpd_host_query_pool_->AcquireQueryIndex(zpd_active_query_index_,
                                               zpd_active_query_generation_)) {
    return QueryOpenResult::kFailed;
  }

  zpd_active_query_is_fsi_ = use_fsi_counter_path;

  // FSI queries don't use Vulkan occlusion queries at all.
  // While the segment is open, the translated pixel shader accumulates passed
  // MSAA samples into one counter slot selected via zpd_fsi_counter_index.
  // Clear the slot here so a recycled index never inherits old counts.
  if (zpd_active_query_is_fsi_) {
    bool fsi_counter_cleared = false;
    if (zpd_host_query_pool_->fsi_counter_initialized()) {
      if (in_render_pass_) {
        VkRenderPass saved_render_pass = current_render_pass_;
        const VulkanRenderTargetCache::Framebuffer* saved_framebuffer =
            current_framebuffer_;
        EndRenderPass();
        zpd_host_query_pool_->ClearFSICounter(deferred_command_buffer_,
                                              zpd_active_query_index_);

        bool saved_pending_begin = zpd_active_segment_.segment_pending_begin;
        zpd_active_segment_.segment_pending_begin = false;
        SubmitBarriersAndEnterRenderTargetCacheRenderPass(saved_render_pass,
                                                          saved_framebuffer);
        zpd_active_segment_.segment_pending_begin = saved_pending_begin;
        fsi_counter_cleared = in_render_pass_;
      } else {
        zpd_host_query_pool_->ClearFSICounter(deferred_command_buffer_,
                                              zpd_active_query_index_);
        fsi_counter_cleared = true;
      }
    }
    if (!fsi_counter_cleared) {
      zpd_host_query_pool_->ReleaseQueryIndex(zpd_active_query_index_,
                                              zpd_active_query_generation_);
      zpd_active_query_index_ = UINT32_MAX;
      zpd_active_query_generation_ = 0;
      zpd_active_query_is_fsi_ = false;
      zpd_fsi_counter_index_force_update_ = true;
      return QueryOpenResult::kFailed;
    }
    zpd_fsi_counter_index_force_update_ = true;
    return QueryOpenResult::kOpened;
  }

  zpd_host_query_pool_->BeginQuery(deferred_command_buffer_,
                                   zpd_active_query_index_);
  return QueryOpenResult::kOpened;
}

bool VulkanCommandProcessor::CloseZPDQuery(ReportHandle report_handle,
                                           uint64_t& out_submission) {
  if (!zpd_active_query_is_fsi_ && !in_render_pass_) {
    XELOGW("ZPD: Split segment requested outside render pass");
    return false;
  }

  if (zpd_active_query_is_fsi_) {
    zpd_host_query_pool_->QueueQueryResolve(zpd_active_query_index_, true);
  } else {
    zpd_host_query_pool_->EndQuery(deferred_command_buffer_,
                                   zpd_active_query_index_);
    zpd_host_query_pool_->QueueQueryResolve(zpd_active_query_index_, false);
  }

  PendingQueryResolve resolve;
  resolve.submission = GetCurrentSubmission();
  resolve.query_index = zpd_active_query_index_;
  resolve.query_generation = zpd_active_query_generation_;
  resolve.uses_fsi_counter = zpd_active_query_is_fsi_;
  resolve.report_handle = report_handle;
  zpd_resolves_in_flight_.push_back(resolve);

  out_submission = resolve.submission;

  zpd_active_query_index_ = UINT32_MAX;
  zpd_active_query_generation_ = 0;
  bool closed_fsi_counter = zpd_active_query_is_fsi_;
  zpd_active_query_is_fsi_ = false;
  if (closed_fsi_counter) {
    zpd_fsi_counter_index_force_update_ = true;
  }
  return true;
}

bool VulkanCommandProcessor::DiscardZPDQuery() {
  if (zpd_active_query_is_fsi_) {
    // The slot counter may be dirty if draws ran between OpenZPDQuery and
    // here, but the next OpenZPDQuery clears it before any new shader adds.
    if (cvars::occlusion_query_log) {
      XELOGI("ZPD/Vulkan: Discarding FSI counter segment index={}",
             zpd_active_query_index_);
    }
    zpd_host_query_pool_->ReleaseQueryIndex(zpd_active_query_index_,
                                            zpd_active_query_generation_);
    zpd_active_query_index_ = UINT32_MAX;
    zpd_active_query_generation_ = 0;
    zpd_active_query_is_fsi_ = false;
    zpd_fsi_counter_index_force_update_ = true;
    return true;
  }

  if (!in_render_pass_) {
    // vkCmdEndQuery is invalid outside a render pass for occlusion queries.
    // Defer the release until the submission containing the stale BeginQuery
    // completes on the GPU.
    XELOGW("ZPD: Discard segment requested outside render pass");
    zpd_deferred_releases_.push_back({GetCurrentSubmission(),
                                      zpd_active_query_index_,
                                      zpd_active_query_generation_});
    zpd_active_query_index_ = UINT32_MAX;
    zpd_active_query_generation_ = 0;
    zpd_active_query_is_fsi_ = false;
    return true;
  }

  // Inside a render pass, EndQuery must be issued before releasing the slot.
  zpd_host_query_pool_->EndQuery(deferred_command_buffer_,
                                 zpd_active_query_index_);
  zpd_host_query_pool_->ReleaseQueryIndex(zpd_active_query_index_,
                                          zpd_active_query_generation_);
  zpd_active_query_index_ = UINT32_MAX;
  zpd_active_query_generation_ = 0;
  zpd_active_query_is_fsi_ = false;
  return true;
}

void VulkanCommandProcessor::PumpQueryResolves() {
  if (GetZPDMode() == ZPDMode::kFake || !zpd_host_query_pool_) {
    return;
  }

  uint64_t completed = GetCompletedSubmission();
  if (completed == 0) {
    return;
  }

  // Drain deferred releases first.
  while (!zpd_deferred_releases_.empty()) {
    auto& entry = zpd_deferred_releases_.front();
    if (entry.submission > completed) {
      break;
    }
    zpd_host_query_pool_->ReleaseQueryIndex(entry.query_index,
                                            entry.query_generation);
    zpd_deferred_releases_.pop_front();
  }

  // Invalidate CPU cache before reading results on non-coherent memory.
  if (!zpd_resolves_in_flight_.empty() &&
      zpd_resolves_in_flight_.front().submission <= completed) {
    zpd_host_query_pool_->InvalidateReadback();
  }

  while (!zpd_resolves_in_flight_.empty()) {
    if (zpd_resolves_in_flight_.front().submission > completed) {
      break;
    }
    PendingQueryResolve resolve = zpd_resolves_in_flight_.front();
    zpd_resolves_in_flight_.pop_front();

    if (zpd_host_query_pool_->GenerationMatches(resolve.query_index,
                                                resolve.query_generation)) {
      uint64_t raw_samples = zpd_host_query_pool_->GetQueryReadbackValue(
          resolve.query_index, resolve.uses_fsi_counter);
      if (cvars::occlusion_query_log) {
        XELOGI("ZPD/Vulkan: Resolved {} segment index={} samples={}",
               resolve.uses_fsi_counter ? "FSI" : "native", resolve.query_index,
               raw_samples);
      }
      zpd_host_query_pool_->ReleaseQueryIndex(resolve.query_index,
                                              resolve.query_generation);
      OnZPDQueryResolved(resolve.report_handle, raw_samples);
    }
  }
}

bool VulkanCommandProcessor::AwaitQueryResolve(ReportHandle report_handle,
                                               uint64_t wait_for_submission) {
  if (GetZPDMode() == ZPDMode::kFake) {
    return false;
  }

  PumpQueryResolves();

  auto it = logical_zpd_reports_.find(report_handle);
  if (it == logical_zpd_reports_.end()) {
    return true;
  }
  if (it->second.pending_segments == 0 && it->second.ended) {
    return true;
  }
  if (wait_for_submission == 0) {
    return false;
  }

  // Ensure the submission is flushed.
  if (wait_for_submission >= GetCurrentSubmission()) {
    if (!submission_open_) {
      return false;
    }
    if (!CanEndSubmissionImmediately()) {
      if (cvars::occlusion_query_log) {
        XELOGI(
            "ZPD/Async: Draining Vulkan async pipeline creation for strict "
            "retirement");
      }
      pipeline_cache_->AwaitPipelineCompletion();
    }
    EndRenderPass();
    if (!EndSubmission(false)) {
      return false;
    }
  }

  if (wait_for_submission > GetCompletedSubmission()) {
    completion_timeline_.AwaitSubmissionAndUpdateCompleted(wait_for_submission);
  }

  PumpQueryResolves();

  it = logical_zpd_reports_.find(report_handle);
  return it == logical_zpd_reports_.end() ||
         (it->second.pending_segments == 0 && it->second.ended);
}

void VulkanCommandProcessor::InitializeTrace() {
  CommandProcessor::InitializeTrace();

  if (!BeginSubmission(true)) {
    return;
  }
  // TODO(Triang3l): Write the EDRAM.
  bool shared_memory_submitted =
      shared_memory_->InitializeTraceSubmitDownloads();
  if (!shared_memory_submitted) {
    return;
  }
  AwaitAllQueueOperationsCompletion();
  if (shared_memory_submitted) {
    shared_memory_->InitializeTraceCompleteDownloads();
  }
}

void VulkanCommandProcessor::LogRecentSubmissions(const char* context) {
  XELOGE(
      "VulkanCommandProcessor: recent submission history ({}) - oldest first, "
      "current in-progress last:",
      context);
  // Walk the ring starting from the oldest entry.
  for (size_t i = 0; i < kSubmissionHistorySize; ++i) {
    const SubmissionSummary& s =
        submission_history_[(submission_history_next_ + i) %
                            kSubmissionHistorySize];
    if (!s.submission_index) {
      continue;
    }
    XELOGE(
        "  sub {:>5} frame {:>5}: draws={} dispatches={} resolves={} "
        "VS={:016X} PS={:016X} RP=0x{:08X}",
        s.submission_index, s.frame_index, s.draw_count, s.dispatch_count,
        s.resolve_count, s.last_vs_hash, s.last_ps_hash,
        uint32_t(s.last_render_pass_key));
  }
  if (submission_open_) {
    const SubmissionSummary& s = submission_in_progress_;
    XELOGE(
        "  sub {:>5} frame {:>5} (in-progress): draws={} dispatches={} "
        "resolves={} VS={:016X} PS={:016X} RP=0x{:08X}",
        s.submission_index, s.frame_index, s.draw_count, s.dispatch_count,
        s.resolve_count, s.last_vs_hash, s.last_ps_hash,
        uint32_t(s.last_render_pass_key));
  }
}

void VulkanCommandProcessor::CheckSubmissionCompletionAndDeviceLoss(
    uint64_t await_submission) {
  // Only report once, no need to retry a wait that won't succeed anyway.
  if (device_lost_) {
    return;
  }

  if (await_submission >= GetCurrentSubmission()) {
    if (submission_open_) {
      EndSubmission(false);
    }
    // A submission won't be ended if it hasn't been started, or if ending
    // has failed - clamp the index.
    await_submission = GetCurrentSubmission() - 1;
  }

  if (cvars::log_gpu_frame_time_breakdown) {
    const uint64_t completed_before = GetCompletedSubmission();
    const uint64_t t0 = FrameStatsNow();
    completion_timeline_.AwaitSubmissionAndUpdateCompleted(await_submission);
    const uint64_t t1 = FrameStatsNow();
    if (await_submission) {
      vk_frame_sync_stats_.awaits++;
      vk_frame_sync_stats_.await_ns += t1 - t0;
      if (await_submission > completed_before) {
        // The await actually blocked; record how far behind the GPU was.
        vk_frame_sync_stats_.blocking_awaits++;
        vk_frame_sync_stats_.await_delta +=
            GetCurrentSubmission() - await_submission;
      }
    }
    const uint64_t completed = GetCompletedSubmission();
    bool timestamps_invalidated = false;
    while (!vk_submit_times_.empty() &&
           vk_submit_times_.front().submission <= completed) {
      const SubmitTimeRecord& record = vk_submit_times_.front();
      const uint64_t latency = t1 - record.submit_ns;
      vk_frame_sync_stats_.sub_latency_ns += latency;
      vk_frame_sync_stats_.sub_latency_max_ns =
          std::max(vk_frame_sync_stats_.sub_latency_max_ns, latency);
      vk_frame_sync_stats_.sub_completions++;
      if (record.timestamp_slot != UINT32_MAX && frame_timestamp_mapping_) {
        if (!timestamps_invalidated) {
          // The readback memory type is not guaranteed to be host-coherent.
          const ui::vulkan::VulkanDevice* const vulkan_device =
              GetVulkanDevice();
          VkMappedMemoryRange invalidate_range = {
              VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
          invalidate_range.memory = frame_timestamp_buffer_memory_;
          invalidate_range.offset = 0;
          invalidate_range.size = VK_WHOLE_SIZE;
          vulkan_device->functions().vkInvalidateMappedMemoryRanges(
              vulkan_device->device(), 1, &invalidate_range);
          timestamps_invalidated = true;
        }
        const uint64_t ts_top =
            frame_timestamp_mapping_[record.timestamp_slot * 2];
        const uint64_t ts_bottom =
            frame_timestamp_mapping_[record.timestamp_slot * 2 + 1];
        const double period_ns =
            double(GetVulkanDevice()->properties().timestampPeriod);
        if (ts_bottom > ts_top) {
          const uint64_t exec_ns = uint64_t((ts_bottom - ts_top) * period_ns);
          vk_frame_sync_stats_.gpu_exec_ns += exec_ns;
          vk_frame_sync_stats_.gpu_exec_max_ns =
              std::max(vk_frame_sync_stats_.gpu_exec_max_ns, exec_ns);
          vk_frame_sync_stats_.gpu_samples++;
        }
        if (frame_timestamp_prev_end_ && ts_top > frame_timestamp_prev_end_) {
          vk_frame_sync_stats_.gpu_gap_ns +=
              uint64_t((ts_top - frame_timestamp_prev_end_) * period_ns);
        }
        frame_timestamp_prev_end_ = ts_bottom;
        if (record.resolve_pair_count && resolve_timestamp_mapping_) {
          VkMappedMemoryRange resolve_invalidate_range = {
              VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
          resolve_invalidate_range.memory = resolve_timestamp_buffer_memory_;
          resolve_invalidate_range.offset = 0;
          resolve_invalidate_range.size = VK_WHOLE_SIZE;
          GetVulkanDevice()->functions().vkInvalidateMappedMemoryRanges(
              GetVulkanDevice()->device(), 1, &resolve_invalidate_range);
          for (uint32_t i = 0; i < record.resolve_pair_count; ++i) {
            const uint64_t r0 =
                resolve_timestamp_mapping_[(record.resolve_slot_base + i) * 2];
            const uint64_t r1 =
                resolve_timestamp_mapping_[(record.resolve_slot_base + i) * 2 +
                                           1];
            if (r1 > r0) {
              const uint64_t resolve_ns = uint64_t((r1 - r0) * period_ns);
              vk_frame_sync_stats_.resolve_gpu_ns += resolve_ns;
              vk_frame_sync_stats_.resolve_gpu_max_ns = std::max(
                  vk_frame_sync_stats_.resolve_gpu_max_ns, resolve_ns);
              vk_frame_sync_stats_.resolve_gpu_samples++;
            }
          }
        }
        if (record.pass_pair_count && pass_timestamp_mapping_) {
          VkMappedMemoryRange pass_invalidate_range = {
              VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
          pass_invalidate_range.memory = pass_timestamp_buffer_memory_;
          pass_invalidate_range.offset = 0;
          pass_invalidate_range.size = VK_WHOLE_SIZE;
          GetVulkanDevice()->functions().vkInvalidateMappedMemoryRanges(
              GetVulkanDevice()->device(), 1, &pass_invalidate_range);
          for (uint32_t i = 0; i < record.pass_pair_count; ++i) {
            const uint32_t pair = record.pass_slot_base + i;
            const uint64_t p0 = pass_timestamp_mapping_[pair * 2];
            const uint64_t p1 = pass_timestamp_mapping_[pair * 2 + 1];
            if (p1 > p0) {
              auto& bucket = pass_bucket_stats_[pass_ts_keys_[pair]];
              bucket.ns += uint64_t((p1 - p0) * period_ns);
              bucket.passes++;
              bucket.draws += pass_ts_draws_[pair];
            }
          }
        }
      }
      vk_submit_times_.pop_front();
    }
  } else {
    completion_timeline_.AwaitSubmissionAndUpdateCompleted(await_submission);
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();

  if (vulkan_device->IsLost()) {
    XELOGE(
        "VulkanCommandProcessor: device lost observed in submission-check - "
        "awaiting submission {}, current {}, completed {}, in-flight {}, "
        "frame {} (frame_open: {}, submission_open: {})",
        await_submission, GetCurrentSubmission(), GetCompletedSubmission(),
        command_buffers_submitted_.size(), frame_current_, frame_open_,
        submission_open_);
    LogRecentSubmissions("submission-check");
    device_lost_ = true;
    graphics_system_->OnHostGpuLossFromAnyThread(true);
    return;
  }

  const uint64_t completed_submission = GetCompletedSubmission();

  // Reclaim semaphores.
  while (!submissions_in_flight_semaphores_.empty()) {
    const auto& semaphore_submission =
        submissions_in_flight_semaphores_.front();
    if (semaphore_submission.first > completed_submission) {
      break;
    }
    semaphores_free_.push_back(semaphore_submission.second);
    submissions_in_flight_semaphores_.pop_front();
  }

  // Reclaim command pools.
  while (!command_buffers_submitted_.empty()) {
    const auto& command_buffer_pair = command_buffers_submitted_.front();
    if (command_buffer_pair.first > completed_submission) {
      break;
    }
    command_buffers_writable_.push_back(command_buffer_pair.second);
    command_buffers_submitted_.pop_front();
  }

  shared_memory_->CompletedSubmissionUpdated();

  primitive_processor_->CompletedSubmissionUpdated();

  render_target_cache_->CompletedSubmissionUpdated();

  texture_cache_->CompletedSubmissionUpdated(completed_submission);

  // Reclaim descriptor pools that the GPU has finished using.
  if (resolve_downscale_descriptor_pool_chain_) {
    resolve_downscale_descriptor_pool_chain_->Reclaim(completed_submission);
  }

  PumpQueryResolves();

  // Destroy objects scheduled for destruction.
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  while (!destroy_framebuffers_.empty()) {
    const auto& destroy_pair = destroy_framebuffers_.front();
    if (destroy_pair.first > completed_submission) {
      break;
    }
    dfn.vkDestroyFramebuffer(device, destroy_pair.second, nullptr);
    destroy_framebuffers_.pop_front();
  }
  while (!destroy_buffers_.empty()) {
    const auto& destroy_pair = destroy_buffers_.front();
    if (destroy_pair.first > completed_submission) {
      break;
    }
    dfn.vkDestroyBuffer(device, destroy_pair.second, nullptr);
    destroy_buffers_.pop_front();
  }
  while (!destroy_image_views_.empty()) {
    const auto& destroy_pair = destroy_image_views_.front();
    if (destroy_pair.first > completed_submission) {
      break;
    }
    dfn.vkDestroyImageView(device, destroy_pair.second, nullptr);
    destroy_image_views_.pop_front();
  }
  while (!destroy_images_.empty()) {
    const auto& destroy_pair = destroy_images_.front();
    if (destroy_pair.first > completed_submission) {
      break;
    }
    dfn.vkDestroyImage(device, destroy_pair.second, nullptr);
    destroy_images_.pop_front();
  }
  while (!destroy_memory_.empty()) {
    const auto& destroy_pair = destroy_memory_.front();
    if (destroy_pair.first > completed_submission) {
      break;
    }
    dfn.vkFreeMemory(device, destroy_pair.second, nullptr);
    destroy_memory_.pop_front();
  }
}

bool VulkanCommandProcessor::BeginSubmission(bool is_guest_command) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  if (device_lost_) {
    return false;
  }

  bool is_opening_frame = is_guest_command && !frame_open_;
  if (submission_open_ && !is_opening_frame) {
    return true;
  }

  // Check the fence - needed for all kinds of submissions (to reclaim transient
  // resources early) and specifically for frames (not to queue too many), and
  // await the availability of the current frame. Also check whether the device
  // is still available, and whether the await was successful.
  uint64_t await_submission =
      is_opening_frame
          ? closed_frame_submissions_[frame_current_ % kMaxFramesInFlight]
          : 0;
  CheckSubmissionCompletionAndDeviceLoss(await_submission);
  const uint64_t completed_submission = GetCompletedSubmission();
  if (device_lost_ || completed_submission < await_submission) {
    return false;
  }

  if (is_opening_frame) {
    // Update the completed frame index, also obtaining the actual completed
    // frame number (since the CPU may be actually less than 3 frames behind)
    // before reclaiming resources tracked with the frame number.
    frame_completed_ = std::max(frame_current_, uint64_t(kMaxFramesInFlight)) -
                       kMaxFramesInFlight;
    for (uint64_t frame = frame_completed_ + 1; frame < frame_current_;
         ++frame) {
      if (closed_frame_submissions_[frame % kMaxFramesInFlight] >
          completed_submission) {
        break;
      }
      frame_completed_ = frame;
    }
  }

  if (!submission_open_) {
    submission_open_ = true;

    submission_in_progress_ = SubmissionSummary{};
    submission_in_progress_.submission_index = GetCurrentSubmission();
    submission_in_progress_.frame_index = frame_current_;

    // Start a new deferred command buffer - will submit it to the real one in
    // the end of the submission (when async pipeline object creation requests
    // are fulfilled).
    deferred_command_buffer_.Reset();
    deferred_setup_command_buffer_.Reset();
    if (shared_memory_) {
      shared_memory_->OnGpuSubmissionOpened();
    }

    // Reset cached state of the command buffer.
    dynamic_viewport_update_needed_ = true;
    dynamic_scissor_update_needed_ = true;
    dynamic_depth_bias_update_needed_ = true;
    dynamic_blend_constants_update_needed_ = true;
    dynamic_stencil_compare_mask_front_update_needed_ = true;
    dynamic_stencil_compare_mask_back_update_needed_ = true;
    dynamic_stencil_write_mask_front_update_needed_ = true;
    dynamic_stencil_write_mask_back_update_needed_ = true;
    dynamic_stencil_reference_front_update_needed_ = true;
    dynamic_stencil_reference_back_update_needed_ = true;
    dynamic_primitive_topology_update_needed_ = true;
    dynamic_primitive_restart_enable_update_needed_ = true;
    dynamic_cull_mode_update_needed_ = true;
    dynamic_front_face_update_needed_ = true;
    dynamic_depth_test_enable_update_needed_ = true;
    dynamic_depth_write_enable_update_needed_ = true;
    dynamic_depth_compare_op_update_needed_ = true;
    dynamic_stencil_test_enable_update_needed_ = true;
    dynamic_stencil_op_front_update_needed_ = true;
    dynamic_stencil_op_back_update_needed_ = true;
    dynamic_depth_clamp_enable_update_needed_ = true;
    dynamic_polygon_mode_update_needed_ = true;
    dynamic_color_blend_enable_update_needed_ = true;
    dynamic_color_blend_equation_update_needed_ = true;
    dynamic_color_write_mask_update_needed_ = true;
    current_render_pass_ = VK_NULL_HANDLE;
    current_framebuffer_ = nullptr;
    in_render_pass_ = false;
    // Fresh deferred command buffer: abandon any open pass-timestamp pair
    // without emitting an end (it would land in the new, unrelated buffer).
    pass_ts_open_pair_ = UINT32_MAX;
    current_guest_graphics_pipeline_ = nullptr;
    current_external_graphics_pipeline_ = VK_NULL_HANDLE;
    current_external_compute_pipeline_ = VK_NULL_HANDLE;
    current_guest_graphics_pipeline_layout_ = nullptr;
    current_graphics_descriptor_sets_bound_up_to_date_ = 0;

    primitive_processor_->BeginSubmission();

    texture_cache_->BeginSubmission(GetCurrentSubmission());
  }

  if (is_opening_frame) {
    frame_open_ = true;

    // Conservatively re-derive all cached sampler parameters in the new
    // frame (cheap - one derivation pass per stage; the handles still go
    // through their per-submission UseSampler revalidation independently).
    current_samplers_fetch_up_to_date_vertex_ = 0;
    current_samplers_fetch_up_to_date_pixel_ = 0;

    // Log guest ZPD report stats every 100 frames.
    if (GetZPDMode() != ZPDMode::kFake && cvars::occlusion_query_log &&
        zpd_host_query_pool_ && zpd_host_query_pool_->capacity() &&
        frame_current_ - zpd_stats_.last_log_frame >= 100) {
      XELOGI(
          "Occlusion Query Stats (last 100 frames): "
          "LogicalBegun={}, LogicalEnded={}, SegBegun={}, SegEnded={}, "
          "PoolExhausted={}, Failed={}, Wraps={}, SameSlotReuse={}",
          zpd_stats_.logical_begun, zpd_stats_.logical_ended,
          zpd_stats_.segments_begun, zpd_stats_.segments_ended,
          zpd_stats_.pool_exhausted, zpd_stats_.failed,
          zpd_stats_.counter_wraps, zpd_stats_.same_slot_reuse);

      zpd_stats_.Reset(frame_current_);
    }

    // Reset bindings that depend on transient data.
    std::memset(current_float_constant_map_vertex_, 0,
                sizeof(current_float_constant_map_vertex_));
    std::memset(current_float_constant_map_pixel_, 0,
                sizeof(current_float_constant_map_pixel_));
    std::memset(current_graphics_descriptor_sets_, 0,
                sizeof(current_graphics_descriptor_sets_));
    current_constant_buffers_up_to_date_ = 0;
    current_graphics_descriptor_sets_
        [SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram] =
            shared_memory_and_edram_descriptor_set_;
    current_graphics_descriptor_set_values_up_to_date_ =
        UINT32_C(1)
        << SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram;
    // MANDATORY: the transient descriptor pool resets at the frame boundary, so
    // every cached texture set written in the previous frame is now invalid.
    // Invalidate in lockstep with the values bitmask reset and the
    // current_graphics_descriptor_sets_ memset above, otherwise a first-draw
    // whose content hash equals the previous frame's could skip the rewrite and
    // sample recycled transient-pool memory (the #1 corruption path).
    current_texture_descriptor_set_hash_valid_vertex_ = false;
    current_texture_descriptor_set_hash_valid_pixel_ = false;
    // Same MANDATORY invalidation for the constants value cache: the cached
    // transient set and the page buffers it references are reclaimable at the
    // frame boundary (the current_graphics_descriptor_sets_ memset above already
    // dropped the cached handle), so any stale per-binding VkBuffer must not let
    // a first draw skip the rewrite.
    constants_descriptor_set_valid_ = false;

    // Reclaim pool pages - no need to do this every small submission since some
    // may be reused.
    // FIXME(Triang3l): This will result in a memory leak if the guest is not
    // presenting.
    uniform_buffer_pool_->Reclaim(frame_completed_);
    while (!single_transient_descriptors_used_.empty()) {
      const UsedSingleTransientDescriptor& used_transient_descriptor =
          single_transient_descriptors_used_.front();
      if (used_transient_descriptor.frame > frame_completed_) {
        break;
      }
      single_transient_descriptors_free_[size_t(
                                             used_transient_descriptor.layout)]
          .push_back(used_transient_descriptor.set);
      single_transient_descriptors_used_.pop_front();
    }
    while (!constants_transient_descriptors_used_.empty()) {
      const std::pair<uint64_t, VkDescriptorSet>& used_transient_descriptor =
          constants_transient_descriptors_used_.front();
      if (used_transient_descriptor.first > frame_completed_) {
        break;
      }
      constants_transient_descriptors_free_.push_back(
          used_transient_descriptor.second);
      constants_transient_descriptors_used_.pop_front();
    }
    while (!texture_transient_descriptor_sets_used_.empty()) {
      const UsedTextureTransientDescriptorSet& used_transient_descriptor_set =
          texture_transient_descriptor_sets_used_.front();
      if (used_transient_descriptor_set.frame > frame_completed_) {
        break;
      }
      auto it = texture_transient_descriptor_sets_free_.find(
          used_transient_descriptor_set.layout);
      if (it == texture_transient_descriptor_sets_free_.end()) {
        it =
            texture_transient_descriptor_sets_free_
                .emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(used_transient_descriptor_set.layout),
                    std::forward_as_tuple())
                .first;
      }
      it->second.push_back(used_transient_descriptor_set.set);
      texture_transient_descriptor_sets_used_.pop_front();
    }

    primitive_processor_->BeginFrame();

    texture_cache_->BeginFrame();
  }

  return true;
}

bool VulkanCommandProcessor::CanEndSubmissionImmediately() const {
  return !submission_open_ || !pipeline_cache_ ||
         !pipeline_cache_->IsCreatingPipelines();
}

bool VulkanCommandProcessor::EndSubmission(bool is_swap) {
  ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Make sure everything needed for submitting exist.
  if (submission_open_) {
    if (!sparse_memory_binds_.empty() && semaphores_free_.empty()) {
      VkSemaphoreCreateInfo semaphore_create_info;
      semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      semaphore_create_info.pNext = nullptr;
      semaphore_create_info.flags = 0;
      VkSemaphore semaphore;
      if (dfn.vkCreateSemaphore(device, &semaphore_create_info, nullptr,
                                &semaphore) != VK_SUCCESS) {
        XELOGE("Failed to create a Vulkan semaphore");
        return false;
      }
      semaphores_free_.push_back(semaphore);
    }
    if (command_buffers_writable_.empty()) {
      CommandBuffer command_buffer;
      VkCommandPoolCreateInfo command_pool_create_info;
      command_pool_create_info.sType =
          VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      command_pool_create_info.pNext = nullptr;
      command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
      command_pool_create_info.queueFamilyIndex =
          vulkan_device->queue_family_graphics_compute();
      if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr,
                                  &command_buffer.pool) != VK_SUCCESS) {
        XELOGE("Failed to create a Vulkan command pool");
        return false;
      }
      VkCommandBufferAllocateInfo command_buffer_allocate_info;
      command_buffer_allocate_info.sType =
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      command_buffer_allocate_info.pNext = nullptr;
      command_buffer_allocate_info.commandPool = command_buffer.pool;
      command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_buffer_allocate_info.commandBufferCount = 1;
      if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info,
                                       &command_buffer.buffer) != VK_SUCCESS) {
        XELOGE("Failed to allocate a Vulkan command buffer");
        dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
        return false;
      }
      command_buffers_writable_.push_back(command_buffer);
    }
  }

  bool is_closing_frame = is_swap && frame_open_;

  if (is_closing_frame) {
    primitive_processor_->EndFrame();
  }

  if (submission_open_) {
    assert_false(scratch_buffer_used_);

    EndRenderPass();

    render_target_cache_->EndSubmission();

    primitive_processor_->EndSubmission();

    shared_memory_->EndSubmission();

    uniform_buffer_pool_->FlushWrites();

    // Submit sparse binds earlier, before executing the deferred command
    // buffer, to reduce latency.
    if (!sparse_memory_binds_.empty()) {
      sparse_buffer_bind_infos_temp_.clear();
      sparse_buffer_bind_infos_temp_.reserve(sparse_buffer_binds_.size());
      for (const SparseBufferBind& sparse_buffer_bind : sparse_buffer_binds_) {
        VkSparseBufferMemoryBindInfo& sparse_buffer_bind_info =
            sparse_buffer_bind_infos_temp_.emplace_back();
        sparse_buffer_bind_info.buffer = sparse_buffer_bind.buffer;
        sparse_buffer_bind_info.bindCount = sparse_buffer_bind.bind_count;
        sparse_buffer_bind_info.pBinds =
            sparse_memory_binds_.data() + sparse_buffer_bind.bind_offset;
      }
      assert_false(semaphores_free_.empty());
      VkSemaphore bind_sparse_semaphore = semaphores_free_.back();
      VkBindSparseInfo bind_sparse_info;
      bind_sparse_info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
      bind_sparse_info.pNext = nullptr;
      bind_sparse_info.waitSemaphoreCount = 0;
      bind_sparse_info.pWaitSemaphores = nullptr;
      bind_sparse_info.bufferBindCount =
          uint32_t(sparse_buffer_bind_infos_temp_.size());
      bind_sparse_info.pBufferBinds =
          !sparse_buffer_bind_infos_temp_.empty()
              ? sparse_buffer_bind_infos_temp_.data()
              : nullptr;
      bind_sparse_info.imageOpaqueBindCount = 0;
      bind_sparse_info.pImageOpaqueBinds = nullptr;
      bind_sparse_info.imageBindCount = 0;
      bind_sparse_info.pImageBinds = 0;
      bind_sparse_info.signalSemaphoreCount = 1;
      bind_sparse_info.pSignalSemaphores = &bind_sparse_semaphore;
      VkResult bind_sparse_result;
      {
        ui::vulkan::VulkanDevice::Queue::Acquisition queue_acquisition =
            vulkan_device->AcquireQueue(
                vulkan_device->queue_family_sparse_binding(), 0);
        bind_sparse_result = dfn.vkQueueBindSparse(
            queue_acquisition.queue(), 1, &bind_sparse_info, VK_NULL_HANDLE);
      }
      if (bind_sparse_result != VK_SUCCESS) {
        XELOGE("Failed to submit Vulkan sparse binds");
        return false;
      }
      current_submission_wait_semaphores_.push_back(bind_sparse_semaphore);
      semaphores_free_.pop_back();
      current_submission_wait_stage_masks_.push_back(
          sparse_bind_wait_stage_mask_);
      sparse_bind_wait_stage_mask_ = 0;
      sparse_buffer_binds_.clear();
      sparse_memory_binds_.clear();
    }
    // Can't cross command buffer boundaries. Close the active segment first.
    CloseQuerySegment();

    SubmitBarriers(true);

    assert_false(command_buffers_writable_.empty());
    CommandBuffer command_buffer = command_buffers_writable_.back();
    if (dfn.vkResetCommandPool(device, command_buffer.pool, 0) != VK_SUCCESS) {
      XELOGE("Failed to reset a Vulkan command pool");
      return false;
    }
    VkCommandBufferBeginInfo command_buffer_begin_info;
    command_buffer_begin_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.pNext = nullptr;
    command_buffer_begin_info.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    command_buffer_begin_info.pInheritanceInfo = nullptr;
    if (dfn.vkBeginCommandBuffer(command_buffer.buffer,
                                 &command_buffer_begin_info) != VK_SUCCESS) {
      XELOGE("Failed to begin a Vulkan command buffer");
      return false;
    }
    uint32_t fs_timestamp_slot = UINT32_MAX;
    if (frame_timestamp_mapping_) {
      fs_timestamp_slot =
          uint32_t(GetCurrentSubmission() % kFrameTimestampSlots);
      dfn.vkCmdResetQueryPool(command_buffer.buffer, frame_timestamp_pool_,
                              fs_timestamp_slot * 2, 2);
      dfn.vkCmdWriteTimestamp(command_buffer.buffer,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              frame_timestamp_pool_, fs_timestamp_slot * 2);
      if (resolve_timestamp_mapping_) {
        // Reset this submission's resolve-pair range before the deferred
        // buffer (which contains the writes) replays.
        dfn.vkCmdResetQueryPool(
            command_buffer.buffer, resolve_timestamp_pool_,
            uint32_t(GetCurrentSubmission() % kResolveTimestampRingSubmissions) *
                kResolveTimestampPairsPerSubmission * 2,
            kResolveTimestampPairsPerSubmission * 2);
      }
      if (pass_timestamp_mapping_) {
        dfn.vkCmdResetQueryPool(
            command_buffer.buffer, pass_timestamp_pool_,
            uint32_t(GetCurrentSubmission() % kPassTimestampRingSubmissions) *
                kPassTimestampPairsPerSubmission * 2,
            kPassTimestampPairsPerSubmission * 2);
      }
    }
    // Submission boundary for asynchronously created pipelines: by default
    // this waits until every pipeline referenced by the recorded deferred
    // binds has finished creation, so no draw is lost. With
    // vulkan_async_skip_draws (or during the startup storage preload) it
    // doesn't block, and deferred binds that still resolve to VK_NULL_HANDLE
    // at replay drop their draws. Also persists the driver pipeline cache
    // (throttled) and flushes the shader/pipeline storage files.
    pipeline_cache_->EndSubmission();
    if (!deferred_setup_command_buffer_.empty()) {
      deferred_setup_command_buffer_.Execute(command_buffer.buffer);
      VkMemoryBarrier setup_barrier;
      setup_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      setup_barrier.pNext = nullptr;
      setup_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      setup_barrier.dstAccessMask =
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      dfn.vkCmdPipelineBarrier(
          command_buffer.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &setup_barrier, 0, nullptr,
          0, nullptr);
    }
    deferred_command_buffer_.Execute(command_buffer.buffer);

    // Record ZPD resolves before submitting.
    if (zpd_host_query_pool_) {
      zpd_host_query_pool_->RecordResolveBatch(command_buffer.buffer);
    }

    if (fs_timestamp_slot != UINT32_MAX) {
      dfn.vkCmdWriteTimestamp(
          command_buffer.buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
          frame_timestamp_pool_, fs_timestamp_slot * 2 + 1);
      dfn.vkCmdCopyQueryPoolResults(
          command_buffer.buffer, frame_timestamp_pool_, fs_timestamp_slot * 2,
          2, frame_timestamp_buffer_,
          fs_timestamp_slot * 2 * sizeof(uint64_t), sizeof(uint64_t),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      if (resolve_timestamp_mapping_ &&
          resolve_ts_submission_ == GetCurrentSubmission() &&
          resolve_ts_count_) {
        // Copy only the pairs actually written this submission - reset-but-
        // unwritten queries under WAIT_BIT would hang the GPU.
        const uint32_t resolve_ts_base =
            uint32_t(GetCurrentSubmission() % kResolveTimestampRingSubmissions) *
            kResolveTimestampPairsPerSubmission;
        dfn.vkCmdCopyQueryPoolResults(
            command_buffer.buffer, resolve_timestamp_pool_, resolve_ts_base * 2,
            resolve_ts_count_ * 2, resolve_timestamp_buffer_,
            resolve_ts_base * 2 * sizeof(uint64_t), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      }
      if (pass_timestamp_mapping_ &&
          pass_ts_submission_ == GetCurrentSubmission() && pass_ts_count_) {
        // Only fully-closed pairs are counted in pass_ts_count_; a pass still
        // open across the split keeps its begin query but is not copied here.
        const uint32_t pass_ts_base =
            uint32_t(GetCurrentSubmission() % kPassTimestampRingSubmissions) *
            kPassTimestampPairsPerSubmission;
        dfn.vkCmdCopyQueryPoolResults(
            command_buffer.buffer, pass_timestamp_pool_, pass_ts_base * 2,
            pass_ts_count_ * 2, pass_timestamp_buffer_,
            pass_ts_base * 2 * sizeof(uint64_t), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      }
      VkMemoryBarrier fs_timestamp_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      fs_timestamp_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      fs_timestamp_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      dfn.vkCmdPipelineBarrier(command_buffer.buffer,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                               &fs_timestamp_barrier, 0, nullptr, 0, nullptr);
    }

    if (dfn.vkEndCommandBuffer(command_buffer.buffer) != VK_SUCCESS) {
      XELOGE("Failed to end a Vulkan command buffer");
      return false;
    }

    const uint64_t submission_index = GetCurrentSubmission();

    // If resolve readback copies are queued for the dedicated transfer queue,
    // have this graphics submission signal a semaphore the transfer copies wait
    // on, so they see the shared-memory writes produced here.
    VkSemaphore transfer_signal_semaphore = VK_NULL_HANDLE;
    if (vulkan_device->queue_family_transfer() != UINT32_MAX &&
        !pending_readback_copies_.empty()) {
      ReclaimCompletedTransferResources();
      if (!transfer_wait_semaphores_free_.empty()) {
        transfer_signal_semaphore = transfer_wait_semaphores_free_.back();
        transfer_wait_semaphores_free_.pop_back();
      } else {
        VkSemaphoreCreateInfo semaphore_create_info = {
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (dfn.vkCreateSemaphore(device, &semaphore_create_info, nullptr,
                                  &transfer_signal_semaphore) != VK_SUCCESS) {
          XELOGE("Failed to create a graphics->transfer readback semaphore");
          transfer_signal_semaphore = VK_NULL_HANDLE;
          // Without the semaphore the copies can't be flushed, so drop them
          // rather than leak them into the next frame.
          pending_readback_copies_.clear();
        }
      }
    }

    VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    if (!current_submission_wait_semaphores_.empty()) {
      submit_info.waitSemaphoreCount =
          uint32_t(current_submission_wait_semaphores_.size());
      submit_info.pWaitSemaphores = current_submission_wait_semaphores_.data();
      submit_info.pWaitDstStageMask =
          current_submission_wait_stage_masks_.data();
    }
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer.buffer;
    if (transfer_signal_semaphore != VK_NULL_HANDLE) {
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores = &transfer_signal_semaphore;
    }
    const VkResult submit_result = completion_timeline_.AcquireFenceAndSubmit(
        vulkan_device->queue_family_graphics_compute(), 0, 1, &submit_info);
    if (submit_result != VK_SUCCESS) {
      XELOGE(
          "VulkanCommandProcessor: Failed to submit a Vulkan command buffer - "
          "VkResult: {} (0x{:08X}), submission: {} (completed: {}, in-flight: "
          "{}), frame: {} (frame_open: {}, is_closing_frame: {}), "
          "wait_semaphores: {}, draw_resolution_scale: {}x{}",
          static_cast<int32_t>(submit_result),
          static_cast<uint32_t>(submit_result), GetCurrentSubmission(),
          GetCompletedSubmission(), command_buffers_submitted_.size(),
          frame_current_, frame_open_, is_closing_frame,
          submit_info.waitSemaphoreCount,
          render_target_cache_ ? render_target_cache_->draw_resolution_scale_x()
                               : 0,
          render_target_cache_ ? render_target_cache_->draw_resolution_scale_y()
                               : 0);
      if (submit_result == VK_ERROR_DEVICE_LOST) {
        XELOGE(
            "VK_ERROR_DEVICE_LOST - GPU crashed or hung. This may be caused by "
            "an invalid shader, out-of-bounds memory access, or driver bug.");
      }
      LogRecentSubmissions("submit-failure");
      if (vulkan_device->IsLost() && !device_lost_) {
        device_lost_ = true;
        graphics_system_->OnHostGpuLossFromAnyThread(true);
      }
      // The graphics work that the queued readback copies depend on wasn't
      // submitted, so drop them. The unsignaled semaphore can be reused.
      if (transfer_signal_semaphore != VK_NULL_HANDLE) {
        transfer_wait_semaphores_free_.push_back(transfer_signal_semaphore);
      }
      pending_readback_copies_.clear();
      return false;
    }
    current_submission_wait_stage_masks_.clear();
    for (VkSemaphore semaphore : current_submission_wait_semaphores_) {
      submissions_in_flight_semaphores_.emplace_back(submission_index,
                                                     semaphore);
    }
    current_submission_wait_semaphores_.clear();
    command_buffers_submitted_.emplace_back(submission_index, command_buffer);
    command_buffers_writable_.pop_back();

    // The graphics work is submitted and will signal the semaphore - run the
    // queued resolve readback copies on the dedicated transfer queue.
    if (transfer_signal_semaphore != VK_NULL_HANDLE) {
      if (!SubmitReadbackCopiesToTransferQueue(transfer_signal_semaphore)) {
        // The semaphore was already signaled by the graphics submit and can't
        // be safely reused; drop it and the copies (rare, device-loss path).
        pending_readback_copies_.clear();
      }
    }

    submission_history_[submission_history_next_] = submission_in_progress_;
    submission_history_next_ =
        (submission_history_next_ + 1) % kSubmissionHistorySize;

    // Mark descriptor pool chains with submission index for reclaim tracking.
    if (resolve_downscale_descriptor_pool_chain_) {
      resolve_downscale_descriptor_pool_chain_->EndSubmission(submission_index);
    }

    submission_open_ = false;
    draws_since_submission_ = 0;
    vk_frame_sync_stats_.submissions++;
    if (cvars::log_gpu_frame_time_breakdown) {
      uint32_t resolve_base = 0;
      uint32_t resolve_pairs = 0;
      if (resolve_timestamp_mapping_ &&
          resolve_ts_submission_ == submission_index) {
        resolve_pairs = resolve_ts_count_;
        resolve_base =
            uint32_t(submission_index % kResolveTimestampRingSubmissions) *
            kResolveTimestampPairsPerSubmission;
      }
      uint32_t pass_base = 0;
      uint32_t pass_pairs = 0;
      if (pass_timestamp_mapping_ && pass_ts_submission_ == submission_index) {
        pass_pairs = pass_ts_count_;
        pass_base =
            uint32_t(submission_index % kPassTimestampRingSubmissions) *
            kPassTimestampPairsPerSubmission;
      }
      vk_submit_times_.push_back({submission_index, FrameStatsNow(),
                                  fs_timestamp_slot, resolve_base,
                                  resolve_pairs, pass_base, pass_pairs});
      resolve_ts_count_ = 0;
      pass_ts_count_ = 0;
    }

    // Process any ZPD resolves that completed with this submission.
    // Block if strict mode has a pending result waiting on the guest sentinel.
    PumpQueryResolves();
    PumpPendingRetire();
  }

  if (is_closing_frame) {
    if (cvars::clear_memory_page_state) {
      shared_memory_->SetSystemPageBlocksValidWithGpuDataWritten();
    }

    frame_open_ = false;
    // Submission already closed now, so minus 1.
    closed_frame_submissions_[(frame_current_++) % kMaxFramesInFlight] =
        GetCurrentSubmission() - 1;

    // Evict old readback buffers once per frame
    EvictOldReadbackBuffers(readback_buffers_);
    EvictOldReadbackBuffers(memexport_readback_buffers_);

    if (cache_clear_requested_ && AwaitAllQueueOperationsCompletion()) {
      cache_clear_requested_ = false;

      DestroyScratchBuffer();

      assert_true(command_buffers_submitted_.empty());
      for (const CommandBuffer& command_buffer : command_buffers_writable_) {
        dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
      }
      command_buffers_writable_.clear();

      ClearTransientDescriptorPools();

      uniform_buffer_pool_->ClearCache();

      texture_cache_->ClearCache();

      render_target_cache_->ClearCache();

      // Not clearing the pipeline layouts and the descriptor set layouts as
      // they're referenced by pipelines, which are not destroyed.

      primitive_processor_->ClearCache();

      shared_memory_->ClearCache();
    }
  }

  return true;
}

void VulkanCommandProcessor::ClearTransientDescriptorPools() {
  texture_transient_descriptor_sets_free_.clear();
  texture_transient_descriptor_sets_used_.clear();
  transient_descriptor_allocator_textures_.Reset();
  // MANDATORY: all transient texture sets are destroyed here, so the value
  // cache for them is no longer valid.
  current_texture_descriptor_set_hash_valid_vertex_ = false;
  current_texture_descriptor_set_hash_valid_pixel_ = false;

  constants_transient_descriptors_free_.clear();
  constants_transient_descriptors_used_.clear();
  // MANDATORY: the uniform-buffer transient allocator is Reset() below,
  // destroying every constants set, so the value cache (the cached set handle in
  // current_graphics_descriptor_sets_[kDescriptorSetConstants] and the
  // per-binding VkBuffer handles) is no longer valid.
  constants_descriptor_set_valid_ = false;
  for (std::vector<VkDescriptorSet>& transient_descriptors_free :
       single_transient_descriptors_free_) {
    transient_descriptors_free.clear();
  }
  single_transient_descriptors_used_.clear();
  transient_descriptor_allocator_storage_buffer_.Reset();
  transient_descriptor_allocator_uniform_buffer_.Reset();
}

void VulkanCommandProcessor::SplitPendingBarrier() {
  size_t pending_buffer_memory_barrier_count =
      pending_barriers_buffer_memory_barriers_.size();
  size_t pending_image_memory_barrier_count =
      pending_barriers_image_memory_barriers_.size();
  if (!current_pending_barrier_.src_stage_mask &&
      !current_pending_barrier_.dst_stage_mask &&
      current_pending_barrier_.buffer_memory_barriers_offset >=
          pending_buffer_memory_barrier_count &&
      current_pending_barrier_.image_memory_barriers_offset >=
          pending_image_memory_barrier_count) {
    return;
  }
  pending_barriers_.emplace_back(current_pending_barrier_);
  current_pending_barrier_.src_stage_mask = 0;
  current_pending_barrier_.dst_stage_mask = 0;
  current_pending_barrier_.buffer_memory_barriers_offset =
      pending_buffer_memory_barrier_count;
  current_pending_barrier_.image_memory_barriers_offset =
      pending_image_memory_barrier_count;
}

void VulkanCommandProcessor::DestroyScratchBuffer() {
  assert_false(scratch_buffer_used_);

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  scratch_buffer_last_usage_submission_ = 0;
  scratch_buffer_last_access_mask_ = 0;
  scratch_buffer_last_stage_mask_ = 0;
  scratch_buffer_size_ = 0;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         scratch_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         scratch_buffer_memory_);
}

void VulkanCommandProcessor::UpdateDynamicState(
    const draw_util::ViewportInfo& viewport_info, bool primitive_polygonal,
    reg::RB_DEPTHCONTROL normalized_depth_control,
    uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
    bool depth_bias_in_pixel_shader,
    const VulkanPipelineCache::DynamicState& pipeline_dynamic_state) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;

  // Window parameters.
  // http://ftp.tku.edu.tw/NetBSD/NetBSD-current/xsrc/external/mit/xf86-video-ati/dist/src/r600_reg_auto_r6xx.h
  // See r200UpdateWindow:
  // https://github.com/freedreno/mesa/blob/master/src/mesa/drivers/dri/r200/r200_state.c
  auto pa_sc_window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();

  // Viewport.
  VkViewport viewport;
  if (viewport_info.xy_extent[0] && viewport_info.xy_extent[1]) {
    viewport.x = float(viewport_info.xy_offset[0]);
    viewport.y = float(viewport_info.xy_offset[1]);
    viewport.width = float(viewport_info.xy_extent[0]);
    viewport.height = float(viewport_info.xy_extent[1]);
  } else {
    // Vulkan viewport width must be greater than 0.0f, but the Xenia  viewport
    // may be empty for various reasons - set the viewport to outside the
    // framebuffer.
    viewport.x = -1.0f;
    viewport.y = -1.0f;
    viewport.width = 1.0f;
    viewport.height = 1.0f;
  }
  viewport.minDepth = viewport_info.z_min;
  viewport.maxDepth = viewport_info.z_max;
  SetViewport(viewport);

  // Scissor.
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  // Scale the scissor to match the render target resolution scale
  scissor.offset[0] *= draw_resolution_scale_x;
  scissor.offset[1] *= draw_resolution_scale_y;
  scissor.extent[0] *= draw_resolution_scale_x;
  scissor.extent[1] *= draw_resolution_scale_y;
  VkRect2D scissor_rect;
  scissor_rect.offset.x = int32_t(scissor.offset[0]);
  scissor_rect.offset.y = int32_t(scissor.offset[1]);
  scissor_rect.extent.width = scissor.extent[0];
  scissor_rect.extent.height = scissor.extent[1];
  SetScissor(scissor_rect);

  if (render_target_cache_->GetPath() ==
      RenderTargetCache::Path::kHostRenderTargets) {
    // Depth bias.
    float depth_bias_constant_factor, depth_bias_slope_factor;
    if (depth_bias_in_pixel_shader) {
      depth_bias_constant_factor = 0.0f;
      depth_bias_slope_factor = 0.0f;
    } else {
      draw_util::GetPreferredFacePolygonOffset(regs, primitive_polygonal,
                                               depth_bias_slope_factor,
                                               depth_bias_constant_factor);
      depth_bias_constant_factor *=
          regs.Get<reg::RB_DEPTH_INFO>().depth_format ==
                  xenos::DepthRenderTargetFormat::kD24S8
              ? draw_util::kD3D10PolygonOffsetFactorUnorm24
              : draw_util::kD3D10PolygonOffsetFactorFloat24;
      // With non-square resolution scaling, make sure the worst-case impact is
      // reverted (slope only along the scaled axis), thus max. More bias is
      // better than less bias, because less bias means Z fighting with the
      // background is more likely.
      depth_bias_slope_factor *=
          xenos::kPolygonOffsetScaleSubpixelUnit *
          float(std::max(render_target_cache_->draw_resolution_scale_x(),
                         render_target_cache_->draw_resolution_scale_y()));
    }
    // std::memcmp instead of != so in case of NaN, every draw won't be
    // invalidating it.
    dynamic_depth_bias_update_needed_ |=
        std::memcmp(&dynamic_depth_bias_constant_factor_,
                    &depth_bias_constant_factor, sizeof(float)) != 0;
    dynamic_depth_bias_update_needed_ |=
        std::memcmp(&dynamic_depth_bias_slope_factor_, &depth_bias_slope_factor,
                    sizeof(float)) != 0;
    if (dynamic_depth_bias_update_needed_) {
      dynamic_depth_bias_constant_factor_ = depth_bias_constant_factor;
      dynamic_depth_bias_slope_factor_ = depth_bias_slope_factor;
      deferred_command_buffer_.CmdVkSetDepthBias(
          dynamic_depth_bias_constant_factor_, 0.0f,
          dynamic_depth_bias_slope_factor_);
      dynamic_depth_bias_update_needed_ = false;
    }

    // Blend constants.
    float blend_constants[] = {
        regs.Get<float>(XE_GPU_REG_RB_BLEND_RED),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA),
    };
    dynamic_blend_constants_update_needed_ |=
        std::memcmp(dynamic_blend_constants_, blend_constants,
                    sizeof(float) * 4) != 0;
    if (dynamic_blend_constants_update_needed_) {
      std::memcpy(dynamic_blend_constants_, blend_constants, sizeof(float) * 4);
      deferred_command_buffer_.CmdVkSetBlendConstants(dynamic_blend_constants_);
      dynamic_blend_constants_update_needed_ = false;
    }

    // Stencil masks and references.
    // Due to pretty complex conditions involving registers not directly related
    // to stencil (primitive type, culling), changing the values only when
    // stencil is actually needed. However, due to the way dynamic state needs
    // to be set in Vulkan, which doesn't take into account whether the state
    // actually has effect on drawing, and because the masks and the references
    // are always dynamic in Xenia guest pipelines, they must be set in the
    // command buffer before any draw.
    if (normalized_depth_control.stencil_enable) {
      Register stencil_ref_mask_front_reg, stencil_ref_mask_back_reg;
      if (primitive_polygonal && normalized_depth_control.backface_enable) {
        if (GetVulkanDevice()->properties().separateStencilMaskRef) {
          stencil_ref_mask_front_reg = XE_GPU_REG_RB_STENCILREFMASK;
          stencil_ref_mask_back_reg = XE_GPU_REG_RB_STENCILREFMASK_BF;
        } else {
          // Choose the back face values only if drawing only back faces.
          auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
          stencil_ref_mask_front_reg =
              (pa_su_sc_mode_cntl.cull_front && !pa_su_sc_mode_cntl.cull_back)
                  ? XE_GPU_REG_RB_STENCILREFMASK_BF
                  : XE_GPU_REG_RB_STENCILREFMASK;
          stencil_ref_mask_back_reg = stencil_ref_mask_front_reg;
        }
      } else {
        stencil_ref_mask_front_reg = XE_GPU_REG_RB_STENCILREFMASK;
        stencil_ref_mask_back_reg = XE_GPU_REG_RB_STENCILREFMASK;
      }
      auto stencil_ref_mask_front =
          regs.Get<reg::RB_STENCILREFMASK>(stencil_ref_mask_front_reg);
      auto stencil_ref_mask_back =
          regs.Get<reg::RB_STENCILREFMASK>(stencil_ref_mask_back_reg);
      // Compare mask.
      dynamic_stencil_compare_mask_front_update_needed_ |=
          dynamic_stencil_compare_mask_front_ !=
          stencil_ref_mask_front.stencilmask;
      dynamic_stencil_compare_mask_front_ = stencil_ref_mask_front.stencilmask;
      dynamic_stencil_compare_mask_back_update_needed_ |=
          dynamic_stencil_compare_mask_back_ !=
          stencil_ref_mask_back.stencilmask;
      dynamic_stencil_compare_mask_back_ = stencil_ref_mask_back.stencilmask;
      // Write mask.
      dynamic_stencil_write_mask_front_update_needed_ |=
          dynamic_stencil_write_mask_front_ !=
          stencil_ref_mask_front.stencilwritemask;
      dynamic_stencil_write_mask_front_ =
          stencil_ref_mask_front.stencilwritemask;
      dynamic_stencil_write_mask_back_update_needed_ |=
          dynamic_stencil_write_mask_back_ !=
          stencil_ref_mask_back.stencilwritemask;
      dynamic_stencil_write_mask_back_ = stencil_ref_mask_back.stencilwritemask;
      // Reference.
      dynamic_stencil_reference_front_update_needed_ |=
          dynamic_stencil_reference_front_ != stencil_ref_mask_front.stencilref;
      dynamic_stencil_reference_front_ = stencil_ref_mask_front.stencilref;
      dynamic_stencil_reference_back_update_needed_ |=
          dynamic_stencil_reference_back_ != stencil_ref_mask_back.stencilref;
      dynamic_stencil_reference_back_ = stencil_ref_mask_back.stencilref;
    }
    // Using VK_STENCIL_FACE_FRONT_AND_BACK for higher safety when running on
    // the Vulkan portability subset without separateStencilMaskRef.
    if (dynamic_stencil_compare_mask_front_update_needed_ ||
        dynamic_stencil_compare_mask_back_update_needed_) {
      if (dynamic_stencil_compare_mask_front_ ==
          dynamic_stencil_compare_mask_back_) {
        deferred_command_buffer_.CmdVkSetStencilCompareMask(
            VK_STENCIL_FACE_FRONT_AND_BACK,
            dynamic_stencil_compare_mask_front_);
      } else {
        if (dynamic_stencil_compare_mask_front_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilCompareMask(
              VK_STENCIL_FACE_FRONT_BIT, dynamic_stencil_compare_mask_front_);
        }
        if (dynamic_stencil_compare_mask_back_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilCompareMask(
              VK_STENCIL_FACE_BACK_BIT, dynamic_stencil_compare_mask_back_);
        }
      }
      dynamic_stencil_compare_mask_front_update_needed_ = false;
      dynamic_stencil_compare_mask_back_update_needed_ = false;
    }
    if (dynamic_stencil_write_mask_front_update_needed_ ||
        dynamic_stencil_write_mask_back_update_needed_) {
      if (dynamic_stencil_write_mask_front_ ==
          dynamic_stencil_write_mask_back_) {
        deferred_command_buffer_.CmdVkSetStencilWriteMask(
            VK_STENCIL_FACE_FRONT_AND_BACK, dynamic_stencil_write_mask_front_);
      } else {
        if (dynamic_stencil_write_mask_front_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilWriteMask(
              VK_STENCIL_FACE_FRONT_BIT, dynamic_stencil_write_mask_front_);
        }
        if (dynamic_stencil_write_mask_back_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilWriteMask(
              VK_STENCIL_FACE_BACK_BIT, dynamic_stencil_write_mask_back_);
        }
      }
      dynamic_stencil_write_mask_front_update_needed_ = false;
      dynamic_stencil_write_mask_back_update_needed_ = false;
    }
    if (dynamic_stencil_reference_front_update_needed_ ||
        dynamic_stencil_reference_back_update_needed_) {
      if (dynamic_stencil_reference_front_ == dynamic_stencil_reference_back_) {
        deferred_command_buffer_.CmdVkSetStencilReference(
            VK_STENCIL_FACE_FRONT_AND_BACK, dynamic_stencil_reference_front_);
      } else {
        if (dynamic_stencil_reference_front_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilReference(
              VK_STENCIL_FACE_FRONT_BIT, dynamic_stencil_reference_front_);
        }
        if (dynamic_stencil_reference_back_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilReference(
              VK_STENCIL_FACE_BACK_BIT, dynamic_stencil_reference_back_);
        }
      }
      dynamic_stencil_reference_front_update_needed_ = false;
      dynamic_stencil_reference_back_update_needed_ = false;
    }
  }

  // Extended dynamic state (VK_EXT_extended_dynamic_state / state2 / state3).
  // The pipeline cache resolved the exact Vulkan values these draw must use
  // (matching what a static pipeline would have baked) into
  // pipeline_dynamic_state; emit only the fields whose capability bit is set,
  // with dirty tracking so redundant setters aren't recorded.
  const VulkanPipelineCache::DynamicStateCapabilities& eds_caps =
      pipeline_cache_->dynamic_state_capabilities();
  if (eds_caps.extended_dynamic_state) {
    const VulkanPipelineCache::DynamicState& ds = pipeline_dynamic_state;

    // Primitive topology (EDS1 core).
    dynamic_primitive_topology_update_needed_ |=
        dynamic_primitive_topology_ != ds.primitive_topology;
    if (dynamic_primitive_topology_update_needed_) {
      dynamic_primitive_topology_ = ds.primitive_topology;
      deferred_command_buffer_.CmdVkSetPrimitiveTopology(
          dynamic_primitive_topology_);
      dynamic_primitive_topology_update_needed_ = false;
    }
    // Primitive restart enable (EDS2 core).
    dynamic_primitive_restart_enable_update_needed_ |=
        dynamic_primitive_restart_enable_ != ds.primitive_restart_enable;
    if (dynamic_primitive_restart_enable_update_needed_) {
      dynamic_primitive_restart_enable_ = ds.primitive_restart_enable;
      deferred_command_buffer_.CmdVkSetPrimitiveRestartEnable(
          dynamic_primitive_restart_enable_);
      dynamic_primitive_restart_enable_update_needed_ = false;
    }
    // Cull mode (EDS1 core).
    dynamic_cull_mode_update_needed_ |= dynamic_cull_mode_ != ds.cull_mode;
    if (dynamic_cull_mode_update_needed_) {
      dynamic_cull_mode_ = ds.cull_mode;
      deferred_command_buffer_.CmdVkSetCullMode(dynamic_cull_mode_);
      dynamic_cull_mode_update_needed_ = false;
    }
    // Front face (EDS1 core).
    dynamic_front_face_update_needed_ |= dynamic_front_face_ != ds.front_face;
    if (dynamic_front_face_update_needed_) {
      dynamic_front_face_ = ds.front_face;
      deferred_command_buffer_.CmdVkSetFrontFace(dynamic_front_face_);
      dynamic_front_face_update_needed_ = false;
    }

    // Depth / stencil dynamic state. The pipeline cache disables the whole
    // extended-dynamic-state capability on the fragment-shader-interlock path
    // (where these aren't fixed-function), so reaching here implies a render
    // pass with real depth / stencil state.
    {
      // Depth test enable (EDS1 core).
      dynamic_depth_test_enable_update_needed_ |=
          dynamic_depth_test_enable_ != ds.depth_test_enable;
      if (dynamic_depth_test_enable_update_needed_) {
        dynamic_depth_test_enable_ = ds.depth_test_enable;
        deferred_command_buffer_.CmdVkSetDepthTestEnable(
            dynamic_depth_test_enable_);
        dynamic_depth_test_enable_update_needed_ = false;
      }
      // Depth write enable (EDS1 core).
      dynamic_depth_write_enable_update_needed_ |=
          dynamic_depth_write_enable_ != ds.depth_write_enable;
      if (dynamic_depth_write_enable_update_needed_) {
        dynamic_depth_write_enable_ = ds.depth_write_enable;
        deferred_command_buffer_.CmdVkSetDepthWriteEnable(
            dynamic_depth_write_enable_);
        dynamic_depth_write_enable_update_needed_ = false;
      }
      // Depth compare op (EDS1 core).
      dynamic_depth_compare_op_update_needed_ |=
          dynamic_depth_compare_op_ != ds.depth_compare_op;
      if (dynamic_depth_compare_op_update_needed_) {
        dynamic_depth_compare_op_ = ds.depth_compare_op;
        deferred_command_buffer_.CmdVkSetDepthCompareOp(
            dynamic_depth_compare_op_);
        dynamic_depth_compare_op_update_needed_ = false;
      }
      // Stencil test enable (EDS1 core).
      dynamic_stencil_test_enable_update_needed_ |=
          dynamic_stencil_test_enable_ != ds.stencil_test_enable;
      if (dynamic_stencil_test_enable_update_needed_) {
        dynamic_stencil_test_enable_ = ds.stencil_test_enable;
        deferred_command_buffer_.CmdVkSetStencilTestEnable(
            dynamic_stencil_test_enable_);
        dynamic_stencil_test_enable_update_needed_ = false;
      }
      // Stencil ops (EDS1 core), tracked per face. memcmp because
      // VkStencilOpState bundles the (dynamic) compare/write masks and
      // reference, which are set separately and must be ignored here.
      auto stencil_ops_equal = [](const VkStencilOpState& a,
                                  const VkStencilOpState& b) {
        return a.failOp == b.failOp && a.passOp == b.passOp &&
               a.depthFailOp == b.depthFailOp && a.compareOp == b.compareOp;
      };
      dynamic_stencil_op_front_update_needed_ |=
          !stencil_ops_equal(dynamic_stencil_op_front_, ds.stencil_front);
      if (dynamic_stencil_op_front_update_needed_) {
        dynamic_stencil_op_front_ = ds.stencil_front;
        deferred_command_buffer_.CmdVkSetStencilOp(
            VK_STENCIL_FACE_FRONT_BIT, ds.stencil_front.failOp,
            ds.stencil_front.passOp, ds.stencil_front.depthFailOp,
            ds.stencil_front.compareOp);
        dynamic_stencil_op_front_update_needed_ = false;
      }
      dynamic_stencil_op_back_update_needed_ |=
          !stencil_ops_equal(dynamic_stencil_op_back_, ds.stencil_back);
      if (dynamic_stencil_op_back_update_needed_) {
        dynamic_stencil_op_back_ = ds.stencil_back;
        deferred_command_buffer_.CmdVkSetStencilOp(
            VK_STENCIL_FACE_BACK_BIT, ds.stencil_back.failOp,
            ds.stencil_back.passOp, ds.stencil_back.depthFailOp,
            ds.stencil_back.compareOp);
        dynamic_stencil_op_back_update_needed_ = false;
      }
    }

    // EDS3 sub-features.
    if (eds_caps.depth_clamp_enable) {
      dynamic_depth_clamp_enable_update_needed_ |=
          dynamic_depth_clamp_enable_ != ds.depth_clamp_enable;
      if (dynamic_depth_clamp_enable_update_needed_) {
        dynamic_depth_clamp_enable_ = ds.depth_clamp_enable;
        deferred_command_buffer_.CmdVkSetDepthClampEnableEXT(
            dynamic_depth_clamp_enable_);
        dynamic_depth_clamp_enable_update_needed_ = false;
      }
    }
    if (eds_caps.polygon_mode) {
      dynamic_polygon_mode_update_needed_ |=
          dynamic_polygon_mode_ != ds.polygon_mode;
      if (dynamic_polygon_mode_update_needed_) {
        dynamic_polygon_mode_ = ds.polygon_mode;
        deferred_command_buffer_.CmdVkSetPolygonModeEXT(dynamic_polygon_mode_);
        dynamic_polygon_mode_update_needed_ = false;
      }
    }
    {
      // Per-render-target blend state. Emit only for the attachments present in
      // the render pass (ds.color_rts_used), as a single contiguous range
      // covering [0, highest used] so the array is dense for the setter.
      uint32_t color_rts_used = ds.color_rts_used;
      uint32_t attachment_count =
          color_rts_used ? (32 - xe::lzcnt(color_rts_used)) : 0;
      if (attachment_count) {
        if (eds_caps.color_blend_enable) {
          bool changed = false;
          for (uint32_t i = 0; i < attachment_count; ++i) {
            if (dynamic_color_blend_enable_[i] != ds.color_blend_enable[i]) {
              dynamic_color_blend_enable_[i] = ds.color_blend_enable[i];
              changed = true;
            }
          }
          dynamic_color_blend_enable_update_needed_ |= changed;
          if (dynamic_color_blend_enable_update_needed_) {
            deferred_command_buffer_.CmdVkSetColorBlendEnableEXT(
                0, attachment_count, dynamic_color_blend_enable_);
            dynamic_color_blend_enable_update_needed_ = false;
          }
        }
        if (eds_caps.color_blend_equation) {
          bool changed = false;
          for (uint32_t i = 0; i < attachment_count; ++i) {
            if (std::memcmp(&dynamic_color_blend_equation_[i],
                            &ds.color_blend_equation[i],
                            sizeof(VkColorBlendEquationEXT)) != 0) {
              dynamic_color_blend_equation_[i] = ds.color_blend_equation[i];
              changed = true;
            }
          }
          dynamic_color_blend_equation_update_needed_ |= changed;
          if (dynamic_color_blend_equation_update_needed_) {
            deferred_command_buffer_.CmdVkSetColorBlendEquationEXT(
                0, attachment_count, dynamic_color_blend_equation_);
            dynamic_color_blend_equation_update_needed_ = false;
          }
        }
        if (eds_caps.color_write_mask) {
          bool changed = false;
          for (uint32_t i = 0; i < attachment_count; ++i) {
            if (dynamic_color_write_mask_[i] != ds.color_write_mask[i]) {
              dynamic_color_write_mask_[i] = ds.color_write_mask[i];
              changed = true;
            }
          }
          dynamic_color_write_mask_update_needed_ |= changed;
          if (dynamic_color_write_mask_update_needed_) {
            deferred_command_buffer_.CmdVkSetColorWriteMaskEXT(
                0, attachment_count, dynamic_color_write_mask_);
            dynamic_color_write_mask_update_needed_ = false;
          }
        }
      }
    }
  }
}

void VulkanCommandProcessor::UpdateSystemConstantValues(
    bool primitive_polygonal,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    bool shader_32bit_index_dma, const draw_util::ViewportInfo& viewport_info,
    uint32_t used_texture_mask, reg::RB_DEPTHCONTROL normalized_depth_control,
    uint32_t normalized_color_mask,
    const draw_util::HostDepthPolygonOffset* host_depth_polygon_offset) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;
  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  auto rb_alpha_ref = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
  auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  auto rb_stencilrefmask = regs.Get<reg::RB_STENCILREFMASK>();
  auto rb_stencilrefmask_bf =
      regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  auto vgt_indx_offset = regs.Get<int32_t>(XE_GPU_REG_VGT_INDX_OFFSET);

  bool edram_fragment_shader_interlock =
      render_target_cache_->GetPath() ==
      RenderTargetCache::Path::kPixelShaderInterlock;
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();

  // Get the color info register values for each render target. Also, for FSI,
  // exclude components that don't exist in the format from the write mask.
  // Don't exclude fully overlapping render targets, however - two render
  // targets with the same base address are used in the lighting pass of
  // 4D5307E6, for example, with the needed one picked with dynamic control
  // flow.
  reg::RB_COLOR_INFO color_infos[xenos::kMaxColorRenderTargets];
  float rt_clamp[4][4];
  // Two UINT32_MAX if no components actually existing in the RT are written.
  uint32_t rt_keep_masks[4][2];
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    auto color_info = regs.Get<reg::RB_COLOR_INFO>(
        reg::RB_COLOR_INFO::rt_register_indices[i]);
    color_infos[i] = color_info;
    if (edram_fragment_shader_interlock) {
      RenderTargetCache::GetPSIColorFormatInfo(
          color_info.color_format, (normalized_color_mask >> (i * 4)) & 0b1111,
          rt_clamp[i][0], rt_clamp[i][1], rt_clamp[i][2], rt_clamp[i][3],
          rt_keep_masks[i][0], rt_keep_masks[i][1]);
    }
  }

  // Disable depth and stencil if it aliases a color render target (for
  // instance, during the XBLA logo in 58410954, though depth writing is already
  // disabled there).
  bool depth_stencil_enabled = normalized_depth_control.stencil_enable ||
                               normalized_depth_control.z_enable;
  if (edram_fragment_shader_interlock && depth_stencil_enabled) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (rb_depth_info.depth_base == color_infos[i].color_base &&
          (rt_keep_masks[i][0] != UINT32_MAX ||
           rt_keep_masks[i][1] != UINT32_MAX)) {
        depth_stencil_enabled = false;
        break;
      }
    }
  }

  bool dirty = false;

  // Flags.
  uint32_t flags = 0;
  // Vertex index shader loading.
  if (shader_32bit_index_dma) {
    flags |= SpirvShaderTranslator::kSysFlag_VertexIndexLoad;
  }
  if (primitive_processing_result.index_buffer_type ==
      PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA) {
    flags |= SpirvShaderTranslator::kSysFlag_ComputeOrPrimitiveVertexIndexLoad;
    if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32) {
      flags |= SpirvShaderTranslator ::
          kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit;
    }
  }
  // W0 division control.
  // http://www.x.org/docs/AMD/old/evergreen_3D_registers_v2.pdf
  // 8: VTX_XY_FMT = true: the incoming XY have already been multiplied by 1/W0.
  //               = false: multiply the X, Y coordinates by 1/W0.
  // 9: VTX_Z_FMT = true: the incoming Z has already been multiplied by 1/W0.
  //              = false: multiply the Z coordinate by 1/W0.
  // 10: VTX_W0_FMT = true: the incoming W0 is not 1/W0. Perform the reciprocal
  //                        to get 1/W0.
  if (pa_cl_vte_cntl.vtx_xy_fmt) {
    flags |= SpirvShaderTranslator::kSysFlag_XYDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_z_fmt) {
    flags |= SpirvShaderTranslator::kSysFlag_ZDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_w0_fmt) {
    flags |= SpirvShaderTranslator::kSysFlag_WNotReciprocal;
  }
  // Whether the primitive is polygonal, and gl_FrontFacing matters.
  if (primitive_polygonal) {
    flags |= SpirvShaderTranslator::kSysFlag_PrimitivePolygonal;
  }
  // Primitive type.
  if (draw_util::IsPrimitiveLine(regs)) {
    flags |= SpirvShaderTranslator::kSysFlag_PrimitiveLine;
  }
  // MSAA sample count.
  flags |= uint32_t(rb_surface_info.msaa_samples)
           << SpirvShaderTranslator::kSysFlag_MsaaSamples_Shift;
  // Depth format.
  if (rb_depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
    flags |= SpirvShaderTranslator::kSysFlag_DepthFloat24;
  }
  // Alpha test.
  xenos::CompareFunction alpha_test_function =
      rb_colorcontrol.alpha_test_enable ? rb_colorcontrol.alpha_func
                                        : xenos::CompareFunction::kAlways;
  flags |= uint32_t(alpha_test_function)
           << SpirvShaderTranslator::kSysFlag_AlphaPassIfLess_Shift;
  // Gamma writing. When gamma is stored as unorm16, the host render target
  // holds linear values (blended in linear space) and the linear -> gamma
  // encode happens on the EDRAM store, so the pixel shader must not pre-encode.
  if (!edram_fragment_shader_interlock &&
      !render_target_cache_->gamma_render_target_as_unorm16()) {
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (color_infos[i].color_format ==
          xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
        flags |= SpirvShaderTranslator::kSysFlag_ConvertColor0ToGamma << i;
      }
    }
  }
  if (edram_fragment_shader_interlock && depth_stencil_enabled) {
    flags |= SpirvShaderTranslator::kSysFlag_FSIDepthStencil;
    if (normalized_depth_control.z_enable) {
      flags |= uint32_t(normalized_depth_control.zfunc)
               << SpirvShaderTranslator::kSysFlag_FSIDepthPassIfLess_Shift;
      if (normalized_depth_control.z_write_enable) {
        flags |= SpirvShaderTranslator::kSysFlag_FSIDepthWrite;
      }
    } else {
      // In case stencil is used without depth testing - always pass, and
      // don't modify the stored depth.
      flags |= SpirvShaderTranslator::kSysFlag_FSIDepthPassIfLess |
               SpirvShaderTranslator::kSysFlag_FSIDepthPassIfEqual |
               SpirvShaderTranslator::kSysFlag_FSIDepthPassIfGreater;
    }
    if (normalized_depth_control.stencil_enable) {
      flags |= SpirvShaderTranslator::kSysFlag_FSIStencilTest;
    }
    // Hint - if not applicable to the shader, will not have effect.
    if (alpha_test_function == xenos::CompareFunction::kAlways &&
        !rb_colorcontrol.alpha_to_mask_enable) {
      flags |= SpirvShaderTranslator::kSysFlag_FSIDepthStencilEarlyWrite;
    }
  }
  dirty |= system_constants_.flags != flags;
  system_constants_.flags = flags;

  // Index buffer address for loading in the shaders.
  if (flags &
      (SpirvShaderTranslator::kSysFlag_VertexIndexLoad |
       SpirvShaderTranslator::kSysFlag_ComputeOrPrimitiveVertexIndexLoad)) {
    dirty |= system_constants_.vertex_index_load_address !=
             primitive_processing_result.guest_index_base;
    system_constants_.vertex_index_load_address =
        primitive_processing_result.guest_index_base;
  }

  // Guest-side vertex index count, used for bounds-safe shared-memory loads in
  // VS expansion (kPointListAsTriangleStrip / kRectangleListAsTriangleStrip)
  // and in the full 32-bit index DMA load path. Out-of-bounds lanes read 0
  // instead of random memory, which otherwise produces scattered/skewed
  // geometry when expansion fans out past the guest draw count.
  const bool is_vs_expansion_draw =
      primitive_processing_result.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kPointListAsTriangleStrip ||
      primitive_processing_result.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kRectangleListAsTriangleStrip;
  const uint32_t vertex_index_count =
      is_vs_expansion_draw ? primitive_processing_result.guest_draw_vertex_count
                           : primitive_processing_result.host_draw_vertex_count;
  dirty |= system_constants_.vertex_index_count != vertex_index_count;
  system_constants_.vertex_index_count = vertex_index_count;

  // Index or tessellation edge factor buffer endianness.
  dirty |= system_constants_.vertex_index_endian !=
           primitive_processing_result.host_shader_index_endian;
  system_constants_.vertex_index_endian =
      primitive_processing_result.host_shader_index_endian;

  // Vertex index offset.
  dirty |= system_constants_.vertex_base_index != vgt_indx_offset;
  system_constants_.vertex_base_index = vgt_indx_offset;

  // Conversion to host normalized device coordinates.
  for (uint32_t i = 0; i < 3; ++i) {
    dirty |= system_constants_.ndc_scale[i] != viewport_info.ndc_scale[i];
    dirty |= system_constants_.ndc_offset[i] != viewport_info.ndc_offset[i];
    system_constants_.ndc_scale[i] = viewport_info.ndc_scale[i];
    system_constants_.ndc_offset[i] = viewport_info.ndc_offset[i];
  }

  // User clip planes, for vertex and domain shaders.
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  if (!pa_cl_clip_cntl.clip_disable && pa_cl_clip_cntl.ucp_ena) {
    float* user_clip_plane_write_ptr = system_constants_.user_clip_planes[0];
    uint32_t user_clip_planes_remaining = pa_cl_clip_cntl.ucp_ena;
    uint32_t user_clip_plane_index;
    while (xe::bit_scan_forward(user_clip_planes_remaining,
                                &user_clip_plane_index)) {
      user_clip_planes_remaining =
          xe::clear_lowest_bit(user_clip_planes_remaining);
      // Validate plane index is within bounds (0-5).
      assert(user_clip_plane_index < 6);
      if (user_clip_plane_index >= 6) {
        continue;
      }
      const void* user_clip_plane_regs =
          &regs[XE_GPU_REG_PA_CL_UCP_0_X + user_clip_plane_index * 4];
      if (std::memcmp(user_clip_plane_write_ptr, user_clip_plane_regs,
                      4 * sizeof(float))) {
        dirty = true;
        std::memcpy(user_clip_plane_write_ptr, user_clip_plane_regs,
                    4 * sizeof(float));
      }
      user_clip_plane_write_ptr += 4;
    }
  } else {
    constexpr float kZeroPlanes[6][4] = {};
    if (std::memcmp(system_constants_.user_clip_planes, kZeroPlanes,
                    sizeof(system_constants_.user_clip_planes))) {
      dirty = true;
      std::memset(system_constants_.user_clip_planes, 0,
                  sizeof(system_constants_.user_clip_planes));
    }
  }

  // Tessellation constants. The factor range has 1.0 added per Xbox 360 docs.
  // fractional_even partitioning needs a minimum of 2.0.
  {
    float tess_min = regs.Get<float>(XE_GPU_REG_VGT_HOS_MIN_TESS_LEVEL) + 1.0f;
    float tess_max = regs.Get<float>(XE_GPU_REG_VGT_HOS_MAX_TESS_LEVEL) + 1.0f;
    dirty |= system_constants_.tessellation_factor_range[0] != tess_min;
    dirty |= system_constants_.tessellation_factor_range[1] != tess_max;
    system_constants_.tessellation_factor_range[0] = tess_min;
    system_constants_.tessellation_factor_range[1] = tess_max;
    uint32_t tess_vie =
        static_cast<uint32_t>(regs.Get<reg::VGT_DMA_SIZE>().swap_mode);
    uint32_t tess_vio = regs[XE_GPU_REG_VGT_INDX_OFFSET];
    uint32_t tess_vmin = regs[XE_GPU_REG_VGT_MIN_VTX_INDX];
    uint32_t tess_vmax = regs[XE_GPU_REG_VGT_MAX_VTX_INDX];
    dirty |= system_constants_.tessellation_vertex_index_endian != tess_vie;
    dirty |= system_constants_.tessellation_vertex_index_offset != tess_vio;
    dirty |=
        system_constants_.tessellation_vertex_index_min_max[0] != tess_vmin;
    dirty |=
        system_constants_.tessellation_vertex_index_min_max[1] != tess_vmax;
    system_constants_.tessellation_vertex_index_endian = tess_vie;
    system_constants_.tessellation_vertex_index_offset = tess_vio;
    system_constants_.tessellation_vertex_index_min_max[0] = tess_vmin;
    system_constants_.tessellation_vertex_index_min_max[1] = tess_vmax;
  }

  // Point size.
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    auto pa_su_point_size = regs.Get<reg::PA_SU_POINT_SIZE>();
    float point_vertex_diameter_min =
        float(pa_su_point_minmax.min_size) * (2.0f / 16.0f);
    float point_vertex_diameter_max =
        float(pa_su_point_minmax.max_size) * (2.0f / 16.0f);
    float point_constant_diameter_x =
        float(pa_su_point_size.width) * (2.0f / 16.0f);
    float point_constant_diameter_y =
        float(pa_su_point_size.height) * (2.0f / 16.0f);
    dirty |= system_constants_.point_vertex_diameter_min !=
             point_vertex_diameter_min;
    dirty |= system_constants_.point_vertex_diameter_max !=
             point_vertex_diameter_max;
    dirty |= system_constants_.point_constant_diameter[0] !=
             point_constant_diameter_x;
    dirty |= system_constants_.point_constant_diameter[1] !=
             point_constant_diameter_y;
    system_constants_.point_vertex_diameter_min = point_vertex_diameter_min;
    system_constants_.point_vertex_diameter_max = point_vertex_diameter_max;
    system_constants_.point_constant_diameter[0] = point_constant_diameter_x;
    system_constants_.point_constant_diameter[1] = point_constant_diameter_y;
    // 2 because 1 in the NDC is half of the viewport's axis, 0.5 for diameter
    // to radius conversion to avoid multiplying the per-vertex diameter by an
    // additional constant in the shader.
    float point_screen_diameter_to_ndc_radius_x =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_x)) /
        std::max(viewport_info.xy_extent[0], uint32_t(1));
    float point_screen_diameter_to_ndc_radius_y =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_y)) /
        std::max(viewport_info.xy_extent[1], uint32_t(1));
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[0] !=
             point_screen_diameter_to_ndc_radius_x;
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[1] !=
             point_screen_diameter_to_ndc_radius_y;
    system_constants_.point_screen_diameter_to_ndc_radius[0] =
        point_screen_diameter_to_ndc_radius_x;
    system_constants_.point_screen_diameter_to_ndc_radius[1] =
        point_screen_diameter_to_ndc_radius_y;
  }

  // Texture signedness / gamma.
  {
    uint32_t textures_remaining = used_texture_mask;
    uint32_t texture_index;
    while (xe::bit_scan_forward(textures_remaining, &texture_index)) {
      textures_remaining &= ~(UINT32_C(1) << texture_index);
      uint32_t& texture_signs_uint =
          system_constants_.texture_swizzled_signs[texture_index >> 2];
      uint32_t texture_signs_shift = 8 * (texture_index & 3);
      uint8_t texture_signs =
          texture_cache_->GetActiveTextureSwizzledSigns(texture_index);
      uint32_t texture_signs_shifted = uint32_t(texture_signs)
                                       << texture_signs_shift;
      uint32_t texture_signs_mask = ((UINT32_C(1) << 8) - 1)
                                    << texture_signs_shift;
      dirty |=
          (texture_signs_uint & texture_signs_mask) != texture_signs_shifted;
      texture_signs_uint =
          (texture_signs_uint & ~texture_signs_mask) | texture_signs_shifted;
    }
  }

  // Texture host swizzle in the shader.
  if (!GetVulkanDevice()->properties().imageViewFormatSwizzle) {
    uint32_t textures_remaining = used_texture_mask;
    uint32_t texture_index;
    while (xe::bit_scan_forward(textures_remaining, &texture_index)) {
      textures_remaining &= ~(UINT32_C(1) << texture_index);
      uint32_t& texture_swizzles_uint =
          system_constants_.texture_swizzles[texture_index >> 1];
      uint32_t texture_swizzle_shift = 12 * (texture_index & 1);
      uint32_t texture_swizzle =
          texture_cache_->GetActiveTextureHostSwizzle(texture_index);
      uint32_t texture_swizzle_shifted = uint32_t(texture_swizzle)
                                         << texture_swizzle_shift;
      uint32_t texture_swizzle_mask = ((UINT32_C(1) << 12) - 1)
                                      << texture_swizzle_shift;
      dirty |= (texture_swizzles_uint & texture_swizzle_mask) !=
               texture_swizzle_shifted;
      texture_swizzles_uint = (texture_swizzles_uint & ~texture_swizzle_mask) |
                              texture_swizzle_shifted;
    }
  }

  // Textures resolution scaled - which textures are from scaled resolve
  // operations.
  {
    uint32_t textures_resolved = 0;
    uint32_t textures_remaining = used_texture_mask;
    uint32_t texture_index;
    while (xe::bit_scan_forward(textures_remaining, &texture_index)) {
      textures_remaining &= ~(UINT32_C(1) << texture_index);
      textures_resolved |=
          uint32_t(
              texture_cache_->IsActiveTextureResolutionScaled(texture_index))
          << texture_index;
    }
    dirty |= system_constants_.textures_resolved != textures_resolved;
    system_constants_.textures_resolved = textures_resolved;
  }

  // Alpha test.
  dirty |= system_constants_.alpha_test_reference != rb_alpha_ref;
  system_constants_.alpha_test_reference = rb_alpha_ref;

  // Alpha to coverage.
  uint32_t alpha_to_mask = rb_colorcontrol.alpha_to_mask_enable
                               ? (rb_colorcontrol.value >> 24) | (1 << 8)
                               : 0;
  dirty |= system_constants_.alpha_to_mask != alpha_to_mask;
  system_constants_.alpha_to_mask = alpha_to_mask;

  // FSI ZPD counter.
  uint32_t zpd_fsi_counter_index = UINT32_MAX;
  if (edram_fragment_shader_interlock &&
      zpd_active_query_index_ != UINT32_MAX && zpd_active_query_is_fsi_ &&
      zpd_host_query_pool_->fsi_counter_initialized()) {
    zpd_fsi_counter_index = zpd_active_query_index_;
  }
  dirty |= zpd_fsi_counter_index_force_update_ ||
           system_constants_.zpd_fsi_counter_index != zpd_fsi_counter_index;
  system_constants_.zpd_fsi_counter_index = zpd_fsi_counter_index;
  zpd_fsi_counter_index_force_update_ = false;

  uint32_t edram_tile_dwords_scaled =
      xenos::kEdramTileWidthSamples * xenos::kEdramTileHeightSamples *
      (draw_resolution_scale_x * draw_resolution_scale_y);

  // EDRAM pitch for FSI render target writing.
  if (edram_fragment_shader_interlock) {
    // Align, then multiply by 32bpp tile size in dwords.
    uint32_t edram_32bpp_tile_pitch_dwords_scaled =
        ((rb_surface_info.surface_pitch *
          (rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 2 : 1)) +
         (xenos::kEdramTileWidthSamples - 1)) /
        xenos::kEdramTileWidthSamples * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_32bpp_tile_pitch_dwords_scaled !=
             edram_32bpp_tile_pitch_dwords_scaled;
    system_constants_.edram_32bpp_tile_pitch_dwords_scaled =
        edram_32bpp_tile_pitch_dwords_scaled;
  }

  // Color exponent bias and FSI render target writing.
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    reg::RB_COLOR_INFO color_info = color_infos[i];
    // Exponent bias is in bits 20:25 of RB_COLOR_INFO.
    int32_t color_exp_bias = color_info.color_exp_bias;
    if (render_target_cache_->GetPath() ==
            RenderTargetCache::Path::kHostRenderTargets &&
        (color_info.color_format == xenos::ColorRenderTargetFormat::k_16_16 &&
             !render_target_cache_->IsFixedRG16TruncatedToMinus1To1() ||
         color_info.color_format ==
                 xenos::ColorRenderTargetFormat::k_16_16_16_16 &&
             !render_target_cache_->IsFixedRGBA16TruncatedToMinus1To1())) {
      // Remap from -32...32 to -1...1 by dividing the output values by 32,
      // losing blending correctness, but getting the full range.
      color_exp_bias -= 5;
    }
    float color_exp_bias_scale;
    *reinterpret_cast<int32_t*>(&color_exp_bias_scale) =
        UINT32_C(0x3F800000) + (color_exp_bias << 23);
    dirty |= system_constants_.color_exp_bias[i] != color_exp_bias_scale;
    system_constants_.color_exp_bias[i] = color_exp_bias_scale;
    if (edram_fragment_shader_interlock) {
      dirty |=
          system_constants_.edram_rt_keep_mask[i][0] != rt_keep_masks[i][0];
      system_constants_.edram_rt_keep_mask[i][0] = rt_keep_masks[i][0];
      dirty |=
          system_constants_.edram_rt_keep_mask[i][1] != rt_keep_masks[i][1];
      system_constants_.edram_rt_keep_mask[i][1] = rt_keep_masks[i][1];
      if (rt_keep_masks[i][0] != UINT32_MAX ||
          rt_keep_masks[i][1] != UINT32_MAX) {
        uint32_t rt_base_dwords_scaled =
            color_info.color_base * edram_tile_dwords_scaled;
        dirty |= system_constants_.edram_rt_base_dwords_scaled[i] !=
                 rt_base_dwords_scaled;
        system_constants_.edram_rt_base_dwords_scaled[i] =
            rt_base_dwords_scaled;
        uint32_t format_flags =
            RenderTargetCache::AddPSIColorFormatFlags(color_info.color_format);
        dirty |= system_constants_.edram_rt_format_flags[i] != format_flags;
        system_constants_.edram_rt_format_flags[i] = format_flags;
        uint32_t blend_factors_ops =
            regs[reg::RB_BLENDCONTROL::rt_register_indices[i]] & 0x1FFF1FFF;
        dirty |= system_constants_.edram_rt_blend_factors_ops[i] !=
                 blend_factors_ops;
        system_constants_.edram_rt_blend_factors_ops[i] = blend_factors_ops;
        // Can't do float comparisons here because NaNs would result in always
        // setting the dirty flag.
        dirty |= std::memcmp(system_constants_.edram_rt_clamp[i], rt_clamp[i],
                             4 * sizeof(float)) != 0;
        std::memcpy(system_constants_.edram_rt_clamp[i], rt_clamp[i],
                    4 * sizeof(float));
      }
    }
  }

  if (!edram_fragment_shader_interlock && host_depth_polygon_offset) {
    draw_util::HostDepthPolygonOffset polygon_offset =
        *host_depth_polygon_offset;
    float scale_factor =
        float(std::max(draw_resolution_scale_x, draw_resolution_scale_y));
    polygon_offset.front_scale *= scale_factor;
    polygon_offset.back_scale *= scale_factor;
    dirty |= system_constants_.edram_poly_offset_front_scale !=
             polygon_offset.front_scale;
    system_constants_.edram_poly_offset_front_scale =
        polygon_offset.front_scale;
    dirty |= system_constants_.edram_poly_offset_front_offset !=
             polygon_offset.front_offset;
    system_constants_.edram_poly_offset_front_offset =
        polygon_offset.front_offset;
    dirty |= system_constants_.edram_poly_offset_back_scale !=
             polygon_offset.back_scale;
    system_constants_.edram_poly_offset_back_scale = polygon_offset.back_scale;
    dirty |= system_constants_.edram_poly_offset_back_offset !=
             polygon_offset.back_offset;
    system_constants_.edram_poly_offset_back_offset =
        polygon_offset.back_offset;
  }

  if (edram_fragment_shader_interlock) {
    uint32_t depth_base_dwords_scaled =
        rb_depth_info.depth_base * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_depth_base_dwords_scaled !=
             depth_base_dwords_scaled;
    system_constants_.edram_depth_base_dwords_scaled = depth_base_dwords_scaled;

    // For non-polygons, front polygon offset is used, and it's enabled if
    // POLY_OFFSET_PARA_ENABLED is set, for polygons, separate front and back
    // are used.
    float poly_offset_front_scale = 0.0f, poly_offset_front_offset = 0.0f;
    float poly_offset_back_scale = 0.0f, poly_offset_back_offset = 0.0f;
    if (primitive_polygonal) {
      if (pa_su_sc_mode_cntl.poly_offset_front_enable) {
        poly_offset_front_scale =
            regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset =
            regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
      }
      if (pa_su_sc_mode_cntl.poly_offset_back_enable) {
        poly_offset_back_scale =
            regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE);
        poly_offset_back_offset =
            regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET);
      }
    } else {
      if (pa_su_sc_mode_cntl.poly_offset_para_enable) {
        poly_offset_front_scale =
            regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset =
            regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
        poly_offset_back_scale = poly_offset_front_scale;
        poly_offset_back_offset = poly_offset_front_offset;
      }
    }
    // With non-square resolution scaling, make sure the worst-case impact is
    // reverted (slope only along the scaled axis), thus max. More bias is
    // better than less bias, because less bias means Z fighting with the
    // background is more likely.
    float poly_offset_scale_factor =
        xenos::kPolygonOffsetScaleSubpixelUnit *
        std::max(draw_resolution_scale_x, draw_resolution_scale_y);
    poly_offset_front_scale *= poly_offset_scale_factor;
    poly_offset_back_scale *= poly_offset_scale_factor;
    dirty |= system_constants_.edram_poly_offset_front_scale !=
             poly_offset_front_scale;
    system_constants_.edram_poly_offset_front_scale = poly_offset_front_scale;
    dirty |= system_constants_.edram_poly_offset_front_offset !=
             poly_offset_front_offset;
    system_constants_.edram_poly_offset_front_offset = poly_offset_front_offset;
    dirty |= system_constants_.edram_poly_offset_back_scale !=
             poly_offset_back_scale;
    system_constants_.edram_poly_offset_back_scale = poly_offset_back_scale;
    dirty |= system_constants_.edram_poly_offset_back_offset !=
             poly_offset_back_offset;
    system_constants_.edram_poly_offset_back_offset = poly_offset_back_offset;

    if (depth_stencil_enabled && normalized_depth_control.stencil_enable) {
      uint32_t stencil_front_reference_masks =
          rb_stencilrefmask.value & 0xFFFFFF;
      dirty |= system_constants_.edram_stencil_front_reference_masks !=
               stencil_front_reference_masks;
      system_constants_.edram_stencil_front_reference_masks =
          stencil_front_reference_masks;
      uint32_t stencil_func_ops =
          (normalized_depth_control.value >> 8) & ((1 << 12) - 1);
      dirty |=
          system_constants_.edram_stencil_front_func_ops != stencil_func_ops;
      system_constants_.edram_stencil_front_func_ops = stencil_func_ops;

      if (primitive_polygonal && normalized_depth_control.backface_enable) {
        uint32_t stencil_back_reference_masks =
            rb_stencilrefmask_bf.value & 0xFFFFFF;
        dirty |= system_constants_.edram_stencil_back_reference_masks !=
                 stencil_back_reference_masks;
        system_constants_.edram_stencil_back_reference_masks =
            stencil_back_reference_masks;
        uint32_t stencil_func_ops_bf =
            (normalized_depth_control.value >> 20) & ((1 << 12) - 1);
        dirty |= system_constants_.edram_stencil_back_func_ops !=
                 stencil_func_ops_bf;
        system_constants_.edram_stencil_back_func_ops = stencil_func_ops_bf;
      } else {
        dirty |= std::memcmp(system_constants_.edram_stencil_back,
                             system_constants_.edram_stencil_front,
                             2 * sizeof(uint32_t)) != 0;
        std::memcpy(system_constants_.edram_stencil_back,
                    system_constants_.edram_stencil_front,
                    2 * sizeof(uint32_t));
      }
    }

    dirty |= system_constants_.edram_blend_constant[0] !=
             regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    system_constants_.edram_blend_constant[0] =
        regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    dirty |= system_constants_.edram_blend_constant[1] !=
             regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    system_constants_.edram_blend_constant[1] =
        regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    dirty |= system_constants_.edram_blend_constant[2] !=
             regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    system_constants_.edram_blend_constant[2] =
        regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    dirty |= system_constants_.edram_blend_constant[3] !=
             regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
    system_constants_.edram_blend_constant[3] =
        regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
  }

  if (dirty) {
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
  }
}

bool VulkanCommandProcessor::UpdateBindings(const VulkanShader* vertex_shader,
                                            const VulkanShader* pixel_shader) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Invalidate constant buffers and descriptors for changed data.

  // Float constants.
  // These are the constant base addresses/ranges for shaders.
  // We have these hardcoded right now cause nothing seems to differ on the Xbox
  // 360 (however, OpenGL ES on Adreno 200 on Android has different ranges).
  assert_true(regs[XE_GPU_REG_SQ_VS_CONST] == 0x000FF000 ||
              regs[XE_GPU_REG_SQ_VS_CONST] == 0x00000000);
  assert_true(regs[XE_GPU_REG_SQ_PS_CONST] == 0x000FF100 ||
              regs[XE_GPU_REG_SQ_PS_CONST] == 0x00000000);
  // Check if the float constant layout is still the same and get the counts.
  const Shader::ConstantRegisterMap& float_constant_map_vertex =
      vertex_shader->constant_register_map();
  uint32_t float_constant_count_vertex = float_constant_map_vertex.float_count;
  for (uint32_t i = 0; i < 4; ++i) {
    if (current_float_constant_map_vertex_[i] !=
        float_constant_map_vertex.float_bitmap[i]) {
      current_float_constant_map_vertex_[i] =
          float_constant_map_vertex.float_bitmap[i];
      // If no float constants at all, any buffer can be reused for them, so not
      // invalidating.
      if (float_constant_count_vertex) {
        current_constant_buffers_up_to_date_ &=
            ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
      }
    }
  }
  uint32_t float_constant_count_pixel = 0;
  if (pixel_shader != nullptr) {
    const Shader::ConstantRegisterMap& float_constant_map_pixel =
        pixel_shader->constant_register_map();
    float_constant_count_pixel = float_constant_map_pixel.float_count;
    for (uint32_t i = 0; i < 4; ++i) {
      if (current_float_constant_map_pixel_[i] !=
          float_constant_map_pixel.float_bitmap[i]) {
        current_float_constant_map_pixel_[i] =
            float_constant_map_pixel.float_bitmap[i];
        if (float_constant_count_pixel) {
          current_constant_buffers_up_to_date_ &= ~(
              UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel);
        }
      }
    }
  } else {
    std::memset(current_float_constant_map_pixel_, 0,
                sizeof(current_float_constant_map_pixel_));
  }

  // Write the new constant buffers.
  constexpr uint32_t kAllConstantBuffersMask =
      (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferCount) - 1;
  assert_zero(current_constant_buffers_up_to_date_ & ~kAllConstantBuffersMask);
  if ((current_constant_buffers_up_to_date_ & kAllConstantBuffersMask) !=
      kAllConstantBuffersMask) {
    current_graphics_descriptor_set_values_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants);
    size_t uniform_buffer_alignment =
        size_t(vulkan_device->properties().minUniformBufferOffsetAlignment);
    // System constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem))) {
      VkDescriptorBufferInfo& buffer_info = current_constant_buffer_infos_
          [SpirvShaderTranslator::kConstantBufferSystem];
      uint8_t* mapping = uniform_buffer_pool_->Request(
          frame_current_, sizeof(SpirvShaderTranslator::SystemConstants),
          uniform_buffer_alignment, buffer_info.buffer, buffer_info.offset);
      if (!mapping) {
        return false;
      }
      buffer_info.range = sizeof(SpirvShaderTranslator::SystemConstants);
      std::memcpy(mapping, &system_constants_,
                  sizeof(SpirvShaderTranslator::SystemConstants));
      current_constant_buffers_up_to_date_ |=
          UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem;
    }
    // Vertex shader float constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex))) {
      VkDescriptorBufferInfo& buffer_info = current_constant_buffer_infos_
          [SpirvShaderTranslator::kConstantBufferFloatVertex];
      // Even if the shader doesn't need any float constants, a valid binding
      // must still be provided (the pipeline layout always has float constants,
      // for both the vertex shader and the pixel shader), so if the first draw
      // in the frame doesn't have float constants at all, still allocate a
      // dummy buffer.
      size_t float_constants_size =
          sizeof(float) * 4 *
          std::max(float_constant_count_vertex, UINT32_C(1));
      uint8_t* mapping = uniform_buffer_pool_->Request(
          frame_current_, float_constants_size, uniform_buffer_alignment,
          buffer_info.buffer, buffer_info.offset);
      if (!mapping) {
        return false;
      }
      buffer_info.range = VkDeviceSize(float_constants_size);
      for (uint32_t i = 0; i < 4; ++i) {
        uint64_t float_constant_map_entry =
            current_float_constant_map_vertex_[i];
        uint32_t float_constant_index;
        while (xe::bit_scan_forward(float_constant_map_entry,
                                    &float_constant_index)) {
          float_constant_map_entry &= ~(1ull << float_constant_index);
          std::memcpy(mapping,
                      &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (i << 8) +
                            (float_constant_index << 2)],
                      sizeof(float) * 4);
          mapping += sizeof(float) * 4;
        }
      }
      current_constant_buffers_up_to_date_ |=
          UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex;
    }
    // Pixel shader float constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel))) {
      VkDescriptorBufferInfo& buffer_info = current_constant_buffer_infos_
          [SpirvShaderTranslator::kConstantBufferFloatPixel];
      size_t float_constants_size =
          sizeof(float) * 4 * std::max(float_constant_count_pixel, UINT32_C(1));
      uint8_t* mapping = uniform_buffer_pool_->Request(
          frame_current_, float_constants_size, uniform_buffer_alignment,
          buffer_info.buffer, buffer_info.offset);
      if (!mapping) {
        return false;
      }
      buffer_info.range = VkDeviceSize(float_constants_size);
      for (uint32_t i = 0; i < 4; ++i) {
        uint64_t float_constant_map_entry =
            current_float_constant_map_pixel_[i];
        uint32_t float_constant_index;
        while (xe::bit_scan_forward(float_constant_map_entry,
                                    &float_constant_index)) {
          float_constant_map_entry &= ~(1ull << float_constant_index);
          std::memcpy(mapping,
                      &regs[XE_GPU_REG_SHADER_CONSTANT_256_X + (i << 8) +
                            (float_constant_index << 2)],
                      sizeof(float) * 4);
          mapping += sizeof(float) * 4;
        }
      }
      current_constant_buffers_up_to_date_ |=
          UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel;
    }
    // Bool and loop constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferBoolLoop))) {
      VkDescriptorBufferInfo& buffer_info = current_constant_buffer_infos_
          [SpirvShaderTranslator::kConstantBufferBoolLoop];
      constexpr size_t kBoolLoopConstantsSize = sizeof(uint32_t) * (8 + 32);
      uint8_t* mapping = uniform_buffer_pool_->Request(
          frame_current_, kBoolLoopConstantsSize, uniform_buffer_alignment,
          buffer_info.buffer, buffer_info.offset);
      if (!mapping) {
        return false;
      }
      buffer_info.range = VkDeviceSize(kBoolLoopConstantsSize);
      std::memcpy(mapping, &regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                  kBoolLoopConstantsSize);
      current_constant_buffers_up_to_date_ |=
          UINT32_C(1) << SpirvShaderTranslator::kConstantBufferBoolLoop;
    }
    // Fetch constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch))) {
      VkDescriptorBufferInfo& buffer_info = current_constant_buffer_infos_
          [SpirvShaderTranslator::kConstantBufferFetch];
      constexpr size_t kFetchConstantsSize = sizeof(uint32_t) * 6 * 32;
      uint8_t* mapping = uniform_buffer_pool_->Request(
          frame_current_, kFetchConstantsSize, uniform_buffer_alignment,
          buffer_info.buffer, buffer_info.offset);
      if (!mapping) {
        return false;
      }
      buffer_info.range = VkDeviceSize(kFetchConstantsSize);
      std::memcpy(mapping, &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0],
                  kFetchConstantsSize);
      current_constant_buffers_up_to_date_ |=
          UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch;
    }
  }

  // Textures and samplers.
  const std::vector<VulkanShader::SamplerBinding>& samplers_vertex =
      vertex_shader->GetSamplerBindingsAfterTranslation();
  const std::vector<VulkanShader::TextureBinding>& textures_vertex =
      vertex_shader->GetTextureBindingsAfterTranslation();
  uint32_t sampler_count_vertex = uint32_t(samplers_vertex.size());
  uint32_t texture_count_vertex = uint32_t(textures_vertex.size());
  const std::vector<VulkanShader::SamplerBinding>* samplers_pixel;
  const std::vector<VulkanShader::TextureBinding>* textures_pixel;
  uint32_t sampler_count_pixel, texture_count_pixel;
  if (pixel_shader) {
    samplers_pixel = &pixel_shader->GetSamplerBindingsAfterTranslation();
    textures_pixel = &pixel_shader->GetTextureBindingsAfterTranslation();
    sampler_count_pixel = uint32_t(samplers_pixel->size());
    texture_count_pixel = uint32_t(textures_pixel->size());
  } else {
    samplers_pixel = nullptr;
    textures_pixel = nullptr;
    sampler_count_pixel = 0;
    texture_count_pixel = 0;
  }
  // Value-cache the texture/sampler descriptor sets: only force a re-write (by
  // clearing the stage's values-up-to-date bit) when the resolved VkImageView /
  // VkSampler handles, the binding counts, or the layout changed since the last
  // successful write, OR an invalidation fired (the *_hash_valid_ flags are
  // reset on new frame / transient-pool reset / pipeline-layout change). When
  // nothing changed the bit stays set, the write-gate (write_*_textures below)
  // is false, and the still-valid cached set is left bound (no
  // vkUpdateDescriptorSets, no re-bind). The handle is hashed (never the fetch
  // constant index) so texture eviction/recreation, which re-resolves the
  // binding to a new handle before this point, always forces a re-write.
  //
  // Hash inputs, in this exact fixed order so an ordering/count change always
  // changes the hash: packed layout key (texture_count | sampler_count<<16 |
  // is_vertex<<31), then each resolved VkImageView in shader binding order,
  // then each resolved VkSampler in order.
  auto compute_texture_set_hash =
      [&scratch = texture_set_hash_scratch_](
          uint32_t texture_count, uint32_t sampler_count, bool is_vertex,
          const std::vector<VulkanShader::TextureBinding>* textures,
          VulkanTextureCache* texture_cache,
          const std::pair<VulkanTextureCache::SamplerParameters, VkSampler>*
              samplers) -> uint64_t {
    // Gather the reuse inputs (in this exact fixed order) into a reused scratch
    // buffer and hash them with one XXH3 call - NEON-accelerated on arm64,
    // matching the rest of the codebase instead of the old byte-at-a-time
    // scalar FNV-1a. The result is only ever compared against a cached value
    // from this same function, so swapping the algorithm is transparent to the
    // reuse gate (different absolute values, identical same-inputs -> same-hash
    // behavior).
    scratch.clear();
    // Packed layout key (texture_count | sampler_count<<16 | is_vertex<<31).
    scratch.push_back(uint64_t(uint32_t(texture_count & 0xFFFF) |
                               ((sampler_count & 0x7FFF) << 16) |
                               (is_vertex ? (UINT32_C(1) << 31) : 0)));
    if (texture_count) {
      // Each resolved VkImageView in shader binding order (the handle, never
      // the fetch constant index - texture eviction/recreation re-resolves to a
      // new handle before this point, so a content change always forces a
      // re-write).
      for (const VulkanShader::TextureBinding& texture_binding : *textures) {
        scratch.push_back(uint64_t(reinterpret_cast<uintptr_t>(
            texture_cache->GetActiveBindingOrNullImageView(
                texture_binding.fetch_constant, texture_binding.dimension,
                bool(texture_binding.is_signed)))));
      }
    }
    if (sampler_count) {
      for (uint32_t i = 0; i < sampler_count; ++i) {
        scratch.push_back(
            uint64_t(reinterpret_cast<uintptr_t>(samplers[i].second)));
      }
      // Handle values can be reused: a destroyed sampler's VkSampler value
      // may come back from vkCreateSampler for different parameters, which
      // would collide with the cached hash and leave a descriptor set
      // internally referencing the destroyed object bound. Folding the
      // destroy generation in forces a rewrite after any sampler
      // destruction.
      scratch.push_back(texture_cache->sampler_destroy_generation());
    }
    return XXH3_64bits(scratch.data(), scratch.size() * sizeof(scratch[0]));
  };
  // Computed below when the cache is enabled; reused in the successful-write
  // blocks to store the new cache entry.
  uint64_t texture_set_hash_vertex = 0;
  uint64_t texture_set_hash_pixel = 0;
  if (cvars::vulkan_cache_texture_descriptors) {
    if (cvars::vulkan_texture_descriptor_reuse_edge) {
      // A/B alternative: upstream edge's reuse gate (ported faithfully from
      // edge b5275a6cb UpdateBindings ~lines 6976-6998). A stage's cached set
      // is kept only when none of these changed since the last write: the
      // shader (and thus its binding list), a used texture's resolved host
      // image view (signalled by the texture cache's per-fetch-constant
      // bindings-changed mask), or any sampler in the stage's vector. The
      // content-hash (FNV) is intentionally NOT computed here so edge mode's
      // gate cost equals edge's gate only.
      uint32_t textures_changed = texture_cache_->texture_bindings_changed();
      uint32_t used_texture_mask_vertex =
          vertex_shader->GetUsedTextureMaskAfterTranslation();
      uint32_t used_texture_mask_pixel =
          pixel_shader ? pixel_shader->GetUsedTextureMaskAfterTranslation() : 0;
      if ((current_graphics_descriptor_set_values_up_to_date_ &
           (UINT32_C(1)
            << SpirvShaderTranslator::kDescriptorSetTexturesVertex)) &&
          (current_textures_vertex_shader_ != vertex_shader ||
           (textures_changed & used_texture_mask_vertex) ||
           current_written_samplers_vertex_ != current_samplers_vertex_)) {
        current_graphics_descriptor_set_values_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
      }
      if ((current_graphics_descriptor_set_values_up_to_date_ &
           (UINT32_C(1)
            << SpirvShaderTranslator::kDescriptorSetTexturesPixel)) &&
          (current_textures_pixel_shader_ != pixel_shader ||
           (textures_changed & used_texture_mask_pixel) ||
           current_written_samplers_pixel_ != current_samplers_pixel_)) {
        current_graphics_descriptor_set_values_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
      }
      texture_cache_->ResetTextureBindingsChanged(used_texture_mask_vertex |
                                                  used_texture_mask_pixel);
    } else {
    // XenDroid's content-hash gate (default).
    // Vertex stage.
    if (texture_count_vertex || sampler_count_vertex) {
      texture_set_hash_vertex = compute_texture_set_hash(
          texture_count_vertex, sampler_count_vertex, true, &textures_vertex,
          texture_cache_.get(), current_samplers_vertex_.data());
      if (!current_texture_descriptor_set_hash_valid_vertex_ ||
          texture_set_hash_vertex !=
              current_texture_descriptor_set_hash_vertex_) {
        current_graphics_descriptor_set_values_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
      }
    }
    // Pixel stage.
    if (texture_count_pixel || sampler_count_pixel) {
      texture_set_hash_pixel = compute_texture_set_hash(
          texture_count_pixel, sampler_count_pixel, false, textures_pixel,
          texture_cache_.get(),
          samplers_pixel ? current_samplers_pixel_.data() : nullptr);
      if (!current_texture_descriptor_set_hash_valid_pixel_ ||
          texture_set_hash_pixel !=
              current_texture_descriptor_set_hash_pixel_) {
        current_graphics_descriptor_set_values_up_to_date_ &= ~(
            UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
      }
    }
    }
  } else {
    // Opt-out: reproduce the original unconditional clear (re-write + re-bind
    // the texture/sampler descriptor sets every draw) for A/B debugging.
    current_graphics_descriptor_set_values_up_to_date_ &=
        ~((UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex) |
          (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel));
  }

  // Make sure new descriptor sets are bound to the command buffer.

  current_graphics_descriptor_sets_bound_up_to_date_ &=
      current_graphics_descriptor_set_values_up_to_date_;

  // Fill the texture and sampler write image infos.

  bool write_vertex_textures =
      (texture_count_vertex || sampler_count_vertex) &&
      !(current_graphics_descriptor_set_values_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex));
  bool write_pixel_textures =
      (texture_count_pixel || sampler_count_pixel) &&
      !(current_graphics_descriptor_set_values_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel));
  // When a needed texture set is skipped (not re-written this draw), the cached
  // VkDescriptorSet must still be a real, allocated set - never the frame-open
  // VK_NULL_HANDLE memset value - or the bind loop below would bind a null
  // handle. A missed hash-validity reset would surface here as a fail-fast.
  assert_true(write_vertex_textures ||
              !(texture_count_vertex || sampler_count_vertex) ||
              current_graphics_descriptor_sets_
                      [SpirvShaderTranslator::kDescriptorSetTexturesVertex] !=
                  VK_NULL_HANDLE);
  assert_true(write_pixel_textures ||
              !(texture_count_pixel || sampler_count_pixel) ||
              current_graphics_descriptor_sets_
                      [SpirvShaderTranslator::kDescriptorSetTexturesPixel] !=
                  VK_NULL_HANDLE);
  descriptor_write_image_info_.clear();
  descriptor_write_image_info_.reserve(
      (write_vertex_textures ? texture_count_vertex + sampler_count_vertex
                             : 0) +
      (write_pixel_textures ? texture_count_pixel + sampler_count_pixel : 0));
  size_t vertex_texture_image_info_offset = descriptor_write_image_info_.size();
  if (write_vertex_textures && texture_count_vertex) {
    for (const VulkanShader::TextureBinding& texture_binding :
         textures_vertex) {
      VkDescriptorImageInfo& descriptor_image_info =
          descriptor_write_image_info_.emplace_back();
      descriptor_image_info.imageView =
          texture_cache_->GetActiveBindingOrNullImageView(
              texture_binding.fetch_constant, texture_binding.dimension,
              bool(texture_binding.is_signed));
      descriptor_image_info.imageLayout =
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
  }
  size_t vertex_sampler_image_info_offset = descriptor_write_image_info_.size();
  if (write_vertex_textures && sampler_count_vertex) {
    for (const std::pair<VulkanTextureCache::SamplerParameters, VkSampler>&
             sampler_pair : current_samplers_vertex_) {
      VkDescriptorImageInfo& descriptor_image_info =
          descriptor_write_image_info_.emplace_back();
      descriptor_image_info.sampler = sampler_pair.second;
    }
  }
  size_t pixel_texture_image_info_offset = descriptor_write_image_info_.size();
  if (write_pixel_textures && texture_count_pixel) {
    for (const VulkanShader::TextureBinding& texture_binding :
         *textures_pixel) {
      VkDescriptorImageInfo& descriptor_image_info =
          descriptor_write_image_info_.emplace_back();
      descriptor_image_info.imageView =
          texture_cache_->GetActiveBindingOrNullImageView(
              texture_binding.fetch_constant, texture_binding.dimension,
              bool(texture_binding.is_signed));
      descriptor_image_info.imageLayout =
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
  }
  size_t pixel_sampler_image_info_offset = descriptor_write_image_info_.size();
  if (write_pixel_textures && sampler_count_pixel) {
    for (const std::pair<VulkanTextureCache::SamplerParameters, VkSampler>&
             sampler_pair : current_samplers_pixel_) {
      VkDescriptorImageInfo& descriptor_image_info =
          descriptor_write_image_info_.emplace_back();
      descriptor_image_info.sampler = sampler_pair.second;
    }
  }

  // Write the new descriptor sets.

  // Consecutive bindings updated via a single VkWriteDescriptorSet must have
  // identical stage flags, but for the constants they vary. Plus vertex and
  // pixel texture images and samplers.
  std::array<VkWriteDescriptorSet,
             SpirvShaderTranslator::kConstantBufferCount + 2 * 2>
      write_descriptor_sets;
  uint32_t write_descriptor_set_count = 0;
  uint32_t write_descriptor_set_bits = 0;
  // Dynamic-constants descriptor buffer infos. These must outlive the single
  // deferred vkUpdateDescriptorSets below (each write_constants.pBufferInfo
  // points into this array), so they live at the same scope as
  // write_descriptor_sets rather than inside the rewrite branch.
  VkDescriptorBufferInfo constants_buffer_infos_dynamic
      [SpirvShaderTranslator::kConstantBufferCount];
  assert_not_zero(
      current_graphics_descriptor_set_values_up_to_date_ &
      (UINT32_C(1)
       << SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram));
  // Constant buffers.
  // The values bit was cleared at the upload site (above) whenever ANY of the 5
  // buffers' content changed, which is what forces the per-draw re-bind below.
  // With UNIFORM_BUFFER_DYNAMIC we DECOUPLE the descriptor WRITE from the bind:
  // the set is re-allocated and re-written only when a binding's backing
  // VkBuffer changed (upload-pool page rollover); otherwise the same set is
  // re-bound with fresh dynamic offsets. With plain UNIFORM_BUFFER (gate off)
  // the original unconditional alloc+write+bind runs.
  if (!(current_graphics_descriptor_set_values_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants))) {
    bool rewrite_needed = true;
    if (use_dynamic_constants_) {
      rewrite_needed = !constants_descriptor_set_valid_;
      for (uint32_t i = 0;
           !rewrite_needed && i < SpirvShaderTranslator::kConstantBufferCount;
           ++i) {
        // Re-write on page rollover (buffer changed) OR a range change. The
        // descriptor's range is baked at write time and only the dynamic offset
        // is re-bound per draw, so a stale (too small) range from a prior
        // shader with fewer float constants would silently clamp the current
        // shader's reads. See constants_descriptor_set_ranges_ in the header.
        if (current_constant_buffer_infos_[i].buffer !=
                constants_descriptor_set_buffers_[i] ||
            current_constant_buffer_infos_[i].range !=
                constants_descriptor_set_ranges_[i]) {
          rewrite_needed = true;
        }
      }
    }
    if (rewrite_needed) {
      const VkDescriptorType constants_descriptor_type =
          use_dynamic_constants_ ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                 : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      VkDescriptorSet constants_descriptor_set;
      if (!constants_transient_descriptors_free_.empty()) {
        constants_descriptor_set = constants_transient_descriptors_free_.back();
        constants_transient_descriptors_free_.pop_back();
      } else {
        VkDescriptorPoolSize constants_descriptor_count;
        // For UNIFORM_BUFFER_DYNAMIC this type is NOT present in the shared
        // allocator's pool sizes (kept at UNIFORM_BUFFER for the compute uniform
        // path), so LinkedTypeDescriptorSetAllocator::Allocate falls back to a
        // dedicated VkDescriptorPool per allocation. That is spec-legal, and
        // bounded: with the dynamic path we allocate only on page rollover
        // (0-2x/frame), not per draw, so the dedicated-pool churn is rare. Do
        // NOT retype the shared pool to fix this - the compute path needs the
        // UNIFORM_BUFFER pool size.
        constants_descriptor_count.type = constants_descriptor_type;
        constants_descriptor_count.descriptorCount =
            SpirvShaderTranslator::kConstantBufferCount;
        constants_descriptor_set =
            transient_descriptor_allocator_uniform_buffer_.Allocate(
                descriptor_set_layout_constants_, &constants_descriptor_count,
                1);
        if (constants_descriptor_set == VK_NULL_HANDLE) {
          return false;
        }
      }
      constants_transient_descriptors_used_.emplace_back(
          frame_current_, constants_descriptor_set);
      // For UNIFORM_BUFFER_DYNAMIC the descriptor base offset MUST be 0: the
      // per-buffer slice offset is supplied as a dynamic offset at bind time, so
      // baking it into the descriptor as well would double-count it. The range
      // stays the per-buffer data size (<= 4 KiB, well below the
      // maxUniformBufferRange spec floor of 16384). With plain UNIFORM_BUFFER
      // the original {buffer, slice_offset, range} info is used directly.
      // (constants_buffer_infos_dynamic is declared at function scope above so
      // it outlives the deferred vkUpdateDescriptorSets.)
      if (use_dynamic_constants_) {
        for (uint32_t i = 0; i < SpirvShaderTranslator::kConstantBufferCount;
             ++i) {
          assert_true(current_constant_buffer_infos_[i].range <=
                      VkDeviceSize(GetVulkanDevice()
                                       ->properties()
                                       .maxUniformBufferRange));
          constants_buffer_infos_dynamic[i].buffer =
              current_constant_buffer_infos_[i].buffer;
          constants_buffer_infos_dynamic[i].offset = 0;
          constants_buffer_infos_dynamic[i].range =
              current_constant_buffer_infos_[i].range;
        }
      }
      // Consecutive bindings updated via a single VkWriteDescriptorSet must have
      // identical stage flags, but for the constants they vary.
      for (uint32_t i = 0; i < SpirvShaderTranslator::kConstantBufferCount;
           ++i) {
        VkWriteDescriptorSet& write_constants =
            write_descriptor_sets[write_descriptor_set_count++];
        write_constants.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_constants.pNext = nullptr;
        write_constants.dstSet = constants_descriptor_set;
        write_constants.dstBinding = i;
        write_constants.dstArrayElement = 0;
        write_constants.descriptorCount = 1;
        write_constants.descriptorType = constants_descriptor_type;
        write_constants.pImageInfo = nullptr;
        write_constants.pBufferInfo =
            use_dynamic_constants_ ? &constants_buffer_infos_dynamic[i]
                                   : &current_constant_buffer_infos_[i];
        write_constants.pTexelBufferView = nullptr;
      }
      write_descriptor_set_bits |=
          UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants;
      current_graphics_descriptor_sets_
          [SpirvShaderTranslator::kDescriptorSetConstants] =
              constants_descriptor_set;
      if (use_dynamic_constants_) {
        for (uint32_t i = 0; i < SpirvShaderTranslator::kConstantBufferCount;
             ++i) {
          constants_descriptor_set_buffers_[i] =
              current_constant_buffer_infos_[i].buffer;
          constants_descriptor_set_ranges_[i] =
              current_constant_buffer_infos_[i].range;
        }
        constants_descriptor_set_valid_ = true;
      }
    } else {
      // Dynamic skip-rewrite path: the cached set's bindings still reference the
      // current page buffers, so only the dynamic offsets (bound below) change.
      // Re-set the values bit (do NOT push to the used list - the set was
      // already accounted for when written) so an unchanged subsequent draw
      // keeps the bit and skips even the re-bind, matching the texture-cache
      // no-rebind behavior. A content change will clear it again at the upload
      // site, re-triggering the rebind with fresh offsets.
      assert_true(current_graphics_descriptor_sets_
                      [SpirvShaderTranslator::kDescriptorSetConstants] !=
                  VK_NULL_HANDLE);
      write_descriptor_set_bits |=
          UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants;
    }
  }
  // Vertex shader textures and samplers.
  if (write_vertex_textures) {
    VkWriteDescriptorSet* write_textures =
        write_descriptor_sets.data() + write_descriptor_set_count;
    uint32_t texture_descriptor_set_write_count = WriteTransientTextureBindings(
        true, texture_count_vertex, sampler_count_vertex,
        current_guest_graphics_pipeline_layout_
            ->descriptor_set_layout_textures_vertex_ref(),
        descriptor_write_image_info_.data() + vertex_texture_image_info_offset,
        descriptor_write_image_info_.data() + vertex_sampler_image_info_offset,
        write_textures);
    if (!texture_descriptor_set_write_count) {
      return false;
    }
    write_descriptor_set_count += texture_descriptor_set_write_count;
    write_descriptor_set_bits |=
        UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex;
    current_graphics_descriptor_sets_
        [SpirvShaderTranslator::kDescriptorSetTexturesVertex] =
            write_textures[0].dstSet;
    // Dual write-back so a live cvar flip between the two reuse gates is safe:
    // after this write BOTH schemes' state describe what was just written.
    // Always refresh edge's shader-pointer + sampler-vector snapshot (cheap).
    current_textures_vertex_shader_ = vertex_shader;
    current_written_samplers_vertex_ = current_samplers_vertex_;
    if (cvars::vulkan_texture_descriptor_reuse_edge) {
      // Edge mode didn't compute the FNV hash, so mark it invalid to force a
      // rebuild if the cvar is later flipped back to the hash gate.
      current_texture_descriptor_set_hash_valid_vertex_ = false;
    } else {
      // Hash mode: record the content hash of the set we just wrote so an
      // unchanged next draw can skip the re-write+re-bind. Valid only until an
      // invalidation (new frame / transient-pool reset / pipeline-layout
      // change).
      current_texture_descriptor_set_hash_vertex_ = texture_set_hash_vertex;
      current_texture_descriptor_set_hash_valid_vertex_ = true;
    }
  }
  // Pixel shader textures and samplers.
  if (write_pixel_textures) {
    VkWriteDescriptorSet* write_textures =
        write_descriptor_sets.data() + write_descriptor_set_count;
    uint32_t texture_descriptor_set_write_count = WriteTransientTextureBindings(
        false, texture_count_pixel, sampler_count_pixel,
        current_guest_graphics_pipeline_layout_
            ->descriptor_set_layout_textures_pixel_ref(),
        descriptor_write_image_info_.data() + pixel_texture_image_info_offset,
        descriptor_write_image_info_.data() + pixel_sampler_image_info_offset,
        write_textures);
    if (!texture_descriptor_set_write_count) {
      return false;
    }
    write_descriptor_set_count += texture_descriptor_set_write_count;
    write_descriptor_set_bits |=
        UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel;
    current_graphics_descriptor_sets_
        [SpirvShaderTranslator::kDescriptorSetTexturesPixel] =
            write_textures[0].dstSet;
    // Dual write-back (see the vertex stage above for the switch-safety
    // rationale): always refresh edge's snapshot, then keep the hash state
    // either correct-and-valid (hash mode) or marked invalid (edge mode).
    current_textures_pixel_shader_ = pixel_shader;
    current_written_samplers_pixel_ = current_samplers_pixel_;
    if (cvars::vulkan_texture_descriptor_reuse_edge) {
      current_texture_descriptor_set_hash_valid_pixel_ = false;
    } else {
      current_texture_descriptor_set_hash_pixel_ = texture_set_hash_pixel;
      current_texture_descriptor_set_hash_valid_pixel_ = true;
    }
  }
  // Write.
  if (write_descriptor_set_count) {
    dfn.vkUpdateDescriptorSets(device, write_descriptor_set_count,
                               write_descriptor_sets.data(), 0, nullptr);
  }
  // Only make valid if all descriptor sets have been allocated and written
  // successfully.
  current_graphics_descriptor_set_values_up_to_date_ |=
      write_descriptor_set_bits;

  // Bind the new descriptor sets.
  uint32_t descriptor_sets_needed =
      (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetCount) - 1;
  if (!texture_count_vertex && !sampler_count_vertex) {
    descriptor_sets_needed &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
  }
  if (!texture_count_pixel && !sampler_count_pixel) {
    descriptor_sets_needed &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
  }
  uint32_t descriptor_sets_remaining =
      descriptor_sets_needed &
      ~current_graphics_descriptor_sets_bound_up_to_date_;
  uint32_t descriptor_set_index;
  while (
      xe::bit_scan_forward(descriptor_sets_remaining, &descriptor_set_index)) {
    uint32_t descriptor_set_mask_tzcnt =
        xe::tzcnt(~(descriptor_sets_remaining |
                    ((UINT32_C(1) << descriptor_set_index) - 1)));
    // When the constants set is bound as UNIFORM_BUFFER_DYNAMIC, supply one
    // dynamic offset per binding (in (set, binding) order). pDynamicOffsets
    // covers the dynamic descriptors of ALL sets in the bound range; the
    // constants set is the ONLY dynamic set in the pipeline layout
    // (shared-memory/edram is STORAGE_BUFFER, textures are SAMPLED_IMAGE/
    // SAMPLER), so any run that includes it contributes exactly these 5 offsets
    // and every other set in the run contributes zero - dynamicOffsetCount is
    // always kConstantBufferCount and the array maps 1:1 onto the constants
    // bindings (set index kDescriptorSetConstants comes before the texture sets,
    // so no earlier set in the run can be dynamic). Each offset is the aligned
    // slice offset from uniform_buffer_pool_->Request() (a multiple of
    // minUniformBufferOffsetAlignment), with slice_offset + range <= page size
    // guaranteed by the pool, so descriptor.offset(0) + dynamicOffset + range <=
    // buffer size holds.
    uint32_t dynamic_offset_count = 0;
    const uint32_t* dynamic_offsets = nullptr;
    uint32_t constants_dynamic_offsets
        [SpirvShaderTranslator::kConstantBufferCount];
    if (use_dynamic_constants_ &&
        descriptor_set_index <=
            SpirvShaderTranslator::kDescriptorSetConstants &&
        SpirvShaderTranslator::kDescriptorSetConstants <
            descriptor_set_mask_tzcnt) {
      for (uint32_t i = 0; i < SpirvShaderTranslator::kConstantBufferCount;
           ++i) {
        constants_dynamic_offsets[i] =
            uint32_t(current_constant_buffer_infos_[i].offset);
      }
      dynamic_offset_count = SpirvShaderTranslator::kConstantBufferCount;
      dynamic_offsets = constants_dynamic_offsets;
    }
    // TODO(Triang3l): Bind to compute for memexport emulation without vertex
    // shader memory stores.
    deferred_command_buffer_.CmdVkBindDescriptorSets(
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        current_guest_graphics_pipeline_layout_->GetPipelineLayout(),
        descriptor_set_index, descriptor_set_mask_tzcnt - descriptor_set_index,
        current_graphics_descriptor_sets_ + descriptor_set_index,
        dynamic_offset_count, dynamic_offsets);
    if (descriptor_set_mask_tzcnt >= 32) {
      break;
    }
    descriptor_sets_remaining &=
        ~((UINT32_C(1) << descriptor_set_mask_tzcnt) - 1);
  }
  current_graphics_descriptor_sets_bound_up_to_date_ |= descriptor_sets_needed;

  return true;
}

uint32_t VulkanCommandProcessor::WriteTransientTextureBindings(
    bool is_vertex, uint32_t texture_count, uint32_t sampler_count,
    VkDescriptorSetLayout descriptor_set_layout,
    const VkDescriptorImageInfo* texture_image_info,
    const VkDescriptorImageInfo* sampler_image_info,
    VkWriteDescriptorSet* descriptor_set_writes_out) {
  assert_true(frame_open_);
  if (!texture_count && !sampler_count) {
    return 0;
  }
  TextureDescriptorSetLayoutKey texture_descriptor_set_layout_key;
  texture_descriptor_set_layout_key.texture_count = texture_count;
  texture_descriptor_set_layout_key.sampler_count = sampler_count;
  texture_descriptor_set_layout_key.is_vertex = uint32_t(is_vertex);
  VkDescriptorSet texture_descriptor_set;
  auto textures_free_it = texture_transient_descriptor_sets_free_.find(
      texture_descriptor_set_layout_key);
  if (textures_free_it != texture_transient_descriptor_sets_free_.end() &&
      !textures_free_it->second.empty()) {
    texture_descriptor_set = textures_free_it->second.back();
    textures_free_it->second.pop_back();
  } else {
    std::array<VkDescriptorPoolSize, 2> texture_descriptor_counts;
    uint32_t texture_descriptor_counts_count = 0;
    if (texture_count) {
      VkDescriptorPoolSize& texture_descriptor_count =
          texture_descriptor_counts[texture_descriptor_counts_count++];
      texture_descriptor_count.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      texture_descriptor_count.descriptorCount = texture_count;
    }
    if (sampler_count) {
      VkDescriptorPoolSize& texture_descriptor_count =
          texture_descriptor_counts[texture_descriptor_counts_count++];
      texture_descriptor_count.type = VK_DESCRIPTOR_TYPE_SAMPLER;
      texture_descriptor_count.descriptorCount = sampler_count;
    }
    assert_not_zero(texture_descriptor_counts_count);
    texture_descriptor_set = transient_descriptor_allocator_textures_.Allocate(
        descriptor_set_layout, texture_descriptor_counts.data(),
        texture_descriptor_counts_count);
    if (texture_descriptor_set == VK_NULL_HANDLE) {
      return 0;
    }
  }
  UsedTextureTransientDescriptorSet& used_texture_descriptor_set =
      texture_transient_descriptor_sets_used_.emplace_back();
  used_texture_descriptor_set.frame = frame_current_;
  used_texture_descriptor_set.layout = texture_descriptor_set_layout_key;
  used_texture_descriptor_set.set = texture_descriptor_set;
  uint32_t descriptor_set_write_count = 0;
  if (texture_count) {
    VkWriteDescriptorSet& descriptor_set_write =
        descriptor_set_writes_out[descriptor_set_write_count++];
    descriptor_set_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_set_write.pNext = nullptr;
    descriptor_set_write.dstSet = texture_descriptor_set;
    descriptor_set_write.dstBinding = 0;
    descriptor_set_write.dstArrayElement = 0;
    descriptor_set_write.descriptorCount = texture_count;
    descriptor_set_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_set_write.pImageInfo = texture_image_info;
    descriptor_set_write.pBufferInfo = nullptr;
    descriptor_set_write.pTexelBufferView = nullptr;
  }
  if (sampler_count) {
    VkWriteDescriptorSet& descriptor_set_write =
        descriptor_set_writes_out[descriptor_set_write_count++];
    descriptor_set_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_set_write.pNext = nullptr;
    descriptor_set_write.dstSet = texture_descriptor_set;
    descriptor_set_write.dstBinding = texture_count;
    descriptor_set_write.dstArrayElement = 0;
    descriptor_set_write.descriptorCount = sampler_count;
    descriptor_set_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptor_set_write.pImageInfo = sampler_image_info;
    descriptor_set_write.pBufferInfo = nullptr;
    descriptor_set_write.pTexelBufferView = nullptr;
  }
  assert_not_zero(descriptor_set_write_count);
  return descriptor_set_write_count;
}

#define COMMAND_PROCESSOR VulkanCommandProcessor
#include "../pm4_command_processor_implement.h"
#undef COMMAND_PROCESSOR
}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

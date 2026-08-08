/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_D3D12_SPIRV_TO_DXIL_COMPILER_H_
#define XENIA_GPU_D3D12_SPIRV_TO_DXIL_COMPILER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace d3d12 {

// Wraps Mesa's spirv_to_dxil (NIR-based SPIR-V to DXIL) for the D3D12 guest
// shader path. Produces DXIL from the SPIR-V emitted by SpirvShaderTranslator.
//
// Supports both render target cache paths. The Mesa fork lowers SPIR-V fragment
// shader interlock to D3D12 rasterizer-ordered views, so the EDRAM ROV path
// goes through this route as well (the EDRAM and ZPD counter UAVs become ROVs).
class SpirvToDxilCompiler {
 public:
  // Mesa git revision of the linked library.
  static uint64_t version();

  // Pipeline stage of the SPIR-V module. Covers the guest vertex/pixel shaders
  // and the host primitive-expansion geometry and tessellation shaders.
  enum class Stage {
    kVertex,
    kTessellationControl,
    kTessellationEvaluation,
    kGeometry,
    kPixel,
  };

  // Translates one SPIR-V module to signed DXIL for the given stage. When
  // lower_to_bindless is set, all descriptor-set resources are lowered to
  // SM 6.6 dynamic resource heap indexing (matching the Dozen driver), for the
  // fully bindless guest path. input_clip_size is the producer's clip distance
  // count, needed by stages that read clip inputs (geometry, tessellation) so
  // the clip / cull split matches the producer. Returns an empty vector on
  // failure and logs the cause.
  static std::vector<uint8_t> Translate(const uint32_t* spirv_words,
                                        size_t spirv_word_count, Stage stage,
                                        bool lower_to_bindless = false,
                                        uint32_t input_clip_size = 0);

  // One SPIR-V stage for TranslateLinked.
  struct LinkedStage {
    const uint32_t* spirv_words;
    size_t spirv_word_count;
    Stage stage;
  };

  // Translates several SPIR-V stages of one pipeline together with cross-stage
  // linking, so the inter-stage signatures match exactly as D3D12 requires (the
  // way tessellation hull and domain shaders reconcile control point counts and
  // patch constants). Stages must be in pipeline order (vertex first). Returns
  // one signed DXIL blob per input stage, or an empty vector on failure.
  static std::vector<std::vector<uint8_t>> TranslateLinked(
      const std::vector<LinkedStage>& stages, bool lower_to_bindless = false);
};

}  // namespace d3d12
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_D3D12_SPIRV_TO_DXIL_COMPILER_H_

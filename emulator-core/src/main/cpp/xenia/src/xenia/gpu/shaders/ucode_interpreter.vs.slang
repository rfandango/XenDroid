// XE_NO_MSL
// Xenos VS ucode interpreter - placeholder vertex shader.
//
// Renders the real geometry of a guest vertex shader by interpreting its ucode
// while the real shader translates+compiles in the background. The pipeline
// cache hot-swaps to the compiled VS when ready. Built once, reused for every
// guest VS. Single source, compiled to SPIR-V (Vulkan) and DXIL (D3D12).
//
// Reads the same descriptor bindings a translated guest VS uses (see
// SpirvShaderTranslator DescriptorSet / ConstantBuffer):
//   set 1 binding 0 system constants (raw view, ucode location + NDC + index)
//   set 1 binding 1 float constants (bound FULL 256 for interpreter, not packed)
//   set 1 binding 3 bool/loop, binding 4 fetch constants
//   set 0 binding 0 shared memory (guest RAM) - both the vertex data AND ucode,
//     so no separate attachment is needed. Shared memory is guest-endian, so
//     ucode dwords are byte-swapped to host order here.
// Interpreter logic is shared with the D3D12 front-end via
// ucode_interpreter_core.hlsli.

#include "../../ui/shaders/xesl.xesli"  // XE_VK_BINDING (SPIR-V set/binding + HLSL register)

#define REG_COUNT 64
#define MAX_INTERP 16

// System constants read as a raw std140 uint4 array, indexed by byte-offset / 16
// so no field-packing assumptions are needed. Only the fields below are touched:
//   [0].x flags   [0].w vertex_index_endian   [1].x vertex_base_index
//   [2].xyz ndc_scale   [3].xyz ndc_offset
//   [34].x interpreter_ucode_base_dwords   [34].y interpreter_cf_instr_count
// DXIL registers match the D3D12 Mesa (spirv_to_dxil) root signature: guest
// constant buffers live in space1 (system b0, float-vertex b1, bool/loop b3,
// fetch b4) and shared memory is the t0 space0 SRV. On Vulkan the register()
// is ignored, the [[vk::binding]] set/binding from XE_VK_BINDING is used.
XE_VK_BINDING(1, 0) cbuffer xe_system_constants : register(b0, space1) { uint4 xe_sys[35]; };
XE_VK_BINDING(1, 1) cbuffer xe_float_constants : register(b1, space1) { float4 xe_float[256]; };
XE_VK_BINDING(1, 3) cbuffer xe_bool_loop_constants : register(b3, space1) { uint4 xe_bool_loop[10]; }; // bool[2]+loop[8]
XE_VK_BINDING(1, 4) cbuffer xe_fetch_constants : register(b4, space1) { uint4 xe_fetch[48]; };          // 32*6 dwords
XE_VK_BINDING(0, 0) ByteAddressBuffer xe_shared_memory : register(t0, space0);

// System-constant field accessors.
uint XeFlags() { return xe_sys[0].x; }
uint XeVertexIndexEndian() { return xe_sys[0].w; }
int XeVertexBaseIndex() { return asint(xe_sys[1].x); }
float3 XeNdcScale() { return asfloat(xe_sys[2].xyz); }
float3 XeNdcOffset() { return asfloat(xe_sys[3].xyz); }
uint XeUcodeBaseDwords() { return xe_sys[34].x; }
uint XeCfInstrCount() { return xe_sys[34].y; }

uint XeBswap32(uint v) {
  return (v >> 24) | ((v >> 8) & 0xFF00) | ((v << 8) & 0xFF0000) | (v << 24);
}
uint EndianSwap(uint v, uint e) {            // 1=8in16, 2=8in32, 3=16in32
  if (e == 1) return ((v & 0x00FF00FF) << 8) | ((v >> 8) & 0x00FF00FF);
  if (e == 2) return XeBswap32(v);
  if (e == 3) return (v >> 16) | (v << 16);
  return v;
}

// ucode lives in shared memory at the guest program address (guest-endian).
uint UcodeLoad(uint dword) {
  return XeBswap32(xe_shared_memory.Load((XeUcodeBaseDwords() + dword) * 4));
}
float4 FloatConst(uint index) { return xe_float[index]; }
bool BoolConst(uint a) {
  uint word = a >> 5;
  return ((xe_bool_loop[word >> 2][word & 3] >> (a & 31)) & 1) != 0;
}
uint4 GatherVertexDwords(uint fetch_const_index, uint index, uint stride, int offset) {
  uint fc = (fetch_const_index * 2);
  uint fc0 = xe_fetch[fc >> 2][fc & 3];
  uint fc1 = xe_fetch[(fc + 1) >> 2][(fc + 1) & 3];
  uint base = fc0 >> 2;          // dword address (type:2 + address:30)
  uint endian = fc1 & 3;
  uint addr = base + index * stride + uint(offset);
  return uint4(EndianSwap(xe_shared_memory.Load(addr * 4), endian),
               EndianSwap(xe_shared_memory.Load((addr + 1) * 4), endian),
               EndianSwap(xe_shared_memory.Load((addr + 2) * 4), endian),
               EndianSwap(xe_shared_memory.Load((addr + 3) * 4), endian));
}

#include "ucode_interpreter_core.hlsli"

// System flags (SpirvShaderTranslator kSysFlag_* bit values).
#define XE_SYS_FLAG_XY_DIVIDED_BY_W 0x8u
#define XE_SYS_FLAG_Z_DIVIDED_BY_W  0x10u
#define XE_SYS_FLAG_W_NOT_RECIPROCAL 0x20u

struct VSOut {
  float4 pos : SV_Position;
};

VSOut main(uint vid : SV_VertexID) {
  // Match the SPIR-V VS r0 seed: endian-swapped host index plus the base index.
  int index = int(EndianSwap(vid, XeVertexIndexEndian())) + XeVertexBaseIndex();
  XeRunInterp(XeCfInstrCount(), float(index));

  // Guest clip-space position to host NDC, matching the translated VS output.
  uint flags = XeFlags();
  float4 p = xe_out_pos;
  float w = p.w;
  if ((flags & XE_SYS_FLAG_W_NOT_RECIPROCAL) == 0u) w = 1.0 / w; // guest gave 1/W
  float2 xy = p.xy;
  if ((flags & XE_SYS_FLAG_XY_DIVIDED_BY_W) != 0u) xy = xy * w;  // revert XY/W
  float z = p.z;
  if ((flags & XE_SYS_FLAG_Z_DIVIDED_BY_W) != 0u) z = z * w;     // revert Z/W
  float3 xyz = float3(xy, z) * XeNdcScale() + XeNdcOffset() * w;

  // Vertex kill: a non-zero oPts.z degenerates the vertex to NaN (OR-mode cull).
  if ((asuint(xe_kill) & 0x7FFFFFFFu) != 0u) w = asfloat(0x7FC00000u);

  VSOut o;
  o.pos = float4(xyz, w);
  return o;
}

#version 460

// GLSL twin of ucode_interpreter.vs.slang (+ ucode_interpreter_core.hlsli) for
// the Android Vulkan build.
//
// The .slang/.hlsli forms are HLSL for the DXIL (spirv_to_dxil) path, which
// glslangValidator can't parse. gen_android_spirv.py prefers this .glsl twin
// over the .slang, so it produces the ucode_interpreter_vs.h SPIR-V bytecode.
// Behavior is kept equivalent to the HLSL source; see it for the annotated
// interpreter logic.
//
// Xenos VS ucode interpreter - placeholder vertex shader. Renders the real
// geometry of a guest vertex shader by interpreting its ucode while the real
// shader translates+compiles in the background; the pipeline cache hot-swaps to
// the compiled VS when ready.
//
// Descriptor bindings match a translated guest VS (SpirvShaderTranslator
// DescriptorSet / ConstantBuffer):
//   set 1 binding 0 system constants (raw uvec4[35] view: ucode location, NDC,
//                   index endian/base)
//   set 1 binding 1 float constants (bound FULL 256 for the interpreter)
//   set 1 binding 3 bool/loop constants
//   set 1 binding 4 fetch constants
//   set 0 binding 0 shared memory (guest RAM) - both vertex data AND ucode.
// Shared memory is guest-endian, so ucode dwords are byte-swapped to host order.

#define REG_COUNT 64u
#define MAX_INTERP 16u
#define MAX_CF_STEPS 1024u

layout(std140, set = 1, binding = 0) uniform xe_system_constants_block {
  uvec4 xe_sys[35];
};
layout(std140, set = 1, binding = 1) uniform xe_float_constants_block {
  vec4 xe_float[256];
};
layout(std140, set = 1, binding = 3) uniform xe_bool_loop_constants_block {
  uvec4 xe_bool_loop[10];  // bool[2] + loop[8]
};
layout(std140, set = 1, binding = 4) uniform xe_fetch_constants_block {
  uvec4 xe_fetch[48];  // 32 * 6 dwords
};
layout(std430, set = 0, binding = 0) readonly buffer xe_shared_memory_block {
  uint xe_shared_memory[];
};

// System-constant field accessors (raw uvec4[] indexed by field).
uint XeFlags() { return xe_sys[0].x; }
uint XeVertexIndexEndian() { return xe_sys[0].w; }
int XeVertexBaseIndex() { return int(xe_sys[1].x); }
vec3 XeNdcScale() { return uintBitsToFloat(xe_sys[2].xyz); }
vec3 XeNdcOffset() { return uintBitsToFloat(xe_sys[3].xyz); }
uint XeUcodeBaseDwords() { return xe_sys[34].x; }
uint XeCfInstrCount() { return xe_sys[34].y; }

uint XeBswap32(uint v) {
  return (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
}
uint EndianSwap(uint v, uint e) {  // 1=8in16, 2=8in32, 3=16in32
  if (e == 1u) return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
  if (e == 2u) return XeBswap32(v);
  if (e == 3u) return (v >> 16) | (v << 16);
  return v;
}

// ucode lives in shared memory at the guest program address (guest-endian).
uint UcodeLoad(uint dword) {
  return XeBswap32(xe_shared_memory[XeUcodeBaseDwords() + dword]);
}
vec4 FloatConst(uint index) { return xe_float[index]; }
bool BoolConst(uint a) {
  uint word = a >> 5;
  return ((xe_bool_loop[word >> 2][word & 3u] >> (a & 31u)) & 1u) != 0u;
}
uvec4 GatherVertexDwords(uint fetch_const_index, uint index, uint stride,
                         int offset) {
  uint fc = fetch_const_index * 2u;
  uint fc0 = xe_fetch[fc >> 2][fc & 3u];
  uint fc1 = xe_fetch[(fc + 1u) >> 2][(fc + 1u) & 3u];
  uint base = fc0 >> 2;  // dword address (type:2 + address:30)
  uint endian = fc1 & 3u;
  uint addr = base + index * stride + uint(offset);
  return uvec4(EndianSwap(xe_shared_memory[addr], endian),
               EndianSwap(xe_shared_memory[addr + 1u], endian),
               EndianSwap(xe_shared_memory[addr + 2u], endian),
               EndianSwap(xe_shared_memory[addr + 3u], endian));
}

// ---------------------------------------------------------------------------
// Interpreter state and core (shared ucode_interpreter_core.hlsli, ported).
// ---------------------------------------------------------------------------
vec4 regs[REG_COUNT];
int xe_a0;
float xe_ps;
bool xe_p0;
vec4 xe_out_pos;
vec4 xe_out_interp[MAX_INTERP];
uint xe_vf_stride;  // vfetch_mini inherits these from the prior full
uint xe_vf_fc;
uint xe_vf_index_reg;
float xe_kill;  // oPts.z vertex-kill value (export register 63)

vec4 RegGet(uint i) { return regs[i]; }
void RegStore(uint i, vec4 v) { regs[i] = v; }
bool Bit(uint v, uint b) { return (v & b) != 0u; }
void WriteMasked(uint i, vec4 v, uint mask) {
  vec4 c = RegGet(i);
  RegStore(i, vec4(Bit(mask, 1u) ? v.x : c.x, Bit(mask, 2u) ? v.y : c.y,
                   Bit(mask, 4u) ? v.z : c.z, Bit(mask, 8u) ? v.w : c.w));
}

float Comp(vec4 v, uint c) {
  return c == 0u ? v.x : (c == 1u ? v.y : (c == 2u ? v.z : v.w));
}
vec4 Swizzle(vec4 v, uint swiz) {
  return vec4(Comp(v, ((swiz >> 0) + 0u) & 3u), Comp(v, ((swiz >> 2) + 1u) & 3u),
              Comp(v, ((swiz >> 4) + 2u) & 3u), Comp(v, ((swiz >> 6) + 3u) & 3u));
}
// sel: register vs constant. const_rel: a0-relative constant.
vec4 Operand(uint reg, uint sel, uint swiz, uint negate, bool const_rel) {
  vec4 src;
  if (sel != 0u) {
    src = RegGet((reg & 0x3Fu) % REG_COUNT);  // no loop-relative reg, aL 0
    if ((reg & 0x80u) != 0u) src = abs(src);  // absolute-value source modifier
  } else {
    uint ci = reg & 0xFFu;
    if (const_rel) ci = uint(clamp(int(ci) + xe_a0, 0, 255));
    src = FloatConst(ci);
  }
  vec4 r = Swizzle(src, swiz & 0xFFu);
  return (negate != 0u) ? -r : r;
}

// vfetch: decode + gather + format conversion.
float SNorm(int x, float maxv) { return clamp(float(x) / maxv, -1.0, 1.0); }
float XeF16(uint v) { return unpackHalf2x16(v).x; }

// Partial format coverage. Unhandled formats fall through to (0,0,0,1).
vec4 ConvertVertex(uint format, bool is_signed, bool is_integer, uvec4 d) {
  vec4 v = vec4(0.0, 0.0, 0.0, 1.0);
  if (format == 57u) {  // 32_32_32_FLOAT
    v.x = uintBitsToFloat(d.x);
    v.y = uintBitsToFloat(d.y);
    v.z = uintBitsToFloat(d.z);
  } else if (format == 38u) {  // 32_32_32_32_FLOAT
    v = vec4(uintBitsToFloat(d.x), uintBitsToFloat(d.y), uintBitsToFloat(d.z),
             uintBitsToFloat(d.w));
  } else if (format == 37u) {  // 32_32_FLOAT
    v.x = uintBitsToFloat(d.x);
    v.y = uintBitsToFloat(d.y);
    v.z = 0.0;
  } else if (format == 36u) {  // 32_FLOAT
    v.x = uintBitsToFloat(d.x);
    v.y = 0.0;
    v.z = 0.0;
  } else if (format == 32u) {  // 16_16_16_16_FLOAT
    v = vec4(XeF16(d.x & 0xFFFFu), XeF16(d.x >> 16), XeF16(d.y & 0xFFFFu),
             XeF16(d.y >> 16));
  } else if (format == 31u) {  // 16_16_FLOAT
    v.x = XeF16(d.x & 0xFFFFu);
    v.y = XeF16(d.x >> 16);
    v.z = 0.0;
  } else if (format == 25u) {  // 16_16
    if (is_signed) {
      int xi = int(d.x << 16) >> 16, yi = int(d.x) >> 16;
      v.x = is_integer ? float(xi) : SNorm(xi, 32767.0);
      v.y = is_integer ? float(yi) : SNorm(yi, 32767.0);
    } else {
      uint xu = d.x & 0xFFFFu, yu = d.x >> 16;
      v.x = is_integer ? float(xu) : float(xu) / 65535.0;
      v.y = is_integer ? float(yu) : float(yu) / 65535.0;
    }
    v.z = 0.0;
  } else if (format == 6u) {  // 8_8_8_8
    for (uint k = 0u; k < 4u; ++k) {
      uint bu = (d.x >> (8u * k)) & 0xFFu;
      if (is_signed) {
        int bi = int(bu << 24) >> 24;
        v[k] = is_integer ? float(bi) : SNorm(bi, 127.0);
      } else {
        v[k] = is_integer ? float(bu) : float(bu) / 255.0;
      }
    }
  }
  return v;
}

// Assumes vfetch. tfetch would mis-decode here, gated out upstream by textures.
void ExecVfetch(uint off) {
  uint w0 = UcodeLoad(off * 3u + 0u);
  uint w1 = UcodeLoad(off * 3u + 1u);
  uint w2 = UcodeLoad(off * 3u + 2u);
  uint src_reg = (w0 >> 5) & 0x3Fu;
  uint dst_reg = (w0 >> 12) & 0x3Fu;
  uint const_index = (w0 >> 20) & 0x1Fu;
  uint const_index_sel = (w0 >> 25) & 0x3u;
  uint src_swiz = (w0 >> 30) & 0x3u;
  uint dst_swiz = (w1 >> 0) & 0xFFFu;
  bool is_signed = Bit(w1, 1u << 12);
  bool is_integer = Bit(w1, 1u << 13);
  uint format = (w1 >> 16) & 0x3Fu;
  bool is_mini = Bit(w1, 1u << 30);
  uint stride = is_mini ? xe_vf_stride : (w2 & 0xFFu);
  uint fc_index = is_mini ? xe_vf_fc : (const_index * 3u + const_index_sel);
  uint index_reg = is_mini ? xe_vf_index_reg : src_reg;
  if (!is_mini) {
    xe_vf_stride = stride;
    xe_vf_fc = fc_index;
    xe_vf_index_reg = index_reg;
  }
  int offset = int((w2 >> 8) & 0x7FFFFFu);

  uint index = uint(max(Comp(RegGet(index_reg % REG_COUNT), src_swiz & 3u), 0.0));
  uvec4 d = GatherVertexDwords(fc_index, index, stride, offset);
  vec4 v = ConvertVertex(format, is_signed, is_integer, d);

  // Destination swizzle (3 bits/comp: 0-3 = xyzw, 4 = 0, 5 = 1, 7 = keep).
  vec4 res = RegGet(dst_reg % REG_COUNT);
  for (uint c = 0u; c < 4u; ++c) {
    uint s = (dst_swiz >> (3u * c)) & 7u;
    if (s < 4u) {
      res[c] = v[s];
    } else if (s == 4u) {
      res[c] = 0.0;
    } else if (s == 5u) {
      res[c] = 1.0;
    }
  }
  RegStore(dst_reg % REG_COUNT, res);
}

// ALU.
void ExecAlu(uint off) {
  uint w0 = UcodeLoad(off * 3u + 0u);
  uint w1 = UcodeLoad(off * 3u + 1u);
  uint w2 = UcodeLoad(off * 3u + 2u);

  uint vector_dest = (w0 >> 0) & 0x3Fu;
  uint scalar_dest = (w0 >> 8) & 0x3Fu;
  uint export_data = (w0 >> 15) & 1u;
  uint vector_write_mask = (w0 >> 16) & 0xFu;
  uint scalar_write_mask = (w0 >> 20) & 0xFu;
  bool vector_clamp = Bit(w0, 1u << 24);
  bool scalar_clamp = Bit(w0, 1u << 25);
  uint scalar_opc = (w0 >> 26) & 0x3Fu;

  uint src3_swiz = (w1 >> 0) & 0xFFu;
  uint src2_swiz = (w1 >> 8) & 0xFFu;
  uint src1_swiz = (w1 >> 16) & 0xFFu;
  uint src3_negate = (w1 >> 24) & 1u;
  uint src2_negate = (w1 >> 25) & 1u;
  uint src1_negate = (w1 >> 26) & 1u;
  uint pred_cond = (w1 >> 27) & 1u;
  bool is_predicated = Bit(w1, 1u << 28);
  uint const_1_rel = (w1 >> 30) & 1u;
  uint const_0_rel = (w1 >> 31) & 1u;

  uint src3_reg = (w2 >> 0) & 0xFFu;
  uint src2_reg = (w2 >> 8) & 0xFFu;
  uint src1_reg = (w2 >> 16) & 0xFFu;
  uint vector_opc = (w2 >> 24) & 0x1Fu;
  uint src3_sel = (w2 >> 29) & 1u;
  uint src2_sel = (w2 >> 30) & 1u;
  uint src1_sel = (w2 >> 31) & 1u;

  bool rel1 = const_0_rel != 0u;
  bool rel2 = (src1_sel != 0u) ? (const_0_rel != 0u) : (const_1_rel != 0u);
  bool rel3 = (src1_sel != 0u && src2_sel != 0u) ? (const_0_rel != 0u)
                                                 : (const_1_rel != 0u);

  vec4 s0 = Operand(src1_reg, src1_sel, src1_swiz, src1_negate, rel1);
  vec4 s1 = Operand(src2_reg, src2_sel, src2_swiz, src2_negate, rel2);
  vec4 s2 = Operand(src3_reg, src3_sel, src3_swiz, src3_negate, rel3);

  vec4 vres;
  switch (vector_opc) {
    case 0u:  vres = s0 + s1; break;
    case 1u:  vres = s0 * s1; break;
    case 2u:  vres = max(s0, s1); break;
    case 3u:  vres = min(s0, s1); break;
    case 4u:  vres = mix(vec4(0.0), vec4(1.0), equal(s0, s1)); break;
    case 5u:  vres = mix(vec4(0.0), vec4(1.0), greaterThan(s0, s1)); break;
    case 6u:  vres = mix(vec4(0.0), vec4(1.0), greaterThanEqual(s0, s1)); break;
    case 7u:  vres = mix(vec4(0.0), vec4(1.0), notEqual(s0, s1)); break;
    case 8u:  vres = fract(s0); break;
    case 9u:  vres = trunc(s0); break;
    case 10u: vres = floor(s0); break;
    case 11u: vres = s0 * s1 + s2; break;
    case 12u: vres = mix(s2, s1, equal(s0, vec4(0.0))); break;
    case 13u: vres = mix(s2, s1, greaterThanEqual(s0, vec4(0.0))); break;
    case 14u: vres = mix(s2, s1, greaterThan(s0, vec4(0.0))); break;
    case 15u: vres = vec4(dot(s0, s1)); break;
    case 16u: vres = vec4(dot(s0.xyz, s1.xyz)); break;
    case 17u: vres = vec4(dot(s0.xy, s1.xy) + s2.x); break;
    // cube(18) passthrough, rare in a VS. Predicate-push 20-23 and kill 24-27
    // unimplemented.
    case 19u: vres = vec4(max(max(s0.x, s0.y), max(s0.z, s0.w))); break;  // max4
    case 28u: vres = vec4(1.0, s0.y * s1.y, s0.z, s1.w); break;           // dst
    case 29u:  // maxa
      xe_a0 = clamp(int(floor(s0.w + 0.5)), -256, 255);
      vres = max(s0, s1);
      break;
    default: vres = s0; break;  // unhandled vector op, passthrough src0
  }
  if (vector_clamp) vres = clamp(vres, 0.0, 1.0);

  float a = s2.x, b = s2.y;
  float sres;
  switch (scalar_opc) {
    case 0u:  sres = a + b; break;                    // adds
    case 1u:  sres = a + xe_ps; break;                // adds_prev
    case 2u:  sres = a * b; break;                    // muls
    case 3u:  sres = a * xe_ps; break;                // muls_prev
    case 4u:  sres = a * xe_ps; break;                // muls_prev2 approx
    case 5u:  sres = max(a, b); break;                // maxs
    case 6u:  sres = min(a, b); break;                // mins
    case 7u:  sres = (a == 0.0) ? 1.0 : 0.0; break;   // seqs
    case 8u:  sres = (a > 0.0) ? 1.0 : 0.0; break;    // sgts
    case 9u:  sres = (a >= 0.0) ? 1.0 : 0.0; break;   // sges
    case 10u: sres = (a != 0.0) ? 1.0 : 0.0; break;   // snes
    case 11u: sres = fract(a); break;                 // frcs
    case 12u: sres = trunc(a); break;                 // truncs
    case 13u: sres = floor(a); break;                 // floors
    case 14u: sres = exp2(a); break;                  // exp
    case 15u:
    case 16u: sres = log2(max(a, 1e-30)); break;      // logc/log (clamp approx)
    case 17u:
    case 18u:
    case 19u: sres = (a == 0.0) ? 0.0 : 1.0 / a; break;  // rcpc/rcpf/rcp
    case 20u:
    case 21u:
    case 22u: sres = inversesqrt(max(a, 1e-30)); break;  // rsqc/rsqf/rsq
    case 23u:  // maxas
      xe_a0 = clamp(int(floor(a + 0.5)), 0, 255);
      sres = max(a, b);
      break;
    case 24u:  // maxasf
      xe_a0 = clamp(int(floor(a)), 0, 255);
      sres = max(a, b);
      break;
    case 25u: sres = a - b; break;                    // subs
    case 26u: sres = a - xe_ps; break;                // subs_prev
    case 27u:  // setp_eq
      xe_p0 = (a == 0.0);
      sres = xe_p0 ? 0.0 : 1.0;
      break;
    case 28u:  // setp_ne
      xe_p0 = (a != 0.0);
      sres = xe_p0 ? 0.0 : 1.0;
      break;
    case 29u:  // setp_gt
      xe_p0 = (a > 0.0);
      sres = xe_p0 ? 0.0 : 1.0;
      break;
    case 30u:  // setp_ge
      xe_p0 = (a >= 0.0);
      sres = xe_p0 ? 0.0 : 1.0;
      break;
    case 40u: sres = sqrt(max(a, 0.0)); break;        // sqrt
    // mulsc/addsc/subsc operand sourcing approximate, uses .x of const and a.
    case 42u:
    case 44u:
    case 46u: {  // mulsc/addsc/subsc (slot 0)
      float cc = FloatConst(src1_reg & 0xFFu).x;
      sres = (scalar_opc == 42u) ? cc * a : (scalar_opc == 44u) ? cc + a : cc - a;
    } break;
    case 43u:
    case 45u:
    case 47u: {  // mulsc/addsc/subsc (slot 1)
      float cc = FloatConst(src2_reg & 0xFFu).x;
      sres = (scalar_opc == 43u) ? cc * a : (scalar_opc == 45u) ? cc + a : cc - a;
    } break;
    case 48u: sres = sin(a); break;
    case 49u: sres = cos(a); break;
    case 50u: sres = xe_ps; break;                    // retain_prev
    default: sres = a; break;  // setp_inv/pop/clr/rstr and kills unimplemented
  }
  if (scalar_clamp) sres = clamp(sres, 0.0, 1.0);

  // Predication gates the writes incl. ps.
  if (is_predicated && (uint(xe_p0) != pred_cond)) return;
  xe_ps = sres;
  vec4 sres4 = vec4(sres);

  if (export_data != 0u) {
    if (vector_dest == 62u) {
      xe_out_pos = vec4(Bit(vector_write_mask, 1u) ? vres.x : xe_out_pos.x,
                        Bit(vector_write_mask, 2u) ? vres.y : xe_out_pos.y,
                        Bit(vector_write_mask, 4u) ? vres.z : xe_out_pos.z,
                        Bit(vector_write_mask, 8u) ? vres.w : xe_out_pos.w);
    } else if (vector_dest < MAX_INTERP) {
      xe_out_interp[vector_dest] = vec4(
          Bit(vector_write_mask, 1u) ? vres.x : xe_out_interp[vector_dest].x,
          Bit(vector_write_mask, 2u) ? vres.y : xe_out_interp[vector_dest].y,
          Bit(vector_write_mask, 4u) ? vres.z : xe_out_interp[vector_dest].z,
          Bit(vector_write_mask, 8u) ? vres.w : xe_out_interp[vector_dest].w);
    } else if (vector_dest == 63u && Bit(vector_write_mask, 4u)) {
      xe_kill = vres.z;  // oPts.z vertex kill; point size .x, edge flag .y N/A
    }  // memexport (dest 32-37) out of scope, eM writes are dropped
  } else {
    WriteMasked(vector_dest % REG_COUNT, vres, vector_write_mask);
  }
  if (scalar_write_mask != 0u && export_data == 0u)
    WriteMasked(scalar_dest % REG_COUNT, sres4, scalar_write_mask);
}

void RunExec(uint address, uint count, uint sequence) {
  for (uint s = 0u; s < count; ++s) {
    uint bits = (sequence >> (2u * s)) & 3u;
    if (Bit(bits, 1u))
      ExecVfetch(address + s);
    else
      ExecAlu(address + s);
  }
}

// Seed state and walk the control flow. Results land in xe_out_pos/interp/kill.
void XeRunInterp(uint cf_instr_count, float vertex_index) {
  for (uint i = 0u; i < REG_COUNT; ++i) regs[i] = vec4(0.0);
  regs[0] = vec4(vertex_index, 0.0, 0.0, 1.0);  // engine seeds r0 w/ vtx index
  xe_a0 = 0;
  xe_ps = 0.0;
  xe_p0 = false;
  xe_kill = 0.0;
  xe_vf_stride = 1u;
  xe_vf_fc = 0u;
  xe_vf_index_reg = 0u;
  xe_out_pos = vec4(0.0, 0.0, 0.0, 1.0);
  for (uint j = 0u; j < MAX_INTERP; ++j) xe_out_interp[j] = vec4(0.0);

  uint cf_index = 0u;
  for (uint step = 0u; step < MAX_CF_STEPS; ++step) {
    if (cf_index >= cf_instr_count) break;
    uint pair = cf_index >> 1;
    uint d0 = UcodeLoad(pair * 3u + 0u), d1 = UcodeLoad(pair * 3u + 1u),
         d2 = UcodeLoad(pair * 3u + 2u);
    uint cfw0, cfw1;
    if ((cf_index & 1u) == 0u) {
      cfw0 = d0;
      cfw1 = d1 & 0xFFFFu;
    } else {
      cfw0 = (d1 >> 16) | (d2 << 16);
      cfw1 = d2 >> 16;
    }
    uint opcode = (cfw1 >> 12) & 0xFu;

    if (opcode == 1u || opcode == 2u || opcode == 3u || opcode == 4u ||
        opcode == 5u || opcode == 6u || opcode == 13u || opcode == 14u) {
      uint address = cfw0 & 0xFFFu;
      uint count = (cfw0 >> 12) & 0x7u;
      uint sequence = (cfw0 >> 16) & 0xFFFu;
      bool run = true;
      if (opcode == 3u || opcode == 4u) {  // cond_exec (bool)
        run = (BoolConst((cfw1 >> 2) & 0xFFu) == (((cfw1 >> 10) & 1u) != 0u));
      } else if (opcode == 5u || opcode == 6u) {  // cond_exec_pred
        run = (xe_p0 == (((cfw1 >> 10) & 1u) != 0u));
      }
      if (run) RunExec(address, count, sequence);
      if (opcode == 2u || opcode == 4u || opcode == 6u || opcode == 14u)
        break;  // *_end
      cf_index += 1u;
    } else if (opcode == 11u) {  // cond_jmp
      uint address = cfw0 & 0x1FFFu;
      bool is_uncond = ((cfw0 >> 13) & 1u) != 0u;
      bool is_pred = ((cfw0 >> 14) & 1u) != 0u;
      bool cond = ((cfw1 >> 10) & 1u) != 0u;
      bool taken =
          is_uncond ||
          (is_pred ? (xe_p0 == cond) : (BoolConst((cfw1 >> 2) & 0xFFu) == cond));
      cf_index = taken ? address : (cf_index + 1u);
    } else {  // nop / alloc / unhandled (loop/endloop, call/return -> nop)
      cf_index += 1u;
    }
  }
}

// System flags (SpirvShaderTranslator kSysFlag_* bit values).
#define XE_SYS_FLAG_XY_DIVIDED_BY_W 0x8u
#define XE_SYS_FLAG_Z_DIVIDED_BY_W 0x10u
#define XE_SYS_FLAG_W_NOT_RECIPROCAL 0x20u

void main() {
  // Match the SPIR-V VS r0 seed: endian-swapped host index plus the base index.
  int index =
      int(EndianSwap(uint(gl_VertexIndex), XeVertexIndexEndian())) +
      XeVertexBaseIndex();
  XeRunInterp(XeCfInstrCount(), float(index));

  // Guest clip-space position to host NDC, matching the translated VS output.
  uint flags = XeFlags();
  vec4 p = xe_out_pos;
  float w = p.w;
  if ((flags & XE_SYS_FLAG_W_NOT_RECIPROCAL) == 0u) w = 1.0 / w;  // guest 1/W
  vec2 xy = p.xy;
  if ((flags & XE_SYS_FLAG_XY_DIVIDED_BY_W) != 0u) xy = xy * w;   // revert XY/W
  float z = p.z;
  if ((flags & XE_SYS_FLAG_Z_DIVIDED_BY_W) != 0u) z = z * w;      // revert Z/W
  vec3 xyz = vec3(xy, z) * XeNdcScale() + XeNdcOffset() * w;

  // Vertex kill: a non-zero oPts.z degenerates the vertex to NaN (OR-mode cull).
  if ((floatBitsToUint(xe_kill) & 0x7FFFFFFFu) != 0u)
    w = uintBitsToFloat(0x7FC00000u);

  gl_Position = vec4(xyz, w);
}

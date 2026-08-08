// Shared Xenos VS ucode interpreter core.
//
// Data-source-agnostic: the CF walk, ALU decode/dispatch, operand decode,
// vfetch decode, and vertex format conversion. Front-ends include this after
// defining the hooks below, so the interpreter logic lives in one place and can
// be reused across host backends (Vulkan SPIR-V, D3D12 DXIL).
//
// The includer MUST define, before #include:
//   REG_COUNT, MAX_INTERP                              (macros)
//   uint   UcodeLoad(uint dword)                       fetch a ucode dword
//   float4 FloatConst(uint index)                      guest float constant c[index]
//   bool   BoolConst(uint addr)                        guest bool constant [0..255]
//   uint4  GatherVertexDwords(uint fetch_const_index,  4 endian-swapped vertex-stream
//            uint index, uint stride, int offset)        dwords at the element
// and call XeRunInterp(cf_instr_count, vertex_index) from main(), then read
// xe_out_pos / xe_out_interp[] / xe_kill.

#ifndef XE_UCODE_INTERP_CORE_HLSLI_
#define XE_UCODE_INTERP_CORE_HLSLI_

#ifndef MAX_CF_STEPS
#define MAX_CF_STEPS 1024u // backstop, no backward jumps in this VS set
#endif

static float4 regs[REG_COUNT];
static int xe_a0;
static float xe_ps;
static bool xe_p0;
static float4 xe_out_pos;
static float4 xe_out_interp[MAX_INTERP];
static uint xe_vf_stride;     // vfetch_mini inherits these from the prior full
static uint xe_vf_fc;
static uint xe_vf_index_reg;
static float xe_kill;         // oPts.z vertex-kill value (export register 63)

float4 RegGet(uint i) { return regs[i]; }
void RegStore(uint i, float4 v) { regs[i] = v; }
bool Bit(uint v, uint b) { return (v & b) != 0; }
void WriteMasked(uint i, float4 v, uint mask) {
  float4 c = RegGet(i);
  RegStore(i, float4(Bit(mask, 1) ? v.x : c.x, Bit(mask, 2) ? v.y : c.y,
                     Bit(mask, 4) ? v.z : c.z, Bit(mask, 8) ? v.w : c.w));
}

float Comp(float4 v, uint c) {
  return c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w));
}
float4 Swizzle(float4 v, uint swiz) {
  return float4(Comp(v, ((swiz >> 0) + 0) & 3), Comp(v, ((swiz >> 2) + 1) & 3),
                Comp(v, ((swiz >> 4) + 2) & 3), Comp(v, ((swiz >> 6) + 3) & 3));
}
// sel: register vs constant. const_rel: a0-relative constant.
float4 Operand(uint reg, uint sel, uint swiz, uint negate, bool const_rel) {
  float4 src;
  if (sel != 0) {
    src = RegGet((reg & 0x3F) % REG_COUNT); // no loop-relative reg addressing, aL assumed 0
    if ((reg & 0x80u) != 0u) src = abs(src); // absolute-value source modifier
  } else {
    uint ci = reg & 0xFF;
    if (const_rel) ci = uint(clamp(int(ci) + xe_a0, 0, 255));
    src = FloatConst(ci);
  }
  float4 r = Swizzle(src, swiz & 0xFF);
  return (negate != 0) ? -r : r;
}

//-----------------------------------------------------------------------------
// vfetch: decode + (hooked) gather + format conversion. Conversion is shared,
// only the gather (where the dwords come from) differs between front-ends.
//-----------------------------------------------------------------------------
float SNorm(int x, float maxv) { return clamp(float(x) / maxv, -1.0, 1.0); }

// Partial format coverage. Unhandled formats fall through to (0,0,0,1).
float4 ConvertVertex(uint format, bool is_signed, bool is_integer, uint4 d) {
  float4 v = float4(0, 0, 0, 1);
  if (format == 57) {            // 32_32_32_FLOAT
    v.x = asfloat(d.x); v.y = asfloat(d.y); v.z = asfloat(d.z);
  } else if (format == 38) {     // 32_32_32_32_FLOAT
    v = float4(asfloat(d.x), asfloat(d.y), asfloat(d.z), asfloat(d.w));
  } else if (format == 37) {     // 32_32_FLOAT
    v.x = asfloat(d.x); v.y = asfloat(d.y); v.z = 0;
  } else if (format == 36) {     // 32_FLOAT
    v.x = asfloat(d.x); v.y = 0; v.z = 0;
  } else if (format == 32) {     // 16_16_16_16_FLOAT
    v = float4(f16tof32(d.x & 0xFFFF), f16tof32(d.x >> 16),
               f16tof32(d.y & 0xFFFF), f16tof32(d.y >> 16));
  } else if (format == 31) {     // 16_16_FLOAT
    v.x = f16tof32(d.x & 0xFFFF); v.y = f16tof32(d.x >> 16); v.z = 0;
  } else if (format == 25) {     // 16_16
    if (is_signed) {
      int xi = int(d.x << 16) >> 16, yi = int(d.x) >> 16;
      v.x = is_integer ? float(xi) : SNorm(xi, 32767.0);
      v.y = is_integer ? float(yi) : SNorm(yi, 32767.0);
    } else {
      uint xu = d.x & 0xFFFF, yu = d.x >> 16;
      v.x = is_integer ? float(xu) : float(xu) / 65535.0;
      v.y = is_integer ? float(yu) : float(yu) / 65535.0;
    }
    v.z = 0;
  } else if (format == 6) {      // 8_8_8_8
    [unroll] for (uint k = 0; k < 4; ++k) {
      uint bu = (d.x >> (8 * k)) & 0xFF;
      if (is_signed) { int bi = int(bu << 24) >> 24; v[k] = is_integer ? float(bi) : SNorm(bi, 127.0); }
      else v[k] = is_integer ? float(bu) : float(bu) / 255.0;
    }
  }
  return v;
}

// Assumes vfetch. tfetch would mis-decode here, gated out upstream by textures.
void ExecVfetch(uint off) {
  uint w0 = UcodeLoad(off * 3 + 0);
  uint w1 = UcodeLoad(off * 3 + 1);
  uint w2 = UcodeLoad(off * 3 + 2);
  uint src_reg = (w0 >> 5) & 0x3F;
  uint dst_reg = (w0 >> 12) & 0x3F;
  uint const_index = (w0 >> 20) & 0x1F;
  uint const_index_sel = (w0 >> 25) & 0x3;
  uint src_swiz = (w0 >> 30) & 0x3;
  uint dst_swiz = (w1 >> 0) & 0xFFF;
  bool is_signed = Bit(w1, 1 << 12);
  bool is_integer = Bit(w1, 1 << 13);
  uint format = (w1 >> 16) & 0x3F;
  bool is_mini = Bit(w1, 1 << 30);
  uint stride = is_mini ? xe_vf_stride : (w2 & 0xFF);
  uint fc_index = is_mini ? xe_vf_fc : (const_index * 3 + const_index_sel);
  uint index_reg = is_mini ? xe_vf_index_reg : src_reg;
  if (!is_mini) { xe_vf_stride = stride; xe_vf_fc = fc_index; xe_vf_index_reg = index_reg; }
  int offset = int((w2 >> 8) & 0x7FFFFF);

  uint index = uint(max(Comp(RegGet(index_reg % REG_COUNT), src_swiz & 3), 0.0));
  uint4 d = GatherVertexDwords(fc_index, index, stride, offset);
  float4 v = ConvertVertex(format, is_signed, is_integer, d);

  // Destination swizzle (3 bits/comp: 0-3 = xyzw, 4 = 0, 5 = 1, 7 = keep).
  float4 res = RegGet(dst_reg % REG_COUNT);
  [unroll] for (uint c = 0; c < 4; ++c) {
    uint s = (dst_swiz >> (3 * c)) & 7;
    if (s < 4) res[c] = v[s]; else if (s == 4) res[c] = 0.0; else if (s == 5) res[c] = 1.0;
  }
  RegStore(dst_reg % REG_COUNT, res);
}

//-----------------------------------------------------------------------------
// ALU.
//-----------------------------------------------------------------------------
void ExecAlu(uint off) {
  uint w0 = UcodeLoad(off * 3 + 0);
  uint w1 = UcodeLoad(off * 3 + 1);
  uint w2 = UcodeLoad(off * 3 + 2);

  uint vector_dest = (w0 >> 0) & 0x3F;
  uint scalar_dest = (w0 >> 8) & 0x3F;
  uint export_data = (w0 >> 15) & 1;
  uint vector_write_mask = (w0 >> 16) & 0xF;
  uint scalar_write_mask = (w0 >> 20) & 0xF;
  bool vector_clamp = Bit(w0, 1 << 24);
  bool scalar_clamp = Bit(w0, 1 << 25);
  uint scalar_opc = (w0 >> 26) & 0x3F;

  uint src3_swiz = (w1 >> 0) & 0xFF;
  uint src2_swiz = (w1 >> 8) & 0xFF;
  uint src1_swiz = (w1 >> 16) & 0xFF;
  uint src3_negate = (w1 >> 24) & 1;
  uint src2_negate = (w1 >> 25) & 1;
  uint src1_negate = (w1 >> 26) & 1;
  uint pred_cond = (w1 >> 27) & 1;
  bool is_predicated = Bit(w1, 1 << 28);
  uint const_1_rel = (w1 >> 30) & 1;
  uint const_0_rel = (w1 >> 31) & 1;

  uint src3_reg = (w2 >> 0) & 0xFF;
  uint src2_reg = (w2 >> 8) & 0xFF;
  uint src1_reg = (w2 >> 16) & 0xFF;
  uint vector_opc = (w2 >> 24) & 0x1F;
  uint src3_sel = (w2 >> 29) & 1;
  uint src2_sel = (w2 >> 30) & 1;
  uint src1_sel = (w2 >> 31) & 1;

  bool rel1 = const_0_rel != 0;
  bool rel2 = (src1_sel != 0) ? (const_0_rel != 0) : (const_1_rel != 0);
  bool rel3 = (src1_sel != 0 && src2_sel != 0) ? (const_0_rel != 0) : (const_1_rel != 0);

  float4 s0 = Operand(src1_reg, src1_sel, src1_swiz, src1_negate, rel1);
  float4 s1 = Operand(src2_reg, src2_sel, src2_swiz, src2_negate, rel2);
  float4 s2 = Operand(src3_reg, src3_sel, src3_swiz, src3_negate, rel3);

  float4 vres;
  switch (vector_opc) {
    case 0:  vres = s0 + s1; break;
    case 1:  vres = s0 * s1; break;
    case 2:  vres = max(s0, s1); break;
    case 3:  vres = min(s0, s1); break;
    case 4:  vres = select(s0 == s1, float4(1, 1, 1, 1), float4(0, 0, 0, 0)); break;
    case 5:  vres = select(s0 > s1,  float4(1, 1, 1, 1), float4(0, 0, 0, 0)); break;
    case 6:  vres = select(s0 >= s1, float4(1, 1, 1, 1), float4(0, 0, 0, 0)); break;
    case 7:  vres = select(s0 != s1, float4(1, 1, 1, 1), float4(0, 0, 0, 0)); break;
    case 8:  vres = frac(s0); break;
    case 9:  vres = trunc(s0); break;
    case 10: vres = floor(s0); break;
    case 11: vres = s0 * s1 + s2; break;
    case 12: vres = select(s0 == 0.0, s1, s2); break;
    case 13: vres = select(s0 >= 0.0, s1, s2); break;
    case 14: vres = select(s0 > 0.0, s1, s2); break;
    case 15: vres = dot(s0, s1); break;
    case 16: vres = dot(s0.xyz, s1.xyz); break;
    case 17: vres = dot(s0.xy, s1.xy) + s2.x; break;
    // cube(18) passthrough, rare in a VS. Predicate-push 20-23 and kill 24-27 unimplemented.
    case 19: vres = max(max(s0.x, s0.y), max(s0.z, s0.w)); break; // max4
    case 28: vres = float4(1.0, s0.y * s1.y, s0.z, s1.w); break;  // dst
    case 29: xe_a0 = clamp(int(floor(s0.w + 0.5)), -256, 255); vres = max(s0, s1); break; // maxa
    default: vres = s0; break; // unhandled vector op, passthrough src0
  }
  if (vector_clamp) vres = saturate(vres);

  float a = s2.x, b = s2.y;
  float sres;
  switch (scalar_opc) {
    case 0:  sres = a + b; break;                  // adds
    case 1:  sres = a + xe_ps; break;              // adds_prev
    case 2:  sres = a * b; break;                  // muls
    case 3:  sres = a * xe_ps; break;              // muls_prev
    case 4:  sres = a * xe_ps; break;              // muls_prev2 approx, ignores the sign-gated path
    case 5:  sres = max(a, b); break;              // maxs
    case 6:  sres = min(a, b); break;              // mins
    case 7:  sres = (a == 0.0) ? 1.0 : 0.0; break; // seqs
    case 8:  sres = (a > 0.0) ? 1.0 : 0.0; break;  // sgts
    case 9:  sres = (a >= 0.0) ? 1.0 : 0.0; break; // sges
    case 10: sres = (a != 0.0) ? 1.0 : 0.0; break; // snes
    case 11: sres = frac(a); break;                // frcs
    case 12: sres = trunc(a); break;               // truncs
    case 13: sres = floor(a); break;               // floors
    case 14: sres = exp2(a); break;                // exp
    case 15: case 16: sres = log2(max(a, 1e-30)); break; // logc/log (clamp approx)
    case 17: case 18: case 19: sres = (a == 0.0) ? 0.0 : 1.0 / a; break; // rcpc/rcpf/rcp
    case 20: case 21: case 22: sres = rsqrt(max(a, 1e-30)); break; // rsqc/rsqf/rsq
    case 23: xe_a0 = clamp(int(floor(a + 0.5)), 0, 255); sres = max(a, b); break; // maxas
    case 24: xe_a0 = clamp(int(floor(a)), 0, 255); sres = max(a, b); break;       // maxasf
    case 25: sres = a - b; break;                  // subs
    case 26: sres = a - xe_ps; break;              // subs_prev
    case 27: xe_p0 = (a == 0.0); sres = xe_p0 ? 0.0 : 1.0; break; // setp_eq
    case 28: xe_p0 = (a != 0.0); sres = xe_p0 ? 0.0 : 1.0; break; // setp_ne
    case 29: xe_p0 = (a > 0.0);  sres = xe_p0 ? 0.0 : 1.0; break; // setp_gt
    case 30: xe_p0 = (a >= 0.0); sres = xe_p0 ? 0.0 : 1.0; break; // setp_ge
    case 40: sres = sqrt(max(a, 0.0)); break;      // sqrt
    // mulsc/addsc/subsc operand sourcing approximate, uses .x of the const and a only.
    case 42: case 44: case 46: {                   // mulsc/addsc/subsc (slot 0)
      float cc = FloatConst(src1_reg & 0xFF).x;
      sres = (scalar_opc == 42) ? cc * a : (scalar_opc == 44) ? cc + a : cc - a;
    } break;
    case 43: case 45: case 47: {                   // mulsc/addsc/subsc (slot 1)
      float cc = FloatConst(src2_reg & 0xFF).x;
      sres = (scalar_opc == 43) ? cc * a : (scalar_opc == 45) ? cc + a : cc - a;
    } break;
    case 48: sres = sin(a); break;
    case 49: sres = cos(a); break;
    case 50: sres = xe_ps; break;                  // retain_prev
    default: sres = a; break; // setp_inv/pop/clr/rstr 31-34 and kills 35-39 unimplemented
  }
  if (scalar_clamp) sres = saturate(sres);

  // Predication gates the writes incl. ps.
  if (is_predicated && (uint(xe_p0) != pred_cond)) return;
  xe_ps = sres;
  float4 sres4 = sres;

  if (export_data != 0) {
    if (vector_dest == 62) {
      xe_out_pos = float4(Bit(vector_write_mask, 1) ? vres.x : xe_out_pos.x,
                          Bit(vector_write_mask, 2) ? vres.y : xe_out_pos.y,
                          Bit(vector_write_mask, 4) ? vres.z : xe_out_pos.z,
                          Bit(vector_write_mask, 8) ? vres.w : xe_out_pos.w);
    } else if (vector_dest < MAX_INTERP) {
      [unroll] for (uint k = 0; k < MAX_INTERP; ++k) if (k == vector_dest) {
        xe_out_interp[k] = float4(Bit(vector_write_mask, 1) ? vres.x : xe_out_interp[k].x,
                                  Bit(vector_write_mask, 2) ? vres.y : xe_out_interp[k].y,
                                  Bit(vector_write_mask, 4) ? vres.z : xe_out_interp[k].z,
                                  Bit(vector_write_mask, 8) ? vres.w : xe_out_interp[k].w);
      }
    } else if (vector_dest == 63 && Bit(vector_write_mask, 4)) {
      xe_kill = vres.z; // oPts.z vertex kill. point size .x and edge flag .y unhandled
    } // memexport (dest 32-37) out of scope, eM writes are dropped
  } else {
    WriteMasked(vector_dest % REG_COUNT, vres, vector_write_mask);
  }
  if (scalar_write_mask != 0 && export_data == 0)
    WriteMasked(scalar_dest % REG_COUNT, sres4, scalar_write_mask);
}

void RunExec(uint address, uint count, uint sequence) {
  [loop] for (uint s = 0; s < count; ++s) {
    uint bits = (sequence >> (2 * s)) & 3;
    if (Bit(bits, 1)) ExecVfetch(address + s); else ExecAlu(address + s);
  }
}

// Seed state and walk the control flow. Results land in xe_out_pos/xe_out_interp.
void XeRunInterp(uint cf_instr_count, float vertex_index) {
  [unroll] for (uint i = 0; i < REG_COUNT; ++i) regs[i] = float4(0, 0, 0, 0);
  regs[0] = float4(vertex_index, 0, 0, 1); // engine seeds r0 with the vertex index
  xe_a0 = 0; xe_ps = 0; xe_p0 = false; xe_kill = 0;
  xe_vf_stride = 1; xe_vf_fc = 0; xe_vf_index_reg = 0;
  xe_out_pos = float4(0, 0, 0, 1);
  [unroll] for (uint j = 0; j < MAX_INTERP; ++j) xe_out_interp[j] = float4(0, 0, 0, 0);

  uint cf_index = 0;
  [loop] for (uint step = 0; step < MAX_CF_STEPS; ++step) {
    if (cf_index >= cf_instr_count) break;
    uint pair = cf_index >> 1;
    uint d0 = UcodeLoad(pair * 3 + 0), d1 = UcodeLoad(pair * 3 + 1), d2 = UcodeLoad(pair * 3 + 2);
    uint cfw0, cfw1;
    if ((cf_index & 1) == 0) { cfw0 = d0; cfw1 = d1 & 0xFFFF; }
    else { cfw0 = (d1 >> 16) | (d2 << 16); cfw1 = d2 >> 16; }
    uint opcode = (cfw1 >> 12) & 0xF;

    if (opcode == 1 || opcode == 2 || opcode == 3 || opcode == 4 ||
        opcode == 5 || opcode == 6 || opcode == 13 || opcode == 14) {
      uint address = cfw0 & 0xFFF;
      uint count = (cfw0 >> 12) & 0x7;
      uint sequence = (cfw0 >> 16) & 0xFFF;
      bool run = true;
      if (opcode == 3 || opcode == 4) {        // cond_exec (bool)
        run = (BoolConst((cfw1 >> 2) & 0xFF) == (((cfw1 >> 10) & 1) != 0));
      } else if (opcode == 5 || opcode == 6) { // cond_exec_pred
        run = (xe_p0 == (((cfw1 >> 10) & 1) != 0));
      }
      if (run) RunExec(address, count, sequence);
      if (opcode == 2 || opcode == 4 || opcode == 6 || opcode == 14) break; // *_end
      cf_index += 1;
    } else if (opcode == 11) {                 // cond_jmp
      uint address = cfw0 & 0x1FFF;
      bool is_uncond = ((cfw0 >> 13) & 1) != 0;
      bool is_pred = ((cfw0 >> 14) & 1) != 0;
      bool cond = ((cfw1 >> 10) & 1) != 0;
      bool taken = is_uncond ||
                   (is_pred ? (xe_p0 == cond) : (BoolConst((cfw1 >> 2) & 0xFF) == cond));
      cf_index = taken ? address : (cf_index + 1);
    } else {                                   // nop / alloc / unhandled
      cf_index += 1; // loop/endloop and call/return unhandled, treated as nop
    }
  }
}

#endif  // XE_UCODE_INTERP_CORE_HLSLI_

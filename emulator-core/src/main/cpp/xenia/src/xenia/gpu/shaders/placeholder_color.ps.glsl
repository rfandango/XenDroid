#version 460

// GLSL twin of placeholder_color.ps.slang for the Android Vulkan build.
//
// The .slang form is HLSL (float4 main() : SV_Target) for the DXIL path, which
// glslangValidator can't parse. gen_android_spirv.py prefers this .glsl twin
// over the .slang, so it produces the placeholder_color_ps.h SPIR-V bytecode.
//
// Debug placeholder pixel shader: flat grey, so the ucode interpreter VS's
// interim geometry is visible while the real shaders compile in the background
// (gated by async_shader_vs_interpreter_debug_color). Host-render-target path
// only - on the ROV/EDRAM path color goes through a UAV, not the color output.
layout(location = 0) out vec4 xe_frag_color;

void main() { xe_frag_color = vec4(0.5, 0.5, 0.5, 1.0); }

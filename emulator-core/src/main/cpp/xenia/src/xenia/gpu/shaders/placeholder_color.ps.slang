// XE_NO_MSL
// Debug placeholder pixel shader: flat grey, so the ucode interpreter VS's
// interim geometry is visible while the real shaders compile in the background
// (gated by async_shader_vs_interpreter_debug_color). Host-render-target path only -
// on the ROV/EDRAM path color goes through a UAV, not SV_Target.
float4 main() : SV_Target { return float4(0.5, 0.5, 0.5, 1.0); }

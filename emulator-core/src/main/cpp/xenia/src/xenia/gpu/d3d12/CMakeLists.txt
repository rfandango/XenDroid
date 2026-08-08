add_library(xenia-gpu-d3d12 STATIC)
xe_platform_sources(xenia-gpu-d3d12 ${CMAKE_CURRENT_SOURCE_DIR})
xe_shader_rules_slang(xenia-gpu-d3d12 ${CMAKE_CURRENT_SOURCE_DIR}/../shaders
                      TARGET dxil)
target_link_libraries(xenia-gpu-d3d12 PUBLIC
  fmt xxhash xenia-base xenia-gpu xenia-ui xenia-ui-d3d12
)
xe_target_defaults(xenia-gpu-d3d12)

# Guest shaders are sourced exclusively from Mesa spirv_to_dxil (SPIR-V emitted
# by SpirvShaderTranslator); there is no HLSL guest shader path. spirv_to_dxil
# is built and linked unconditionally (the D3D12 backend is Windows-only, the
# only place the Mesa tree is built - see third_party/CMakeLists.txt).
target_link_libraries(xenia-gpu-d3d12 PUBLIC spirv_to_dxil)
# spirv_shader_translator.h pulls in glslang's SpvBuilder, like the Vulkan and
# Metal backends, so it needs glslang's include root and symbols.
target_include_directories(xenia-gpu-d3d12 PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/glslang)
target_link_libraries(xenia-gpu-d3d12 PRIVATE glslang-spirv)
# The linked tessellation path feeds the SPIR-V host vertex/hull shaders through
# Mesa, so it includes their vulkan_spirv headers, which the Vulkan backend
# generates (always built when D3D12 is). Depend on that generation so the
# headers exist on a clean build.
foreach(_spirv_shaders_dep
        xenia-gpu-vulkan-spirv-shaders xenia-gpu-vulkan-slang-spirv-shaders)
  if(TARGET ${_spirv_shaders_dep})
    add_dependencies(xenia-gpu-d3d12 ${_spirv_shaders_dep})
  endif()
endforeach()

if(XENIA_BUILD_MISC)
  add_executable(xenia-gpu-d3d12-trace-viewer WIN32
    ${CMAKE_CURRENT_SOURCE_DIR}/d3d12_trace_viewer_main.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/ui/windowed_app_main_qt.cc
  )
  target_link_libraries(xenia-gpu-d3d12-trace-viewer PRIVATE
    xenia-apu xenia-apu-nop xenia-base xenia-core xenia-cpu
    xenia-gpu xenia-gpu-d3d12 xenia-hid xenia-hid-nop xenia-hid-portal
    xenia-kernel xenia-patcher xenia-ui xenia-ui-d3d12 xenia-vfs
    aes_128 capstone dxbc fmt imgui libavcodec libavutil mspack snappy xxhash
  )
  if(XE_TARGET_X86_64)
    target_link_libraries(xenia-gpu-d3d12-trace-viewer PRIVATE xenia-cpu-backend-x64)
  endif()
  xe_target_defaults(xenia-gpu-d3d12-trace-viewer)

  # D3D12 trace dump
  add_executable(xenia-gpu-d3d12-trace-dump
    ${CMAKE_CURRENT_SOURCE_DIR}/d3d12_trace_dump_main.cc
    ${PROJECT_SOURCE_DIR}/src/xenia/base/console_app_main_win.cc
  )
  target_link_libraries(xenia-gpu-d3d12-trace-dump PRIVATE
    xenia-apu xenia-apu-nop xenia-base xenia-core xenia-cpu
    xenia-gpu xenia-gpu-d3d12 xenia-hid xenia-hid-nop xenia-hid-portal
    xenia-kernel xenia-patcher xenia-ui xenia-ui-d3d12 xenia-vfs
    aes_128 capstone dxbc fmt imgui libavcodec libavutil mspack snappy xxhash
  )
  if(XE_TARGET_X86_64)
    target_link_libraries(xenia-gpu-d3d12-trace-dump PRIVATE xenia-cpu-backend-x64)
  endif()
  xe_target_defaults(xenia-gpu-d3d12-trace-dump)
endif()

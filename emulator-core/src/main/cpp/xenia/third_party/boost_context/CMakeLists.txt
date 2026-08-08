# boost_context: Boost.Context fcontext_t fiber primitives (BSL-1.0).
#
# Builds make_/jump_/ontop_fcontext straight out of the boostorg/context
# submodule (./context). We consume only these three asm primitives; the
# extern "C" declarations and the guarded-stack allocator live in
# src/xenia/base/threading_fiber.cc, so no Boost headers are pulled in and the
# submodule stays pristine.

set(_bc_asm_dir "${CMAKE_CURRENT_SOURCE_DIR}/context/src/asm")
if(NOT EXISTS "${_bc_asm_dir}")
  message(FATAL_ERROR
    "boostorg/context submodule missing at ${_bc_asm_dir}. "
    "Run: git submodule update --init third_party/boost_context/context")
endif()

# The project sets global C/C++ compile options (/utf-8, /Zc:__cplusplus,
# /arch:AVX, ...). ml64 (MASM) rejects these as invalid and then exits 0 WITHOUT
# emitting an object, which later breaks the link. None are needed to assemble
# fcontext, so drop the inherited options for this directory.
set_directory_properties(PROPERTIES COMPILE_OPTIONS "")

# Select the asm variant by arch + object format + assembler dialect.
if(WIN32)
  if(XE_TARGET_AARCH64)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
      set(_bc_variant "arm64_aapcs_pe_armclang.S")
      set(_bc_asm_lang ASM)
    else()
      # ASM_MARMASM is CMake's built-in Microsoft armasm64 language (3.26+),
      # the submodule's homegrown ASM_ARMASM modules predate it and lack the
      # linker-information files newer CMake demands.
      if(CMAKE_VERSION VERSION_LESS 3.26)
        message(FATAL_ERROR "Windows ARM64 requires CMake 3.26+ for ASM_MARMASM")
      endif()
      set(_bc_variant "arm64_aapcs_pe_armasm.asm")
      set(_bc_asm_lang ASM_MARMASM)
    endif()
  else()
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
      set(_bc_variant "x86_64_ms_pe_clang_gas.S")
      set(_bc_asm_lang ASM)
    else()
      set(_bc_variant "x86_64_ms_pe_masm.asm")
      set(_bc_asm_lang ASM_MASM)
    endif()
  endif()
elseif(APPLE)
  set(_bc_asm_lang ASM)
  if(XE_TARGET_AARCH64)
    set(_bc_variant "arm64_aapcs_macho_gas.S")
  else()
    set(_bc_variant "x86_64_sysv_macho_gas.S")
  endif()
else()  # Linux / other ELF
  set(_bc_asm_lang ASM)
  if(XE_TARGET_AARCH64)
    set(_bc_variant "arm64_aapcs_elf_gas.S")
  else()
    set(_bc_variant "x86_64_sysv_elf_gas.S")
  endif()
endif()

enable_language(${_bc_asm_lang})

if(_bc_asm_lang STREQUAL "ASM_MASM")
  # ml64 prints " Assembling: <file>" (and its diagnostics) to stdout, with no
  # flag to silence just the progress line. Redirect stdout to NUL for a clean
  # build log. ninja still fails the build on a nonzero exit, so a break is
  # never silent -- only ml64's own error text would be hidden, and the asm is
  # frozen upstream code. Drop the redirect temporarily if you ever debug it.
  string(APPEND CMAKE_ASM_MASM_COMPILE_OBJECT " > nul")
endif()

add_library(boost_context STATIC
  "${_bc_asm_dir}/make_${_bc_variant}"
  "${_bc_asm_dir}/jump_${_bc_variant}"
  "${_bc_asm_dir}/ontop_${_bc_variant}"
)

# Boost's MASM/PE sources tag the routines with BOOST_CONTEXT_EXPORT; for a
# static lib it must be empty.
target_compile_definitions(boost_context PRIVATE "BOOST_CONTEXT_EXPORT=")

if(MSVC)
  target_compile_options(boost_context PRIVATE
    $<$<COMPILE_LANGUAGE:ASM_MASM>:/nologo>
  )
endif()

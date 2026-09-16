#
# Attempt to find the python package that uses the same python executable as
# `EXECUTABLE` and is one of the `SUPPORTED_VERSIONS`.
#
macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  set(_VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()

#
# Run `EXPR` in python.  The standard output of python is stored in `OUT` and
# has trailing whitespace stripped.  If an error is encountered when running
# python, a fatal message `ERR_MSG` is issued.
#
function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

# Run `EXPR` in python after importing `PKG`. Use the result of this to extend
# `CMAKE_PREFIX_PATH` so the torch cmake configuration can be imported.
macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

#
# Get additional GPU compiler flags from torch.
#
function (get_torch_gpu_compiler_flags OUT_GPU_FLAGS GPU_LANG)
    #
    # Get common NVCC flags from torch.
    #
  run_python(GPU_FLAGS
    "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
    "Failed to determine torch nvcc compiler flags")

  # CUDA_VERSION reflects the gstub nvcc toolkit version (parsed from
  # `nvcc --version` in CMakeLists.txt). It replaces the former SUDA_VERSION
  # that the suda cmake package used to provide.
  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND GPU_FLAGS "-DENABLE_FP8")
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
    list(REMOVE_ITEM GPU_FLAGS
      "-D__CUDA_NO_HALF_OPERATORS__"
      "-D__CUDA_NO_HALF_CONVERSIONS__"
      "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
      "-D__CUDA_NO_HALF2_OPERATORS__"
      "--expt-relaxed-constexpr"
      )
  endif()
  list(APPEND GPU_FLAGS "-DUSE_SUDA")
set(${OUT_GPU_FLAGS} ${GPU_FLAGS} PARENT_SCOPE)
endfunction()

# Macro for converting a `gencode` version number to a cmake version number.
macro(string_to_ver OUT_VER IN_STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

#
# For a specific file set the `-gencode` flag in compile options conditionally
# for the CUDA language.
#
# Example:
#   set_gencode_flag_for_srcs(
#     SRCS "foo.cu"
#     ARCH "compute_75"
#     CODE "sm_75")
#   adds: "-gencode arch=compute_75,code=sm_75" to the compile options for
#    `foo.cu` (only for the CUDA language).
#
macro(set_gencode_flag_for_srcs)
  set(options)
  set(oneValueArgs ARCH CODE)
  set(multiValueArgs SRCS)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )
  set(_FLAG -gencode arch=${arg_ARCH},code=${arg_CODE})
  set_property(
    SOURCE ${arg_SRCS}
    APPEND PROPERTY
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_FLAG}>"
  )

  message(DEBUG "Setting gencode flag for ${arg_SRCS}: ${_FLAG}")
endmacro(set_gencode_flag_for_srcs)

#
# For a list of source files set the `-gencode` flags in the files specific
#  compile options (specifically for the CUDA language).
#
# arguments are:
#  SRCS: list of source files
#  CUDA_ARCHS: list of CUDA architectures in the form `<major>.<minor>[letter]`
#  BUILD_PTX_FOR_ARCH: if set to true, then the PTX code will be built
#    for architecture `BUILD_PTX_FOR_ARCH` if there is a CUDA_ARCH in CUDA_ARCHS
#    that is larger than BUILD_PTX_FOR_ARCH.
#
macro(set_gencode_flags_for_srcs)
  set(options)
  set(oneValueArgs BUILD_PTX_FOR_ARCH)
  set(multiValueArgs SRCS CUDA_ARCHS)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

  foreach(_ARCH ${arg_CUDA_ARCHS})
    # handle +PTX suffix: generate both sm and ptx codes if requested
    string(FIND "${_ARCH}" "+PTX" _HAS_PTX)
    if(NOT _HAS_PTX EQUAL -1)
      string(REPLACE "+PTX" "" _BASE_ARCH "${_ARCH}")
      string(REPLACE "." "" _STRIPPED_ARCH "${_BASE_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "sm_${_STRIPPED_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "compute_${_STRIPPED_ARCH}")
    else()
      string(REPLACE "." "" _STRIPPED_ARCH "${_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "sm_${_STRIPPED_ARCH}")
    endif()
  endforeach()

  if (${arg_BUILD_PTX_FOR_ARCH})
    list(SORT arg_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)
    list(GET arg_CUDA_ARCHS -1 _HIGHEST_ARCH)
    if (_HIGHEST_ARCH VERSION_GREATER_EQUAL ${arg_BUILD_PTX_FOR_ARCH})
      string(REPLACE "." "" _PTX_ARCH "${arg_BUILD_PTX_FOR_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_PTX_ARCH}"
        CODE "compute_${_PTX_ARCH}")
    endif()
  endif()
endmacro()

#
# For the given `SRC_CUDA_ARCHS` list of gencode versions in the form
#  `<major>.<minor>[letter]` compute the "loose intersection" with the
#  `TGT_CUDA_ARCHS` list of gencodes. We also support the `+PTX` suffix in
#  `SRC_CUDA_ARCHS` which indicates that the PTX code should be built when there
#  is a CUDA_ARCH in `TGT_CUDA_ARCHS` that is equal to or larger than the
#  architecture in `SRC_CUDA_ARCHS`.
# The loose intersection is defined as:
#   { max{ x \in tgt | x <= y } | y \in src, { x \in tgt | x <= y } != {} }
#  where `<=` is the version comparison operator.
# In other words, for each version in `TGT_CUDA_ARCHS` find the highest version
#  in `SRC_CUDA_ARCHS` that is less or equal to the version in `TGT_CUDA_ARCHS`.
# We have special handling for x.0a, if x.0a is in `SRC_CUDA_ARCHS` and x.0 is
#  in `TGT_CUDA_ARCHS` then we should remove x.0a from `SRC_CUDA_ARCHS` and add
#  x.0a to the result (and remove x.0 from TGT_CUDA_ARCHS).
# The result is stored in `OUT_CUDA_ARCHS`.
#
# Example:
#   SRC_CUDA_ARCHS="7.5;8.0;8.6;9.0;9.0a"
#   TGT_CUDA_ARCHS="8.0;8.9;9.0"
#   cuda_archs_loose_intersection(OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
#   OUT_CUDA_ARCHS="8.0;8.6;9.0;9.0a"
#
# Example With PTX:
#   SRC_CUDA_ARCHS="8.0+PTX"
#   TGT_CUDA_ARCHS="9.0"
#   cuda_archs_loose_intersection(OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
#   OUT_CUDA_ARCHS="8.0+PTX"
#
function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
  set(_SRC_CUDA_ARCHS "${SRC_CUDA_ARCHS}")
  set(_TGT_CUDA_ARCHS ${TGT_CUDA_ARCHS})

  # handle +PTX suffix: separate base arch for matching, record PTX requests
  set(_PTX_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "\\+PTX$")
      string(REPLACE "+PTX" "" _base "${_arch}")
      list(APPEND _PTX_ARCHS "${_base}")
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      list(APPEND _SRC_CUDA_ARCHS "${_base}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _PTX_ARCHS)
  list(REMOVE_DUPLICATES _SRC_CUDA_ARCHS)

  # If x.0a or x.0f is in SRC_CUDA_ARCHS and x.0 is in CUDA_ARCHS then we should
  # remove x.0a or x.0f from SRC_CUDA_ARCHS and add x.0a or x.0f to _CUDA_ARCHS
  set(_CUDA_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "[af]$")
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      string(REGEX REPLACE "[af]$" "" _base "${_arch}")
      if ("${_base}" IN_LIST TGT_CUDA_ARCHS)
        list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_base}")
        list(APPEND _CUDA_ARCHS "${_arch}")
      endif()
    endif()
  endforeach()

  list(SORT _SRC_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)

  # for each ARCH in TGT_CUDA_ARCHS find the highest arch in SRC_CUDA_ARCHS that
  # is less or equal to ARCH (but has the same major version since SASS binary
  # compatibility is only forward compatible within the same major version).
  foreach(_ARCH ${_TGT_CUDA_ARCHS})
    set(_TMP_ARCH)
    # Extract the major version of the target arch
    string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" TGT_ARCH_MAJOR "${_ARCH}")
    foreach(_SRC_ARCH ${_SRC_CUDA_ARCHS})
      # Extract the major version of the source arch
      string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" SRC_ARCH_MAJOR "${_SRC_ARCH}")
      # Check version-less-or-equal, and allow PTX arches to match across majors
      if (_SRC_ARCH VERSION_LESS_EQUAL _ARCH)
        if (_SRC_ARCH IN_LIST _PTX_ARCHS OR SRC_ARCH_MAJOR STREQUAL TGT_ARCH_MAJOR)
          set(_TMP_ARCH "${_SRC_ARCH}")
        endif()
      else()
        # If we hit a version greater than the target, we can break
        break()
      endif()
    endforeach()

    # If we found a matching _TMP_ARCH, append it to _CUDA_ARCHS
    if (_TMP_ARCH)
      list(APPEND _CUDA_ARCHS "${_TMP_ARCH}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _CUDA_ARCHS)

  # reapply +PTX suffix to architectures that requested PTX
  set(_FINAL_ARCHS)
  foreach(_arch ${_CUDA_ARCHS})
    if(_arch IN_LIST _PTX_ARCHS)
      list(APPEND _FINAL_ARCHS "${_arch}+PTX")
    else()
      list(APPEND _FINAL_ARCHS "${_arch}")
    endif()
  endforeach()
  set(_CUDA_ARCHS ${_FINAL_ARCHS})

  set(${OUT_CUDA_ARCHS} ${_CUDA_ARCHS} PARENT_SCOPE)
endfunction()

#
# Define a target named `MOD_NAME` for a single extension. The
# arguments are:
#
# DESTINATION <dest>         - Module destination directory.
# LANGUAGE <lang>            - The language for this module, e.g. CUDA, HIP,
#                              CXX, etc.
# SOURCES <sources>          - List of source files relative to CMakeLists.txt
#                              directory.
#
# Optional arguments:
#
# ARCHITECTURES <arches>     - A list of target architectures in cmake format.
#                              For GPU, refer to CMAKE_CUDA_ARCHITECTURES and
#                              CMAKE_HIP_ARCHITECTURES for more info.
#                              ARCHITECTURES will use cmake's defaults if
#                              not provided.
# COMPILE_FLAGS <flags>      - Extra compiler flags passed to NVCC/hip.
# INCLUDE_DIRECTORIES <dirs> - Extra include directories.
# LIBRARIES <libraries>      - Extra link libraries.
# WITH_SOABI                 - Generate library with python SOABI suffix name.
# USE_SABI <version>         - Use python stable api <version>
#
# Note: optimization level/debug info is set via cmake build type.
#
function (define_extension_target MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    ARG
    "WITH_SOABI"
    "DESTINATION;LANGUAGE;USE_SABI"
    "SOURCES;ARCHITECTURES;COMPILE_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES")

  if (ARG_WITH_SOABI)
    set(SOABI_KEYWORD WITH_SOABI)
  else()
    set(SOABI_KEYWORD "")
  endif()

  run_python(IS_FREETHREADED_PYTHON
    "import sysconfig; print(1 if sysconfig.get_config_var(\"Py_GIL_DISABLED\") else 0)"
    "Failed to determine whether interpreter is free-threaded")

  # Free-threaded Python doesn't yet support the stable ABI (see PEP 803/809),
  # so avoid using the stable ABI under free-threading only.
  message(STATUS "TorchSUPA_CXX_FLAGS: ${TorchSUPA_CXX_FLAGS}")
  list(APPEND SUPA_BRCC_FLAGS ${VLLM_GPU_FLAGS} ${TorchSUPA_CXX_FLAGS})
  list(APPEND SUPA_BRCC_FLAGS "-Wno-deprecated-builtins" "-Wno-missing-exception-spec" "-Wno-ignored-pragmas" "-Wno-ignored-attributes" "-Wno-macro-redefined" "-Wno-string-compare" "-Wno-pass-failed" )
  # list(APPEND SUPA_BRCC_FLAGS ${VLLM_GPU_FLAGS} ${TorchSUPA_CXX_FLAGS} "-fno-gpu-rdc -use-mira=true -Xmira-as -br-tcore-async-src-register=0 -Xmira-as -br-allow-tensor-reg-spill=1 -Xmira-as -br-preallocate-tensor-regs=0")
  if (ARG_USE_SABI AND NOT IS_FREETHREADED_PYTHON)
    Python_add_library(${MOD_NAME} MODULE USE_SABI ${ARG_USE_SABI} ${SOABI_KEYWORD} "${ARG_SOURCES}")
  else()
    Python_add_library(${MOD_NAME} MODULE ${SOABI_KEYWORD} "${ARG_SOURCES}")
  endif()
  target_link_libraries(${MOD_NAME} PRIVATE CUDA::cuda_driver)

  target_include_directories(${MOD_NAME} PRIVATE csrc ${ARG_INCLUDE_DIRECTORIES})

  if (ARG_ARCHITECTURES)
    set_target_properties(${MOD_NAME} PROPERTIES
      ${ARG_LANGUAGE}_ARCHITECTURES "${ARG_ARCHITECTURES}")
  endif()

  target_compile_options(${MOD_NAME} PRIVATE
    ${VLLM_GPU_FLAGS})

  target_compile_options(${MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-builtins>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-missing-exception-spec>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-ignored-pragmas>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-ignored-attributes>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-string-compare>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-pass-failed>)

  foreach(FLAG ${TorchSUPA_CXX_FLAGS})
    target_compile_options(${MOD_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${FLAG}>)
  endforeach()

  if(ENABLE_COVERAGE)
    target_compile_options(${MOD_NAME} PRIVATE --coverage)
    target_link_libraries(${MOD_NAME} PRIVATE --coverage)
  endif()

  target_compile_definitions(${MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${MOD_NAME}")

  message(STATUS "SUPA_ARCH: ${SUPA_ARCH}")
  target_compile_definitions(${MOD_NAME} PRIVATE SUPA_ARCH=${SUPA_ARCH})

  target_include_directories(${MOD_NAME} BEFORE PRIVATE ${TORCH_SUPA_PATH}/include)

  message(STATUS "TORCH_SUPA_LIB: ${TORCH_SUPA_LIB}")
  target_link_libraries(${MOD_NAME} PRIVATE
    torch
    ${TORCH_SUPA_LIB}
    CUDA::cuda_driver
    "-Wl,--no-as-needed"
    CUDA::cudart
    "-Wl,--as-needed"
    ${ARG_LIBRARIES})

  target_link_libraries(${MOD_NAME} PRIVATE CUDA::cublas)

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION ${ARG_DESTINATION} COMPONENT ${MOD_NAME})

  set_target_properties(${MOD_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

endfunction()

#
# Build fake CUDA stub shared libraries and link them to a target.
# This loads empty libtorch_cuda.so, libnvrtc.so.13, and libc10_cuda.so
# at runtime so that cumem_allocator can resolve symbols.
#
function(link_fake_cuda_stubs TARGET)
  set(FAKE_CUDA_STUB_SRC ${CMAKE_CURRENT_BINARY_DIR}/fake_cuda_stub.cpp)
  if(NOT EXISTS ${FAKE_CUDA_STUB_SRC})
    file(WRITE ${FAKE_CUDA_STUB_SRC} "")
  endif()

  # Only create the stub targets once; subsequent calls just link them.
  if(NOT TARGET fake_libtorch_cuda)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libtorch_cuda.so
        COMMAND ${CMAKE_CXX_COMPILER} -shared -fPIC -o ${CMAKE_CURRENT_BINARY_DIR}/libtorch_cuda.so ${FAKE_CUDA_STUB_SRC}
        DEPENDS ${FAKE_CUDA_STUB_SRC}
        COMMENT "Compiling fake shared library: libtorch_cuda.so"
        VERBATIM
    )
    add_custom_target(fake_libtorch_cuda DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtorch_cuda.so)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libnvrtc.so.13
        COMMAND ${CMAKE_CXX_COMPILER} -shared -fPIC -Wl,-soname,libnvrtc.so.13 -o ${CMAKE_CURRENT_BINARY_DIR}/libnvrtc.so.13 ${FAKE_CUDA_STUB_SRC}
        DEPENDS ${FAKE_CUDA_STUB_SRC}
        COMMENT "Compiling fake shared library: libnvrtc.so.13"
        VERBATIM
    )
    add_custom_target(fake_libnvrtc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libnvrtc.so.13)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libc10_cuda.so
        COMMAND ${CMAKE_CXX_COMPILER} -shared -fPIC -o ${CMAKE_CURRENT_BINARY_DIR}/libc10_cuda.so ${FAKE_CUDA_STUB_SRC}
        DEPENDS ${FAKE_CUDA_STUB_SRC}
        COMMENT "Compiling fake shared library: libc10_cuda.so"
        VERBATIM
    )
    add_custom_target(fake_libc10_cuda DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libc10_cuda.so)
  endif()

  add_dependencies(${TARGET} fake_libtorch_cuda fake_libnvrtc fake_libc10_cuda)
  target_link_libraries(${TARGET} PRIVATE
      "-Wl,--no-as-needed"
      "${CMAKE_CURRENT_BINARY_DIR}/libtorch_cuda.so"
      "-Wl,${CMAKE_CURRENT_BINARY_DIR}/libnvrtc.so.13"
      "${CMAKE_CURRENT_BINARY_DIR}/libc10_cuda.so"
      "-Wl,--as-needed")
endfunction()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

if(DEFINED ENV{BIREN_HOME})
    # fullstack environment
    set(FULLSTACK_PATH $ENV{BIREN_HOME})
else()
    # SDK environment
    set(FULLSTACK_PATH /usr/local/birensupa/sdk/latest/)
endif()

function(ADD_SUPA_PYTHON_LIBRARY TARGET_NAME)
    CMAKE_PARSE_ARGUMENTS(PARSE_ARGV 1
    ARG
    ""
    "DESTINATION;ARCH"
    "CXX_FLAGS;SU_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES;LIBRARY_DIRS;SOURCES")


    set(SOURCES "")
    set(OBJECTS "")
    set(BRCC_INCLUDES "")
    set(SYSTEM_INCLUDES "")
    set(DEFINES "")
    set(GPU_ARCH "arch_20")
    set(CPP_BASE_FLAGS
    "-Wl,-z,relro,-z,now,-z,noexecstack;-fstack-protector-all;-fPIE;-pie;-fPIC;-faligned-new"
    "-fvisibility=hidden;-fno-math-errno;-fno-trapping-math;-finline-functions;-fno-omit-frame-pointer;-rdynamic"
    "-Werror;-Wall;-Wextra;-Wno-terminate;-Wno-error=terminate;-Wno-narrowing;-Wno-missing-field-initializers"
    "-Wno-type-limits;-Wno-array-bounds;-Wno-unknown-pragmas;-Wno-sign-compare;-Wno-unused-parameter;-Wno-unused-function"
    "-Wno-unused-result;-Wno-strict-overflow;-Wno-strict-aliasing;-Wno-deprecated-declarations;-Wno-ignored-qualifiers"
    "-Wno-write-strings;-Wno-deprecated-copy;-Wno-dangling-reference;-Wno-stringop-overflow;-Wno-error=pedantic"
    "-Wno-error=redundant-decls;-Wno-error=old-style-cast;-Wno-unused-but-set-variable;-Wno-uninitialized")

    if(ENABLE_COVERAGE)
      list(APPEND CPP_BASE_FLAGS --coverage)
    endif()

    if(DEFINED ARG_ARCH)
      set(GPU_ARCH ${ARG_ARCH})
    endif()

    set(SU_BASE_FLAGS
    "--supa-gpu-arch=${GPU_ARCH};-std=c++17;-fopenmp;-fdeclspec;-fPIC"
    "-Werror;-Wno-pass-failed;-Wno-deprecated-builtins;-Wno-pass-failed;-Wno-absolute-value")


    if(NOT TORCH_INCLUDE_DIRS)
      message(FATAL_ERROR "TORCH_INCLUDE_DIRS is empty. Please run 'find_package(Torch REQUIRED)' first.")
    endif()
    if(NOT Python_INCLUDE_DIRS)
      message(FATAL_ERROR "Python_INCLUDE_DIRS is empty. Please run 'find_package(Python REQUIRED)' first.")
    endif()

    # process python and torch headers as system in order to disable some warnings
    foreach(INC ${Python_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
      list(APPEND SYSTEM_INCLUDES -isystem "${INC}")
    endforeach()

    run_python(ABI "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" "Failed to query ABI version.")
    list(APPEND DEFINES -DTORCH_EXTENSION_NAME=${TARGET_NAME} -DPy_LIMITED_API=3)

    # add custom command for su files. process args for brcc
    foreach(INC ${ARG_INCLUDE_DIRECTORIES})
      list(APPEND BRCC_INCLUDES "-I${INC}")
    endforeach()

    set(SU_FLAGS ${ARG_SU_FLAGS} ${SU_BASE_FLAGS} ${DEFINES})

    separate_arguments(REL_FLAGS NATIVE_COMMAND ${CMAKE_CXX_FLAGS_DEBUG})
    if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      separate_arguments(REL_FLAGS NATIVE_COMMAND ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
      separate_arguments(REL_FLAGS NATIVE_COMMAND ${CMAKE_CXX_FLAGS_RELEASE})
    endif()

    foreach(SRC ${ARG_SOURCES})
        get_filename_component(SRC_EXT ${SRC} LAST_EXT)
        if(SRC_EXT STREQUAL ".su")
            set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET_NAME}.dir/${SRC}.o")
            set(DEP_FILE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET_NAME}.dir/${SRC}.d")
            set(CMDLINE brcc ${SYSTEM_INCLUDES} ${BRCC_INCLUDES} ${SU_FLAGS} ${REL_FLAGS} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SRC} -o ${OBJ_FILE})
            add_custom_command(
                OUTPUT ${OBJ_FILE}
                COMMAND ${CMAKE_CXX_COMPILER_LAUNCHER} ${CMDLINE}
                -MMD -MF ${DEP_FILE}
                DEPFILE ${DEP_FILE}
                DEPENDS ${SRC}
                COMMENT "Building BRCC (Device) object ${OBJ_FILE}"
            )

            if(VERBOSE)
              # save command line to su_compile_commands for debug.
              string(REPLACE ";" " " CMD_LINE_STR "${CMDLINE}")
              set(CUSTOM_ENTRY "{\"directory\": \"${CMAKE_BINARY_DIR}\",\n\"command\": \"${CMD_LINE_STR}\",\n\"file\": \"${CMAKE_CURRENT_SOURCE_DIR}/${SRC}\"},\n")
              file(APPEND "${CMAKE_BINARY_DIR}/su_compile_commands.json" "${CUSTOM_ENTRY}")
            endif()

            list(APPEND OBJECTS ${OBJ_FILE})
        else()
            list(APPEND CPP_SOURCES ${SRC})
        endif()
    endforeach()

    # create python extension module with stable ABI.
    Python_add_library(${TARGET_NAME} MODULE USE_SABI 3 WITH_SOABI ${CPP_SOURCES})
    target_sources(${TARGET_NAME} PRIVATE ${OBJECTS})

    target_compile_definitions(${TARGET_NAME} PRIVATE ${DEFINES})
    target_compile_options(${TARGET_NAME} PRIVATE ${CPP_BASE_FLAGS})
    target_include_directories(${TARGET_NAME} PRIVATE ${SYSTEM_INCLUDES} ${ARG_INCLUDE_DIRECTORIES})

    target_link_directories(${TARGET_NAME} PRIVATE ${ARG_LIBRARY_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${ARG_LIBRARIES})

    install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION ${ARG_DESTINATION} COMPONENT ${TARGET_NAME})

    set_target_properties(${TARGET_NAME} PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_RPATH_USE_ORIGIN TRUE
      INSTALL_RPATH_USE_LINK_PATH TRUE
    )

endfunction()

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_ir_mem CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Desktop: discover Torch via the active Python environment ----
# Officially recommended: query torch.utils.cmake_prefix_path if PyTorch installed via pip. 
if(NOT DEFINED PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if(TORCH_CMAKE_PREFIX STREQUAL "")
  message(FATAL_ERROR "Could not query torch.utils.cmake_prefix_path. Activate your venv and ensure torch is installed.")
endif()

# Satisfy potential Torch interface expectations (some builds expect CUDA::nvToolsExt).
# If your TorchConfig already defines it, we won't override it.
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")

# Data path baked into binary
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(DATA_DIR "${REPO_ROOT}/data")
message(STATUS "DATA_DIR = ${DATA_DIR}")

add_executable(torch_ir_mem main.cpp)
target_link_libraries(torch_ir_mem "${TORCH_LIBRARIES}")
target_compile_definitions(torch_ir_mem PRIVATE DATA_DIR="${DATA_DIR}")
target_compile_options(torch_ir_mem PRIVATE ${TORCH_CXX_FLAGS})

# Make runtime loader find torch libs without LD_LIBRARY_PATH.
# TorchConfig typically sets TORCH_INSTALL_PREFIX.
if(DEFINED TORCH_INSTALL_PREFIX)
  set_target_properties(torch_ir_mem PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
endif()

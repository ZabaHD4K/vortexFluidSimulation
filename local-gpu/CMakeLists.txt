cmake_minimum_required(VERSION 3.24)
project(VortexGPU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)  # SM 12.0 for RTX 5060

# ---- Dependencies via FetchContent ----
include(FetchContent)

FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4 GIT_SHALLOW ON)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1 GIT_SHALLOW ON)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.8 GIT_SHALLOW ON)
FetchContent_MakeAvailable(imgui)

# ---- GLAD 2 (OpenGL 4.6 core loader) ----
add_library(glad STATIC extern/glad/src/gl.c)
target_include_directories(glad PUBLIC extern/glad/include)

# ---- ImGui library ----
add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_lib PUBLIC glfw glad)

# ---- Main executable ----
add_executable(vortex_gpu
  src/main.cpp
  src/app.cpp
  src/sph/sph_system.cu
  src/render/renderer.cpp
  src/render/fluid_renderer.cpp
  src/render/box_renderer.cpp
  src/ui/ui.cpp
  src/util/gl_utils.cpp
)

target_include_directories(vortex_gpu PRIVATE src)
target_compile_options(vortex_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)
target_link_libraries(vortex_gpu PRIVATE
  glfw glad glm::glm imgui_lib
  CUDA::cudart
)

# CUDA-GL interop
find_package(CUDAToolkit REQUIRED)
target_link_libraries(vortex_gpu PRIVATE CUDA::cuda_driver)

# Copy shaders to build directory
add_custom_command(TARGET vortex_gpu POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/render/shaders
    $<TARGET_FILE_DIR:vortex_gpu>/shaders
)

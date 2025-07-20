cmake_minimum_required (VERSION 3.8)

project(lava CXX)

configure_file("${PROJECT_SOURCE_DIR}/lava/utils/path.h.in"
               "${PROJECT_BINARY_DIR}/lava/path.h")

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(QUARTZCORE_LIBRARY QuartzCore)

MARK_AS_ADVANCED(METAL_LIBRARY FOUNDATION_LIBRARY QUARTZCORE_LIBRARY)
SET(EXTRA_LIBS ${METAL_LIBRARY} ${FOUNDATION_LIBRARY} ${QUARTZCORE_LIBRARY})


pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)
pkg_check_modules(onnxruntime REQUIRED IMPORTED_TARGET libonnxruntime)
pkg_check_modules(tbb REQUIRED IMPORTED_TARGET tbb)
pkg_check_modules(glfw3 REQUIRED IMPORTED_TARGET glfw3)

set(SRC_FILES ${PROJECT_SOURCE_DIR}/lava/imgui/imgui_demo.cpp
              ${PROJECT_SOURCE_DIR}/lava/imgui/imgui_draw.cpp
              ${PROJECT_SOURCE_DIR}/lava/imgui/imgui_tables.cpp
              ${PROJECT_SOURCE_DIR}/lava/imgui/imgui_widgets.cpp 
              ${PROJECT_SOURCE_DIR}/lava/imgui/imgui_impl_glfw.cpp
              ${PROJECT_SOURCE_DIR}/lava/imgui/imgui.cpp
              ${PROJECT_SOURCE_DIR}/lava/imgui/imgui_impl_metal.mm)

ADD_EXECUTABLE(exe main.cpp ${SRC_FILES})
target_compile_definitions(exe PRIVATE IMGUI_IMPL_METAL_CPP)
target_link_libraries(exe PUBLIC   ${EXTRA_LIBS} PkgConfig::glfw3 PkgConfig::tbb PkgConfig::opencv PkgConfig::onnxruntime Threads::Threads)
#for Metal <metal/metal.h> != "metal/metal.h"
target_include_directories(exe SYSTEM PUBLIC ${CMAKE_SOURCE_DIR})

#add_subdirectory(sandbox)

# Locate GTest
#enable_testing()
#add_subdirectory(test)



cmake_minimum_required (VERSION 3.10)

project(lava CXX)

configure_file("${PROJECT_SOURCE_DIR}/lava/utils/path.h.in"
               "${PROJECT_BINARY_DIR}/lava/path.h")

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)
pkg_check_modules(onnxruntime REQUIRED IMPORTED_TARGET libonnxruntime)
pkg_check_modules(tbb REQUIRED IMPORTED_TARGET tbb)

ADD_EXECUTABLE(exe main.cpp lava/app/application.cpp ${SRC_FILES})
target_link_libraries(exe PUBLIC   ${EXTRA_LIBS} PkgConfig::tbb PkgConfig::opencv PkgConfig::onnxruntime Threads::Threads)


cmake_minimum_required (VERSION 3.10)

project(lava CXX)

configure_file("${PROJECT_SOURCE_DIR}/lava/utils/path.h.in"
               "${PROJECT_BINARY_DIR}/lava/path.h")

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

find_package(Threads REQUIRED)

if(UNIX)
    # --- Linux / macOS ---
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)
    pkg_check_modules(onnxruntime REQUIRED IMPORTED_TARGET libonnxruntime)
    pkg_check_modules(tbb REQUIRED IMPORTED_TARGET tbb)

    set(LAVA_DEPS
        PkgConfig::opencv
        PkgConfig::onnxruntime
        PkgConfig::tbb
        Threads::Threads
    )

elseif(WIN32)
    # --- Windows ---
    # OpenCV
    find_package(OpenCV REQUIRED)
    # oneTBB
    find_package(TBB REQUIRED tbb)
    # ONNX Runtime (adjust PATHS to where you unpacked the SDK)
    set(ONNXRUNTIME_DIR "C:\\opt\\onnxruntime\\Microsoft.ML.OnnxRuntime.DirectML.1.22.1")
    # OpenVINO
    set(OPENVINO_DIR "C:\\opt\\openvino")

    # Includes
    include_directories( ${ONNXRUNTIME_DIR}/build/native/include)
#   include_directories( ${OPENVINO_DIR}/include)
    # Link dirs (contains onnxruntime.lib and provider DLLs)
    link_directories(${ONNXRUNTIME_DIR}/runtimes/win-x64/native)
#   link_directories(${OPENVINO_DIR}/lib)

    set(LAVA_DEPS
        ${OpenCV_LIBS}
        TBB::tbb
        Threads::Threads
        onnxruntime
    )
endif()

add_executable(exe
    main.cpp
    lava/app/application.cpp
    ${SRC_FILES}
)

target_link_libraries(exe PUBLIC ${EXTRA_LIBS} ${LAVA_DEPS})

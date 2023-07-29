ADD_EXECUTABLE(exe main.cpp)
target_link_libraries(exe PUBLIC PkgConfig::tbb PkgConfig::opencv PkgConfig::onnxruntime Threads::Threads)

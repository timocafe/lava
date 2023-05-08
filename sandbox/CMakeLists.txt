ADD_EXECUTABLE(exe main.cpp)
target_link_libraries(exe PUBLIC PkgConfig::tbb TBB::tbbmalloc_proxy PkgConfig::opencv PkgConfig::onnxruntime Threads::Threads)

add_library(lavarandom SHARED lava.cpp)
target_link_libraries(lavarandom PUBLIC PkgConfig::tbb PkgConfig::opencv PkgConfig::onnxruntime)

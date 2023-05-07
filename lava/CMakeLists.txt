add_library(lavarandom SHARED lava.cpp)
target_link_libraries(lavarandom PUBLIC Threads::Threads TBB::tbb PkgConfig::opencv PkgConfig::openssl)

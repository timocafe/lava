//
//  utils.h
//  experiment
//
//  Created by timothee.ewart on 5/6/20.
//

#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "lava/sha256/sha256.h"

namespace lava {

struct tiny_timer {
  tiny_timer(const std::string &name) : name_(name) {
    start_ = std::chrono::system_clock::now();
  }

  ~tiny_timer() {
    end_ = std::chrono::system_clock::now();
    auto sim_ms =
        std::chrono::duration<float, std::chrono::milliseconds::period>(end_ -
                                                                        start_);
    std::string s = "[qnt]: " + name_ + " ";
    std::cout << s << sim_ms.count() << " [ms]\n";
  }

  std::chrono::time_point<std::chrono::system_clock> start_;
  std::chrono::time_point<std::chrono::system_clock> end_;
  std::string name_;
};

// compute the sha256 from the image
inline std::string make_sha256(const std::vector<uint8_t> &v,
                               const std::string &model) {
  std::string hash_hex_str;
  picosha2::hash256_hex_string(v, hash_hex_str);
  return hash_hex_str;
}

} // namespace lava

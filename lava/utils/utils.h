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

} // namespace lava

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

#include <openssl/sha.h>

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

// hex array to string readable
inline std::string
sha256_to_str(const std::array<uint8_t, SHA256_DIGEST_LENGTH> array) {
  std::stringstream ss;
  ss << std::hex;
  for (int i = 0; i < array.size(); ++i)
    ss << std::setw(2) << std::setfill('0') << (int)array[i];
  return ss.str();
}

// compute the sha256 from the image
inline std::string make_sha256(const std::vector<uint8_t> &v,
                               const std::string &model) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> sha256;
  SHA256(v.data(), v.size(), (unsigned char *)sha256.data());
  return sha256_to_str(sha256);
}

} // namespace lava

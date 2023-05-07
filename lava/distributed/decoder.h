
//
//  decoder.h
//  experiment
//
//  Created by timothee.ewart on 5/6/20.
//

#pragma once


#include "oneapi/tbb/parallel_pipeline.h"

#include "lava/distributed/node.h"

namespace lava {

struct lavadom {

  explicit lavadom(const std::string &model = "lava.onnx") : ml_(ml(model)) {}

  // functor for the pipeline
  void operator()() {
    try {
      oneapi::tbb::parallel_pipeline(
          ntokens_,
          // get the images from the camera
          oneapi::tbb::make_filter<void, message<uint8_t>>(
              oneapi::tbb::filter_mode::parallel,
              [&](oneapi::tbb::flow_control &fc) -> message<uint8_t> {
                return generator_(fc);
              }) &
              // perform ML model
              oneapi::tbb::make_filter<message<uint8_t>, message<uint8_t>>(
                  oneapi::tbb::filter_mode::parallel,
                  [&](const message<uint8_t> &m) -> message<uint8_t> {
                    return ml_(m);
                  }) &
              // show image with the randomnumber
              oneapi::tbb::make_filter<message<uint8_t>, void>(
                  oneapi::tbb::filter_mode::parallel,
                  [&](const message<uint8_t> &m) { return show_(m); }));
    } catch (std::out_of_range &e) {
      std::cerr << "ERROR: somthing else" << std::endl;
      throw e;
    }
  }
  // accessor number of tokens (batchs) for oneapi::tbb
  inline std::size_t &ntokens() { return ntokens_; }
  inline const std::size_t &ntokens() const { return ntokens_; }

  std::size_t ntokens_ = {1}; // number of tokens available
  generator generator_;
  ml ml_;
  show show_;
};

} // namespace lava

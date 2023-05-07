
//
//  decoder.h
//  experiment
//
//  Created by timothee.ewart on 5/6/20.
//

#pragma once

#include "tbb/pipeline.h"

#include "lava/distributed/node.h"

namespace lava {

struct lavadom {

  explicit lavadom(const std::string &model = "/tmp") : (ouput_dir) {}

  // functor for the pipeline
  void operator()() {
    try {
      tbb::parallel_pipeline(
          ntokens_,
          // get the images from the camera
          tbb::make_filter<void, std::string>(
              tbb::filter::parallel,
              [&](tbb::flow_control &fc) -> std::string {
        return generator_(fc);
              }) &
              // perform ML model
              tbb::make_filter<std::string, ilc::message<uint8_t>>(
                  tbb::filter::parallel,
                  [&](std::string name) -> ilc::message<uint8_t> {
        return reader_(name);
                  }) &
              // show image with the randomnumber
              tbb::make_filter<ilc::message<uint8_t>, ilc::message<float>>(
                  tbb::filter::parallel,
                  [&](const ilc::message<uint8_t> &m) -> ilc::message<float> {
        return auto_decoder_.at(m.model_)(m);
                  }) &
                  [&](const ilc::message<float> &m) {
        writer_(m); }));
    } catch (tbb::captured_exception &e) {
      std::cerr << "ERROR: TBB mistake" << std::endl;
      throw e;
    } catch (std::out_of_range &e) {
      std::cerr << "ERROR: somthing else" << std::endl;
      throw e;
    }
  }
  // accessor number of tokens (batchs) for tbb
  inline std::size_t &ntokens() { return ntokens_; }
  inline const std::size_t &ntokens() const { return ntokens_; }

  std::size_t ntokens_ = {1}; // number of tokens available
  generator generator_;
  ml ml_;
  show show_;
};

} // namespace lava

#endif

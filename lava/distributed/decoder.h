
// MIT License

// Copyright (c) 2023 Timothee EWART

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <opencv2/opencv.hpp>

#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/parallel_pipeline.h>

#include "lava/distributed/node.h"

namespace lava {

struct lavadom {

  explicit lavadom(
      const std::string &model = "lava.onnx",
      std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>> q =
          std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>>(),
      std::shared_ptr<oneapi::tbb::concurrent_queue<cv::Mat>> qm =
          std::shared_ptr<oneapi::tbb::concurrent_queue<cv::Mat>>())
      : ml_(ml(model)), chat_(q){};

  // functor for the pipeline
  void operator()() {
    try {
      oneapi::tbb::parallel_pipeline(
          ntokens_,
          // get the images from the camera
          oneapi::tbb::make_filter<void, cv::Mat>(
              oneapi::tbb::filter_mode::parallel,
              [&](oneapi::tbb::flow_control &fc) -> cv::Mat {
                return generator_(fc);
              }) &
              // perform ML model
              oneapi::tbb::make_filter<cv::Mat, cv::Mat>(
                  oneapi::tbb::filter_mode::serial_in_order,
                  [&](const cv::Mat &m) -> cv::Mat { return ml_(m); }) &
              // show image with the randomnumber
              oneapi::tbb::make_filter<cv::Mat, void>(
                  oneapi::tbb::filter_mode::parallel,
                  [&](const cv::Mat &m) { chat_(m); }));
    } catch (std::out_of_range &e) {
      std::cerr << "ERROR: somthing else" << std::endl;
      throw e;
    }
  }

  std::size_t ntokens_ = {2}; // number of tokens available
  generator generator_;
  ml ml_;
  chat chat_;
};

} // namespace lava

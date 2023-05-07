//
//  node.h
//  experiment
//
//  Created by timothee.ewart on 5/6/20.
//

#ifndef node__h
#define node__h
// std
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <variant>
// OPenCV
#include <opencv2/opencv.hpp>
// ONNX
#include <onnxruntime_cxx_api.h>
// TBB
#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/parallel_pipeline.h"
// ME
#include "lava/distributed/message.h"
#include "lava/utils/utils.h"

namespace lava {

struct helper_onnx {
  helper_onnx(const std::string &model = std::string(),
              const std::size_t threads = 1) {
    if (!std::filesystem::exists(model))
      throw std::runtime_error("ml model does not exist \n");
    memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    session_options_ = Ort::SessionOptions();
    session_options_.SetInterOpNumThreads(threads);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tutu");
    session_ = Ort::Session(env_, model.c_str(), session_options_);
  }

  Ort::MemoryInfo memory_info_{nullptr};
  Ort::SessionOptions session_options_{nullptr};
  Ort::Session session_{nullptr};
  Ort::Env env_{nullptr};
};

struct generator {

  generator() {
    int deviceID = 0;        // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API
    // open selected camera using selected API
    cap_.open(deviceID, apiID);
    if (!cap_.isOpened())
      throw std::runtime_error("Error! Unable to open camera.\n");
  }

  message<uint8_t> operator()(tbb::flow_control &fc) {
    message<uint8_t> m;
    m.data().reserve(1080 * 1920);
    auto data = m.data();
    cv::Mat frame(data);
    if (cap_.read(frame)) {
      return std::move(m);
    } else {
      fc.stop();
      return message<uint8_t>();
    }
  }

  tbb::concurrent_queue<std::string> qname_;
  cv::VideoCapture cap_;
};

struct show {
  // using auto because the type is insane ...
  void operator()(const message<uint8_t> &m) {}
};

struct ml {
  explicit ml(const std::string &model_path = std::string(),
              const std::size_t threads = 1)
      : model_path_(model_path), ho_(model_path), threads_(threads) {}

  message<uint8_t> operator()(const message<uint8_t> &image) {
    message<uint8_t> m2;
    auto &memory_info = ho_.memory_info_;
    auto &session = ho_.session_;
    auto input_shape =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_shape =
        session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    const auto size_input = input_shape[0] * input_shape[1];
    const auto size_output = output_shape[0] * output_shape[1];

    // Ort::Value input_tensor_ =
    //     Ort::Value::CreateTensor<float>(memory_info, x_data, size_input,
    //                                     input_shape.data(),
    //                                     input_shape.size());
    // Ort::Value output_tensor_ = Ort::Value::CreateTensor<float>(
    //     memory_info, y_data, size_output, output_shape.data(),
    //     output_shape.size());
    // const char *input_names[] = {"x"};
    // const char *output_names[] = {"dense_3"};

    // session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1,
    //             output_names, &output_tensor_, 1);
    return std::move(m2);
  }

  std::filesystem::path model_path_ = std::filesystem::path();
  helper_onnx ho_;
  std::size_t threads_;
};

} // namespace lava

#endif /* node__h */

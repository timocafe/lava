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
#include "tbb/concurrent_queue.h"
#include "tbb/pipeline.h"
// ME
#include "lava/distributed/message.h"
#include "lava/utils/utils.h"

namespace lava {

struct helper_onnx {
  helper_onnx(const std::string &model = std::string(),
              const std::size_t threads = 1) {
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
    capi_.open(deviceID, apiID);
    if (!cap.isOpened())
      throw std::runtime_error("Error! Unable to open camera.\n");
  }

  std::string operator()(tbb::flow_control &fc) {
    std::vector while (!qname_.empty()) {
      std::string name;
      if (qname_.try_pop(name)) {
        return name;
      }
    }

    fc.stop();
    return std::string();
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
      : model_path_(model_path), threads_(threads) {
    if (std::filesystem::exists(model_path_))
      ho_ = ho_(model, threads);
    else
      throw std::runtime_error("ml model does not exist \n");
  }

  message<uint8_t> operator()(const message<int32_t> &image) {
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
  }

  std::filesystem::path model_path_ = std::filesystem::path();
  helper_onnx ho_;
};

} // namespace lava

#endif /* node__h */

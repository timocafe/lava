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
#include <numeric>
#include <variant>
// OPenCV
#include <opencv2/opencv.hpp>
// ONNX
#include <onnxruntime_cxx_api.h>
// TBB
#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/parallel_pipeline.h"
// ME
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

  cv::Mat operator()(tbb::flow_control &fc) {
    cv::Mat frame;
    if (cap_.read(frame)) {
      return std::move(frame);
    } else {
      fc.stop();
      return cv::Mat();
    }
  }

  cv::VideoCapture cap_;
};

struct chat {
  chat(
      const std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>> &q =
          std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>>())
      : q_(q) {}
  void operator()(const cv::Mat &image) {
    std::string sha = picosha2::hash256_hex_string(image.begin<uint8_t>(),
                                                   image.end<uint8_t>());
    cv::Mat frame;
    frame = image;
    write_text(frame, sha);
    q_->try_push(frame);
  }
  void write_text(cv::Mat &image, const std::string &sha) {
    cv::Point text_position(300, 80);     // Declaring the text position
    int font_size = 1;                    // Declaring the font size
    cv::Scalar font_Color(255, 255, 255); // Declaring the color of the font
    int font_weight = 2;                  // Declaring the font weight
    cv::putText(image, sha, text_position, cv::FONT_HERSHEY_COMPLEX, font_size,
                font_Color, font_weight);
  }

  std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>> q_;
};

struct ml {
  explicit ml(const std::string &model_path = std::string(),
              const std::size_t threads = 1)
      : model_path_(model_path), ho_(model_path), threads_(threads) {}

  cv::Mat operator()(const cv::Mat &image) {
    auto &memory_info = ho_.memory_info_;
    auto &session = ho_.session_;
    Ort::AllocatorWithDefaultOptions allocator;
    int input_count = session.GetInputCount();
    int output_count = session.GetOutputCount();
    std::vector<std::string> input_names(input_count);
    std::vector<std::string> output_names(output_count);
    input_names.resize(input_count);
    output_names.resize(output_count);
    for (int i = 0; i < input_count; i++)
      input_names[i] = &*session.GetInputNameAllocated(i, allocator);

    for (int i = 0; i < output_count; i++)
      output_names[i] = &*session.GetOutputNameAllocated(i, allocator);

    // auto toto = session.GetInputNames();

    // auto input_shape =
    //     session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // auto output_shape =
    //     session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    // const auto size_input =
    //     std::accumulate(input_shape.begin(), input_shape.end(), 1,
    //                     [](int a, int b) { return a * b; });
    // const auto size_output =
    //     std::accumulate(output_shape.begin(), output_shape.end(), 1,
    //                     [](int a, int b) { return a * b; });

    // // resize to 640x640 the input
    // cv::Size new_size(640, 640);
    // cv::Mat imgRGBFloat;
    // image.convertTo(imgRGBFloat, CV_32F);
    // cv::resize(imgRGBFloat, imgRGBFloat, new_size);

    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    //     memory_info, reinterpret_cast<float *>(imgRGBFloat.data),
    //     size_input, input_shape.data(), input_shape.size());

    // std::vector<Ort::Value> outputTensors =
    //     session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
    //     &input_tensor,
    //                 1, outputNames.data(), outputNames.size());

    // session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1,
    //             output_names, &output_tensor_, 1);
    return std::move(image);
  }

  std::filesystem::path model_path_ = std::filesystem::path();
  helper_onnx ho_;
  std::size_t threads_;
};

} // namespace lava

#endif /* node__h */


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
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/parallel_for.h"
// MIAOU
#include "lava/sha256/sha256.h"

namespace lava {

struct constant {
  const static int nimages = 30;
  const static int model_width = 640;
  const static int model_high = 640;
};

std::vector<std::string> get_input_name(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::size_t input_count = session.GetInputCount();
  std::vector<std::string> input_names(input_count);
  for (int i = 0; i < input_count; i++)
    input_names[i] = &*session.GetInputNameAllocated(i, allocator);

  return input_names;
};

std::vector<std::string> get_output_name(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::size_t output_count = session.GetOutputCount();
  std::vector<std::string> output_names(output_count);
  for (int i = 0; i < output_count; i++)
    output_names[i] = &*session.GetOutputNameAllocated(i, allocator);

  return output_names;
};

std::vector<const char *> cnames(const std::vector<std::string> &names) {
  std::vector<const char *> ctnames(names.size());
  std::transform(names.begin(), names.end(), ctnames.begin(),
                 [](const std::string &s) { return s.c_str(); });
  return ctnames;
}

const int64_t get_size(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         [](int a, int b) { return a * b; });
}

struct helper_onnx {
  helper_onnx(const std::string &model = std::string(),
              const std::size_t threads = 1) {
    if (!std::filesystem::exists(model))
      throw std::runtime_error("ml model does not exist \n");
    memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    session_options_ = Ort::SessionOptions();
    uint32_t coreml_flags = 0;
    session_options_.SetInterOpNumThreads(threads);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
    //     session_options_, coreml_flags));
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tutu");
    session_ = Ort::Session(env_, model.c_str(), session_options_);
    input_name_ = get_input_name(session_);
    output_name_ = get_output_name(session_);
  }

  Ort::MemoryInfo memory_info_{nullptr};
  Ort::SessionOptions session_options_{nullptr};
  Ort::Session session_{nullptr};
  Ort::Env env_{nullptr};
  std::vector<std::string> input_name_;
  std::vector<std::string> output_name_;
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

  std::vector<cv::Mat> operator()(tbb::flow_control &fc) {
    std::vector<cv::Mat> v(constant::nimages);
    cv::Mat frame;
    for (int i = 0; i < constant::nimages; ++i) {
      if (cap_.read(frame)) {
        v[i] = frame;
      } else {
        fc.stop();
      }
    }
    return v;
  }

  cv::VideoCapture cap_;
};

void write(cv::Mat &image, const std::string &sha, cv::Point text_position) {
  int font_size = 1;                    // Declaring the font size
  cv::Scalar font_Color(255, 255, 255); // Declaring the color of the font
  int font_weight = 2;                  // Declaring the font weight
  cv::putText(image, sha, text_position, cv::FONT_HERSHEY_COMPLEX, font_size,
              font_Color, font_weight);
}

struct chat {
  chat(
      const std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>> &q =
          std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>>())
      : q_(q) {}
  void operator()(const std::vector<cv::Mat> &images) {
    // std::string sha = picosha2::hash256_hex_string(image.begin<uint8_t>(),
    //                                                image.end<uint8_t>());
    //  write big chat
    //   write_text(frame, sha);
    for (auto &image : images)
      q_->try_push(image);
  }

  void write_text(cv::Mat &image, const std::string &sha) {
    cv::Point text_position(200, 80); // Declaring the text position
    write(image, sha, text_position);
  }

  std::shared_ptr<oneapi::tbb::concurrent_bounded_queue<cv::Mat>> q_;
};

cv::Mat make_input_image(const std::vector<cv::Mat> &image) {
  std::vector<cv::Mat> clone(constant::nimages);

  for (int i = 0; i < constant::nimages; ++i) {
    clone[i] = image[i].clone();
    cv::resize(clone[i], clone[i], cv::Size(640, 640));
  }
  return cv::dnn::blobFromImages(clone, 1. / 255, cv::Size(640, 640),
                                 cv::Scalar(), true, false);
}

struct Detection {
  float confidence;
  cv::Rect box;
  std::string sha_;
};

void detect(const cv::Mat &input_image, Ort::Value &output_tensor,
            std::vector<Detection> &output, std::vector<float> &confidences,
            std::vector<cv::Rect> &boxes) {
  // model trains with 640,640 inputs
  float x_factor = input_image.cols / 640.;
  float y_factor = input_image.rows / 640.;

  float *data = output_tensor.GetTensorMutableData<float>();
  std::vector<int64_t> outputShape =
      output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  size_t count = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  // OpenCV is a pain with coordinates which are != from usual (x,y) orthogonal
  // basis
  cv::Mat l_Mat =
      cv::Mat(outputShape[1], outputShape[2], CV_32FC1, (void *)data);
  cv::Mat l_Mat_t = l_Mat.t();
  // first 5 elements are box[4] and obj confidence
  int numClasses = l_Mat_t.cols - 4;

  for (int l_Row = 0; l_Row < l_Mat_t.rows; l_Row++) {
    cv::Mat l_MatRow = l_Mat_t.row(l_Row);
    const float objConf = l_MatRow.at<float>(0, 4);

    if (objConf > 0.4) {
      confidences.push_back(objConf);
      float x = (l_MatRow.at<float>(0, 0));
      float y = (l_MatRow.at<float>(0, 1));
      float w = (l_MatRow.at<float>(0, 2));
      float h = (l_MatRow.at<float>(0, 3));
      int left = int((x - 0.5 * w) * x_factor);
      int top = int((y - 0.5 * h) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);
      boxes.push_back(cv::Rect(left, top, width, height));
    }
  }

  const float SCORE_THRESHOLD = 0.2;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.4;

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    nms_result);
  for (int i = 0; i < nms_result.size(); i++) {
    int idx = nms_result[i];
    output.push_back({confidences[idx], boxes[idx]});
  }
}

void compute_sha_buble(const cv::Mat &image,
                       std::vector<Detection> &detections) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, detections.size()),
      [&](tbb::blocked_range<int> r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          auto &detection = detections[i];
          auto box = detection.box;

          if (box.x >= 0 && box.y >= 0 && box.width + box.x < image.cols &&
              box.height + box.y < image.rows) {
            cv::Mat sub_image = image(cv::Range(box.y, box.y + box.height),
                                      cv::Range(box.x, box.x + box.width));
            std::string sha = picosha2::hash256_hex_string(
                sub_image.begin<uint8_t>(), sub_image.end<uint8_t>());
            detection.sha_ = sha.substr(1, 8);
          }
        }
      });
}

void mark_image(const cv::Mat &image,
                const std::vector<Detection> &detections) {
  for (const auto &detection : detections) {
    auto box = detection.box;
    const auto color = cv::Scalar(187, 114, 0); // bgr french blue
    cv::rectangle(image, box, color, 3);
    cv::rectangle(image, cv::Point(box.x, box.y - 20),
                  cv::Point(box.x + box.width, box.y), color, cv::FILLED);
    cv::putText(image, detection.sha_, cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }
}

struct ml {
  explicit ml(const std::string &model_path = std::string(),
              const std::size_t threads = 1)
      : model_path_(model_path), ho_(model_path), threads_(threads) {
    detections_.reserve(32);
    confidences_.reserve(32);
    boxes_.reserve(32);
  }

  std::vector<cv::Mat> operator()(const std::vector<cv::Mat> &images) {
    auto &memory_info = ho_.memory_info_;
    auto &session = ho_.session_;
    cinput_name_ = cnames(ho_.input_name_);
    coutput_name_ = cnames(ho_.output_name_);

    std::vector<int64_t> input_shape = {constant::nimages, 3, 640, 640};

    const auto input_size = get_size(input_shape);

    const auto &batch = make_input_image(images);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, reinterpret_cast<float *>(batch.data), input_size,
        input_shape.data(), input_shape.size());

    auto output = session.Run(Ort::RunOptions{nullptr}, cinput_name_.data(),
                              &input_tensor, cinput_name_.size(),
                              coutput_name_.data(), coutput_name_.size());

    for (int i = 0; i < constant::nimages; ++i) {
      auto &image = images[i];
      auto &output_tensor = output[i];

      // detect the bubbles
      detect(image, output_tensor, detections_, confidences_, boxes_);
      // compute every sha
      compute_sha_buble(image, detections_);
      // mark the original image with rectangle
      mark_image(image, detections_);
      detections_.clear();
      confidences_.clear();
      boxes_.clear();
    }
    return std::move(images);
  }

  std::vector<const char *> cinput_name_;
  std::vector<const char *> coutput_name_;
  std::vector<float> confidences_;
  std::vector<cv::Rect> boxes_;
  std::vector<Detection> detections_;
  std::filesystem::path model_path_ = std::filesystem::path();
  helper_onnx ho_;
  std::size_t threads_;
};

} // namespace lava

#endif /* node__h */

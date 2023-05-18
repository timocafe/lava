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

std::vector<std::string> get_input_name(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  int input_count = session.GetInputCount();
  std::vector<std::string> input_names(input_count);
  for (int i = 0; i < input_count; i++)
    input_names[i] = &*session.GetInputNameAllocated(i, allocator);

  return input_names;
};

std::vector<std::string> get_output_name(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  int output_count = session.GetOutputCount();
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
    session_options_.SetInterOpNumThreads(threads);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
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

cv::Mat make_input_image(const cv::Mat &image) {
  auto nimage = image.clone();
  cv::resize(nimage, nimage, cv::Size(640, 640));
  return cv::dnn::blobFromImage(nimage, 1. / 255, cv::Size(640, 640),
                                cv::Scalar(), true, false);
}

struct Detection {
  int class_id;
  float confidence;
  cv::Rect box;
};

void getBestClassInfo(const cv::Mat &p_Mat, const int &numClasses,
                      float &bestConf, int &bestClassId) {
  bestClassId = 0;
  bestConf = 0;

  if (p_Mat.rows && p_Mat.cols) {
    for (int i = 0; i < numClasses; i++) {
      if (p_Mat.at<float>(0, i + 4) > bestConf) {
        bestConf = p_Mat.at<float>(0, i + 4);
        bestClassId = i;
      }
    }
  }
}

void detect(const cv::Mat &input_image, Ort::Value &output_tensor,
            std::vector<Detection> &output) {

  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  float x_factor = input_image.cols / 640.;
  float y_factor = input_image.rows / 640.;

  float *data = output_tensor.GetTensorMutableData<float>();
  std::vector<int64_t> outputShape =
      output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  size_t count = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

  cv::Mat l_Mat =
      cv::Mat(outputShape[1], outputShape[2], CV_32FC1, (void *)data);
  cv::Mat l_Mat_t = l_Mat.t();
  // first 5 elements are box[4] and obj confidence
  int numClasses = l_Mat_t.cols - 4;

  for (int l_Row = 0; l_Row < l_Mat_t.rows; l_Row++) {
    cv::Mat l_MatRow = l_Mat_t.row(l_Row);
    float objConf;
    int classId;

    getBestClassInfo(l_MatRow, numClasses, objConf, classId);

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

  std::vector<int> class_ids;
  const float SCORE_THRESHOLD = 0.2;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.4;

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    nms_result);
  for (int i = 0; i < nms_result.size(); i++) {
    int idx = nms_result[i];
    Detection result;
    result.class_id = 0;
    result.confidence = confidences[idx];
    result.box = boxes[idx];
    output.push_back(result);
  }
}

struct ml {
  explicit ml(const std::string &model_path = std::string(),
              const std::size_t threads = 1)
      : model_path_(model_path), ho_(model_path), threads_(threads) {}

  cv::Mat operator()(const cv::Mat &image) {
    auto &memory_info = ho_.memory_info_;
    auto &session = ho_.session_;
    const auto &cinput_name = cnames(ho_.input_name_);
    const auto &coutput_name = cnames(ho_.output_name_);

    auto input_shape =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_shape =
        session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    const auto input_size = get_size(input_shape);

    const auto &nimage = make_input_image(image);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, reinterpret_cast<float *>(nimage.data), input_size,
        input_shape.data(), input_shape.size());

    auto output = session.Run(Ort::RunOptions{nullptr}, cinput_name.data(),
                              &input_tensor, cinput_name.size(),
                              coutput_name.data(), coutput_name.size());

    auto &output_tensor = output.front();
    std::vector<Detection> detections;
    detect(image, output_tensor, detections);

    for (const auto &detection : detections) {
      auto box = detection.box;
      auto classId = detection.class_id;
      const auto color = cv::Scalar(0, 255, 0);
      cv::rectangle(image, box, color, 3);

      cv::rectangle(image, cv::Point(box.x, box.y - 20),
                    cv::Point(box.x + box.width, box.y), color, cv::FILLED);
      cv::putText(image, "buble", cv::Point(box.x, box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return std::move(image);
  }

  std::filesystem::path model_path_ = std::filesystem::path();
  helper_onnx ho_;
  std::size_t threads_;
};

} // namespace lava

#endif /* node__h */

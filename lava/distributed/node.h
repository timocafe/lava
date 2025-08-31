
// MIT License
// Copyright (c) 2023 Timothee EWART
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// ============================================================================
// INCLUDES
// ============================================================================

// Standard library
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <variant>

// OpenCV
#include <opencv2/opencv.hpp>

// ONNX Runtime
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

// Intel TBB
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/parallel_for.h"

// Project headers
#include "lava/sha256/sha256.h"

namespace lava {

// Forward declaration for shutdown coordination
extern std::atomic<bool> pipeline_should_stop;

// ============================================================================
// TYPE ALIASES AND CONSTANTS
// ============================================================================

using chrono_type = std::chrono::time_point<std::chrono::high_resolution_clock>;

// ============================================================================
// FORWARD DECLARATIONS AND ENUMS
// ============================================================================

enum class ExecutionProvider {
  AUTO,   // Try CoreML first, fallback to CPU
  COREML, // Force CoreML (Mac GPU)
  CPU     // Force CPU
};

struct Detection {
  float confidence;
  cv::Rect box;
  std::string sha_;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Extract input names from ONNX session
 */
inline std::vector<std::string> get_input_names(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  const std::size_t input_count = session.GetInputCount();
  std::vector<std::string> input_names(input_count);

  for (std::size_t i = 0; i < input_count; ++i) {
    input_names[i] = &*session.GetInputNameAllocated(i, allocator);
  }

  return input_names;
}

/**
 * @brief Extract output names from ONNX session
 */
inline std::vector<std::string> get_output_names(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  const std::size_t output_count = session.GetOutputCount();
  std::vector<std::string> output_names(output_count);

  for (std::size_t i = 0; i < output_count; ++i) {
    output_names[i] = &*session.GetOutputNameAllocated(i, allocator);
  }

  return output_names;
}

/**
 * @brief Convert string vector to C-style string array
 */
inline std::vector<const char *>
to_c_strings(const std::vector<std::string> &names) {
  std::vector<const char *> c_names(names.size());
  std::transform(names.begin(), names.end(), c_names.begin(),
                 [](const std::string &s) { return s.c_str(); });
  return c_names;
}

/**
 * @brief Calculate total size from tensor shape
 */
inline constexpr int64_t
calculate_tensor_size(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>{});
}

/**
 * @brief Create preprocessed input image for ONNX model
 */
inline cv::Mat create_model_input(const cv::Mat &image) {
  const cv::Size MODEL_INPUT_SIZE{640, 640};
  constexpr double NORMALIZATION_FACTOR = 1.0 / 255.0;

  return cv::dnn::blobFromImage(image, NORMALIZATION_FACTOR, MODEL_INPUT_SIZE,
                                cv::Scalar(), true, false);
}

// ============================================================================
// MOCK PROVIDERS (for testing)
// ============================================================================

/**
 * @brief Mock CoreML provider that always fails for testing fallback mechanism
 */
inline OrtStatus *
OrtSessionOptionsAppendExecutionProvider_CoreML(OrtSessionOptions *options,
                                                uint32_t coreml_flags) {
  // Simulate CoreML provider failure - throw exception that will be caught
  throw Ort::Exception(
      "Mock CoreML provider: CoreML not available on this system", ORT_FAIL);
}

// ============================================================================
// ONNX SESSION MANAGEMENT
// ============================================================================

/**
 * @brief RAII wrapper for ONNX Runtime session with execution provider support
 */
class ONNXSession {
public:
  explicit ONNXSession(const std::string &model_path = std::string(),
                       size_t threads = 1,
                       ExecutionProvider provider = ExecutionProvider::AUTO) {
    if (model_path.empty()) {
      throw std::runtime_error("Model path cannot be empty");
    }

    if (!std::filesystem::exists(model_path)) {
      throw std::runtime_error("ML model does not exist: " + model_path);
    }

    initialize_session(model_path, threads, provider);
    cache_session_info();
  }

  // Non-copyable but movable
  ONNXSession(const ONNXSession &) = delete;
  ONNXSession &operator=(const ONNXSession &) = delete;
  ONNXSession(ONNXSession &&) = default;
  ONNXSession &operator=(ONNXSession &&) = default;

  // Public member access
  Ort::MemoryInfo memory_info_{nullptr};
  Ort::Session session_{nullptr};
  std::vector<std::string> input_name_;
  std::vector<std::string> output_name_;

private:
  void initialize_session(const std::string &model_path, size_t threads,
                          ExecutionProvider provider) {
    // Create memory info
    memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Initialize session options
    session_options_ = Ort::SessionOptions();

    // Configure threading (avoiding SetIntraOpNumThreads for better
    // performance)
    session_options_.SetInterOpNumThreads(static_cast<int>(threads));
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Set execution provider based on user preference
    setup_execution_provider(provider);

    // Create environment and session
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LavaONNX");
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);
  }

  void cache_session_info() {
    input_name_ = get_input_names(session_);
    output_name_ = get_output_names(session_);
  }

  void setup_execution_provider(ExecutionProvider provider) {
    switch (provider) {
    case ExecutionProvider::COREML:
      try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
            session_options_, 0));
        std::cout << "Using CoreML execution provider (GPU acceleration)"
                  << std::endl;
      } catch (const std::exception &e) {
        throw std::runtime_error("Failed to set CoreML execution provider: " +
                                 std::string(e.what()));
      }
      break;

    case ExecutionProvider::CPU:
      try {
        Ort::ThrowOnError(
            OrtSessionOptionsAppendExecutionProvider_CPU(session_options_, {}));
        std::cout << "Using CPU execution provider" << std::endl;
      } catch (const std::exception &e) {
        throw std::runtime_error("Failed to set CPU execution provider: " +
                                 std::string(e.what()));
      }
      break;

    case ExecutionProvider::AUTO:
    default:
      // Try CoreML first, fallback to CPU
      try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
            session_options_, 0));
        std::cout << "Using CoreML execution provider (GPU acceleration)"
                  << std::endl;
      } catch (const std::exception &e) {
        std::cout << "CoreML provider failed, falling back to CPU: " << e.what()
                  << std::endl;
        try {
          Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(
              session_options_, {}));
          std::cout << "Using CPU execution provider" << std::endl;
        } catch (const std::exception &cpu_e) {
          throw std::runtime_error("Failed to set any execution provider: " +
                                   std::string(cpu_e.what()));
        }
      }
      break;
    }
  }

  Ort::SessionOptions session_options_{nullptr};
  Ort::Env env_{nullptr};
};

// ============================================================================
// CAMERA AND INPUT HANDLING
// ============================================================================

/**
 * @brief Camera frame generator for real-time video processing
 */
class CameraGenerator {
public:
  CameraGenerator() {
    constexpr int DEVICE_ID = 0;        // Default camera
    constexpr int API_ID = cv::CAP_ANY; // Auto-detect API

    cap_.open(DEVICE_ID, API_ID);
    if (!cap_.isOpened()) {
      throw std::runtime_error("Error! Unable to open camera.");
    }
  }

  std::pair<cv::Mat, chrono_type> operator()(tbb::flow_control &fc) {
    // Check shutdown signal or camera failure - both stop the pipeline
    if (pipeline_should_stop.load() || !cap_.read(frame_)) {
      fc.stop();
      return std::make_pair(cv::Mat(),
                            std::chrono::high_resolution_clock::now());
    }

    // Return captured frame with timestamp
    return std::make_pair(std::move(frame_),
                          std::chrono::high_resolution_clock::now());
  }

private:
  cv::VideoCapture cap_;
  cv::Mat frame_; // Reusable frame buffer
};

// ============================================================================
// DETECTION PROCESSING
// ============================================================================

/**
 * @brief Perform object detection post-processing with NMS
 */
inline void detect(const cv::Mat &input_image, Ort::Value &output_tensor,
                   std::vector<Detection> &output,
                   std::vector<float> &confidences,
                   std::vector<cv::Rect> &boxes) {

  // Clear previous results
  output.clear();
  confidences.clear();
  boxes.clear();

  // Model constants
  constexpr float MODEL_SIZE = 640.0f;
  constexpr float DETECTION_THRESHOLD = 0.4f;
  constexpr float SCORE_THRESHOLD = 0.2f;
  constexpr float NMS_THRESHOLD = 0.4f;

  // Calculate scaling factors
  const float x_factor = static_cast<float>(input_image.cols) / MODEL_SIZE;
  const float y_factor = static_cast<float>(input_image.rows) / MODEL_SIZE;

  // Get tensor data and shape
  float *data = output_tensor.GetTensorMutableData<float>();
  const auto output_shape =
      output_tensor.GetTensorTypeAndShapeInfo().GetShape();

  // Create OpenCV matrix from tensor data (transpose for easier access)
  cv::Mat output_mat(static_cast<int>(output_shape[1]),
                     static_cast<int>(output_shape[2]), CV_32FC1, data);
  cv::Mat transposed;
  cv::transpose(output_mat, transposed);

  // Process each detection
  for (int row = 0; row < transposed.rows; ++row) {
    const float *row_data = transposed.ptr<float>(row);
    const float confidence = row_data[4];

    if (confidence > DETECTION_THRESHOLD) {
      // Extract bounding box coordinates
      const float cx = row_data[0];
      const float cy = row_data[1];
      const float width = row_data[2];
      const float height = row_data[3];

      // Convert to corner coordinates and scale to image size
      const int left = static_cast<int>((cx - width * 0.5f) * x_factor);
      const int top = static_cast<int>((cy - height * 0.5f) * y_factor);
      const int box_width = static_cast<int>(width * x_factor);
      const int box_height = static_cast<int>(height * y_factor);

      confidences.push_back(confidence);
      boxes.emplace_back(left, top, box_width, box_height);
    }
  }

  // Apply Non-Maximum Suppression
  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    nms_indices);

  // Build final detection results
  output.reserve(nms_indices.size());
  for (int idx : nms_indices) {
    output.push_back({confidences[idx], boxes[idx], ""});
  }
}

/**
 * @brief Compute SHA256 hash for detected regions in parallel
 */
inline void compute_detection_hashes(const cv::Mat &image,
                                     std::vector<Detection> &detections) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, static_cast<int>(detections.size())),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
          auto &detection = detections[i];
          const auto &box = detection.box;

          // Validate bounding box
          if (box.x >= 0 && box.y >= 0 && box.x + box.width < image.cols &&
              box.y + box.height < image.rows) {

            cv::Mat sub_image = image(cv::Range(box.y, box.y + box.height),
                                      cv::Range(box.x, box.x + box.width));

            const uint8_t *data = sub_image.ptr<uint8_t>(0);
            size_t total_bytes = sub_image.total() * sub_image.elemSize();
            std::string sha =
                picosha2::hash256_hex_string(data, data + total_bytes);
            detection.sha_ =
                sha.substr(1, 8); // Take first 8 chars after prefix
          }
        }
      });
}

/**
 * @brief Draw detection results on image
 */
inline void draw_detections(cv::Mat &image,
                            const std::vector<Detection> &detections) {
  const cv::Scalar BOX_COLOR(187, 114, 0); // French blue (BGR)
  const cv::Scalar TEXT_COLOR(0, 0, 0);    // Black text
  constexpr int BOX_THICKNESS = 3;
  constexpr int TEXT_OFFSET = 20;
  constexpr double FONT_SCALE = 0.5;

  for (const auto &detection : detections) {
    const auto &box = detection.box;

    // Draw bounding box
    cv::rectangle(image, box, BOX_COLOR, BOX_THICKNESS);

    // Draw text background
    cv::rectangle(image, cv::Point(box.x, box.y - TEXT_OFFSET),
                  cv::Point(box.x + box.width, box.y), BOX_COLOR, cv::FILLED);

    // Draw SHA text
    cv::putText(image, detection.sha_, cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR);
  }
}

// ============================================================================
// OUTPUT HANDLING
// ============================================================================

/**
 * @brief Text overlay utility
 */
inline void write_text_overlay(cv::Mat &image, const std::string &text,
                               cv::Point position) {
  constexpr int FONT_SIZE = 1;
  const cv::Scalar FONT_COLOR(255, 255, 255); // White
  constexpr int FONT_WEIGHT = 2;

  cv::putText(image, text, position, cv::FONT_HERSHEY_COMPLEX, FONT_SIZE,
              FONT_COLOR, FONT_WEIGHT);
}

/**
 * @brief Frame output handler for pipeline processing
 */
class FrameOutput {
public:
  using QueueType =
      oneapi::tbb::concurrent_bounded_queue<std::pair<cv::Mat, chrono_type>>;

  explicit FrameOutput(std::shared_ptr<QueueType> queue = nullptr)
      : queue_(queue) {}

  void operator()(const std::pair<cv::Mat, chrono_type> &frame_data) {
    if (queue_) {
      queue_->try_push(frame_data);
    }
  }

  void write_text(cv::Mat &image, const std::string &text) {
    const cv::Point TEXT_POSITION(200, 80);
    write_text_overlay(image, text, TEXT_POSITION);
  }

private:
  std::shared_ptr<QueueType> queue_;
};

// ============================================================================
// MACHINE LEARNING PROCESSOR
// ============================================================================

/**
 * @brief High-performance ML inference processor with optimized ONNX session
 * management
 */
class MLProcessor {
public:
  explicit MLProcessor(const std::string &model_path = std::string(),
                       ExecutionProvider provider = ExecutionProvider::CPU,
                       std::size_t threads = 1)
      : model_path_(model_path), onnx_session_(model_path, threads, provider),
        threads_(threads) {

    // Pre-allocate vectors for better performance
    detections_.reserve(32);
    confidences_.reserve(32);
    boxes_.reserve(32);

    // Cache the C-style name arrays once during construction
    input_name_ptrs_ = to_c_strings(onnx_session_.input_name_);
    output_name_ptrs_ = to_c_strings(onnx_session_.output_name_);

    // Cache tensor shape information to avoid repeated queries
    cached_input_shape_ = onnx_session_.session_.GetInputTypeInfo(0)
                              .GetTensorTypeAndShapeInfo()
                              .GetShape();
    cached_input_size_ = calculate_tensor_size(cached_input_shape_);
  }

  std::pair<cv::Mat, chrono_type>
  operator()(const std::pair<cv::Mat, chrono_type> &input) {

    auto &[image, timestamp] = input;

    if (image.empty()) {
      return input; // Return early if image is empty
    }

    // Use cached references for performance
    auto &memory_info = onnx_session_.memory_info_;
    auto &session = onnx_session_.session_;

    // Use cached tensor shape to avoid repeated type info queries
    const auto input_size = cached_input_size_;

    // Preprocess the image
    const auto processed_image = create_model_input(image);

    // Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(
            reinterpret_cast<const float *>(processed_image.data)),
        input_size, cached_input_shape_.data(), cached_input_shape_.size());

    // Run inference using cached name arrays
    auto outputs =
        session.Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(),
                    &input_tensor, input_name_ptrs_.size(),
                    output_name_ptrs_.data(), output_name_ptrs_.size());

    // Process the results
    auto &output_tensor = outputs.front();
    detect(image, output_tensor, detections_, confidences_, boxes_);
    compute_detection_hashes(image, detections_);

    // Make a mutable copy of the image for drawing
    cv::Mat image_with_detections = image.clone();
    draw_detections(image_with_detections, detections_);

    // Clear buffers for next iteration
    detections_.clear();
    confidences_.clear();
    boxes_.clear();

    return std::make_pair(image_with_detections, timestamp);
  }

private:
  // Session and model management
  std::filesystem::path model_path_;
  ONNXSession onnx_session_;
  std::size_t threads_;

  // Cached C-style name arrays (computed once in constructor)
  std::vector<const char *> input_name_ptrs_;
  std::vector<const char *> output_name_ptrs_;

  // Cached tensor shape information (computed once in constructor)
  std::vector<int64_t> cached_input_shape_;
  int64_t cached_input_size_;

  // Working buffers (reused across calls)
  std::vector<float> confidences_;
  std::vector<cv::Rect> boxes_;
  std::vector<Detection> detections_;
};

// Legacy type aliases for backward compatibility
using generator = CameraGenerator;
using chat = FrameOutput;
using ml = MLProcessor;

} // namespace lava

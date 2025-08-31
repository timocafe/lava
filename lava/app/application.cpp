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

#include "application.h"
#include <iostream>

namespace lava {

Application::Application() {
  // Initialize queue
  queue_ = std::make_shared<
      oneapi::tbb::concurrent_bounded_queue<std::pair<cv::Mat, chrono_type>>>();

  // Set up model path
  const std::string name = "model_colab.onnx";
  model_path_ = helper_build_path::model_path() + name;

  // Configure FPS stream
  fps_stream_ << std::fixed << std::setprecision(2);
}

Application::~Application() { shutdown(); }

int Application::run() {
  // Get user's provider choice
  current_provider_ = selectExecutionProvider();

  // Create initial pipeline
  pipeline_ = std::make_unique<lavadom>(model_path_, current_provider_, queue_);
  pipeline_thread_ = std::thread(std::ref(*pipeline_));

  // Display runtime controls
  displayControls();

  // Main application loop
  while (!done_) {
    std::pair<cv::Mat, chrono_type> frame_data;
    if (queue_->try_pop(frame_data)) {
      processFrame(frame_data);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  return 0;
}

ExecutionProvider Application::selectExecutionProvider() {
  std::cout << "\n=== Execution Provider Selection ===" << std::endl;
  std::cout << "1. CPU (Default - Always available)" << std::endl;
  std::cout << "2. CoreML (Mac GPU acceleration)" << std::endl;
  std::cout << "3. AUTO (Try CoreML, fallback to CPU)" << std::endl;
  std::cout << "Select execution provider (1-3): ";

  int choice;
  std::cin >> choice;

  switch (choice) {
  case 2:
    std::cout << "Selected: CoreML Provider" << std::endl;
    return ExecutionProvider::COREML;
  case 3:
    std::cout << "Selected: AUTO Provider (CoreML with CPU fallback)"
              << std::endl;
    return ExecutionProvider::AUTO;
  case 1:
  default:
    std::cout << "Selected: CPU Provider" << std::endl;
    return ExecutionProvider::CPU;
  }
}

std::string Application::getProviderName(ExecutionProvider provider) const {
  switch (provider) {
  case ExecutionProvider::CPU:
    return "CPU";
  case ExecutionProvider::COREML:
    return "CoreML";
  case ExecutionProvider::AUTO:
    return "AUTO";
  default:
    return "Unknown";
  }
}

void Application::handleProviderSwitch(char key) {
  if (key != '1' && key != '2' && key != '3') {
    return;
  }

  ExecutionProvider newProvider;
  switch (key) {
  case '1':
    newProvider = ExecutionProvider::CPU;
    break;
  case '2':
    newProvider = ExecutionProvider::COREML;
    break;
  case '3':
    newProvider = ExecutionProvider::AUTO;
    break;
  default:
    return;
  }

  if (newProvider != current_provider_) {
    restartPipeline(newProvider);
  }
}

void Application::restartPipeline(ExecutionProvider newProvider) {
  current_provider_ = newProvider;
  std::cout << "Restarting pipeline with " << getProviderName(current_provider_)
            << " provider..." << std::endl;

  // Signal pipeline to stop
  pipeline_should_stop.store(true);

  // Wait for current pipeline to finish
  pipeline_thread_.join();

  // Reset shutdown flag
  pipeline_should_stop.store(false);

  // Create new pipeline with new provider
  pipeline_ = std::make_unique<lavadom>(model_path_, current_provider_, queue_);

  // Start new pipeline
  pipeline_thread_ = std::thread(std::ref(*pipeline_));

  std::cout << "Pipeline restarted successfully!" << std::endl;
}

void Application::processFrame(
    const std::pair<cv::Mat, chrono_type> &frame_data) {
  current_image_ = std::move(frame_data.first);
  const auto start = frame_data.second;

  // Handle keyboard input
  const char key = static_cast<char>(cv::waitKey(1));
  if (key == 27 || key == 'q' || key == 'Q') {
    done_ = true;
    return;
  }

  // Handle provider switching
  handleProviderSwitch(key);

  // Calculate and display FPS
  const auto duration = std::chrono::high_resolution_clock::now() - start;
  const float microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  const float fps = 1e6f / microseconds;

  // Reuse stream buffer for performance
  fps_stream_.str("");
  fps_stream_.clear();
  fps_stream_ << fps << " fps";

  // Add overlays
  write_text_overlay(current_image_, fps_stream_.str(), fps_position_);

  std::string provider_text = "Provider: " + getProviderName(current_provider_);
  write_text_overlay(current_image_, provider_text, provider_position_);

  // Display frame
  cv::imshow("LavaRandom", current_image_);
}

void Application::displayControls() const {
  std::cout << "\n=== Runtime Controls ===" << std::endl;
  std::cout << "ESC/Q: Quit application" << std::endl;
  std::cout << "1: Switch to CPU provider" << std::endl;
  std::cout << "2: Switch to CoreML provider" << std::endl;
  std::cout << "3: Switch to AUTO provider" << std::endl;
}

void Application::shutdown() {
  // needed to avoid race condition
  pipeline_should_stop.store(true);
  pipeline_thread_.join();
}

} // namespace lava

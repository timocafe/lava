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

#include <chrono>
#include <memory>
#include <sstream>
#include <thread>

#include "lava/lava.h"

namespace lava {

/**
 * @brief Main application class that manages the computer vision pipeline
 */
class Application {
public:
  /**
   * @brief Constructor
   */
  Application();

  /**
   * @brief Destructor - ensures clean shutdown
   */
  ~Application();

  /**
   * @brief Run the main application loop
   * @return Exit code (0 for success)
   */
  int run();

private:
  /**
   * @brief Display provider selection menu and get user choice
   */
  ExecutionProvider selectExecutionProvider();

  /**
   * @brief Get provider name for display
   */
  std::string getProviderName(ExecutionProvider provider) const;

  /**
   * @brief Handle keyboard input for provider switching
   */
  void handleProviderSwitch(char key);

  /**
   * @brief Restart pipeline with new execution provider
   */
  void restartPipeline(ExecutionProvider newProvider);

  /**
   * @brief Process a single frame
   */
  void processFrame(const std::pair<cv::Mat, chrono_type> &frame_data);

  /**
   * @brief Display runtime controls information
   */
  void displayControls() const;

  /**
   * @brief Clean shutdown of the pipeline
   */
  void shutdown();

  // Application state
  bool done_{false};
  ExecutionProvider current_provider_{ExecutionProvider::CPU};
  std::string model_path_;

  // Pipeline components
  std::shared_ptr<
      oneapi::tbb::concurrent_bounded_queue<std::pair<cv::Mat, chrono_type>>>
      queue_;
  std::unique_ptr<lavadom> pipeline_;
  std::thread pipeline_thread_;

  // UI components
  cv::Mat current_image_;
  cv::Point fps_position_{10, 60};
  cv::Point provider_position_{10, 30};
  std::ostringstream fps_stream_;
};

} // namespace lava

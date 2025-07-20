
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

#include <chrono>
#include <sstream>
#include <thread>

#include "lava/lava.h"

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "lava/imgui/imgui.h"
#include "lava/imgui/imgui_impl_glfw.h"
#include "lava/imgui/imgui_impl_metal.h"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

// #include <Metal/Metal.hpp>

volatile bool done = false; // volatile is enough here. We don't need a mutex
                            // for this simple flag.

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char **) {

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

  // Setup style
  ImGui::StyleColorsDark();

  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

  // Create window with graphics context
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window = glfwCreateWindow(
      1280, 720, "Dear ImGui GLFW+Metal example", nullptr, nullptr);
  if (window == nullptr)
    return 1;

  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  MTL::CommandQueue *queue = device->newCommandQueue();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplMetal_Init(device);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();

  auto q = std::make_shared<oneapi::tbb::concurrent_bounded_queue<
      std::pair<cv::Mat, lava::chrono_type>>>(
      oneapi::tbb::concurrent_bounded_queue<
          std::pair<cv::Mat, lava::chrono_type>>());

  std::string name("model_colab.onnx");
  const auto &model = lava::helper_build_path::model_path() + name;
  lava::lavadom l(model, q);
  auto pipelineRunner = std::thread(std::ref(l));

  std::pair<cv::Mat, lava::chrono_type> p;
  cv::Mat image;
  cv::Point text_position(200, 80);
  while (!glfwWindowShouldClose(window)) {
    if (q->try_pop(p)) {
      image = p.first;
      auto start = p.second;
      char c = (char)cv::waitKey(1);
      if (c == 27 || c == 'q' || c == 'Q') {
        done = true;
      }
      auto end = std::chrono::high_resolution_clock::now() - start;
      float microseconds =
          std::chrono::duration_cast<std::chrono::microseconds>(end).count();
      float fps = 1000000. / microseconds;
      std::stringstream stream;
      stream << std::fixed << std::setprecision(2) << fps;
      lava::write_text_overlay(image, stream.str() + " fps", text_position);
      cv::imshow("LavaRandom", image);
    }
  }
  pipelineRunner.join();

  // Cleanup
  ImGui_ImplMetal_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

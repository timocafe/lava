
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

volatile bool done = false; // volatile is enough here. We don't need a mutex
                            // for this simple flag.

int main(int, char **) {

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
  for (; !done;) {
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
      lava::write(image, stream.str() + " fps", text_position);
      cv::imshow("LavaRandom", image);
    }
  }
  pipelineRunner.join();
  return 0;
}

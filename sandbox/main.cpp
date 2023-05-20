
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

#include <thread>

#include "lava/lava.h"

volatile bool done = false; // volatile is enough here. We don't need a mutex
                            // for this simple flag.

int main(int, char **) {

  auto q = std::make_shared<oneapi::tbb::concurrent_bounded_queue<cv::Mat>>(
      oneapi::tbb::concurrent_bounded_queue<cv::Mat>());

  std::string name("model_colab.onnx");
  const auto &model = lava::helper_build_path::model_path() + name;
  lava::lavadom l(model, q);
  auto pipelineRunner = std::thread(std::ref(l));

  cv::Mat image;
  for (; !done;) {
    if (q->try_pop(image)) {
      char c = (char)cv::waitKey(1);
      if (c == 27 || c == 'q' || c == 'Q') {
        done = true;
      }
      cv::imshow("result", image);
    }
  }
  pipelineRunner.join();
  return 0;
}

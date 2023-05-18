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

  int counter = 0;

  for (; !done;) {
    cv::Mat image;
    if (q->try_pop(image)) {
      char c = (char)cv::waitKey(1);
      if (c == 27 || c == 'q' || c == 'Q') {
        done = true;
      }
      cv::imshow("result", image);
      //    auto name = "image" + std::to_string(counter++) + ".png";
      //    cv::imwrite(name, image);
    }
  }
  pipelineRunner.join();
  return 0;
}

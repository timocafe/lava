#include <iostream>

#include "lava/lava.h"

int main(int, char **) {

  char model[] = "lava.onnx";
  generate(model);
  std::cout << " hello \n";
  return 0;
  // cv::VideoCapture camera;
  // int device_counts = 0;
  // while (true)
  // {
  //     if (!camera.open(device_counts++))
  //     {
  //         break;
  //     }
  // }
  // camera.release();
  // std::cout << "devices count : " << device_counts - 1 << std::endl;

  // Mat frame;
  // //--- INITIALIZE VIDEOCAPTURE
  // VideoCapture cap;
  // // open the default camera using default API
  // // cap.open(0);
  // // OR advance usage: select any API backend
  // int deviceID = 0;        // 0 = open default camera
  // int apiID = cv::CAP_ANY; // 0 = autodetect default API
  // // open selected camera using selected API
  // cap.open(deviceID, apiID);
  // // check if we succeeded
  // if (!cap.isOpened())
  // {
  //     cerr << "ERROR! Unable to open camera\n";
  //     return -1;
  // }
  // //--- GRAB AND WRITE LOOP
  // cout << "Start grabbing" << endl
  //      << "Press any key to terminate" << endl;
  // for (;;)
  // {
  //     // wait for a new frame from camera and store it into 'frame'
  //     cap.read(frame);
  //     // check if we succeeded
  //     if (frame.empty())
  //     {
  //         cerr << "ERROR! blank frame grabbed\n";
  //         break;
  //     }
  //     // show live and wait for a key with timeout long enough to show
  //     images imshow("Live", frame); if (waitKey(5) >= 0)
  //         break;
  // }
  // // the camera will be deinitialized automatically in VideoCapture
  // destructor
  return 0;
}

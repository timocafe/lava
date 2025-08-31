# LavaRandom

A random number generator based on a lava lamp.

<img src="./movie/lavalamp.gif"/>

## A Bit of Physics

A lava lamp consists of two fluids with different densities—typically a paraffin derivative for the wax and water for the second fluid. When the lamp is turned on, the light bulb heats the liquids. The wax (paraffin) expands, and since density equals mass divided by volume, when the volume increases, the density of the wax decreases, causing it to float to the top of the lamp. As the wax reaches the cooler liquid at the top, its volume decreases, resulting in higher density, which causes it to sink back down.

The shape of the wax can be psychedelic and belongs to the realm of Rayleigh-Taylor instability (a fond memory from my master's degree). The physics behind it is quite complex—the same phenomenon occurs in thermonuclear reactors like Tokamaks (where the fluids are protons and electrons).

Since the shape of the wax is largely unpredictable and different each time, it makes an excellent source for random number generation. I simply compute the SHA256 hash of the image.

This idea isn't new and was patented by SGI. Cloudflare uses it daily for internet security... ^_^

## What Do You Need?

A lava lamp ($30), a USB webcam ($30), and a computer.

## Platform 
I have personally tested on the following platforms:
```
- Apple M1
- Windows 11 + WSL (Ubuntu 20) - note: this requires rebuilding the kernel to support USB webcams 
  https://github.com/timocafe/wsl2_linux_kernel_usbcam_enable_conf
```
## Build

The software requires oneAPI, ONNX, and OpenCV. If the dependencies have been correctly installed and the packages are properly configured, follow these steps:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
./exe
```

## Technique - AI

The problem is straightforward: detect the wax in the lava lamp. I created a basic dataset of 256 images from a video, then prepared the data by manually labeling the wax regions and performed training with YOLOv8 Nano (already pre-trained on COCO). YOLO is a standard for object detection—lightweight and highly efficient.

I performed all my AI work in the cloud:

```
- Roboflow to prepare and annotate my data
- Ultralytics for the YOLOv8 model
- Google Colab for training and exporting to ONNX format
- ONNX Runtime for inference (generic and cross-platform)
```

## Technique - HPC

Intel oneAPI provides excellent tools for performance optimization. The main components used here are:
- oneAPI/TBB for the processing pipeline and parallel containers
- oneAPI/oneDNN for efficient ONNX execution

Note that ONNX is platform-agnostic, so regardless of your machine, you can select the optimal backend for your hardware.

## Disclaimer

I have not conducted a formal study on the efficiency of this generator, nor do I understand the influence of SHA256 on the randomness quality. Therefore, there is no guarantee whatsoever regarding the mathematical quality of the random numbers generated. 


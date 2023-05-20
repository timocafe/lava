# LavaRandom

Random generator from a lava lamp.

<img src="./movie/lavalamp.gif"/>

## A Bit of Physics

A lava lamp consists of two fluids with different densities, typically a derivative of paraffin for the wax and water for the second fluid. When the lamp is switched on, the light bulb warms up the liquids. The wax (paraffin) expands, and since density is equal to mass divided by volume, when the volume increases, the density of the wax decreases, and it floats, reaching the top of the lamp. The liquid water is cooler, so the volume of the wax decreases, resulting in higher density, causing it to drop.

The form of the wax can be psychedelic and belongs to the world of Rayleigh-Taylor instability (souvenir my master degree). The physics behind it is not trivial. It is the same in thermonuclear reactors like a Tokamak (in this case, the fluids are the protons and electrons).

As the form of the wax is more or less unpredictable and different every time, it becomes a good idea for a random generator. I simply compute the SHA256 of the image.

The idea is not new and was patented by SGI. Cloudflare uses it daily for the net... ^_^

## What Do You Need?

A lava lamp ($30), a USB webcam ($30), and a computer.

## Build

The software needs oneAPI, ONNX, and OpenCV. If the dependencies have been correctly installed and the package is correctly configured, follow these steps:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
./sandbox/exe
```

## Technique - AI

The problem is basic; it consists of detecting the wax. From a movie, I created a basic dataset of 256 images. Then, I prepared the data by marking the wax 
and performed basic training with YOLOv8 Nano (already pretrained on COCO). YOLO is a standard for detection, light and very efficient. 

I did all my AI work on the cloud:

```
- Roboflow to prepare my data
- Ultralytics for YOLOv8 model
- Colab for training and exporting to ONNX format
- ONNX for inference because it is generic
```

## Technique - HPC

Intel oneAPI provides amazing tools for performance. Here, the main components are oneAPI/TBB for the pipeline and parallel container, oneAPI/DNN for a decent execution of ONNX. Note that ONNX is really generic, so regardless of your machine, you should select the best backend.

## Disclamer

There is no garantee at all, it is a perfect random generator ! No idea of the SHA256 algo for the randomness. 


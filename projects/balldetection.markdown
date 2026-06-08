---
layout: project
title: Ball Detection in Soccer Videos
description: Detecting the soccer ball in match videos using YOLO and modern object detection techniques.
image: {{ site.baseurl }}/assets/images/footballdetection.png
github_link: https://github.com/Pietro-D/balltracking
tools: [Python, Object Detection, Yolo, OpenCV]
permalink: /projects/balldetection/
featured: true
---

This project focuses on building an object detection system
using YOLO, a state-of-the art family of models for computer vision
applications. In particular, we will fine-tune a YOLOv8 model to
perform real-time ball tracking during soccer matches.

# Table of Contents
* TOC
{:toc}
## Introduction
<b> Object detection</b> is a computer vision task whose goal is
to identify and locate multiple objects within a single image.
More specifically, it can be seen as the combination of two subtasks:<br>
1) <b> Classification </b>: What kind of object is present in the image?<br>
2) <b> Localization </b>: Where is the object located?<br>

Over the last decade, many object detection models have been released.
For this project, I decided to use YOLO, specifically version 8.
The main reason behind this decision is speed.
YOLO is a <b>one stage</b> object detection architecture, meaning that
localization and classification are performed in a single forward pass.
This design significantly improves inference speed and makes the model
particularly well suited for
real-time applications. In fact, YOLO can process on average more
the 100 frames per second (FPS), whereas other models such as Faster R-CNN
and R-CNN typically process around 10 FPS.

<img src="{{ site.baseurl }}/assets/images/yoloarchitecture.jpg" alt="YOLOv8 Architecture">


Before diving deeper into the design choices, let's take a closer look
at the architecture, which can be divided  into three main
components: <br>
<ol>
<li>The <b> Backbone</b>, composed of convolutional layers for feature extraction,
interleaved with <b>C2f</b> modules that provide richer gradient flow information. At the end of the backbone, a <b>SPPF</b>
module is used to further enhance feature representation.</li>
<li>A <b>Path Aggregation Network (PAN)</b>, an improvement over the traditional <b>Feature Pyramid Network (FPN)</b>, which enables the
extraction of information at multiple feature scales. This has been
shown to improve performance in tasks like object detection and segmentation.</li>
<li>The <b>Head</b>, which is responsible for predicting bounding box coordinates, class labels and confidence scores for each detected object.</li>
</ol>
{: .stylish-list}
## Dataset Preparation
For this project, I collected four videos for fine-tuning the model,
along with two additional videos reserved for testing.
Since the test videos are only available at inference time,
one of the main challenges was deciding how to split the
four remaining videos into a training set and a validation set.
A possible approach would be to randomly sample frames to build
the validation set, however this is not a good choice, as it
might lead to <b> future data leakege </b>, which can
distort validation metrics. To avoid this, I used all the frames from a
randomly selected video (722 frames containing the ball) as the
<b>validation set</b>, while frames from the remaining videos (3.693
frames containing the ball) were used as the
<b>training set</b>. <br>
As a design choice, I decided to discard almost all frames in which
the ball was not present, keeping only a small subset of them.
This approach proved to be a good compromise between geralization
performace and training speed.
Another important aspect to consider is the intrinsic difficulty of the task:
the object to be detected (the ball) is relatively small compared
to the rest of the scene, which has a native resolution of 1920x1080.
Therefore, the frames were resized to 1024x1024
to ensure that the model could still reliably detect the ball while
reducing memory usage.

## Augmentation

Due to the limited amount of data, <b>mosaic augmentation </b> was
used to increase data diversity and improve generalization.
With this technique, four randomly selected images are augmented and then
combined into a single mosaic image. For each image, the following
transformations are applied:
<ol>
<li> <b> Horizontal flip </b> with a probability of 50%</li>
<li> <b> Vertical flip </b> with a probability of 30%</li>
<li> <b> Brightness variation </b> up to 40%</li>
<li> <b> Saturation variation </b> up to 70%</li>
</ol>
{: .stylish-list}
The image below shows a batch of four images after mosaic augmentation
has been applied.

<img src="/assets/images/mosaicaugmentation.png" alt="batch of 4 images with mosaic augmentation ">

## Loss and Training Setting

Diving deeper into the training setup, I chose the nano version
of the model, which contains approximately <b>3.2 million parameters</b>.
Although this is
the smallest variant with the lowest capacity, it's particularly well suited
for scenarios where applications must run on resource-constrained
devices.<br>

The model was trained using a composite loss function, defined as follows: <br>
<center>$$ \lambda_{box}*L_{box}+\lambda_{cls}*L_{cls}+\lambda_{dfl}*L_{dfl}$$</center>
where:<br>
1) $$L_{box}$$ measures the error between the predicted bounding box and
the ground truth. <br>
2) $$L_{cls}$$ evaluates whether the model correcly predicts the object class.<br>
3) $$L_{dfl}$$ (<b> Distribution Focal Loss </b>) is designed to help the
model focus on hard examples
by reducing the contribution of easy ones during training.<br>

The table below summarizes the main training hyperparameters:<br>

| $$\lambda_{box}$$ | 6.5 |
| $$\lambda_{cls}$$ | 0.5 |
| $$\lambda_{dfl}$$ | 1.5 |
| epochs | 25 |
| batch size | 6 |
| Optimizer | AdamW |
| learning rate | 0.002 |
| momentum | 0.93 |
{: .tech-table}
<br>
<div style="display: flex; justify-content: center;">
  <img src="/assets/images/yolores.png" alt="yolo loss trend" style="width: 60%;">
</div>

## Inference Results

After finetuning, the model was evaluated on the two videos from
the test set. In particular, the MSE was computed between
the predicted bounding boxes
and the ground truth annotations. Across the two videos, the model
achieved a final MSE of <b> 0.03 </b>, which is a strong result given
the difficulty of the
task, the limited amount of data, and the small size of the model. <br>


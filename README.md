# YOLO v3 Detector engine

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/210550734.svg)](https://zenodo.org/badge/latestdoi/210550734)

## Introduction

The purpose of this project is to separate the detector script from the training script of the CNN YOLO. With this, the package itself gets ultra lightweight by using only the detection as main feature whereas the traning with all other parts will be served in a separate package to keep the extensibility and the deployability of the detection code.

---

## Quick Start

1. Copy your YOLOv3 weights into the `model` folder
1. Install requirements by running `pip install -r requirements.txt`. Note: it is expected that you already have a working Keras installation on your machine.
3. Run `python yolo_video.py`.

## Usage

### Use `yolo_video.py` directly

If you want to use the detector script directly, you can do so by just calling `yolo_video.py`.
Use `--help` to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

### Use the detection method in your script

1 Include `yolo.py` in your project
```python
from yolo import YOLO
from PIL import Image as PIL_Image
```

2 Set your args

```python
args = {
    "model_path": 'model/weights.h5',
    "anchors_path": 'model/anchors.txt',
    "classes_path": 'model/classes.txt',
    "score": 0.7,
    "iou": 0.15,
    "model_image_size": (608, 608),
    "gpu_num": 1,
}
```

3 Convert your image into an PIL image (if you're using OpenCV for example)

```python
pil_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
color_image = PIL_Image.fromarray(pil_image)
```
4 Run the detector

```python
yolo = YOLO(**args)
print('Run detection...')
r_image, out_boxes, out_scores, out_classes = yolo.detect_image(color_image)
print('Closing YOLO session...')
yolo.close_session()
```

### Examples / Cookbook

#### Full code example by using OpenCV as image source

```python
from yolo import YOLO
from PIL import Image as PIL_Image
import cv2

args = {
    "model_path": 'model/weights.h5',
    "anchors_path": 'model/anchors.txt',
    "classes_path": 'model/classes.txt',
    "score": 0.7,
    "iou": 0.15,
    "model_image_size": (608, 608),
    "gpu_num": 1,
}

cv2_img = cv2.imread('your_image.png')

pil_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
color_image = PIL_Image.fromarray(pil_image)

yolo = YOLO(**args)
print('Run detection...')
r_image, out_boxes, out_scores, out_classes = yolo.detect_image(color_image)
print('Closing YOLO session...')
yolo.close_session()
```

#### Continuous input via Image Stream inside ROS

This piece of code converts the image message from a ROS topic into a CV2 image and runs the detection within

```python
import rospy
from sensor_msgs.msg import Image
from yolo import YOLO
from PIL import Image as PIL_Image
import cv2
from cv_bridge import CvBridge
import numpy as np

args = {
    "model_path": 'model/weights.h5',
    "anchors_path": 'model/anchors.txt',
    "classes_path": 'model/classes.txt',
    "score": 0.7,
    "iou": 0.15,
    "model_image_size": (608, 608),
    "gpu_num": 1,
}

color_image = PIL_Image.new('RGB', (1920, 1080))

def preprocess_data(image_message):
    global color_image
    global timestamp

    bridge = CvBridge()
    cv2_img = bridge.imgmsg_to_cv2(image_message, 'bgr8')
    pil_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    color_image = PIL_Image.fromarray(pil_image)

if __name__ == '__main__':

    rospy.init_node('Detector')

    color_img_topic = rospy.get_param('~image_topic', '/YOUR_IMAGE_TOPIC')
    rospy.Subscriber(color_img_topic, Image, preprocess_data)

    # Start listener
    try:
        print('Start YOLO...')
        yolo = YOLO(**args)
        while not rospy.is_shutdown():
            print('Run detection...')
            r_image, out_boxes, out_scores, out_classes = yolo.detect_image(color_image)
            print('Got detection. Convert it to CV2 image format...')
            cv2_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)

            # Do here some other magic (like publish detection result)

        print('Closing YOLO session...')
        yolo.close_session()

    except rospy.ROSInterruptException:
        pass
```

## Attribution
Thanks to [@qqwweee](https://github.com/qqwweee/keras-yolo3) for his awesome implementation of the YOLOv3 detector in Keras which is the base of this project.

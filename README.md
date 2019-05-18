## Project Wiki

*Feel free to add resources you find usefull*

### List of some SOTA models (with code in pytorch/keras)

**YOLO**

 - [YOLO original paper](https://arxiv.org/abs/1506.02640)

 - [Followup YOLO9000](https://arxiv.org/abs/1612.08242) | [pytorch](https://github.com/longcw/yolo2-pytorch)

 - [YOLO V.3 (Latest)](https://arxiv.org/abs/1804.02767) | [pytorch](https://github.com/ayooshkathuria/pytorch-yolo-v3)


**Region Proposal Networks**


- [R-CNN (Original Paper)](https://arxiv.org/abs/1311.2524)
- [Faster R-CNN (Followup)](https://arxiv.org/abs/1506.01497) | [keras](https://github.com/jinfagang/keras_frcnn) |  [pytorch](https://github.com/jwyang/faster-rcnn.pytorch)
- [Mask R-CNN (Followup, but with masks)](https://arxiv.org/abs/1703.06870) | [keras](https://github.com/matterport/Mask_RCNN) | [pytorch](https://github.com/multimodallearning/pytorch-mask-rcnn)

**RetinaNet**
- [RetinaNet (Original Paper)](https://arxiv.org/abs/1708.02002) | [keras](https://github.com/fizyr/keras-retinanet) | [pytorch](https://github.com/kuangliu/pytorch-retinanet)

**SSD**
- [SSD (Original paper)](https://arxiv.org/abs/1512.02325) | [keras](https://github.com/pierluigiferrari/ssd_keras) | [pytorch](https://github.com/amdegroot/ssd.pytorch)

### Papers what might be usefull

- [Learning a Rotation Invariant Detector with Rotatable Bounding Box](https://arxiv.org/pdf/1711.09405.pdf)

### Video Ressources

- [Non-maximum suppresion (Andrew Ng)](https://www.youtube.com/watch?v=A46HZGR5fMw)
- [Anchor Boxes (Andrew Ng)](https://www.youtube.com/watch?v=Pf7iFeRPYK8)


### Misc

- [Not bad comparison of this model zoo with pros and cons (Stanford)](http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds06.pdf)

### Implementations

#### YOLO

- [Blog post](http://guanghan.info/blog/en/my-works/train-yolo/) on how to train original implementation of YOLO v2 (written in C and cuda) on your own data. In theory this is exactly what we need. If this works out of the box we are done. Another + is that it works with video out of the box. Downside is that if something does not work we have little marge de manoeuvre here, since none of us know C. 

- [Github repo](https://github.com/marvis/pytorch-yolo2) with YOLO V2 implementation. The upside is that it is in pytorch, so we can modify everything. The downside is that there are not much instructions on how to fine-tune on your data, so we might need to go throught fair amount of code (n000 lines).

- [Another blog post](https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/) discussing how to train original implementatio of YOLO V2 on your data. Again, C and cuda.

- [And yet another one](http://bennycheung.github.io/yolo-for-real-time-food-detection) This time for food detection.

#### RetinaNet

- [Discussion on how to implement focal loss in pytorch](https://discuss.pytorch.org/t/how-to-implement-focal-loss-in-pytorch/6469)

- [Pretrained keras model applied to some challenge](https://medium.com/data-from-the-trenches/object-detection-with-deep-learning-on-aerial-imagery-2465078db8a9). We might follow their path. I think it should be possible to re-define the classification and regression heads in keras. 

- [The keras model repo itself](https://github.com/fizyr/keras-retinanet)

#### MASK RCNN

- [Repo with keras model](https://github.com/fizyr/keras-retinanet). Strictly speaking this designed to generate masks around the objects not bounding boxes (even thought it does generate boxes as a preprocessing step before generating the mask). But, in order to deal with rotation, we might use this approach:
     - First, predict mask around object. 
     - Then, [from mask extrapolate to rotated bounding box](https://github.com/fizyr/keras-retinanet/issues/484). 

## On the image format and preprocessing

The images were converted from  RAW to PNG using [dcraw](https://www.cybercom.net/~dcoffin/dcraw/). The options that were used during the conversion are **-D -4 -T**. According to the [man](https://www.cybercom.net/~dcoffin/dcraw/dcraw.1.html) page of dcraw that means:

- -D: Show the raw data as a grayscale image with no interpolation. Good for photographing black-and-white documents. Original unscaled pixel values.

- -4: Linear 16-bit

- -T: Write TIFF with metadata instead of PGM/PPM/PAM.

## How to run web downloader?
In bash type:
```bash
brew cask install chromedriver
pip install selenium
pip install beautifulsoup4 
```
Then in `data_downloader.py` change path variable:

```python
if __name__ == "__main__":
    path = r"absolute/path/to/you/download/folder"
```

## Missing RetinaNet Code
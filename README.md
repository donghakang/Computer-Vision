# HOG Descriptor and Face Detection

Histogram of Oriented Gradients (HOG) and Face Detection using Histogram of Oriented Gradients.

## Getting Started

To run the program.

```
python3 HOG_ver1.py
```

7 functions that helps to reach the HOG descriptor and also Face detection.
*get_differential_filter* and *filter_image* function help to filter the image by differentiation (both x and y).
With the filtered image, and *get_gradient* function, the image can be visualized by magnitude and angle of the gradient.
By gradient image and *build_histogram*, *get_block_descriptor* function, The image can be described as
Histogram. With the histogram, one can finally detect the face with target image and template image using *face_detection* function.


## Running the tests

* extract_hog
* get_differential_filter
* filter_image
* get_gradient
* build_histogram
* get_block_descriptor
* face_detection

### Test Images

<img src="./img/original.png" width="50%" height="50%">
original image
<img src="./img/im_dx.png" width="50%" height="50%">
im_dx, x direction filter
<img src="./img/im_dy.png" width="50%" height="50%">
im_dy, y direction filter
<img src="./img/grad_mag.png" width="50%" height="50%">
grad_mag gradient by magnitude
<img src="./img/grad_angle.png" width="50%" height="50%">
grad_angle gradient by angle
<img src="./img/HOG.png" width="50%" height="50%">
HOG Descriptor

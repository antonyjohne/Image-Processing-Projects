# Image Processing Projects
Image Processing Projects that provide pre-requisite knowledge for advanced Computer Vision projects

This Project repository is a part of my Digital Image Processing Coursework that helped me understand the fundamentals of Image Processing for Computer Vision

### Project 01: Image Convolution & Filtering

Implemented Convolution in the Spatial Domain and applied various edge detection filters using Convolution. Problem 2 helped understand how to apply Histogram Equalization on the input image, followed by converting the image into the Frequency Domain and back into the Spatial domain. The various visualization techniques of the image in the frequency domain (Ex: Log Filter) was also recognized. 

### Project 02: Image Blending
Developed an Image blending model that allowed the user to blend 2 images using Gaussian and Laplacian Pyramids. Programmed a GUI that allows the user to crop a mask from the source image and apply it to the target image. This project helped understand how Gaussian and Laplacian Pyramids help downsample an image to lower resolutions with loss and allow for the perfect reconstruction of the image after performing the desired operation at the lowest scale.

### Project 03: Blob Detection
Implemented a Blob/Feature Detection model using principles learned from the previous projects. The basic idea for implementing a blob detector is to convolve the image with a blob filter at multiple scale and look for extrema at the various scales. After blob detection a lot of blobs are found to be overlapping. To reduce this overlap, area between the overlapping blobs is compared with a threshold ratio. If this area is greater than the threshold, the blob with the smaller radius is discarded. This operation of reducing the overlapping blobs is known as non-maximum suppression (NMS).

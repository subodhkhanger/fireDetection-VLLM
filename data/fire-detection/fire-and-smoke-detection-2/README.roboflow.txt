
fire and smoke detection - v2 2023-09-18 11:37am
==============================

This dataset was exported via roboflow.com on October 30, 2023 at 10:06 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 15345 images.
Fire and smoke are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-contrast via histogram equalization

The following augmentation was applied to create 3 versions of each source image:

The following transformations were applied to the bounding boxes of each image:
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random Gaussian blur of between 0 and 2 pixels
* Salt and pepper noise was applied to 5 percent of pixels



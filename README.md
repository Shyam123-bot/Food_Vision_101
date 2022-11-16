# Food_Vision_101 

As an introductory project to myself, I built an **end-to-end CNN Image Classification Model** which identifies the food in your image.

I worked out with a pretrained Image Classification Model that comes with Keras and then retrained it on the infamous **Food101 Dataset**.


**Fun Fact :**

The Model actually beats the DeepFood Paper's model which also trained on the same dataset.
 
The Accuracy of [**DeepFood**](https://arxiv.org/abs/1606.05675) was **77.4%** and our model's is **85%**. Difference of **8%** ain't much but the interesting thing is, DeepFood's model took 2-3 days to train while our's was around 60min.

> **Dataset :** `Food101`

> **Model :** `EfficientNetB1`
We're going to be building Food Vision Big™, using all of the data from the Food101 dataset.

Yep. All 75,750 training images and 25,250 testing images.

And guess what...

This time we've got the goal of beating DeepFood, a 2016 paper which used a Convolutional Neural Network trained for 2-3 days to achieve 77.4% top-1 accuracy.

🔑 Note: Top-1 accuracy means "accuracy for the top softmax activation value output by the model" (because softmax ouputs a value for every class, but top-1 means only the highest one is evaluated). Top-5 accuracy means "accuracy for the top 5 softmax activation values output by the model", in other words, did the true label appear in the top 5 activation values? Top-5 accuracy scores are usually noticeably higher than top-1.

Alongside attempting to beat the DeepFood paper, we're going to learn about two methods to significantly improve the speed of our model training:

Prefetching
Mixed precision training
But more on these later.

What we're going to cover
Using TensorFlow Datasets to download and explore data
Creating preprocessing function for our data
Batching & preparing datasets for modelling (making our datasets run fast)
Creating modelling callbacks
Setting up mixed precision training
Building a feature extraction model (see transfer learning part 1: feature extraction)
Fine-tuning the feature extraction model (see transfer learning part 2: fine-tuning)
Viewing training results on TensorBoard

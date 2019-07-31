---
title: "Fashion Recommender based on Image Similarity with Deep Learning"
layout: post
date: 2019-07-31 22:10
tag: CNN, deep learning, pytorch
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
description: "An attempt on image recognition with transfer learning"
category: project
author: yubin
externalLink: false
---


See codes - [Link](https://github.com/yubin627/ga_projects). 

---

When it comes to what to wear, I often seek inspiration from what is around me - it could be from Instagram photos or some random passers-by I spot on the street. For an avid online (almost exclusively) shopper like me, I would start browsing immediately on the websites for similar items, but it has been quite a challenge for me to find correct words to put in the search bar to describe the items accurately. 

Image search engine is an answer to my problem. In reality it actually has already been a default product feature for most of the major e-commerce sites these days. As a beginner in deep learnnig, I am very intrigued to get some hands-on and better understanding of the algorithms under the hood. 


as shown above, is what we will be looking at, given a base image, recommend visually similar images. Image-based recommendations come in very handy in many scenarios especially in cases where visual tastes of the user are involvedthis project a classification problem that requires CNN (Convolutional Neural Network) to extract image features that are comprehensive enough in striking a balance between inter-class and intra-class variation, followed by a search problem that could shorten the time retrieving the items that are the most similar.

## Goal

My goal would be to build a search engine based on image similarity-based recommendations. 
Essentially the workflow is divided into three steps:
1. Modeling
2. Feature extraction
3. Image retrieval

During each step there are a few points to consider:
1. Modeling
- Can simple CNN handle this task? (TBC)
- If not, which pre-trained model works the best, in terms of accuracy and training time? 

2. Feature extraction
- How to extract feature vectors to capture most of the information contained in the images?

3. Image retrieval
- Given a large dataset (130k images in my project), what is the optimal algorithm for image retrieval?

## Preparation Work
### Dataset

I used the [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset that has been meticulously gathered and labeled by the The Multimedia Laboratory at the Chinese University of Hong Kong. 

In total it contains over 800,000 diverse fashion images with 50 categories ranging from well-posed shop images to unconstrained consumer photos. Due to limitation of computation resources and time, I only used the upper body clothes images in this project.

As shown below is the data folder structure.

{% highlight html %}

├── img
│   ├── img_subfolder_1
│   ├── ...
│   └── img_subfolder_n
├── Anno
│   ├── list_bbox.txt
│   ├── list_category_cloth.txt
│   └── list_category_img.txt
└── Eval
    └── list_eval_partition.txt

{% endhighlight %}


### Hardware

|---INSTANCE---|---	CPU CORES---|---MEMORY---|---GPU TYPE---|---GPU MEMORY---|
|GPU|4|64 GB|Nvidia Tesla K-80|12 GB dedicated Memory|

Notes:
https://www.datascience.com/blog/transfer-learning-in-pytorch-part-two
Each model can be considered as composed of two parts:

The convolutional neural network backbone (a CNN architecture with several blocks comprising of convolutions with varying number of filters, non-linearities, max or average pooling layers, batch normalizations, dropout layers, etc.)
A head with a fully connected classifier at the output end

In most cases, the output layer does not have any fully connected hidden layers. However, we have the option to replace the classifier layer with our own, and add more hidden layers by replacing the output layer with our own. We may easily use our own FC class (defined in Part 1 of this tutorial) for this purpose. 



DataLoaders
PyTorch DataLoaders are objects that act as Python generators. They supply data in chunks or batches while training and validation. We can instantiate DataLoader objects and pass our datasets to them. DataLoaders store the dataset objects internally.

When the application asks for the next batch of data, a DataLoader uses its stored dataset as a Python iterator to get the next element (row or image in our case) of data. Then it aggregates a batch worth of data and returns it to the application.

The following is an example of calling the DataLoader constructor:

Here we are creating a DataLoader object for our training dataset with a batch size of 50. The sampler parameter specifies the strategy with which we want to sample data while constructing batches.

We have different samplers available in torch.utils.data.sampler. The explanation is straightforward. You can read about them in the Pytorch Documentation here.

The num_workers argument specifies how many processes (or cores) we want to use while loading our data. This provides parallelism while loading large datasets. Default is 0 which means load all data in main process.

DataLoader reports its length in number of batches. Since we created this DataLoader with a batch size of 50 and we had 50,000 images in our train dataset, we have the length of dataloader = 1000 batches.




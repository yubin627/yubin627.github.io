---
title: "Fashion Recommender based on Image Similarity - An Exploratory Deep Learning Project"
layout: post
date: 2017-07-31 22:10
tag: CNN, deep learning, pytorch
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
description: "An attempt on image recognition with transfer learning"
category: project
author: johndoe
externalLink: false
---


See codes - [Link](http://sergiokopplin.github.io/indigo/). 

---

When it comes to what to wear, there are often cases when people are inspired by what they see around them - it could come from passers-by, celebrities in the news or social influencers on the Internet. For online shoppers, it'd be natural for people to start browsing immediately the similar items on their usual go-to e-commerce platforms/retailers and purchase the garments that resemble what they've seen. 

Nowadays image search has become a default product feature for most of the e-commerce apps. I am very intrigued to understand the algorithms underneath and the intricacies involved that could influence the accuracy and performance of the algorithms.

Essentially, this becomes a classification problem that requires CNN (Convolutional Neural Network) to extract image features that are comprehensive enough in striking a balance between inter-class and intra-class variation, followed by a search problem that could shorten the time retrieving the items that are the most similar.


Dataset
In my exploratory study, I had used the dataset that has been meticulously gathered and labeled by the The Multimedia Laboratory at the Chinese University of Hong Kong. You can access the dataset on their website DeepFashion: a large-scale fashion database. It contains over 800,000 diverse fashion images ranging from well-posed shop images to unconstrained consumer photos. Each image in this dataset is labeled with 50 categories, 1000 descriptive attributes, bounding box and clothing landmarks.
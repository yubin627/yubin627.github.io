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
See the [codes](https://github.com/yubin627/ga_projects)


When it comes to what to wear, I often seek inspiration from what is around me - it could be from Instagram photos or some random passers-by I spotted on the street. For an avid online (almost exclusively) shopper like me, I would start browsing immediately on the websites, but it has been quite a challenge for me to find the exact words to describe the items accurately and to type in the search bar. 

Image search engine is an answer to my problem. In reality it actually has already been a default product feature for most of the major e-commerce sites these days. As a beginner in deep learnnig myself, I am very intrigued to get some hands-on and better understanding of the algorithms under the hood. 


## Goal

My goal would be to build a search engine based on image similarity-based recommendations. 
Essentially the workflow is divided into three steps:
1. Modeling
2. Feature extraction
3. Image retrieval

Following this workflow, there are a few points to consider:
- Can simple CNN handle this task? (TBC)
- If not, which pre-trained model works the best, in terms of accuracy and training time? 
- How to extract the feature vectors (or embeddings) to capture most of the information contained in the images?
- Given a large dataset (130k images in this project), what is the optimal algorithm for image retrieval?

## Preparation Work
### Dataset

I used the [DeepFashion Attribute Prediction Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) that has been meticulously gathered and labeled by the The Multimedia Laboratory at the Chinese University of Hong Kong. 

In total it contains over 800,000 diverse fashion images with 50 categories ranging from well-posed shop images to unconstrained consumer photos. Due to limitation of computation resources and time, I only used the upper body clothes images in this project, which contains 139,709 images from 20 categories (please refer to category_label in Anno/list_category_img.txt in the dataset). 

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


### Hardware & Environment

I trained the model on floydhub for its easy set-up. The configuration is as follows:

| Instance 	| CPU Core 	| Memory 	| GPU Type          	| GPU Memory 	|
|----------	|----------	|--------	|-------------------	|------------	|
| GPU      	| 4        	| 64GB   	| Nvidia Tesla K-80 	| 12GB       	|

- torch==1.1
- torchvision==0.3
on floydhub's pytorch docker image

## Model Training

### Prepare dataloader
PyTorch DataLoaders are objects that act as Python generators. They supply data in chunks or batches while training and validation. We can instantiate DataLoader objects and pass our datasets to them. DataLoaders store the dataset objects internally.

When the application asks for the next batch of data, a DataLoader uses its stored dataset as a Python iterator to get the next element (row or image in our case) of data. Then it aggregates a batch worth of data and returns it to the application.

The following is a snippet of the codes that I used to create the training dataset (click the triangle to expand):

<details>
<summary>
<i>dataloader class to prepare train/test/all datasets</i>
</summary>
<p>{% highlight python %}
class Fashion_attr_prediction(data.Dataset):
    def __init__(self, type="train", transform=None, target_transform=None, crop=False, img_path=None):
        self.transform = transform
        self.target_transform = target_transform
        self.crop = crop
        # type_all = ["train", "test", "all", "triplet"]
        self.type = type
        self.train_list = []
        self.train_dict = {i: [] for i in range(CATEGORIES)}
        self.test_list = []
        self.all_list = []
        self.bbox = dict()
        self.anno = dict()
        self.read_partition_category()
        self.read_bbox()

    def __len__(self):
        if self.type == "all":
            return len(self.all_list)
        elif self.type == "train":
            return len(self.train_list)
        elif self.type == "test":
            return len(self.test_list)
        else:
            return 1

    def read_partition_category(self):
        #print("current directory"+os.getcwd()) #testing
        list_eval_partition = os.path.join(DATASET_BASE, r'Eval', r'list_eval_partition.txt')
        list_category_img = os.path.join(DATASET_BASE, r'Anno', r'list_category_img.txt')
        partition_pairs = self.read_lines(list_eval_partition)
        category_img_pairs = self.read_lines(list_category_img)
        for k, v in category_img_pairs:
            v = int(v)
            if v <= 20:
                self.anno[k] = v - 1
        for k, v in partition_pairs:
            if k in self.anno:
                if v == "train":
                    self.train_list.append(k)
                    self.train_dict[self.anno[k]].append(k)
                else:
                    # Test and Val
                    self.test_list.append(k)
        self.all_list = self.test_list + self.train_list
        random.shuffle(self.train_list)
        random.shuffle(self.test_list)
        random.shuffle(self.all_list)

    def read_bbox(self):
        list_bbox = os.path.join(DATASET_BASE, r'Anno', r'list_bbox.txt')
        pairs = self.read_lines(list_bbox)
        for k, x1, y1, x2, y2 in pairs:
            self.bbox[k] = [x1, y1, x2, y2]

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def read_crop(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.crop:
            x1, y1, x2, y2 = self.bbox[img_path]
            if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
                img = img.crop((x1, y1, x2, y2))
        return img

    def __getitem__(self, index):
        if self.type == "triplet":
            img_path = self.train_list[index]
            target = self.anno[img_path]
            img_p = random.choice(self.train_dict[target])
            img_n = random.choice(self.train_dict[random.choice(list(filter(lambda x: x != target, range(20))))])
            img = self.read_crop(img_path)
            img_p = self.read_crop(img_p)
            img_n = self.read_crop(img_n)
            if self.transform is not None:
                img = self.transform(img)
                img_p = self.transform(img_p)
                img_n = self.transform(img_n)
            return img, img_p, img_n

        if self.type == "all":
            img_path = self.all_list[index]
        elif self.type == "train":
            img_path = self.train_list[index]
        else:
            img_path = self.test_list[index]
        target = self.anno[img_path]
        img = self.read_crop(img_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_path if self.type == "all" else target
{% endhighlight %} 
</p>
</details>
Here I created a class to prepare the dataset object. There are various options (self.type) created to cater for the needs of training and feature extraction later. To be more specific, `train/test` data will be used for training the model; `all` will be used for feature extraction, `triplet` will be used to generate triplet margin loss function for the backpropagation. More on loss function later.

### Preprocessing and Transforming the Dataset
One more step before we move on to defining our network and start training - we need to preprocess our datasets. Specifically, this incluces Resizing, Data Augmentation, Conversion to PyTorch Tensors and Normalizing. 

<details>
<summary>
<i>proprocessing</i>
</summary>
<p>{% highlight python %}
data_transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE), #224*224
    transforms.RandomResizedCrop(CROP_SIZE),  #224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #ImageNet's mean/std parameters
    ])

data_transform_test = transforms.Compose([
    transforms.Resize(CROP_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

{% endhighlight %} 
</p>
</details>

Now we can instantiate the class to create the data loader objects as shown in the snippets below.
<details>
<summary>
<i>dataloader objects</i>
</summary>
<p>{% highlight python %}
#Refer to config.py for the settings on batch size and number of workers
train_loader = torch.utils.data.DataLoader(
    Fashion_attr_prediction(type="train", transform=data_transform_train),
    batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    Fashion_attr_prediction(type="test", transform=data_transform_test),
    batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

# For calculating triplet margin loss    
triplet_loader = torch.utils.data.DataLoader(
    Fashion_attr_prediction(type="triplet", transform=data_transform_train),
    batch_size=TRIPLET_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)
{% endhighlight %} 
</p>
</details>

### Modeling with Transfer Learning

Here I selected [ResNet-50](https://arxiv.org/abs/1512.03385) pretrained on ImageNet dataset as my base architectures due to its proved benchmark performance in accuracy and training time (see [Stanford University's DAWNBench](https://dawn.cs.stanford.edu/benchmark/)). I also had an attempt on [ResNeXt-50](https://arxiv.org/abs/1611.05431) due to its reportedly [higher accuracy](https://github.com/facebookresearch/ResNeXt) but at the point of writing the speed to comparable accuracy is not on par with ResNet-50 on my current settings. It might require further tuning.

With transfer learning, I froze the architecture and weights from all the layers except the last two - pooling and fully connected layers and fine tuned these two. This way I don't need to train the whole CNN from scrach (which is also not necessary). Here is a great visualization showing the building blocks of [ResNet-50 architecture](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006). 

Reason for training the 

- main module, pooling module, color module



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





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
See the [codes](https://github.com/yubin627/ga_projects/tree/master/Capstone_Project/codes)

Check out the [app](https://deepfashion-finder.herokuapp.com/)

---

When it comes to what to wear, I often need inspiration from what is around me - it could be from Instagram photos or some passers-by I spotted on the street. For an avid online (almost exclusively) shopper like me, I would start browsing immediately on the websites if I see something that I like, but it has been quite a challenge for me to find the exact words to describe the items accurately in the search bar. 

Image search engine is an answer to my problem. In reality it has already become a prevalent product feature on the major e-commerce sites these days. For myself as a beginner just starting the learning journey in deep learnnig, I am very intrigued to get some hands-on and better understanding of the algorithms under the hood. 


## Goal

My goal would be to build a search engine based on image similarity for clothing recommendation. 
Essentially the workflow consists of three steps:
1. Modeling
2. Feature extraction
3. Image retrieval

Following this workflow, there are a few points to consider:
- Can shallow CNN handle this task? 
- If not, which pre-trained model works the best, in terms of accuracy and training time? 
- How to extract the feature vectors (or embeddings) to capture most of the information contained in the images?
- Given a large dataset (130k images in this project), what is the optimal algorithm for image retrieval?

## Set-Up
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

I trained the model on floydhub for its user-friendly set-up process. The configuration is as follows:

| Instance 	| Docker Image                                  	| torchvision 	|  
|:----------|:-----------------------------------------------	|:-------------	|
| GPU      	| floydhub/pytorch:1.0.1-gpu.cuda9cudnn7-py3.42 	| install 0.3   | 

## Data Preprocessing

### Preparing Dataset
PyTorch DataLoaders are objects that act as Python generators. They supply data in chunks or batches while training and validation. We can instantiate DataLoader objects and pass our datasets to them. DataLoaders store the dataset objects internally.

When the application asks for the next batch of data, a DataLoader uses its stored dataset as a Python iterator to get the next element (row or image in our case) of data. Then it aggregates a batch worth of data and returns it to the application.

The following is a snippet of the codes that I used to create the training dataset containing the cropped images and the target variable. Click the triangle to expand the code block:

<details>
<summary>
<i>dataset class to prepare train/test/all set</i>
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

### Preprocessing and Data Augmentation
One more step before we move on to defining our network and start training - we need to preprocess our datasets. Specifically, this incluces Resizing, Data Augmentation, Conversion to PyTorch Tensors and Normalizing. 

<details>
<summary>
<i>preprocessing</i>
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

## Modeling
### Shallow CNN Modeling
I had an attempt on a shallow network with 3 layers and was only able to achieve 41% accuracy. This is understandable since 3 layers could possibly only detect basic patterns (circles, grids etc.) but would not be able to take care of the complexities of clothing.

### Modeling with Transfer Learning

Here I used [ResNet-50](https://arxiv.org/abs/1512.03385) pretrained on ImageNet dataset as my base architectures due to its proved benchmark performance in accuracy and training time (see [Stanford University's DAWNBench](https://dawn.cs.stanford.edu/benchmark/)). I also had an attempt on [ResNeXt-50](https://arxiv.org/abs/1611.05431) due to its reportedly [higher accuracy](https://github.com/facebookresearch/ResNeXt) but at the point of writing the training speed to comparable accuracy is much slower than (~0.6x) ResNet-50 on my current hardware. As future work, I will play with fine-tuning of the ResNeXt version.

With transfer learning, I froze the architecture and weights from all the layers up till the last two - pooling and fully connected (FC) layers and fine tuned these two. This way I don't need to train the whole CNN from scrach (which is also not necessary since ImageNet that the model is pretrained on is quite versatile - it covers >1M images spanning 1000 categories). Here is a great visualization showing the building blocks of [ResNet-50 architecture](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006). 

Instead of outputing to a 1000-d FC layer, I took the output of pooling layer to 2 FC layers in sequence, with the first one being a 512-d vector for feature extraction purpose later on, and the second being a 20-d vector for image classification (recall that I am only using 20 categories in the dataset).

The weights of the last two layers are trained using a combination of [Cross Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) and [Triplet Margin Loss](https://www.coursera.org/lecture/convolutional-neural-networks/triplet-loss-HuUtN?utm_source=linkshare&siteID=je6NUbpObpQ-v1d8hWFJabwDXRzs0wbHQg&ranEAID=je6NUbpObpQ&utm_content=10&ranMID=40328&ranSiteID=je6NUbpObpQ-v1d8hWFJabwDXRzs0wbHQg&utm_campaign=je6NUbpObpQ&utm_medium=partners) function and Stochastic Gradient Descent optimizer. 

See below for the snippet of the codes on the model adjustment:
<details>
<summary>
<i>model constructor used for transfer learning</i>
</summary>
<p>{% highlight python %}
    
# Refer to config.py file for the settings on INTER_DIM, CATEGORIES, learning rate and momentum
class f_model(nn.Module):
    '''
    input: N * 3 * 224 * 224
    output: N * num_classes (20), N * inter_dim (512), N * C' (2048) * 7 * 7
    '''
    def __init__(self, freeze_param=True, inter_dim=INTER_DIM, num_classes=CATEGORIES, model_path=None):
        super(f_model, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=True)
        state_dict = self.backbone.state_dict()
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.backbone.load_state_dict(model_dict)
        if freeze_param:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.avg_pooling = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, inter_dim)
        self.fc2 = nn.Linear(inter_dim, num_classes)
        state = load_model(model_path)
        if state:
            new_state = self.state_dict()
            new_state.update({k: v for k, v in state.items() if k in new_state})
            self.load_state_dict(new_state)

    def forward(self, x):
        x = self.backbone(x)
        pooled = self.avg_pooling(x)
        inter_out = self.fc(pooled.view(pooled.size(0), -1))
        out = self.fc2(inter_out)
        return out, inter_out, x
    
model = f_model(freeze_param=FREEZE_PARAM, model_path=DUMPED_MODEL).cuda()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=MOMENTUM)
{% endhighlight %} 
</p>
</details>

#### Results

Plot for training/test accuracy on ResNet-50 and ResNeXt-50
![alt-text-1](/assets/images/resnet.png "resnet") ![alt-text-2](/assets/images/resnext.png "resnext")

The train/test accuracy (top-1) reached 59%/68% for the ResNet-50 model. On ResNeXt-50, I stopped it at the 5th epoch as no further improvement is seen.

## Feature Extraction

I used the output of the second last fully connected layer as the embeddings, which are vectors of dimension (512, 1). 

See codes below for the feature extraction.
<details>
<summary>
<i>feature extractor</i>
</summary>
<p>{% highlight python %}
class FeatureExtractor(nn.Module):
    def __init__(self, deep_module, color_module, pooling_module):
        super(FeatureExtractor, self).__init__()
        self.deep_module = deep_module
        self.deep_module.eval()

    def forward(self, x):
        cls, feat, conv_out = self.deep_module(x)
        return feat.cpu().data.numpy()

main_model = f_model(model_path=DUMPED_MODEL).cuda()   
extractor = FeatureExtractor(main_model)
all_loader = torch.utils.data.DataLoader(
        Fashion_attr_prediction(type="all", transform=data_transform_test),
        batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

def dump_dataset(loader, deep_feats, labels):
    for batch_idx, (data, data_path) in enumerate(loader):
        data = Variable(data).cuda()
        deep_feat = extractor(data)
        for i in range(len(data_path)):
            path = data_path[i]
            feature_n = deep_feat[i].squeeze()
            deep_feats.append(feature_n)
            labels.append(path)

        if batch_idx % LOG_INTERVAL == 0:
            print("{} / {}".format(batch_idx * EXTRACT_BATCH_SIZE, len(loader.dataset)))
    
        feat_all = '/output/all_feat_pca.npy'
        feat_list = '/output/all_feat.list'
        with open(feat_list, "w") as fw:
            fw.write("\n".join(labels))
        np.save(feat_all, np.vstack(deep_feats_reduced))
        print("Dumped to all_feat.npy and all_feat.list.")  

deep_feats = []
labels = []
dump_dataset(all_loader, deep_feats, labels)
{% endhighlight %} 
</p>
</details>

## Image Retrieval
Now with the feature vector of all images available, it's time to build a search function(s) to get similar images given an image input. I tried three approaches and compared the time taken in each method.
The methods included:

**A naive approach** that computes the similarity score between the given image to every other image in the dataset and get the top-n images.
<details>
<summary>
<i>naive query</i>
</summary>
<p>{% highlight python %}
def dump_single_feature(img_path):
    deep_feats, labels = load_feat_db()
    
    deep_feats = np.array(deep_feats)
    labels = np.array(labels)
    deep_feat = deep_feats[labels == img_path][0,:]

    return deep_feat
    
def get_similarity(feature, feats, metric='cosine'):
    dist = cdist(np.expand_dims(feature, axis=0), feats, metric)[0]
    return dist  

def get_top_n(dist, labels, retrieval_top_n):
    ind = np.argpartition(dist, retrieval_top_n)[0:retrieval_top_n]
    ret = list(zip([labels[i] for i in ind], dist[ind]))
    ret = sorted(ret, key=lambda x: x[1], reverse=False)
    print(ret)
    return ret
    
def get_deep_top_n(features, deep_feats, labels, retrieval_top_n=5):
    deep_scores = get_similarity(features, deep_feats, DISTANCE_METRIC[0])
    results = get_top_n(deep_scores, labels, retrieval_top_n)
    return results
    
def naive_query(features, deep_feats, labels, retrieval_top_n=5):
    results = get_deep_color_top_n(features, deep_feats, labels, retrieval_top_n)
    return results
    
feats, labels = load_feat_db()   
f = dump_single_feature_npy(example) 
result = naive_query(f, feats, labels, 5)
    
{% endhighlight %} 
</p>
</details>  

**A K-Means approach** that added an intermediate step to classify the images to a number of clusters (50 in my case) in the features space. The query would firstly look for the cluster, followed by similarity search within the cluster. 
<details>
<summary>
<i>K-Means query</i>
</summary>
<p>{% highlight python %}
feats, labels = load_feat_db()
model = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_jobs=-1).fit(feats)

def kmeans_query(clf, features, deep_feats, labels, retrieval_top_n=5):
    label = clf.predict(features[0].reshape(1, features[0].shape[0]))
    ind = np.where(clf.labels_ == label)
    d_feats = deep_feats[ind]
    n_labels = list(np.array(labels)[ind])
    results = get_deep_top_n(features, d_feats, n_labels, retrieval_top_n)
    return results
{% endhighlight %} 
</p>
</details>

**A PCA (Principal Component Analysis) approach** that reduced dimensionality of the feature vectors. It appears that we could reduce the features from 512 to 30 to explain at least 90% of the variance.
<details>
<summary>
<i>PCA on feature vectors</i>
</summary>
<p>{% highlight python %}
#Reduce dimensionality on deep features
scaler = MinMaxScaler(feature_range=[-1, 1])
feats_rescaled = scaler.fit_transform(feats)
pca = PCA(n_components=30)
feats_reduced = pca.fit_transform(feats_rescaled)

with open(feat_list, "w") as fw:
    fw.write("\n".join(labels))
np.save(feat_all, np.vstack(feats_reduced))
{% endhighlight %} 
</p>
</details>

Comparing the retrieval time taken by these three approaches, PCA scored the top place. In addition to the improvement in algorithm time complexity, it also eases the burden on server in database loading, as the features data file is reduced from 286MB to 34MB
![alt-text-1](/assets/images/retrieval-time.png "retrieval")
Therefore 

## Deploying the model on a web interface

I built a simple [web application](https://deepfashion-finder.herokuapp.com/) based on the search engine generated above. The front page looks like this:
![alt-text-1](/assets/images/deepfashion-home.png "app homepage")
The app would firstly pick one image randomly from the upper wear dataset (139,709 images) within DeepFashion. You can refresh the page till you see an image that you like, and click the image to check out the results. 
Have fun!

See the [codes](https://github.com/yubin627/ga_projects/tree/master/Capstone_Project/app) for the flask deployment.

---
## Additional Notes 

### Features
Color features 

I extracted the RGB information by average pooling the original 224*224*3 images to 7*7*3 tensors, and then took the 2048*7*7 last convolutional layer trained by the model and applied average pooling to get 7*7 matrices that captures coarse information about the images. Subsequently I used the positions of the maximum 10 values from the 7*7 matrix to extract 10*3 vectors from the image 7*7*3 tensors. This way tese feature vectors capture most of the color information.

<details>
<summary>
<i>color features generator</i>
</summary>
<p>{% highlight python %}
class c_model(nn.Module):
    '''
    Extract color tensors from original images
    input: N * C * 224 * 224
    output: N * C * 7 * 7
    '''
    def __init__(self, pooling_size=32):
        super(c_model, self).__init__()
        self.pooling = nn.AvgPool2d(pooling_size)

    def forward(self, x):
        return self.pooling(x)

class p_model(nn.Module):
    '''
    Apply average pooling to obtain coarse information from 
    the last conv layer
    input: N * C * W * H
    output: N * 1 * W * H
    '''
    def __init__(self):
        super(p_model, self).__init__()

    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.avg_pool1d(x, c)
        return pooled.view(n, 1, w, h)

main_model = f_model(model_path=DUMPED_MODEL).cuda()
color_model = c_model().cuda()
pooling_model = p_model().cuda()
extractor = FeatureExtractor(main_model, color_model, pooling_model)

all_loader = torch.utils.data.DataLoader(
        Fashion_attr_prediction(type="all", transform=data_transform_test),
        batch_size=EXTRACT_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

def dump_dataset(loader, deep_feats, color_feats, labels):
    for batch_idx, (data, data_path) in enumerate(loader):
        data = Variable(data).cuda()
        deep_feat, color_feat = extractor(data)
        for i in range(len(data_path)):
            path = data_path[i]
            feature_n = deep_feat[i].squeeze()
            color_feature_n = color_feat[i]
            # dump_feature(feature, path)

            deep_feats.append(feature_n)
            color_feats.append(color_feature_n)
            labels.append(path)

        if batch_idx % LOG_INTERVAL == 0:
            print("{} / {}".format(batch_idx * EXTRACT_BATCH_SIZE, len(loader.dataset)))
    
        feat_all = '/output/all_feat_pca.npy'
        color_feat_all = '/output/all_color_feat.npy'
        feat_list = '/output/all_feat.list'
        with open(feat_list, "w") as fw:
            fw.write("\n".join(labels))
        np.save(feat_all, np.vstack(deep_feats_reduced))
        np.save(color_feat_all, np.vstack(color_feats))
        print("Dumped to all_feat.npy, all_color_feat.npy and all_feat.list.")  

deep_feats = []
color_feats = []
labels = []
dump_dataset(all_loader, deep_feats, color_feats, labels)
{% endhighlight %} 
</p>
</details>

So now we have two types of features, one being the 512-dim deep features and 10*3-dim color features. We can play around with the similarity score by customizing the weightage to these two components.

As follows are some of the examples with various weights.


### Libraries
There are [various deep learning frameworks](https://skymind.ai/wiki/comparison-frameworks-dl4j-tensorflow-pytorch) that can achieve the same task. PyTorch is easier for me to pick up as it is similar to Python NumPy in the way that it manages computations, while TensorFlow appears more complicated to me for its similarity to C++ (which I have little experience in). However it is still a bit of a challege to write the training codes and to visualize the training progress in PyTorch.

I came across FastAi which is a fantastic wrapper sitting on top of PyTorch that simplifies the code dramatically. It also provides lots of function that makes model fine-tuning so much easier. Here is a [notebook](https://github.com/yubin627/ga_projects/blob/master/Capstone_Project/codes/fasti_ai.ipynb) where I attempted a simple trial of ResNet-34 and ResNet-50 using the same dataset on FastAi. Love to play with it more!

## Future Work
- Explore crop detector to enable input from external source without bbox information.
- Expand the application of the model to other image dataset to build use cases that answer real-world problems. I have wanted to join the [volunteer project](https://www.meetup.com/DataKind-SG/events/261014834/?_xtd=gqFypzUzMTY4NDihcKZpcGhvbmU&from=ref) in DataKind to identify water points around the globe. Comparing to the 'ideal' DeepFashion dataset, this would require much more efforts in the data preprocessing involving data cleaning, labeling etc.
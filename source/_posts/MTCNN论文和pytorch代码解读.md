title: MTCNN论文和pytorch代码解读
tags:
  - 计算机视觉
categories: []
author: whut ykh
date: 2021-04-12 08:39:00
---
# MTCNN人脸检测和pytorch代码实现解读

## 传送门
> 论文地址：[https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)
> 我的论文笔记：[https://khany.top/file/paper/mtcnn.pdf](https://khany.top/file/paper/mtcnn.pdf)
> 本文csdn链接：[https://blog.csdn.net/Jack___E/article/details/115601474](https://blog.csdn.net/Jack___E/article/details/115601474)
> github参考：[https://github.com/Sierkinhane/mtcnn-pytorch](https://github.com/Sierkinhane/mtcnn-pytorch)
> github参考：[https://github.com/GitHberChen/MTCNN_Pytorch](https://github.com/GitHberChen/MTCNN_Pytorch)

## abstract
<!-- more -->
> abstract—Face detection and alignment in unconstrained environment are challenging due to various poses, illuminations and occlusions. Recent studies show that deep learning approaches can achieve impressive performance on these two tasks. In this paper, we propose a deep cascaded multi-task framework which exploits the inherent correlation between detection and alignment 
to boost up their performance. In particular, our framework leverages a cascaded architecture with three stages of carefully designed deep convolutional networks to predict face and landmark location in a coarse-to-fine manner. In addition, we propose a new online hard sample mining strategy that further improves the performance in practice. Our method achieves superior accuracy over the state-of-the-art techniques on the challenging FDDB and WIDER FACE benchmarks for face detection, and AFLW benchmark for face alignment, while keeps real time performance. 

在无约束条件的环境下进行人脸检测和校准人脸检测和校准是非常具挑战性的，因为你要考虑复杂的姿势，光照因素以及面部遮挡问题的影响，最近的研究表明，使用深度学习的方法能够在这两项任务上有不错的效果。本文作者探索和研究了人脸检测和校准之间的内在联系，提出了一个深层级的多任务框架有效提升了网络的性能。我们的深层级联合架构包含三个阶段的卷积神经网络从粗略到细致逐步实现人脸检测和面部特征点标记。此外，我们还提出了一种新的线上进行困难样本的预测的策略可以在实际使用过程中有效提升网络的性能。我们的方法目前在FDDB和WIDER FACE人脸检测任务和AFLW面部对准任务上超越了最佳模型方法的性能标准，同时该模型具有不错的实时检测效果。
> my English is not so well，if there are some mistakes in translations, please contact me in blog comments.


## 简介
人脸检测和脸部校准任务对于很多面部任务和应用来说都是十分重要的，比如说人脸识别，人脸表情分析。不过在现实生活当中，面部特征时常会因为一些遮挡物，强烈的光照对比，复杂的姿势发生很大的变化，这使得这些任务变得具有很大挑战性。Viola和Jones 提出的级联人脸检测器利用[Haar特征和AdaBoost训练级联分类器](https://www.cnblogs.com/zyly/p/9410563.html)，实现了具有实时效率的良好性能。 然而，也有相当一部分的工作表明，这种分类器可能在现实生活的应用中效果显着降低，即使具有更高级的特征和分类器。，之后人们又提出了[DPM](https://blog.csdn.net/weixin_41798111/article/details/79989794)的方式，这些年来基于CNN的一些解决方案也层出不穷，基于CNN的方案主要是为了识别出面部的一些特征来实现。**作者认为这种方式其实是需要花费更多的时间来对面部进行校准来训练同时还忽略了面部的重要特征点位置和边框问题。**
对于人脸校准的话目前也有大致的两个方向：一个是基于回归的方法还有一个就是基于模板拟合的方式进行，主要是对特征识别起到一个辅助的工作。
作者认为关于人脸识别和面部对准这两个任务其实是有内在关联的，而目前大多数的方法则忽略了这一点，所以本文所提出的方案就是将这两者都结合起来，构建出一个三阶段的模型实现一个端到端的人脸检测并校准面部特征点的过程。
**第一阶段：通过浅层CNN快速生成候选窗口。**
**第二阶段：通过more complex的CNN拒绝大量非面部窗口来细化窗口。**
**第三阶段：使用more powerful的CNN再次细化结果并输出五个面部标志位置。**

## 方法
#### Image pyramid图像金字塔
在进入stage-1之前，作者先构建了一组多尺度的图像金字塔缩放，这一块要理解起来还是有一点费力的，这里我们需要了解的是在训练网络的过程当中，作者都是把WIDER FACE的图片随机剪切成12x12size的大小进行单尺度训练的，不能满足任意大小的人脸检测，所以为了检测到不同尺寸的人脸，所以在推理时需要先生成图像金字塔生成一组不同scale的图像输入P-net进行来实现人脸的检测，同时也这是为什么P-net要使用FCN的原因。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021041118210630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
```python
def calculateScales(img):
	min_face_size = 20 # 最小的人脸尺寸
	width, height = img.size
	min_length = min(height, width)
	min_detection_size = 12
	factor = 0.707  # sqrt(0.5)
	scales = []
	m = min_detection_size / min_face_size
	min_length *= m
	factor_count = 0
	# 图像尺寸缩小到不大于min_detection_size
	while min_length > min_detection_size:
	    scales.append(m * factor ** factor_count)
	    min_length *= factor
	    factor_count += 1
```
$$\text{length} =\text{length} \times (\frac{\text{min detection size}}{\text{min face size}} )\times factor^{count} $$
$\text{here  we define in our code:} factor=\frac{1}{\sqrt{2}}, \text{min face size}=20,\text{min detection size}=12$

**min_face_size和factor对推理过程会产生什么样的影响？**
`min_face_size`越大，`factor`越小，图像最短边就越快缩放到接近`min_detect_size`,从而生成图像金字塔的耗时就越短，同时各尺度的跨度也更大。因此，加大`min_face_size`、减小`factor`能加速图像金字塔的生成，但同时也更易造成漏检。
#### Stage 1 P-Net(Proposal Network)
这是一个全卷积网络，也就是说这个网络可以兼容任意大小的输入，他的整个网络其实十分简单，该层的作用主要是为了获得人脸检测的大量候选框，这一层我们最大的作用就是要尽可能的提升recall，然后在获得许多候选框后再使用非极大抑制的方法合并高度重叠的候选区域。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411181845202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
P-Net的示意图当中我们也可以看到的是作者输入12x12大小最终的output为一个1x1x2的face label classification概率和一个1x1x4的boudingbox(左上角和右下角的坐标)以及一个1x1x10的landmark(双眼，鼻子，左右嘴角的坐标)
**注意虽然这里作者举例是12x12的图片大小，但是他的意思并不是说这个P-Net的图片输入大小必须是12，我们可以知道这个网络是FCN，这就意味着不同输入的大小都是可以输入其中的，最后我们所得到的featuremap$[m \times n \times (2+4+10) ]$每一个小像素点映射到输入图像所在的12x12区域是否包含人脸的分类结果,候选框左上和右下基准点以及landmark，然后根据图像金字塔的我们再根据scale和resize的逆过程将其映射到原图像上**

**P-net structure**
```python
class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            # 12x12x3
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            # 10x10x10
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            # 5x5x10
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            # 3x3x16
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            # 1x1x32
            nn.PReLU()  # PReLU3
        )
        # detection
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        # weight initiation with xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        det = torch.sigmoid(self.conv4_1(x))
        box = self.conv4_2(x)
        landmark = self.conv4_3(x)
        # det:[,2,1,1], box:[,4,1,1], landmark:[,10,1,1]
        return det, box, landmark
```
这里要提一下`nn.PReLU()`，有点类似Leaky ReLU，可以看下面的博客深入了解一下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411201535463.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
> [https://blog.csdn.net/qq_23304241/article/details/80300149](https://blog.csdn.net/qq_23304241/article/details/80300149)
> [https://blog.csdn.net/shuzfan/article/details/51345832](https://blog.csdn.net/shuzfan/article/details/51345832)

然后在P-Net之后我们通过**非极大抑制**的方法和将**所有的boudingbox都修正为框的最长边为边长的正方形框**，**主要是避免后面的Rnet在resize的时候因为尺寸原因出现信息的损失。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411201152243.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
#### Non-Maximum Suppression非极大抑制
其实关于非极大抑制这个trick最初是在目标检测任务当中提出的来的，其思想是搜素局部最大值，抑制极大值，主要用在目标检测当中，最传统的非极大抑制所采用的评价指标就是**交并比IoU**(intersection-over-union)即两个groud truth和bounding box的交集部分除以它们的并集.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210412002012403.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
$$IoU = \frac{area(C) \cap area(G)}{area(C) \cup area(G)}$$

```python
def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy1 = np.maximum(box[1], boxes[:, 1])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter,(box_area + area - inter))
    #ovr = inter / (box_area + area - inter)
    return ovr
```

**使用非极大抑制的前提是，我们已经得到了一组候选框和对应label的置信分数，以及groud truth的信息，通过设定阈值来删除重合度较高的候选框。**
算法流程如下：
- 根据置信度得分进行排序
- 选择置信度最高的比边界框添加到最终输出列表中，将其从边界框列表中删除
- 计算所有边界框的面积
- 计算置信度最高的边界框与其它候选框的IoU。
- 删除IoU大于阈值的边界框
- 重复上述过程，直至边界框列表为空。

```python
def nms(dets,threshod,mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param threshod: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:,0]
    y1 = dets[;,1]
    x2 = dets[:,2]
    y2 = dets[:,3]

    scores = dets[:,4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1] # reverse

    keep=[]

    while order.size()>0:
        i = order[0]
        keep.append(i)
        # A & B left top position 
        xx1 = np.maximun(x1[i],x1[order[1,:]])
        yy1 = np.maximun(y1[i],y1[order[1,:]])
        # A & B right down position
        xx2 = np.minimum(x2[i],x2[order[1,:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        # cacaulate the IOU between box which have largest score with other boxes
        if mode == "Union":
            # area[i]: the area of largest score
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # delete the IoU that higher than threshod 
        inds = np.where(ovr <= threshod)[0]
        order = order[inds + 1] # +1: eliminates the first element in order
    
    return keep 
```
#### 边框修正
以最大边作为边长将矩形修正为正方形，同时包含的信息也更多，以免在后面resize输入下一个网络时减少信息的损失。
```python
def convert_to_square(bboxes):
    """
    Convert bounding boxes to a square form.
    """
    # 将矩形对称扩大为正方形
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes
```

#### Stage 2 R-Net(Refine Network)
R-net的输入是固定的，必须是24x24，所以对于P-net产生的大量boundingbox我们需要先进行resize然后再输入R-Net，**在论文中我们了解到该网络层的作用主要是对大量的boundingbox进行有效过滤，获得更加精细的候选框。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411203248732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

**R-net structure**
```python
class R_Net(nn.Module):
    def __init__(self):
        super(R_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            # 24x24x3
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            # 22x22x28
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            # 10x10x28
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            # 8x8x48
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            # 3x3x48
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            # 2x2x64
            nn.PReLU()  # prelu3
        )
        # 2x2x64
        self.conv4 = nn.Linear(64 * 2 * 2, 128)  # conv4
        # 128
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        landmark = self.conv5_3(x)
        return det, box, landmark
```

然后在P-Net之后我们通过**非极大抑制**的方法和将**所有的boudingbox都修正为框的最长边为边长的正方形框**，**也是避免后面的Onet在resize的时候出现因为尺寸原因出现信息的损失。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411202906999.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
#### Stage 3 O-Net(Output?[作者未指出命名] Network)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411203345893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
Onet与Rnet工作流程类似。只不过输入的尺寸变成了48x48，对于R-net当中的框再次进行处理，得到的网络结构的输出则是最终的label classfication，boundingbox，landmark。
**O-net structure**
```python
class O_Net(nn.Module):
    def __init__(self):
        super(O_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Linear(128 * 2 * 2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Linear(256, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        det = torch.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        return det, box, landmark
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411204039760.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
**注意，其实在之前的两个网络当中我们也预测了landmark的位置，只是在论文的图示当中没有展示而已，从作者描述的网络结构图的输出当中是详细指出了landmarkposition的卷积块的**

## 损失函数
在研究完网络结构以后我们可以来看看这个mtcnn模型的一个损失函数，作者将损失函数分为了3个部分：Face classification、Bounding box regression、Facial landmark localization。
#### Face classification
对于第一部分的损失就是用的是最常用的交叉熵损失函数，就本质上来说人脸识别还是一个二分类问题，这里就使用Cross Entropy是最简单也是最合适的选择。
$$L_i^{det} = -(y_i^{det} \log(p_i) + (1-y_i^det)(1-\log(p_i)))$$
$p_i$是预测为face的可能性，$y_i^{det}$指的是真实标签也就是groud truth label 
#### Bounding box regression 
对于目标边界框的损失来说，对于每一个候选框我们都需要对与他最接近的真实目标边界框进行比较，the bounding box$（left, top, height, and width）$
$$L_i^{box} = ||y_j^{box}-y_i^{box} ||_2^2$$
#### Facial landmark localization
而对于boundingbox和landmark来说整个框的调整过程其实可以看作是一个连续的变化过程，固使用的是欧氏距离回归损失计算方法。比较的是各个点的坐标与真实面部关键点坐标的差异。
$$L_i^{landmark} = ||y_j^{landmark}-y_i^{landmark} ||_2^2$$
#### total loss
最终我们得到的损失函数是有上述这三部分加权而成
$$\min{\sum_{i=1}^{N}{\sum_{j \in \{det,box,landmark\}} \alpha_j \beta_i^j L_i^j}}$$
其中$\alpha_j$表示权重，$\beta_i^j$表示第i个样本的类型，也可以说是第i个样本在任务j中是否需要贡献loss，如果不存在人脸即label为0时则无需计算loss。
对于不同的网络我们所设置的权重是不一样的
**in P-net & R-net(在这两层我们更注重classification的识别)**
$$alpha_{det}=1, alpha_{box}=0.5,alpha_{landmark}=0.5$$
**in O-net(在这层我们提高了对于注意关键点的预测精度)**
$$alpha_{det}=1, alpha_{box}=0.5,alpha_{landmark}=1 $$
文中也有提到，采用的是随机梯度下降优化器进行的训练。
## OHEM（Online Hard Example Mining）
作者对于困难样本的在线预测所做的trick也比较简单，就是挑选损失最大的前70%作为困难样本，在反向传播时仅使用这70%困难样本产生的损失，这样就剔除了很容易预测的easy sample对训练结果的影响，不过在我参考的这两个版本的代码里面似乎没有这么做。
## 实验
实验这块的话也比较充分，验证了每个修改部分对于实验结果的有效性验证，主要是讲述了实验过程对于训练数据集的处理和划分，验证了在线硬样本挖掘的有效性，联合检测和校准的有效性，分别评估了face detection和lanmark的贡献，面部检测评估，面部校准评估以及与SOTA的对比，这一块的话就没有细看了，分享的网站也有详细解释，论文原文也写了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411214338982.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210411214409844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 总结
这篇论文是中科院深圳先进技术研究院的乔宇老师团队所作，2016ECCV，看完这篇文章的话给我的感觉的话就是idea和实现过程包括实验都是很充分的，创新点这块的话主要是挖掘了人脸特征关键点和目标检测的一些联系，结合了人脸检测和对准两个任务来进行人脸检测，相比其他经典的目标检测网络如yolov3,R-CNN系列，在网络和损失函数上的创新确实略有不足，但是也让我收到了一些启发，看似几个简单的model和常见的loss联合，在对数据进行有效处理的基础上也是可以实现达到十分不错的效果的，不过这个方案的话在训练的时候其实是很花费时间的，毕竟需要对于不同scale的图片都进行输入训练，然后就是这种输入输出的结构其实还是存在一些局限性的，对于图像检测框和关键点的映射我个人觉得也比较繁杂或者说浪费了一些时间，毕竟是一篇2016年的论文之后应该会有更好的实现方式。

## 参考资料
>参考链接： [https://zhuanlan.zhihu.com/p/337690783](https://zhuanlan.zhihu.com/p/337690783)
>参考链接：[https://zhuanlan.zhihu.com/p/60199964?utm_source=qq&utm_medium=social&utm_oi=1221207556791963648](https://zhuanlan.zhihu.com/p/60199964?utm_source=qq&utm_medium=social&utm_oi=1221207556791963648)
>参考链接： [https://blog.csdn.net/weixin_44791964/article/details/103530206](https://blog.csdn.net/weixin_44791964/article/details/103530206)
>参考链接：[https://blog.csdn.net/qq_34714751/article/details/85536669](https://blog.csdn.net/qq_34714751/article/details/85536669)
>参考链接：[https://www.bilibili.com/video/BV1fJ411C7AJ?from=search&seid=2691296178499711503](https://www.bilibili.com/video/BV1fJ411C7AJ?from=search&seid=2691296178499711503)

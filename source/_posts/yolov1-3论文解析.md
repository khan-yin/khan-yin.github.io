title: yolov1-3论文解析
tags:
  - 计算机视觉
categories: []
author: whut ykh
date: 2021-04-22 10:20:00
---
## yolov1-3论文解析
最近在看经典目标检测算法yolo的思想，为了更好的了解yolo系列的相关文章，我从最初版本的论文思想开始看的，之后有时间会把yolov4和yolov5再认真看看，目前来说yolov3的spp版本是使用得最为广泛的一种，整体上来说yolo的设计思想还是很有创造性的数学也比较严谨。
<!--more-->
## yolov1论文思想
物体检测主流的算法框架大致分为one-stage与two-stage。two-stage算法代表有R-CNN系列，one-stage算法代表有Yolo系列。按笔者理解，two-stage算法将步骤一与步骤二分开执行，输入图像先经过候选框生成网络（例如faster rcnn中的RPN网络），再经过分类网络；one-stage算法将步骤一与步骤二同时执行，输入图像只经过一个网络，生成的结果中同时包含位置与类别信息。two-stage与one-stage相比，精度高，但是计算量更大，所以运算较慢。

> We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. **Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities.**

yolo相比其他R-CNN等方法最大的一个不同就是他将这个识别和定位的过程转化成了一个空间定位并分类的回归问题，这也是他为什么能够利用网络进行端到端直接同时预测的重要原因。
#### yolo的特点
**1.YOLO速度非常快。由于我们的检测是当做一个回归问题，不需要很复杂的流程。在测试的时候我们只需将一个新的图片输入网络来检测物体。**
`First, YOLO is extremely fast. Since we frame detection as a regression problem we don’t need a complex pipeline`
**2.Yolo会基于整张图片信息进行预测，与基于滑动窗口和候选区域的方法不同，在训练和测试期间YOLO可以看到整个图像，所以它隐式地记录了分类类别的上下文信息及它的全貌**
`Second, YOLO reasons globally about the image when making predictions Unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance`
**3.第三，YOLO能学习到目标的泛化表征。作者尝试了用自然图片数据集进行训练，用艺术画作品进行预测，Yolo的检测效果更佳。**
`Third, YOLO learns generalizable representations of objects. Since YOLO is highly generalizable it is less likely to break down when applied to new domains or unexpected inputs`
#### backbone-darknet
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421180804103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
yolov1这块的话，由于是比较早期的网络所以在设计时没有使用batchnormal，激活函数上采用leaky relu，输入图像大小为448x448，经过许多卷积层与池化层，变为7x7x1024张量，最后经过两层全连接层，输出张量维度为7x7x30。除了最后一层的输出使用了线性激活函数，其他层全部使用Leaky Relu激活函数。
$$\phi(x)= \begin{cases}
x,& \text{if x>0} \\
0.1x,& \text{otherwise}
\end{cases}$$
#### 输出维度
yolov1最精髓的地方应该就是他的输出维度，论文当中有放出这样一张图片，看完以后我们能对yolo算法的特点有更加深刻的理解。在上图的backbone当中我们可以看到的是
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421182245129.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)yolo的输出相当于把原图片分割成了$S \times S$个区域，然后对预测目标标定他的boundingbox，boundingbox由中心点的坐标,长宽以及物体的置信分数组成$x,y,w,h,confidence$，图中的$B$表示的是boundingbox的个数。置信分数在yolo当中是这样定义的：$Pr(Object)*IOU_{pred}^{truth}$对于$Pr(Object)=1 \text{ if detect object else 0}$，其实**置信分数就是boudingbox与groundtruth之间的IOU**，并对每个小网格对应的$C$个类别都预测出他的**条件概率：**$Pr(Class_i|Object)$。
在推理时，我们可以得到当前网格对于每个类别的预测置信分数：$$Pr(Class_i)*IOU_{pred}^{truth} = Pr(Class_i|Object)*Pr(Object)*IOU_{pred}^{truth}$$
这样一种严谨的条件概率的方式得到我们的最终概率是十分可行和合适的。
`YOLO imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict. Our model struggles with small objects that appear in groups, such as flocks of birds.`
**不过在原论文当中也有提到的一点是：YOLO给边界框预测强加空间约束，因为每个网格单元只预测两个框和只能有一个类别。这个空间约束限制了我们的模型可以预测的邻近目标的数量。我们的模型难以预测群组中出现的小物体（比如鸟群）。**
#### 损失函数
在损失函数方面，由于公式比较长所以就直接截图过来了，损失函数的设计放方面作者也考虑的比较周全，
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421183909382.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
预测框的中心点$(x,y)$。预测框的宽高$w,h$，其中$\mathbb{1}_{i j}^{\text {obj }}$为控制函数，**在标签中包含物体的那些格点处，该值为 1 ，若格点不含有物体，该值为 0**

在计算损失函数的时候作者最开始使用的是最简单的平方损失函数来计算检测，因为它是计算简单且容易优化的，但是它并不完全符合我们最大化平均精度的目标，原因如下：
`It weights localization error equally with classification error which may not be ideal.`
**1.它对定位错误的权重与分类错误的权重相等，这可能不是理想的选择。**
`Also, in every image many grid cells do not contain any object. This pushes the “confidence” scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability, causing training to diverge early on.`
而且，在每个图像中，许多网格单元不包含任何对象。这就把这些网格的置信度分数推到零，往往超过了包含对象的网格的梯度。**2.这可能导致模型不稳定，导致训练早期发散难以训练。**

**解决方案：增加边界框坐标预测的损失，并且减少了不包含对象的框的置信度预测的损失。我们使用两个参数来实现这一点。**$\lambda_{coord}=5，\lambda_{noobj}=0.5$，其实这也是yolo面临的一个样本数目不均衡的典型例子。主要目的是让含有物体的格点，在损失函数中的权重更大，让模型更加注重含有物体的格点所造成的损失。
**我们的损失函数则只会对那些有真实物体所属的格点进行损失计算，若该格点不包含物体，那么预测数值不对损失函数造成影响。**

**这里对$w,h$在损失函数中的处理分别取了根号，原因在于，如果不取根号，损失函数往往更倾向于调整尺寸比较大的预测框**。例如，20个像素点的偏差，对于800x600的预测框几乎没有影响，此时的IOU数值还是很大，但是对于30x40的预测框影响就很大。**取根号是为了尽可能的消除大尺寸框与小尺寸框之间的差异。**

预测框的置信度$C_i$。当该格点不含有物体时，该置信度的标签为0；若含有物体时，该置信度的标签为预测框与真实物体框的IOU数值（。
物体类别概率$P_i$，对应的类别位置，该标签数值为1，其余位置为0，与分类网络相同。
## yolov2：Better, Faster, Stronger
yolov2的论文里面其实更多的是对训练数据集的组合和扩充，然后再添加了很多计算机视觉方面新出的trick，然后达到了性能的提升。同时yolov2方面所使用的trick也是现在计算机视觉常用的方法，我认为这些都是我们需要好好掌握的，同时也是面试和kaggle上都经常使用的一些方法。
#### Batch Normalization
BN是由Google于2015年提出，论文是《Batch Normalization_ Accelerating Deep Network Training by Reducing Internal Covariate Shift》，这是一个深度神经网络训练的技巧，主要是让数据的分布变得一致，从而使得训练深层网络模型更加容易和稳定。
这里有一些相关的链接可以参考一下

> [https://www.cnblogs.com/itmorn/p/11241236.html](https://www.cnblogs.com/itmorn/p/11241236.html)
> [https://zhuanlan.zhihu.com/p/24810318](https://zhuanlan.zhihu.com/p/24810318)
> [https://zhuanlan.zhihu.com/p/93643523](https://zhuanlan.zhihu.com/p/93643523)
> [莫烦python](https://www.bilibili.com/video/BV1Lx411j7GT?from=search&seid=16898413208243461324)
> [李宏毅yyds](https://www.bilibili.com/video/BV1bx411V798?from=search&seid=16898413208243461324)

算法具体过程：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421213340208.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

这里要提一下这个$\gamma和\beta$是可训练参数，维度等于张量的通道数，主要是为了在反向传播更新网络时使得标准化后的数据分布与原始数据尽可能保持一直，从而很好的抽象和保留整个batch的数据分布。

Batch Normalization的作用：
将这些输入值或卷积网络的张量在batch维度上进行类似标准化的操作，将其放缩到合适的范围，加快模型训练时的收敛速度，使得模型训练过程更加稳定，避免梯度爆炸或者梯度消失，并且起到一定的正则化作用。
#### High Resolution Classifier
在Yolov1中，网络的backbone部分会在ImageNet数据集上进行预训练，训练时网络输入图像的分辨率为224x224。在v2中，将分类网络在输入图片分辨率为448x448的ImageNet数据集上训练10个epoch，再使用检测数据集（例如coco）进行微调。高分辨率预训练使mAP提高了大约4%。
#### Convolutional With Anchor Boxes
`predicts these offsets at every location in a feature map`
`Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn.`
第一篇解读v1时提到，每个格点预测两个矩形框，在计算loss时，只让与ground truth最接近的框产生loss数值，而另一个框不做修正。这样规定之后，作者发现两个框在物体的大小、长宽比、类别上逐渐有了分工。在v2中，神经网络不对预测矩形框的宽高的绝对值进行预测，而是预测与Anchor框的偏差（offset），每个格点指定n个Anchor框。在训练时，最接近ground truth的框产生loss，其余框不产生loss。在引入Anchor Box操作后，mAP由69.5下降至69.2，原因在于，每个格点预测的物体变多之后，召回率大幅上升，准确率有所下降，总体mAP略有下降。
#### Dimension Clusters
`Instead of choosing priors by hand, we run k-means clustering on the training set bounding boxes to automatically find good priors.`
这里算是作者数据处理上的一个创新点，这里作者提到之前的工作当中先Anchor Box都是认为设定的比例和大小，而这里作者时采用无监督的方式将训练数据集中的矩形框全部拿出来，用kmeans聚类得到先验框的宽和高。**使用（1-IOU）数值作为两个矩形框的的距离函数**，这个处理也十分的聪明。
$$
d(\text { box }, \text { centroid })=1-\operatorname{IOU}(\text { box }, \text { centroid })
$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421214909877.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
#### Direct location prediction
Yolo中的位置预测方法是预测的左上角的格点坐标预测偏移量。网络预测输出要素图中每个单元格的5个边界框。网络为每个边界框预测$t_x，t_y，t_h，t_w和t_o$这5个坐标。如果单元格从图像的左上角偏移了$(c_x，c_y)$并且先验边界框的宽度和高度为$p_w，p_h$，则预测对应于：
$$
\begin{array}{l}
x=\left(t_{x} * w_{a}\right)-x_{a} \\
y=\left(t_{y} * h_{a}\right)-y_{a}
\end{array}
$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421222704145.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
$$
\begin{aligned}
b_{x} &=\sigma\left(t_{x}\right)+c_{x} \\
b_{y} &=\sigma\left(t_{y}\right)+c_{y} \\
b_{w} &=p_{w} e^{t_{w}} \\
b_{h} &=p_{h} e^{t_{h}} \\
\operatorname{Pr}(\text { object }) * I O U(b, \text { object }) &=\sigma\left(t_{o}\right)
\end{aligned}
$$
**由于我们约束位置预测，参数化更容易学习，使得网络更稳定**使用维度集群以及直接预测边界框中心位置使YOLO比具有anchor box的版本提高了近5％的mAP。
####  Fine-Grained Features
在26*26的特征图，经过卷积层等，变为13*13的特征图后，作者认为损失了很多细粒度的特征，导致小尺寸物体的识别效果不佳，所以在此加入了**passthrough层**。
**传递层通过将相邻特征堆叠到不同的通道而不是堆叠到空间位置，将较高分辨率特征与低分辨率特征相连**，类似于ResNet中的标识映射。这将26×26×512特征映射转换为13×13×2048特征映射，其可以与原始特征连接。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421223145797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
#### Multi-Scale Training
我觉得这个也是yolo为什么性能提升的一个很重要的点，解决了yolo1当中对于多个小物体检测的问题。
原始的YOLO使用448×448的输入分辨率。添加anchor box后，我们将分辨率更改为416×416。然而，**由于我们的模型只使用卷积层和池化层,相比yolo1删除了全连接层，它可以在运行中调整大小（应该是使用了1x1卷积不过我没看代码）**。我们希望YOLOv2能够在不同大小的图像上运行，因此我们将其训练到模型中。
`instead of fixing the input image size we change the network every few iterations. Every 10 batches our network randomly chooses a new image dimension size.`
不固定输入图像的大小，我们每隔几次迭代就更改网络。在每10个batch之后，我们的网络就会随机resize成{320, 352, ..., 608}中的一种。不同的输入，最后产生的格点数不同，比如输入图片是320*320，那么输出格点是10*10，如果每个格点的先验框个数设置为5，那么总共输出500个预测结果；如果输入图片大小是608*608，输出格点就是19*19，共1805个预测结果。
`YOLO’s convolutional layers downsample the image by a factor of 32 so by using an input image of 416 we get an output feature map of 13 × 13.`
` We do this because we want an odd number of locations in our feature map so there is a single center cell.`
关于416×416也是有说法的，主要是下采样是32的倍数，而且最后的输出是13x13是奇数然后会有一个中心点更加易于目标检测。
**这种训练方法迫使网络学习在各种输入维度上很好地预测。这意味着相同的网络可以预测不同分辨率的检测。**
**关于yolov2最大的一个提升还有一个原因就是WordTree组合数据集的训练方法，不过这里我就不再赘述，可以看参考资料有详细介绍，这里主要讲网络和思路**
## yolov3论文改进
yolov3的论文改进就有很多方面了，而且yolov3-spp的网络效果很不错，和yolov4，yolov5的效果差别也不是特别大，这也是为什么yolov3网络被广泛应用的原因之一。

## backbone：Darknet-53
相比yolov1的网络，在网络上有了很大的改进，借鉴了Resnet、Densenet、FPN的做法结合了当前网络中十分有效的
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042122500522.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
1.Yolov3中，全卷积，对于输入图片尺寸没有特别限制，可通过调节卷积步长控制输出特征图的尺寸。
2.Yolov3借鉴了**金字塔特征图**思想，**小尺寸特征图用于检测大尺寸物体，而大尺寸特征图检测小尺寸物体**。特征图的输出维度为$N \times N \times (3 \times (4+1+80))$， NxN为输出特征图格点数，一共3个Anchor框，每个框有4维预测框数值 $t_x,t_y,t_w,t_h$ ，1维预测框置信度，80维物体类别数。所以第一层特征图的输出维度为8x8x255
3.**Yolov3总共输出3个特征图**，第一个特征图下采样32倍，第二个特征图下采样16倍，第三个下采样8倍。每个特征图进行一次3X3、步长为2的卷积，然后保存该卷积layer，再进行一次1X1的卷积和一次3X3的卷积，并把这个结果加上layer作为最后的结果。**三个特征层进行5次卷积处理，处理完后一部分用于输出该特征层对应的预测结果，一部分用于进行反卷积UmSampling2d后与其它特征层进行结合。**
4.concat和resiual加操作，借鉴DenseNet将特征图按照通道维度直接进行拼接，借鉴Resnet添加残差边，缓解了在深度神经网络中增加深度带来的梯度消失问题。
5.上采样层(upsample)：作用是将小尺寸特征图通过插值等方法，生成大尺寸图像。例如使用最近邻插值算法，将8*8的图像变换为16*16。上采样层不改变特征图的通道数。

对于yolo3的模型来说，**网络最后输出的内容就是三个特征层每个网格点对应的预测框及其种类，即三个特征层分别对应着图片被分为不同size的网格后，每个网格点上三个先验框对应的位置、置信度及其种类**。然后对输出进行解码，解码过程也就是yolov2上面的Direct location prediction方法一样，每个网格点加上它对应的x_offset和y_offset，加完后的结果就是预测框的中心，然后再利用 先验框和h、w结合 计算出预测框的长和宽。这样就能得到整个预测框的位置了。
#### CIoU loss
在yolov3当中我们使用的不再是传统的IoU loss，而是一个非常优秀的loss设计，整个发展过程也可以多了解了解IoU->GIoU->DIoU->CIoU，完整的考虑了**重合面积，中心点距离，长宽比**。
$$CIoU=IoU-(\frac{\rho(b,b^{gt}}{c^2}+\alpha v))$$
$$v=\frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}}-\arctan\frac{w}{h})^2$$
$$\alpha = \frac{v}{(1-IoU)+v}$$
其中， $b$ ， $b^{gt}$ 分别代表了预测框和真实框的中心点，且 $\rho$代表的是计算两个中心点间的欧式距离。 [公式] 代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离。
**在介绍CIoU之前，我们可以先介绍一下DIoU loss**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422092054189.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
其实这两项就是我们的DIoU
$$DIoU = IoU-(\frac{\rho(b,b^{gt}}{c^2})$$
则我们可以提出DIoU loss函数
$$
L_{\text {DIoU }}=1-D I o U \\
0 \leq L_{\text {DloU}}  \leq 2
$$
**DloU损失能够直接最小化两个boxes之间的距离，因此收敛速度更快。**
**而我们的CIoU则比DIoU考虑更多的东西，也就是长宽比即最后一项。而$v$用来度量长宽比的相似性,$\alpha$是权重**
在损失函数这块，有知乎大佬写了一下，其实就是对yolov1上的损失函数进行了一下变形。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210421233832256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
#### Spatial Pyramid Pooling
在yolov3的spp版本中使用了spp模块，实现了不同尺度的特征融合，spp模块最初来源于何凯明大神的论文SPP-net，主要用来解决输入图像尺寸不统一的问题，而目前的图像预处理操作中，resize，crop等都会造成一定程度的图像失真，因此影响了最终的精度。SPP模块，使用固定分块的池化操作，可以对不同尺寸的输入实现相同大小的输出，因此能够避免这一问题。
同时SPP模块中不同大小特征的融合，有利于待检测图像中目标大小差异较大的情况，也相当于增大了多重感受野吧，尤其是对于yolov3一般针对的复杂多目标图像。
在yolov3的darknet53输出到第一个特征图计算之前我们插入了一个spp模块。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422093246740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
SPP的这几个模块stride=1，而且会对外补padding，所以最后的输出特征图大小是一致的，然后在深度方向上进行concatenate，使得深度增大了4倍。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422093556765.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422093809296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
可以看出随着输入网络尺度的增大，spp的效果会更好。多个spp的效果增大也就是spp3和spp1的效果是差不多的，为了保证推理速度就只使用了一个spp

#### Mosaic图像增强
在yolov3-spp包括yolov4当中我们都使用了mosaic数据增强，有点类似cutmix。cutmix是合并两张图片进行预测，mosaic数据增强利用了四张图片，对四张图片进行拼接，每一张图片都有其对应的框框，将四张图片拼接之后就获得一张新的图片，同时也获得这张图片对应的框框，然后我们将这样一张新的图片传入到神经网络当中去学习，相当于一下子传入四张图片进行学习了。论文中说这极大丰富了检测物体的背景，且在标准化BN计算的时候一下子会计算四张图片的数据。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422084715366.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
作用：增加数据的多样性，增加目标个数，BN能一次性归一化多张图片组合的分布，会更加接近原数据的分布。
#### focal loss
最后我们提一下focal loss，虽然在原论文当中作者说focal loss效果反而下降了，但是我认为还是有必要了解一下何凯明的这篇bestpaper。作者提出focal loss的出发点也是希望one-stage detector可以达到two-stage detector的准确率，同时不影响原有的速度。作者认为造成这种情况的原因是**样本的类别不均衡导致的**。
对于二分类的CrossEntropy，我们有如下表示方法：
$$
\mathrm{CE}(p, y)=\left\{\begin{array}{ll}
-\log (p) & \text { if } y=1 \\
-\log (1-p) & \text { otherwise. }
\end{array}\right.
$$
我们定义$p_t$：
$$
p_{\mathrm{t}}=\left\{\begin{array}{ll}
p & \text { if } y=1 \\
1-p & \text { otherwise }
\end{array}\right.
$$
则重写交叉熵
$$
\operatorname{CE}(n, y)=C E(p_t)=-\log (p_t)
$$
接着我们提出改进思路就是在计算CE时添加了一个平衡因子，是一个可训练的参数
$$
\mathrm{CE}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}} \log \left(p_{\mathrm{t}}\right)
$$
其中
$$
\alpha_{\mathrm{t}}=\left\{\begin{array}{ll}
\alpha_{\mathrm{t}} & \text { if } y=1 \\
1-\alpha_{\mathrm{t}} & \text { otherwise }
\end{array}\right.
$$
相当于给正负样本加上权重，负样本出现的频次多，那么就降低负样本的权重，正样本数量少，就相对提高正样本的权重。因此可以通过设定a的值来控制正负样本对总的loss的共享权重。a取比较小的值来降低负样本（多的那类样本）的权重。
**但是何凯明认为这个还是不能完全解决样本不平衡的问题，虽然这个平衡因子可以控制正负样本的权重，但是没法控制容易分类和难分类样本的权重**
> As our experiments will show, the large class imbalance encountered during training of dense detectors overwhelms the cross entropy loss. Easily classified negatives comprise the majority of the loss and dominate the gradient. **While
α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples. Instead, we propose to reshape the loss function to down-weight easy examples and thus focus training on hard negatives**

于是提出了一种新的损失函数Focal Loss
$$
\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)
$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422095440770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
#### focal loss的两个重要性质
1、当一个样本被分错的时候，pt是很小的，那么调制因子（1-Pt）接近1，损失不被影响；当Pt→1，因子（1-Pt）接近0，那么分的比较好的（well-classified）样本的权值就被调低了。因此调制系数就趋于1，也就是说相比原来的loss是没有什么大的改变的。当pt趋于1的时候（此时分类正确而且是易分类样本），调制系数趋于0，也就是对于总的loss的贡献很小。

2、当γ=0的时候，focal loss就是传统的交叉熵损失，**当γ增加的时候，调制系数也会增加**。 专注参数γ平滑地调节了易分样本调低权值的比例。γ增大能增强调制因子的影响，实验发现γ取2最好。直觉上来说，调制因子减少了易分样本的损失贡献，拓宽了样例接收到低损失的范围。当γ一定的时候，比如等于2，一样easy example(pt=0.9)的loss要比标准的交叉熵loss小100+倍，当pt=0.968时，要小1000+倍，但是对于hard example(pt < 0.5)，loss最多小了4倍。这样的话hard example的权重相对就提升了很多。这样就增加了那些误分类的重要性

**focal loss的两个性质算是核心，其实就是用一个合适的函数去度量难分类和易分类样本对总的损失的贡献。**

然后作者又提出了最终的focal loss形式
$$
\mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)
$$
$$
F L(p)\left\{\begin{array}{ll}
-\alpha(1-p)^{\gamma} \log (p) & \text { if } y=1 \\
-(1-\alpha) p^{\gamma} \log (1-p) & \text { otherwise }
\end{array}\right.
$$
**这样既能调整正负样本的权重，又能控制难易分类样本的权重**
在实验中a的选择范围也很广，一般而言当γ增加的时候，a需要减小一点（实验中γ=2，a=0.25的效果最好）

## 总结
yolo和yolov2的严谨数学推理和网络相结合的做法令人十分惊艳，而yolov3相比之前的两个版本不管是在数据处理还是网络性质上都有很大的改变，也使用了很多流行的tirck来实现目标检测，基本上结合了很多cv领域的精髓，博客中有很多部分由于时间关系没有太细讲，其实每个部分都可以去原论文中找到很多可以学习的地方，之后有时候会补上yolov4和yolov5的讲解，后面的网络添加了注意力机制模块效果应该是更加work的。
## 参考文献

> [yolov3-spp](https://www.bilibili.com/video/BV1yi4y1g7ro?p=4)
> [focal loss](https://zhuanlan.zhihu.com/p/49981234)
> [batch normal](https://www.cnblogs.com/itmorn/p/11241236.html)
> [IoU系列](https://zhuanlan.zhihu.com/p/94799295)
> [yolov1论文翻译](https://blog.csdn.net/shuiyixin/article/details/82533849)
> [yolo1-3三部曲](https://zhuanlan.zhihu.com/p/76802514)
> [Bubbliiiing的教程里](https://www.bilibili.com/video/BV1Hp4y1y788?from=search&seid=2780145238002767073)
> 还有很多但是没找到之后会补上的

title: EfficientNet论文解读和pytorch代码实现
tags:
  - 计算机视觉
categories: []
author: whut ykh
date: 2021-05-09 17:44:00
---
## EfficientNet论文解读和pytorch代码实现
## 传送门

> 论文地址：[https://arxiv.org/pdf/1905.11946.pdf](https://arxiv.org/pdf/1905.11946.pdf)
> 官方github：[https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
> github参考：[https://github.com/qubvel/efficientnet](https://github.com/qubvel/efficientnet)
> github参考：[https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## 摘要

谷歌的文章总是能让我印象深刻，不管从实验上还是论文的书写上都让人十分的佩服，不得不说这确实是一个非常creative的公司!
`Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.
`
卷积神经网络通常是在固定的资源预算下进行计算和发展的，如果有更多可用的资源，模型能通过缩放和扩展获得更好的效果，本文**系统研究（工作量很充足）**了模型缩放，并且证明了**细致地平衡网络的深度，宽度，和分辨率能够得到更好的效果**，基于这个观点，我们提出了一个新的尺度缩放的方法，我们提出了一个新的尺度缩放的方法即：**使用一个简单且高效的复合系数统一地缩放所有网络层的深度/宽度/分辨率的维度**。我们证明了该方法在扩大MobileNets和ResNet方面的有效性。
<!--more-->

`To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at`[https://github.com/tensorflow/tpu/tree/
master/models/official/efficientnet.](https://github.com/tensorflow/tpu/tree/%20master/models/official/efficientnet.)

更进一步，我们使用**神经网络结构搜索**设计了一个新的baseline网络架构并且对其进行缩放得到了一组模型，我们把他叫做EfficientNets。EfficientNets相比之前的ConvNets，有着更好的准确率和更高的效率。在ImageNet上达到了最好的水平即top-1准确率84.4%/top-5准确率97.1%，然而却比已有的最好的ConvNet模型小了8.4倍并且推理时间快了6.1倍。我们的EfficientNet迁移学习的效果也好，达到了最好的准确率水平CIFAR-100（91.7%），Flowers（98.8%），和其他3个迁移学习数据集合，**参数少了一个数量级（with an order of magnitude fewer parameters）**。

## 论文思想
在之前的一些工作当中，我们可以看到，有的会通过增加网络的width即增加卷积核的个数（从而增大channels）来提升网络的性能，有的会通过增加网络的深度即使用更多的层结构来提升网络的性能，有的会通过增加输入网络的分辨率来提升网络的性能。而在Efficientnet当中则重新探究了这个问题，并提出了一种组合条件这三者的网络结构，同时量化了这三者的指标。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021050914370943.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

`The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem`
增加网络的深度depth能够得到更加丰富、复杂的特征并且能够在其他新任务中很好的泛化性能。但是，由于逐渐消失的梯度问题，更深层的网络也更难以训练。

`wider networks tend to be able to capture more fine-grained features and are easier to train. However, extremely wide but shallow networks tend to have difficulties in capturing higher level features`
with更广的网络往往能够捕获更多细粒度的功能，并且更易于训练。但是，极其宽泛但较浅的网络在捕获更高级别的特征时往往会遇到困难

`With higher resolution input images, ConvNets can potentially capture more fine-grained pattern,but the accuracy gain diminishes for very high resolutions`
使用更高分辨率的输入图像，ConvNets可以捕获更细粒度的图案，但对于非常高分辨率的图片，准确度会降低。

作者在论文中对整个网络的运算过程和复合扩展方法进行了抽象：
首先定义了每一层卷积网络为$\mathcal{F}_{i}\left(X_{i}\right)$，而$X_i$是输入张量，$Y_i$是输出张量，而tensor的形状是$<H_i,W_i,C_i>$
整个卷积网络由 k 个卷积层组成，可以表示为$\mathcal{N}=\mathcal{F}_{k} \odot \ldots \odot \mathcal{F}_{2} \odot \mathcal{F}_{1}\left(X_{1}\right)=\bigodot_{j=1 \ldots k} \mathcal{F}_{j}\left(X_{1}\right)$
从而得到我们的整个卷积网络:
$$
\mathcal{N}=\bigodot_{i=1 \ldots s} \mathcal{F}_{i}^{L_{i}}\left(X_{\left\langle H_{i}, W_{i}, C_{i}\right\rangle}\right)
$$

下标 i(从 1 到 s) 表示的是 stage 的序号，$\mathcal{F}_{i}^{L_{i}}$表示第 i 个 stage ，它表示卷积层 $\mathcal{F}_{i}$重复了${L_{i}}$ 次。

为了探究$d , r , w$这三个因子对最终准确率的影响，则将$d , r , w$加入到公式中，我们可以得到抽象化后的优化问题（在指定资源限制下），其中$s.t.$代表限制条件：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509151026858.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
- $d$用来缩放深度$\widehat{L}_i $。
- $r$用来缩放分辨率即影响$\widehat{H}_i$以及$\widehat{W}_i$。
- $w$就是用来缩放特征矩阵的channels即 $\widehat{C}_i$。
- target_memory为memory限制
- target_flops为FLOPs限制
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509151622471.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
`Bigger networks with larger width, depth, or resolution tend to achieve higher accuracy, but the accuracy gain quickly saturate after reaching 80%, demonstrating the limitation of single dimension scaling`
具有较大宽度，深度或分辨率的较大网络往往会实现较高的精度，但是精度增益在达到80％后会迅速饱和，**这表明了单维缩放的局限性**。
## compound scaling method
`In this paper, we propose a new compound scaling method, which use a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way:`
在本文中，我们提出了一种新的复合缩放方法，该方法使用一个统一的复合系数$\phi$对网络的宽度，深度和分辨率进行均匀缩放。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509152344159.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
其中$\alpha,\beta,\gamma$是通过一个小格子搜索的方法决定的常量。通常来说，$\phi$是一个用户指定的系数来控制有多少的额外资源能够用于模型的缩放，$\alpha,\beta,\gamma$指明了怎么支配这些额外的资源分别到网络的宽度，深度，和分辨率上。尤其是，一个标准卷积操作的运算量的比例是$d,w^2,r^2$双倍的网络深度将带来双倍的运算量，但是双倍的网络宽度或分辨率将会增加运算为4倍。因为卷积操作通常在ConvNets中占据绝大部分计算量，通过3式来缩放ConvNet大约将增加$(\alpha,\beta^2,\gamma^2)$运算量。
**我们限制$\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$所以对于任意给定的$\phi$值，我们总共的运算量将大约增加$2^{\phi}$**

## 网络结构
文中给出了Efficientnet-B0的基准框架，在Efficientnet当中，我们所使用的激活函数是swish激活函数，整个网络分为了9个stage，在第2到第8stage是一直在堆叠MBConv结构，**表格当中MBConv结构后面的数字（1或6）表示的就是MBConv中第一个1x1的卷积层会将输入特征矩阵的channels扩充为n倍。Channels表示通过该Stage后输出特征矩阵的Channels。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509153603684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## MBConv结构
MBConv的结构相对来说还是十分好理解的，这里是我自己画的一张结构图。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021050916551423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
MBConv结构主要由一个1x1的`expand conv`（升维作用，包含BN和Swish激活函数）提升维度的倍率正是之前网络结构中看到的1和6，一个KxK的**卷积深度可分离**卷积`Depthwise Conv`（包含BN和Swish）k的具体值可看EfficientNet-B0的网络框架主要有3x3和5x5两种情况，一个SE模块，一个1x1的`pointer conv`降维作用，包含BN），一个`Droupout`层构成，一般情况下都添加上了残差边模块`Shortcut`，从而得到输出特征矩阵。
首先关于深度可分离卷积的概念是是我们需要了解的，相当于将卷积的过程拆成运算和合并两个过程，分别对应depthwise和point两个过程，这里给一个参考连接给大家就不在详细描述了[Depthwise卷积与Pointwise卷积](https://blog.csdn.net/tintinetmilou/article/details/81607721)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509170104747.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
**1.由于该模块最初是用tensorflow来实现的与pytorch有一些小差别，所以在使用pytorch实现的时候我们最好将tensorflow当中的samepadding操作重新实现一下**
**2.在batchnormal时候的动量设置也是有一些不一样的，论文当中设置的batch_norm_momentum=0.99,在pytorch当中需要先执行1-momentum再设为参数与tensorflow不同。**
```python
# Batch norm parameters
momentum= 1 - batch_norm_momentum  # pytorch's difference from tensorflow
eps= batch_norm_epsilon
nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=eps)
```

```python
class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size,padding=0, stride=1, dilation=1,
                 groups=1, bias=True, padding_mode = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:] # input
        kh, kw = self.weight.size()[-2:] # because kernel size equals to weight size
        sh, sw = self.stride # stride

        # change the output size according to stride
        oh, ow = math.ceil(ih/sh), math.ceil(iw/sw) # output
        """
            kernel effective receptive field size: (kernel_size-1) x dilation + 1
        """
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

```

## MBConvBlock模块结构

```python
class ConvBNActivation(nn.Module):
    """Convolution BN Activation"""
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,groups=1,
                 bias=False,momentum=0.01,eps=1e-3,active=False):
        super().__init__()
        self.layer=nn.Sequential(
            Conv2dSamePadding(in_channels,out_channels,kernel_size=kernel_size,
                              stride=stride,groups=groups, bias=bias),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=eps)
        )
        self.active=active
        self.swish = Swish()

    def forward(self, x):
        x = self.layer(x)
        if self.active:
            x=self.swish(x)
        return x




class MBConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,
                 expand_ratio=1, momentum=0.01,eps=1e-3,
                 drop_connect_ratio=0.2,training=True,
                 se_ratio=0.25,skip=True, image_size=None):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.skip = skip and stride == 1 and in_channels == out_channels

        # 1x1 convolution channels expansion phase
        expand_channels = in_channels * self.expand_ratio  # number of output channels

        if self.expand_ratio != 1:
            self.expand_conv = ConvBNActivation(in_channels=in_channels,out_channels=expand_channels,
                                                momentum=momentum,eps=eps,
                                                kernel_size=1,bias=False,active=False)

        # Depthwise convolution phase
        self.depthwise_conv = ConvBNActivation(in_channels=expand_channels,out_channels=expand_channels,
                                               momentum=momentum,eps=eps,
                                               kernel_size=kernel_size,stride=stride,groups=expand_channels,bias=False,active=False)

        # Squeeze and Excitation module
        if self.has_se:
            sequeeze_channels = max(1,int(expand_channels*self.se_ratio))
            self.se = SEModule(expand_channels,sequeeze_channels)

        # Pointwise convolution phase
        self.project_conv = ConvBNActivation(expand_channels,out_channels,momentum=momentum,eps=eps,
                                             kernel_size=1,bias=False,active=True)

        self.drop_connect = DropConnect(drop_connect_ratio,training)

    def forward(self,x):
        input_x = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.has_se:
            x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.drop_connect(x)
            x = x + input_x
        return x

```

## SE模块
**SE模块是通道注意力模块，该模块能够关注channel之间的关系，可以让模型自主学习到不同channel特征的重要程度**。由一个全局平均池化，两个全连接层组成。第一个全连接层的节点个数是输入该MBConv特征矩阵channels的1/4 ，且使用Swish激活函数。第二个全连接层的节点个数等于Depthwise Conv层输出的特征矩阵channels，且使用Sigmoid激活函数，这里引用一下别人画好的示意图。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509170601162.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021050917075028.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

```python
class SEModule(nn.Module):
    """Squeeze and Excitation module for channel attention"""
    def __init__(self,in_channels,squeezed_channels):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels,squeezed_channels,1),  # use 1x1 convolution instead of linear
            Swish(),
            nn.Conv2d(squeezed_channels,in_channels,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x= self.global_avg_pool(x)
        y = self.sequential(x)
        return x * y
```

## swish激活函数
Swish是Google提出的一种新型激活函数,其原始公式为:f(x)=x * sigmod(x),变形Swish-B激活函数的公式则为f(x)=x * sigmod(b * x),其拥有不饱和,光滑,非单调性的特征,Google在论文中的多项测试表明Swish以及Swish-B激活函数的性能即佳,在不同的数据集上都表现出了要优于当前最佳激活函数的性能。
$$f(x)=x \cdot \operatorname{sigmoid}(\beta x)$$
其中$\beta$是个常数或可训练的参数，Swish 具备无上界有下界、平滑、非单调的特性。

```python
class Swish(nn.Module):
    """ swish activation function"""
    def forward(self,x):
        return x * torch.sigmoid(x)
```
## droupout代码实现

```python
class DropConnect(nn.Module):
    """Drop Connection"""
    def __init__(self,ratio,training):
        super().__init__()
        assert 0 <= ratio <= 1
        self.keep_prob = 1 - ratio
        self.training=training

    def forward(self,x):
        if not self.training:
            return x
        batch_size = x.shape[0]

        random_tensor = self.keep_prob
        random_tensor += torch.rand([batch_size,1,1,1],dtype=x.dtype,device=x.device)
        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        binary_tensor = torch.floor(random_tensor)

        output = x / self.keep_prob * binary_tensor
        return output
```

## 总结
这里我们只给出了b0结构的代码实现，对于其他结构的实现过程就是在b0的基础上对wdith，depth，resolution都通过倍率因子统一缩放，这个部分在这个博客里面有详细的介绍[EfficientNet(B0-B7)参数设置](https://blog.csdn.net/qq_37541097/article/details/114434046)。
在本文中，我们系统地研究了ConvNet的缩放比例，分析了各因素对网络结构的影响，并确定仔细平衡网络的宽度，深度和分辨率，这是目前网络训练最重要但缺乏研究的部分，也是我们无法获得更好的准确性和效率忽略的一个部分。为解决此问题，本文作者提出了一种简单而高效的复合缩放方法，通过一个倍率因子统一缩放各部分比例并添加上通道注意力机制和残差模块，从而改善网络架构得到SOTA的效果，同时对模型各成分因子进行量化十分具有创新性。
> 参考连接
> [EfficientNet网络详解](https://blog.csdn.net/qq_37541097/article/details/114434046)
> [神经网络学习小记录50——Pytorch EfficientNet模型的复现详解](https://blog.csdn.net/weixin_44791964/article/details/106733795)
> [论文翻译](https://blog.csdn.net/weixin_42464187/article/details/100939130)
> [令人拍案叫绝的EfficientNet和EfficientDet](https://zhuanlan.zhihu.com/p/96773680)
> [swish激活函数](https://www.cnblogs.com/makefile/p/activation-function.html)
> [Depthwise卷积与Pointwise卷积](https://blog.csdn.net/tintinetmilou/article/details/81607721)
> [CV领域常用的注意力机制模块（SE、CBAM）](https://blog.csdn.net/weixin_42907473/article/details/106525668)
> [Global Average Pooling](https://blog.csdn.net/fjsd155/article/details/88953153)
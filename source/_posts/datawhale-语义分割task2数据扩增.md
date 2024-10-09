title: datawhale-语义分割task2数据扩增
tags:
  - datawhale
  - 计算机视觉
categories: []
author: whut ykh
date: 2021-02-22 19:04:00
---
# 语义分割-Task2 数据扩增

本章对语义分割任务中常见的数据扩增方法进行介绍，并使用OpenCV和albumentations两个库完成具体的数据扩增操作。
干货链接：
[albumentations 数据增强工具的使用](https://zhuanlan.zhihu.com/p/107399127/)
[Pytorch：transforms的二十二个方法](https://blog.csdn.net/weixin_38533896/article/details/86028509)
[Pytorch使用albumentations实现数据增强](https://blog.csdn.net/zhangyuexiang123/article/details/107705311)
<!--more-->

## 2 数据扩增方法
**简单来说数据扩充主要分成两类，一类是基于图像处理的数据扩增，一类是基于深度学习方法的数据扩充**  这里简单介绍一下
**基于图像处理的数据扩增—几何变换**
旋转，缩放，翻转，裁剪，平移，仿射变换
作用:几何变换可以有效地对抗数据中存在的位置偏差、视角偏差、尺寸偏差，而且易于实现，非常常用。

**基于图像处理的数据扩增—灰度和彩色空间变换**
·亮度调整，对比度饱和度调整，颜色空间转换，色彩调整，gamma变换
作用：对抗数据中存在的光照，色彩，亮度，对比度偏差

**基于图像处理的数据扩增——添加噪声和滤波**
添加高斯噪声，椒盐噪声
滤波：模糊，锐化，雾化等
作用：应对噪声干扰，恶劣环境，成像异常等特殊情况，帮助CNN学习更泛化的特征
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222185335244.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222185326950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222184733887.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

本章主要内容为数据扩增方法、OpenCV数据扩增、albumentations数据扩增和Pytorch读取赛题数据四个部分组成。

### 2.1 学习目标
- 理解基础的数据扩增方法
- 学习OpenCV和albumentations完成数据扩增
- Pytorch完成赛题读取


### 2.2 常见的数据扩增方法

![v](https://img-blog.csdnimg.cn/20210222184303792.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/202102221844428.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)

数据扩增是一种有效的正则化方法，可以防止模型过拟合，在深度学习模型的训练过程中应用广泛。数据扩增的目的是增加数据集中样本的数据量，同时也可以有效增加样本的语义空间。

需注意：

1. 不同的数据，拥有不同的数据扩增方法；

2. 数据扩增方法需要考虑合理性，不要随意使用；

3. 数据扩增方法需要与具体任何相结合，同时要考虑到标签的变化；

对于图像分类，数据扩增方法可以分为两类：

1. 标签不变的数据扩增方法：数据变换之后图像类别不变；
2. 标签变化的数据扩增方法：数据变换之后图像类别变化；

而对于语义分割而言，常规的数据扩增方法都会改变图像的标签。如水平翻转、垂直翻转、旋转90%、旋转和随机裁剪，这些常见的数据扩增方法都会改变图像的标签，即会导致地标建筑物的像素发生改变。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222183636488.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)


### 2.3 OpenCV数据扩增

OpenCV是计算机视觉必备的库，可以很方便的完成数据读取、图像变化、边缘检测和模式识别等任务。为了加深各位对数据可做的影响，这里首先介绍OpenCV完成数据扩增的操作。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222183659950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
```python
# 首先读取原始图片
img = cv2.imread(train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(mask)
```
```python
# 垂直翻转
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.flip(img, 0))

plt.subplot(1, 2, 2)
plt.imshow(cv2.flip(mask, 0))
```

```python
# 水平翻转
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.flip(img, 0))

plt.subplot(1, 2, 2)
plt.imshow(cv2.flip(mask, 0))
```

```python
# 随机裁剪
x, y = np.random.randint(0, 256), np.random.randint(0, 256)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(img[x:x+256, y:y+256])

plt.subplot(1, 2, 2)
plt.imshow(mask[x:x+256, y:y+256])
```
### 2.4 albumentations数据扩增

albumentations是基于OpenCV的快速训练数据增强库，拥有非常简单且强大的可以用于多种任务（分割、检测）的接口，易于定制且添加其他框架非常方便。

albumentations也是计算机视觉数据竞赛中最常用的库：

- GitHub： [https://github.com/albumentations-team/albumentations](https://link.zhihu.com/?target=https%3A//github.com/albumentations-team/albumentations)
- 示例：[https://github.com/albumentations-team/albumentations_examples](https://link.zhihu.com/?target=https%3A//github.com/albumentations-team/albumentations_examples)

与OpenCV相比albumentations具有以下优点：

- albumentations支持的操作更多，使用更加方便；
- albumentations可以与深度学习框架（Keras或Pytorch）配合使用；
- albumentations支持各种任务（图像分流）的数据扩增操作

albumentations它可以对数据集进行逐像素的转换，如模糊、下采样、高斯造点、高斯模糊、动态模糊、RGB转换、随机雾化等；也可以进行空间转换（同时也会对目标进行转换），如裁剪、翻转、随机裁剪等。

```python
import albumentations as A

# 水平翻转
augments = A.HorizontalFlip(p=1)(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']

# 随机裁剪
augments = A.RandomCrop(p=1, height=256, width=256)(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']

# 旋转
augments = A.ShiftScaleRotate(p=1)(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']
```

albumentations还可以组合多个数据扩增操作得到更加复杂的数据扩增操作：

```python
trfm = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

augments = trfm(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(augments['image'])

plt.subplot(1, 2, 2)
plt.imshow(augments['mask'])aug
```

这里是我之前打kaggle的时候用到albumentations的一组增强方法，可以参考参考

```python
def get_train_transforms():
    return Compose([
            #将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            # 转置
            Transpose(p=0.5),
            HorizontalFlip(p=0.5), #img翻转 
            VerticalFlip(p=0.5),# 依据概率p对PIL图片进行垂直翻转
            ShiftScaleRotate(p=0.5),# 随机放射变换（ShiftScaleRotate），该方法可以对图片进行平移（translate）、缩放（scale）和旋转（roatate）
            # 随机改变图片的 HUE、饱和度和值
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            #随机亮度对比度
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            #将像素值除以255 = 2 ** 8 - 1，减去每个通道的平均值并除以每个通道的std
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # 在图像上生成矩形区域。
            CoarseDropout(p=0.5),
            # 在图像中生成正方形区域。
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
```

### 2.7 课后作业-添加噪声

```python
import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
```

```python
new_img = sp_noise(img,0.1)
plt.subplot(1,2,1)
plt.imshow(new_img)
new_img = gasuss_noise(img)
plt.subplot(1,2,2)
plt.imshow(new_img)
plt.savefig('1.png')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222184217813.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
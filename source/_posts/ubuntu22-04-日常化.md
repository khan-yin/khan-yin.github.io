---
title: ubuntu22.04 日常化
date: 2024-10-09 18:17:37
tags: linux
---

最近有在折腾一些Ubuntu日常化的东西，给电脑重新装了双系统，相比前几年，ubuntu22.04之后系统确实变得非常好用了，软件生态支持也变多了，为了更加日常化的使用折腾了一些相关软件的安装，这边做一个简单的记录。
<!--more-->

## ubuntu安装卸载
ubuntu的安装卸载以及分区教程还是有些麻烦的，我也是参考了b战上面一个不错的up主的教学，期间有差点把自己windows引导搞出问题，然后用驱动精灵修复了。注意计算机专业同学还是尽量用英语安装，且安装以后禁用掉软件和内核更新，在[一生一芯](https://ysyx.oscc.cc/docs/ics-pa/0.1.html#installing-ubuntu)中有提到，禁用方法很简单自行百度即可。
参考链接：[Windows 和 Ubuntu 双系统的安装和卸载](https://www.bilibili.com/video/BV1554y1n7zv/?share_source=copy_web&vd_source=5a36bf224e0b99f26a95b15bd816244b)


## Easyconnect
目前华科的VPN是支持ubuntu系统的，这确实提供了学生在连接校园网和服务器上的便利。当然安装可能会打不开的小问题。
参考链接：[解决方案，需下载的软件内有可用的百度云网址](https://yunwei365.blog.csdn.net/article/details/114263954?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-3-114263954-blog-108071231.t0_edu_mix&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-3-114263954-blog-108071231.t0_edu_mix&utm_relevant_index=4)

## VPN-Clash
找了一个不错的ubuntu支持的带有客户端的clash，亲测可用，导入任意可用的订阅链接就行。
参考链接：[Devpn](https://devpn.github.io)

## QQ
QQ 9可以直接在ubuntu22.04上直接使用。
参考链接：[QQ 9官网](https://im.qq.com/linuxqq/index.shtml)

## QQ 音乐
官方目前已经有可支持的版本，再也不用担心冲了会员用不了啦！偶尔会有闪退问题，但是不影响。
参考链接：[QQ音乐官网](https://y.qq.com/download/download.html)

## WPS
WPS是在linux系统下可支持office三件套的工具，有一些中文汉化的问题，解决方案跟下面一样，**不过实际下载之后发现，在`/opt/kingsoft/wps-office/office6/mui`下面本来就有语言包，只需要像下面的链接一样，移除下载后的其他语言安装包，只保留`default  zh_CN`就能直接汉化**，虽然在选择字体上好像还有点问题没解决，但是算是能用了至少。
参考链接：[参考链接](https://blog.csdn.net/qq_34358193/article/details/132079878)

## 中文输入法
在b战找到了一个简单不错的中文输入法——rime输入法，感觉非常好用，安装后如果没有效果记得重启下电脑。
参考链接：[rime输入法安装](https://www.bilibili.com/video/BV1Ks421A7R8/?share_source=copy_web&vd_source=5a36bf224e0b99f26a95b15bd816244b)

## 微信
微信的安装好像有很多版本了，官方也有在做统一的OS支持，这里我用的可能不一定是最好的版本，但是可以支持文字和图片截图，还能打开小程序和公众号，只是看不了朋友圈，安装过程没有踩坑。
参考链接：[2024如何在Ubuntu上安装原生微信wechat weixin](https://zhuanlan.zhihu.com/p/690854988)

## 浏览器
Edge & Chrome 目前都支持linux的相关系统了，不用担心使用问题，编程相关的软件就更方便了。

## End
一生一芯可以启动了！

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It&#39;s time to use ubuntu now😍 <a href="https://t.co/BSuh7xTN0M">pic.twitter.com/BSuh7xTN0M</a></p>&mdash; kehan yin (@jack_kehan) <a href="https://twitter.com/jack_kehan/status/1834059136491024780?ref_src=twsrc%5Etfw">September 12, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
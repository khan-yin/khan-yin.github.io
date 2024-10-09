title: Hexo博客搭建
author: whut ykh
tags:
  - Hexo
categories: []
date: 2020-07-10 23:01:00
---
# Hexo博客搭建
鸽了半年的hexo博客搭建，阿里云都快过了半年了，把自己的一些踩坑和修改写一下吧，首先放一下我hexo博客搭建的时候的一些参考吧，这几个链接应该按道理没有很多踩坑的地方，我使用的主题是yilia主题所以说下面几个链接主要是关于`yilia`的，不过其实hexo博客框架的那个js和css名字都一样，配置文件的语法和格式也是一致的，所以说还是可以提供到一些帮助的，而且我自己在搭建一些操作的时候也学习了其他主题的一些方法。
从0开始搭建一个hexo博客无坑教程，是真的无坑版本，一切顺利（这里需要感谢一位b站的良心up主codesheep，他其他的视频也不错哦）：
[手把手教你从0开始搭建自己的个人博客 |无坑版视频教程| hexo](https://www.bilibili.com/video/BV1Yb411a7ty?from=search&seid=15936694823282570499)
美化相关链接：
- [参考链接1](https://zhousiwei.gitee.io/)
- [参考链接2](https://blog.zscself.com/posts/70677220/)
- [参考链接3](https://shu-ren-yu.github.io/2020/02/08/hexo-%E5%BB%BA%E7%AB%99%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/)
- [参考链接4](https://ccs.zone/post/65edc760.html)
 
好了说完了这些现在开始讲我搭建我的博客的过程<!--more-->
## 安装git，nodejs
安装git和nodejs这个我就不过多介绍了，大家去csdn找找最新的相关安装教程就行了，在codesheep的视频里面也有教大家怎么做，记得切换一下node源到淘宝镜像cnpm，有的时候npm下载特别慢或者是报错。
## 官网下载hexo博客框架
[hexo官网](https://hexo.io/)，如果在hexo的使用过程有什么问题也可以去看hexo的官方文档
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200710212314229.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
在git或者cmd下面都可以使用该命令完成hexo博客的安装
```shell
npm install -g hexo-cli#建议用这个全局安装
```
高级用户可能更喜欢安装和使用hexo软件包。
```shell
npm install hexo
```
## 初始化hexo博客
安装Hexo后，输入以下命令以创建一个新文件夹初始化Hexo，注意这个文件夹就是我们的以后存放博客等各种主题等配置文件的文件夹了，如果配置过程博客出了问题，重新建一个新文件夹就行了，问题不大，注意信息备份。
```bash
hexo init <folder>
cd <folder>
npm install
```
npm instal以后目录中会出现这些文件

```bash
├── _config.yml  最重要的配置文件
├── package.json
├── scaffolds
├── source 这里会存放你的博客文件
|   ├── _drafts
|   └── _posts
└── themes 这里存放各种主题
```

这样完了以后其实我们的博客就已经搭建完成了，你可以在你初始化后的博客目录下输入一下命令，然后就会开启hexo服务，在浏览器输入http://localhost:4000  
就可以看到你生成的博客了，默认情况下hexo博客主题是`landspace`
```bash
hexo s
```
![开启hexo](https://img-blog.csdnimg.cn/20200710214046575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![默认主题](https://img-blog.csdnimg.cn/20200710214711645.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

## github部署
对于github上进行部署的话我们首先需要新建一个repo，**并且必须命名为：你的github账户名.github.io**，建议大家最好配置好ssh这样就不用每次输入密码了
然后我们需要在博客根目录下npm一个用于部署的模块`deploy-git` 

```bash
npm install hexo-deployer-git --save
```
github对于hexo博客的部署是非常方便的，只需要在主配置文件`_config.yml`下添加一项配置就可以了，配置到服务器上也是如此
这里我直接将github和服务器都配置好了提交，你如果没有服务器端的配置就直接删掉那一行就行。

```bash
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repo: 
        aliyun: git@你的域名:/path/blog.git,#这个我好像已经忘了如何设置了大家可以自行百度hexo部署到阿里云
        github: git@github.com:xxxx/xxxx.github.io.git
  branch: master #一般用master分支就够了
```
配置好以后我们使用

```bash
#建议每次提交都执行这三个命令，
#当然了你也可以在直接写成bat文件或者sh文件，就不用每次自己手动输入了
hexo clean
hexo g
hexo d
```
他就会自动将我们所写的博客和所需要展示的内容转成htmlcssjs等文件push到我们的github上去,所以说大家如果有配置和修改自己选择的主题的话，可能你需要备份一下，不然到时候就没了，在部署的网页上这些npmmodules和你修改的样式都是直接打包好了的，你的源代码是不会提交上去的。
## 常用操作
好了其实到这里我们的博客搭建就已经完成了，这里我们介绍几个常用的命令和操作吧，
```bash
hexo clean 清除生成的文件
hexo g 重新生成博客的文件
hexo s 开启hexo服务，我们可以预览自己新写的博客在网页上的样子
hexo d 提交部署到服务器上包括github

hexo new "my first blog" 新建一个名为my first blog的博客，会显示在你的博客根目录的`source/_posts`下面
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200710221527596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
大家可能会注意到我这里好像不仅生成了文件还生成了一个同名的文件夹，这个文件夹是用于引入同名博客的图片的。
当然了其实我建议最好在csdn上写好自己的博客发表以后就变成网图了，然后直接复制csdn的博文在commit到hexo上面这样就不用存图片啥的，不过这里还是介绍一下本地图片引入吧

 1. 把主页配置文件`_config.yml` 里的`post_asset_folder:`这个选项设置为true
 2. 在你的hexo目录下执行这样一句话`npm install hexo-asset-image --save`，这是下载安装一个可以上传本地图片的插件
 3. 然后你在执行`hexo new "xxx"`的时候会同时生成这个文件夹了，里面存入图片就行，然后使用markdown的语法引入图片就行。

## 代码高亮设置
代码高亮设置就在你所选的主题目录下有一个`layout/_partial/head.ejs`里面引入一下对应的css和highlight.js就行，我选择的格式是`atom-one-dark.css`，去highlight.js官网下载一个highlightjs的压缩包里面有很多中代码风格的css。
最好的方法就是把这个文件夹放到服务器上用链接形式引入，别的方法我没有尝试成功，毕竟我前端没学很多哈哈哈
![引入](https://img-blog.csdnimg.cn/20200710225154715.png)
![这个部分丢入服务器](https://img-blog.csdnimg.cn/20200710225107993.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200710225024894.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
## 看板娘
这个看板娘是来自一位github上的大佬，但我好像忘了叫什么了，在以前的star记录里也没找到，以后找到了再补上吧，就引入几个链接就行。
对了如果不想让博客内容全部展示的话还需要在markdown的任意位置加一句`<!--more-->`这样就会隐去后面的内容让别人点击展开详情查看。
这里放一张我自己的博客图片吧哈哈哈
[ykhblog 一路向北](https://khany.top/)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200710223950744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

## 博客迁移
为了更便于hexo原始博客资源和主题自定义在不同机器之间的迁移，我们可以借助Github平台托管创建新的分支来存储这些原始文件，其实主要就是将`source`, `themes`, `scaffolds`, `_config.yml`, `package.json`等文件提交即可，Hexo框架自带生成的`.gitignore`已经帮我们整理了不需要提交的文件。可以参考我的代码仓库分支设置[khan-yin.github.io/hexo-sources-env](https://github.com/khan-yin/khan-yin.github.io/tree/hexo-sources-env)。
参考链接：[hexo博客同步管理及迁移](https://www.jianshu.com/p/fceaf373d797)

title: docker镜像操作
tags:
  - docker
categories: []
author: whut ykh
date: 2021-02-25 00:03:00
---
## commit镜像
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224162521454.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 数据卷操作实战：mysql同步
**mysql运行容器，需要做数据挂载，安装启动mysql是需要配置密码的这一点要注意，所以要去docker hub官方文档上面去看官方配置**
```powershell
docker pull mysql:5.7
```

docker运行，docker run的常用参数这里我们再次回顾一下

```powershell
-d 后台运行
-p 端口映射
-v 卷挂载
-e 环境配置
--name 环境名字
```
<!--more-->

通过docker hub我们找到了官方的命令：`docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -d mysql:tag`，在修改一下得到我们最终的输入命令

```powershell
docker run -d -p 3310:3306 -v /home/mysql/conf:/etc/mysql/conf.d -v /home/mysql/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=12345678 --name mysql01 mysql:5.7
```

```sql
CREATE DATABASE test_db;
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224233439169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224171410273.png)
删除这个镜像后数据则依旧保存下来了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224171612792.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 具名/匿名挂载
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224172157943.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224172227379.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224172304119.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
大多数情况下，为了方便，我们会使用具名挂载
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224172628376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## Dockerfile
Dockerfile就是用来构建docker镜像的文件，命令脚本，通过这个脚本可以生成镜像。
构建步骤

 1. 编写一个Dockerfile
 2. docker build 构建成为一个镜像
 3. docker run 运行镜像
 4. docker push 发布镜像(DockerHub，阿里云镜像)
 
 这里我们可以先看看Docker Hub官方是怎么做的
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224203344446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224203405252.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
官方镜像是比较基础的，有很多命令和功能都省去了，所以我们通常需要在基础的镜像上来构建我们自己的镜像
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224203837702.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

## Dockerfile命令
|常用命令|用法  |
|--|--|
| FROM | 基础镜像，一切从这开始构建  |
| MAINTAINER |镜像是谁写的，姓名+邮箱  |
| RUN | 镜像构建的时候需要运行的命令  |
| ADD | 添加内容，如tomcat压缩包  |
| WORKDIR | 镜像的工作目录  |
| VOLUME | 挂载的目录 |
| CMD |  指定这个容器启动时要运行的命令，只有最后一个会生效，可被替代  |
| ENTRYPOINT | 指定这个容器启动时要运行的命令，可以追加命令  |
| ONBUILD | 当构建一个被继承DockerFile这个时候就会执行ONBUILD命令  |
| COPY| 类似ADD将文件拷贝到镜像当中  |
| ENV| 构建的时候设置环境变量  |

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224204049431.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

## 创建一个自己的centos
Dockerfile中99%的镜像都来自于这个scratch镜像，然后配置需要的软件和配置来进行构建。

```powershell
mkdir dockerfile
cd dockerfile
vim mydockerfile-centos
```
编写mydockerfile-centos

```powershell
FROM centos
MAINTAINER khan<khany@foxmail.com>
ENV MYPATH /usr/local
WORKDIR $MYPATH
RUN yum -y install vim
RUN yum -y install net-tools
EXPOSE 80
CMD echo $MYPATH
CMD echo "---end---"
CMD /bin/bash
```
然后我们进入`docker build`,**注意后面一定要有一个.号**

```powershell
docker build -f mydockerfile-centos  -t mycentos:1.0 .
```
然后我们通过`docker run -it mycentos:1.0
`命令进入我们自己创建的镜像测试运行我们新安装的包和命令是否能正常运行。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224220131565.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
我们可以通过 `docker history +容器名称/容器id`看到这个容器的构建过程。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224220552645.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## CMD和ENTRYPOINT区别
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224222435489.png)

dockerfile-cmd-test：
```powershell
FROM centos
CMD ["ls","-a"]
```
dockerfile-entrypoint-test：
```powershell
FROM centos
ENTRYPOINT ["ls","-a"]

```
执行命令` docker run 容器名称 -l`在CMD下会报错，命令会被后面追加的`-l`替代，而`-l`并不是有效的linux命令，所以报错，而ENTRYPOINT则是可以追加的则该命令会变为`ls -al`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224222324355.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 发布镜像

> 发布到Docker Hub

1. 地址 https://hub.docker.com/ 注册自己的账号
2. 确定这个账号可以登录
3. 在我们服务器上提交自己的镜像
4. 登录成功，通过`push`命令提交镜像,**记得注意添加版本号**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224223425131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
这里出了一点小问题：
**在build自己的镜像的时候添加tag时必须在前面加上自己的dockerhub的username，然后再push就可以了**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224230121332.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

```powershell
docker tag 镜像id YOUR_DOCKERHUB_NAME/firstimage
docker push YOUR_DOCKERHUB_NAME/firstimage
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224231803691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
提交成功，可以在docker hub上找到你提交的镜像
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224232108402.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

> 阿里云镜像提交

在阿里云的容器镜像服务里面，创建一个新的镜像仓库，然后就会有详细的教学，做法与docker hub基本一致，提交成功能在镜像版本当中查看到，这里就不再重复讲解了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/202102242325274.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## Docker镜像操作过程总结
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224232749397.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224232831852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

> 参考链接：[【狂神说Java】Docker最新超详细版教程通俗易懂](https://www.bilibili.com/video/BV1og4y1q7M4?t=201&p=33)
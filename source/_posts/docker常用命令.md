title: docker常用命令
tags:
  - docker
categories: []
author: whut ykh
date: 2021-02-24 15:30:00
---
# Docker 常用命令
## 帮助命令

```bash
docker version #显示docker版本信息
docker info #显示docker的系统信息，包括镜像和容器的数量
docker 命令 --help # 帮助命令
```

## 镜像命令
**1.`docker images`查看所有本地的主机镜像**

| docker images显示字段| 解释|
|:------------|:------------|
|  REPOSITORY |  镜像的仓库源|
|  TAG | 镜像的标签 |
|  IMAGE ID | 镜像的id |
|  CREATED  | 镜像的创建时间 |
|  SIZE| 镜像的大小 |

<!--more-->
```bash
(base) [root@iZuf69rye0flkbn4kbxrobZ ~]# docker images
REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
hello-world   latest    bf756fb1ae65   13 months ago   13.3kB
(base) [root@iZuf69rye0flkbn4kbxrobZ ~]# docker images --help

Usage:  docker images [OPTIONS] [REPOSITORY[:TAG]]

List images

Options:
  -a, --all             Show all images 
  -q, --quiet           Only show image IDs

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224130929333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
**2.`docker search`命令搜索镜像**
搜索镜像可以去docker hub网站上直接搜索，也可以通过命令行来搜索，通过万能帮助命令能更快的看到他的一些用法，这两种方法结果是一样的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224131759589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224131506578.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

我们也可以通过`--filter`来进行条件筛选
比如`docker search mysql --filter=STARS=3000`

```bash
(base) [root@iZuf69rye0flkbn4kbxrobZ ~]# docker search mysql --filter=STARS=3000
NAME      DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
mysql     MySQL is a widely used, open-source relation…   10538     [OK]       
mariadb   MariaDB is a community-developed fork of MyS…   3935      [OK]  
```

**3.`docker pull`下载镜像**
**这个命令其实信息量很大，这也是docker高明的地方，关于指定版本下载一定要是docker hub官网上面支持和提供的版本**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022413252511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
我这里使用了

```bash
docker pull mysql
docker pull mysql:5.7
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224132844566.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
**4.`docker rmi`删除镜像**
删除可以通过`REPOSITORY`来删，也可以通过`IMAGE ID`来删除
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224133232500.png)
## 容器命令
**说明：我们有了镜像才可以创建容器，linux，下载一个centos镜像来测试学习**
```bash
docker pull centos
```
**1.新建容器并启动**
通过`docker run`命令进入下载的centos容器里面后我们可以发现的是，我们的rootname不一样了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224134041815.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224134219210.png)

**2.列出所有运行的容器**
`docker ps`命令
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022413511697.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
**3.`exit`退出命令**

```powershell
exit #直接容器停止并退出
Ctrl + P + Q  #容器不停止并退出
```
在执行exit命令后，我们看到rootname又变回来了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224134531160.png)


**4.删除容器**

```powershell
docker rm 容器id #删除指定的容器，不能删除正在运行的容器，如果要强制删除，需要使用 rm -f
docker rm $(docker ps -aq) #删除全部的容器
docker ps -a -q|xargs docker rm #删除全部容器
```
**5.启动和停止容器**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224135856699.png)
## 日志元数据进程查看
![](https://img-blog.csdnimg.cn/20210224141011643.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224141603263.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
**1.`docker top 容器id`查看容器中的进程**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224141846922.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
**2.`docker inspect 容器id`查看元数据**


**3.进入当前正在运行的容器**

方式1： `docker exec -it 容器id bashshell`并可通过`ps -ef`查看容器当中的进程
![](https://img-blog.csdnimg.cn/20210224143749283.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)



方式2：`docker attach 容器id`进入容器，如果当前有正在执行的容器则会直接进入到当前正在执行的进程当中

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224142610940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 从容器内拷贝到主机上
即使容器已经停止也是可以进行拷贝的
```powershell
docker cp 容器id:容器内路径 目的主机路径
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224144038919.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

## docker部署nginx
```powershell
$ docker search nginx
$ docker pull nginx
$ docker run -d --name nginx01 -p 8083:80 nginx
$ docker ps
$ curl localhost:8083
```
`docker stop` 后则无法再访问
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224145948506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224150019256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224144959653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## portainer可视化管理

```powershell
docker run -d -p 8088:9000 --restart=always -v /var/run/docker.sock:/var/run/docker.sock --privileged=true portainer/portainer
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224152421776.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
进入后选择**local模式**，然后就能看到这个版面了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224152631399.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
参考链接：
[【狂神说Java】Docker最新超详细版教程通俗易懂](https://www.bilibili.com/video/BV1og4y1q7M4?p=9)
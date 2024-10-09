title: docker安装和简易原理
tags:
  - docker
categories: []
author: whut ykh
date: 2021-02-24 12:53:00
---
## docker安装和简易原理
最近参加了阿里云datawhale天池的一个比赛里面需要用docker进行提交，所以借此机会学习了一下docker，b站上有个很好的视频[【狂神说Java】Docker最新超详细版教程通俗易懂](https://www.bilibili.com/video/BV1og4y1q7M4?p=6)

## docker基本组成
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224124633537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

<!--more-->

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224124854806.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)


## docker安装
centos7安装
先查看centos版本，新版本的docker都只支持centos7以上

```bash
(base) [root@iZuf69rye0flkbn4kbxrobZ ~]# cat /etc/os-release
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"
```

然后我们的操作均按照帮助文档进行操作即可
[dockerCentos安装](https://docs.docker.com/engine/install/centos/)

```bash
# 卸载旧版本
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine

# 安装
sudo yum install -y yum-utils

#配置镜像，官方是国外的很慢，这里我们使用阿里云镜像
sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

# 推荐使用这个
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

#更新yum
yum makecache fast

# 安装相关的包
 sudo yum install docker-ce docker-ce-cli containerd.io

```

## Start Docker

```bash
sudo systemctl start docker
```
使用`docker version`查看是否安装成功
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022412185923.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
`sudo docker run hello-world`![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224122343176.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

使用`docker images` 查看所安装的所有镜像
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224122509960.png)

## Uninstall Docker Engine
1.Uninstall the Docker Engine, CLI, and Containerd packages:

```bash
sudo yum remove docker-ce docker-ce-cli containerd.io
```
2.删除默认工作目录和资源

```bash
$ sudo rm -rf /var/lib/docker 
```

## 阿里云镜像加速器
这一块的话推荐天池的一个[docker学习赛](https://tianchi.aliyun.com/competition/entrance/231759/tab/226)
登录阿里云找到容器服务，并创建新容器,然后找到里面的镜像加速器对centos进行配置。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224123245412.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

```bash
sudo mkdir -p /etc/docker

sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://fdm7vcvf.mirror.aliyuncs.com"]
}
EOF

sudo systemctl daemon-reload

sudo systemctl restart docker
```

## 回归Hello World镜像的运行过程
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224123453853.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 底层原理
Docker是什么工作的?
Docker是一个Client - Server结构的系统，Docker的守护进行运行在主机上。通过Socket从客户端访问!DockerServer接收到Docker-Client的指令，就会执行这个命令!
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224124010560.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
docker为什么比虚拟机快？
[参考链接](https://www.cnblogs.com/fanqisoft/p/10440220.html)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224124320560.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210224124350819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
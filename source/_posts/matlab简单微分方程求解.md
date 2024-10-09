title: matlab简单微分方程求解
author: whut ykh
tags:
  - 数学建模
categories: []
date: 2020-07-16 23:36:00
---
时隔半年，我又重回微分方程的学习了，现在学确实挺难搞的，很多知识和理论思路都忘了，数学还是很重要啊，其实一个蛮简单的东西我看了很久很久才慢慢的又懂了，话不多说，直接写文。
首先要明确的一点就是，我们求微分方程的时候，要注意有解析解和数值解，解析解又有通解和特解，这在我们编写代码的时候可以通过初始点的值来获得特解。其实今天老师讲的还挺不错的，举出了很多的例子，基本上与物理有关，**其实要说这一个模块最稳妥的办法其实是如果能够求出通解，一般最好手动进行微分方程的求解，然后用计算机检验，用计算机求微分方程的情况大多数是求数值解。所以说这一部分的话，其实对建模的要求更高，求解我们到时候可以看到matlab调用函数就可以了**
<!--more-->
## dsolve函数的妙用
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716225724889.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020071622590399.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
#### 求解常微分方程通解
$求微分方程\frac{du}{dx}=1+u^2和x^2+y+(x-2y)y'的通解$
```matlab
dsolve('Du=1+u^2','t');
dsolve('x^2+y+(x-2*y)*Dy=0','x');
```
#### 求解常微分方程特解
$求y'''-y''=x，y(1)=8，y'(1)=7，y''(2)=4的特解$
```matlab
y=dsolve('D3y-D2y=x','y(1)=8','Dy(1)=7','D2y(2)=4','x');
```
```js
```
#### 求解微分方程组
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716230752510.png#pic_center)
```matlab
[x,y,z]=dsolve('Dx=2*x-3*y+3*z','Dy=4*x-5*y+3*z','Dz=4*x-4*y+2*z','t');
x=simplify(x);%化简
y=simplify(y);
z=simplify(z);
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716230944302.png#pic_center)

```matlab
equ1='D2f+3*g=sin(x)';
equ2='Dg+Df=cos(x)';
[general_f,general_g]=dsolve(equ1,equ2,'x');  %通解
[f,g]=dsolve(equ1,equ2,'Df(2)=0,f(3)=3,g(5)=1','x');
```
## 龙哥库塔求数值解
在介绍龙哥库塔求数值解的时候，我们先要介绍一下这个刚性方程组的定义
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716231241550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
然后matlab里面提供了几个非常好的解刚性常微分方程的功能函数如`ode15s,ode45,ode113,ode23,ode23s,ode23t,ode23tb,`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716231441765.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
然后在介绍求解的知识之前，我需要先让你对这个过程做一个了解，以至于你写的时候不会觉得很晕，其实很简单的，估计我是忘了老师怎么教的了，看了半天我才慢慢明白。
其实他的思路就是，用当前可以得到的函数值和自变量来导出我们所需要的求得微分变量，如果出现了高阶得情况，则将每一个高阶都逐一替换成导数用函数值表示的公式，前几阶导数完全可以嵌套定义表示出来，可能你还是没用很懂，不过等你看到我下面的分析的时候，你应该就懂了，而且通过我的发现，基本上告诉你几个初始值，一般你就要求多少阶的导数。
#### 简单的一阶方程初值问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716232119387.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716232057499.png#pic_center)
F2.m
```matlab
function dy = F2(t,y)
%F2 此处显示有关此函数的摘要
%   此处显示详细说明
dy=-2*y+2*t^2+2*t;
end
```
main.m
```matlab
[T,Y]=ode45('F2',[0,500],[1]);
plot(T,Y(:,1),'-');
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716232314803.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
#### 高阶微分方程
做变量替换，转化成多个一阶微分方程
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716232648679.png#pic_center)
F.m

```matlab
%y1=y
%y2=y1'=y'
%y2'=y''
%y0=2,y1=0
function dy=F(t,y)%虽然这里我们没用上自变量但是这个参数还是要加，毕竟是调库函数所以说还是得按他的规矩
dy=zeros(2,1);
dy(1)=y(2);
dy(2)=1000*(1-y(1)^2)*y(2)-y(1);
```
main.m

```matlab
[T,Y]=ode45('F',[0,3000],[2 0]);
plot(T,Y(:,1),'-');
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020071623290151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716232930676.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200716232957721.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
**发现了吗，其实他就是不停的做变量代换，然后比如说这里第二题就是用y1,y2,y3表示我们需要迭代得微分y1',y2',y3'，为了便于计算和迭代，我们将初值写成了一个列向量，因此我们后面的参数就是一个列向量来表示的。其实这个事情一点也不难，但是自己确实还是数学忘了，还是要多复习多学习**

title: 模拟退火matlab实现（TSP为例）
author: whut ykh
tags:
  - 数学建模
categories: []
date: 2020-07-15 11:38:00
mathjax: true
---
最近学习了模拟退火智能算法，顺便学习了一下matlab，顺带一提matlba真香，对矩阵的操作会比python的numpy操作要更加方便，这里我们是以TSP问题为例子，因为比较好理解。
## 模拟退火介绍
模拟退火总的来说还是一种优化算法，他模拟的是淬火冶炼的一个过程，通过升温增强分子的热运动，然后再慢慢降温，使其达到稳定的状态。
<!--more-->
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715101056191.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
**初始解**  
通常是以一个随机解作为初始解. 并保证理论上能够生成解空间中任意的解，也可以是一个经挑选过的较好的解，初始解不宜“太好”, 否则很难从这个解的邻域跳出，针对问题去分析。  
**扰动邻解**  
邻解生成函数应尽可能保证产生的侯选解能够遍布解空间，邻域应尽可能的小，能够在少量循环步中允分探测.，但每次的改变不应该引起太大的变化。  
**初始温度**  
- 初始温度应该设置的尽可能的高, 以确保最终解不受初始解
影响. 但过高又会增加计算时间.
- 均匀抽样一组状态，以各状态目标值的方差为初温.
- 如果能确定邻解间目标函数(COST 函数) 的最大差值, 就可以确定出初始温度$T_{0}$,以使初始接受概率$P=e^{-|\Delta C|\_{max}/T}$足够大。$|\Delta C|\_{max}$ 可由随机产生一组状态的最大目标值差来替代.
- 在正式开始退火算法前, 可进行一个升温过程确定初始温度:逐渐增加温度, 直到所有的尝试尝试运动都被接受, 将此时的温度设置为初始温度.
- 由经验给出, 或通过尝试找到较好的初始温度.  
**等温步数**  
- 等温步数也称Metropolis 抽样稳定准则, 用于决定在各温度下产生候选解的数目. 通常取决于解空间和邻域的大小.
- 等温过程是为了让系统达到平衡, 因此可通过检验目标函数的均值是否稳定(或连续若干步的目标值变化较小) 来确定等温步数.
- 等温步数受温度的影响. 高温时, 等温步数可以较小, 温度较小时, 等温步数要大. 随着温度的降低, 增加等温步数.
- 有时为了考虑方便, 也可直接按一定的步数抽样。  
**Metropolis法则**  
Metropolis法则是SA接受新解的概率。  
$$P(x=>x')=
\begin{cases}
1 &\text{$,f(x')<f(x)$} \\\\
e^{-\frac{f(x')-f(x)}{T}} & \text{$,f(x')>f(x)$}\\
\end{cases}
$$
$x$是表示当前解，$x'$是新解，其实这也是模拟退火区别于贪心的一点，我们在更新新解的时候对于不满足条件的情况，我们也有一定的概率来进行选取，从而可以使得退火模拟可以跳出局部最优解。

**降温公式**  
经典模拟退火算法的降温方式：
$$T(t)=\frac{T_0}{log(1+t)}$$
快速模拟退火算法的降温方式:
$$T(t)=\frac{T_0}{1+t}$$
常用的模拟退火算法的降温方式还有(通常$0.8<\alpha<0.99$)
$$T(t+\Delta t)=\alpha T(t)$$
终止条件自己设定阈值即可。  
**花费函数COST**  
这个基本就是分析题目，一般设置为我们需要求解的目标函数最值。
## TSP问题
已知中国34 个省会城市(包括直辖市) 的经纬度, 要求从北京出发, 游遍34 个城市, 最后回到北京. 用模拟退火算法求最短路径。  
1.main.m  
主函数
```matlab
clc
clear;
%% 
load('china.mat');
plotcities(province, border, city);%画出地图
cityaccount=length(city);%城市数量
dis=distancematrix(city)%距离矩阵

route =randperm(cityaccount);%路径格式
temperature=1000;%初始化温度
cooling_rate=0.95;%温度下降比率
item=1;%用来控制降温的循环记录次数
distance=totaldistance(route,dis);%计算路径总长度
temperature_iterations = 1;
% This is a flag used to plot the current route after 200 iterations
plot_iterations = 1;

plotroute(city, route, distance,temperature);%画出路线

%% 
while temperature>1.0 %循环条件
    temp_route=change(route,'reverse');%产生扰动。分子序列变化
%     fprintf("%d\n",temp_route(1));
    temp_distance=totaldistance(temp_route,dis);%产生变化后的长度
    dist=temp_distance-distance;%两个路径的差距
    if(dist<0)||(rand < exp(-dist/(temperature)))
        route=temp_route;
        distance=temp_distance;
        item=item+1;
        temperature_iterations=temperature_iterations+1;
        plot_iterations=plot_iterations+1;
    end
    if temperature_iterations>=10
        temperature=cooling_rate*temperature;
        temperature_iterations=0;
    end
    
    if plot_iterations >= 20
       plotroute(city, route, distance,temperature);%画出路线
       plot_iterations = 0;
    end
%     fprintf("it=%d",item);
end
```
2.distance.m  
计算两点之间的距离
```matlab
function d = distance(lat1, long1, lat2, long2, R)
% DISTANCE
% d = DISTANCE(lat1, long1, lat2, long2, R) compute distance between points
% on sphere with radians R.
%
% Latitude/Longitude Distance Calculation:
% http://www.mathforum.com/library/drmath/view/51711.html
 
y1 = lat1/180*pi; x1 = long1/180*pi;
y2 = lat2/180*pi; x2 = long2/180*pi;
dy = y1-y2; dx = x1-x2;
d = 2*R*asin(sqrt(sin(dy/2)^2+sin(dx/2)^2*cos(y1)*cos(y2)));
end
```
3.totaldistance.m  
一条route路线的总路程，也是我们需要优化的目标函数
```matlab
function totaldis = totaldistance(route,dis)%传入距离矩阵和当前路线
%TOTALDISTANCE 此处显示有关此函数的摘要
%   此处显示详细说明

% fprintf("%d\n",route(1));
totaldis=dis(route(end),route(1));
% totaldis=dis(route(end),route(1));
for k=1:length(route)-1
    totaldis=totaldis+dis(route(k),route(k+1));%直接加两个点之间的距离
end

```
4.distancematrix.m  
任意两点之间的距离矩阵
```matlab
function dis = distancematrix(city)
%DISTANCEMATRIX 此处显示有关此函数的摘要
%   此处显示详细说明
cityacount=length(city);
R=6378.137;%地球半径用于求两个城市的球面距离
  for i = 1:cityacount
      for j = i+1:cityacount
          dis(i,j)=distance(city(i).lat,city(i).long,...
                              city(j).lat,city(j).long,R);%distance函数原来是设计来计算球面上距离的
          dis(j,i)=dis(i,j);%对称，无向图
      end
  end
end
```

5.change.m  
产生分子扰动，求当前解的邻解
```matlab
function route = change(pre_route,method)
%CHANGE 此处显示有关此函数的摘要
%   此处显示详细说明
route=pre_route;
cityaccount=length(route);
%随机取数，相当于把0-34个城市映射到0-1的等概率抽取上，再取整
city1=ceil(cityaccount*rand); % [1, 2, ..., n-1, n]
city2=ceil(cityaccount*rand); % 1<=city1, city2<=n
switch method
    case 'reverse' %[1 2 3 4 5 6] -> [1 5 4 3 2 6]
        cmin = min(city1,city2);
        cmax = max(city1,city2);
        route(cmin:cmax) = route(cmax:-1:cmin);%反转某一段
    case 'swap' %[1 2 3 4 5 6] -> [1 5 3 4 2 6]
        route([city1, city2]) = route([city2, city1]);
end
```
6.plotcities.m  
画出city
```matlab
function h = plotcities(province, border, city)
% PLOTCITIES
% h = PLOTCITIES(province, border, city) draw the map of China, and return 
% the route handle.

global h;
% draw the map of China
plot(province.long, province.lat, 'color', [0.7,0.7,0.7])
hold on
plot(border.long  , border.lat  , 'color', [0.5,0.5,0.5], 'linewidth', 1.5);
 

% plot a NaN route, and global the handle h.
h = plot(NaN, NaN, 'b-', 'linewidth', 1);

% plot cities as green dots
plot([city(2:end).long], [city(2:end).lat], 'o', 'markersize', 3, ...
                              'MarkerEdgeColor','b','MarkerFaceColor','g');
% plot Beijing as a red pentagram
plot([city(1).long],[city(1).lat],'p','markersize',5, ...
                              'MarkerEdgeColor','r','MarkerFaceColor','g');
axis([70 140 15 55]);
```
7.plotroute.m  
画出路线
```matlab
function plotroute(city, route, current_distance, temperature)
% PLOTROUTE
% PLOTROUTE(city, route, current_distance, temperature) plots the route and
% display current temperautre and distance.

global h;
cycle = route([1:end, 1]);
% update route
set(h,'Xdata',[city(cycle).long],'Ydata',[city(cycle).lat]);

% display current temperature and total distance
xlabel(sprintf('T = %6.1f        Total Distance = %6.1f', ...
                    temperature,                  current_distance));
drawnow

```
一开始的路径
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715110631713.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
最后运行出来的最优化路径
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715110536364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70#pic_center)
有需要的朋友可以下载相关的数据集[下载链接](https://khany.top/database/china.mat)
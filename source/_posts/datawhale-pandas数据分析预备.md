title: datawhale-pandas数据分析预备
tags:
  - pandas
  - datawhale
author: whut ykh
date: 2020-12-21 12:40:31
---
## datawhale-pandas数据分析预备
## 列表推导式


```python
def my_func(x):
    return 2*x
```

## [* for i in *] 
其中，第一个 * 为映射函数，其输入为后面 i 指代的内容，第二个 * 表示迭代的对象。
<!--more-->

```python
[my_func(i) for i in range(5)]
```
out:[0, 2, 4, 6, 8]

列表表达式支持多层嵌套


```python
[m+'_'+n for m in['a','b'] for n in['c','d']]
```
out:['a_c', 'a_d', 'b_c', 'b_d']



## 条件赋值

## value = a if condition else b ：


```python
value = 'cat' if 2>1 else 'dog'
```


```python
value
```
out:'cat'



下面举一个例子，截断列表中超过5的元素，即超过5的用5代替，小于5的保留原来的值：


```python
L=[1,2,3,4,5,6,7]

[i if i<=5 else 5 for i in L]
```
out:[1, 2, 3, 4, 5, 5, 5]



## lambda


```python
my_func=lambda x:2*x
```


```python
my_func(2)
```
out:4




```python
f2=lambda a,b:a+b
```


```python
f2(1,2)
```
out:3




```python
[ (lambda i:2*i)(x) for x in range(5)]
```
out:[0, 2, 4, 6, 8]



对于上述的这种列表推导式的匿名函数映射， Python 中提供了 map 函数来完成，它返回的是一个 map 对象，需要通过 list 转为列表：


```python
list(map(lambda x: 2*x, range(5)))
```
[0, 2, 4, 6, 8]



对于多个输入值的函数映射，可以通过追加迭代对象实现：


```python
list(map(lambda x, y: str(x)+'_'+y, range(5), list('abcde')))
```
 ['0_a', '1_b', '2_c', '3_d', '4_e']



## zip

zip函数能够把多个可迭代对象打包成一个元组构成的可迭代对象，它返回了一个 zip 对象，通过 tuple, list 可以得到相应的打包结果：


```python
L1, L2, L3 = list('abc'), list('def'), list('hij')
```


```python
list(zip(L1, L2, L3))
```
[('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]




```python
tuple(zip(L1, L2, L3))
```
(('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j'))




```python
for i,j,k in zip(L1,L2,L3):
    print(i,j,k)
```
a d h
b e i
c f j
    

## enumerate 

`enumerate` 是一种特殊的打包，它可以在迭代时绑定迭代元素的遍历序号：


```python
L = list('abcd')
```


```python
for index, value in enumerate(L):
    print(index, value)
```
0 a
1 b
2 c
3 d
    
    

用zip实现这个功能


```python
[*zip(range(len(L)),L)]
```
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]




```python
for index,value in zip(range(len(L)),L):
    print(index,value)
```
0 a
1 b
2 c
3 d
    
    

当需要对两个列表建立字典映射时，可以利用 zip 对象：


```python
dict(zip(L1,L2))
```
{'a': 'd', 'b': 'e', 'c': 'f'}




```python
zipped = list(zip(L1, L2, L3))
```


```python
zipped
```
[('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]




```python
list(zip(*zipped))
```

[('a', 'b', 'c'), ('d', 'e', 'f'), ('h', 'i', 'j')]



## numpy回顾

## 1. np数组的构造 


```python
import numpy as np
```

最一般的方法是通过 array 来构造：


```python
np.array([1,2,3])
```
array([1, 2, 3])



等差序列： `np.linspace`, `np.arange`


```python
np.linspace(1,5,9) # 起始、终止（包含）、样本个数
```
array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])




```python
np.arange(1,5,2) # 起始、终止（不包含）、步长
```
array([1, 3])



特殊矩阵： `zeros`, `eye`, `full`


```python
np.zeros((2,3)) # 传入元组表示各维度大小
```
array([[0., 0., 0.],
      [0., 0., 0.]])




```python
np.eye(3) #代表维度3*3的单位矩阵
```
array([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])



    




```python
np.eye(3, k=1) # 偏移主对角线1个单位的伪单位矩阵
```
array([[0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 0.]])




```python
np.eye(3, k=-1) # 偏移主对角线1个单位的伪单位矩阵
```
array([[0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.]])



    




```python
np.eye(3, k=2) # 偏移主对角线1个单位的伪单位矩阵
```
array([[0., 0., 1.],
        [0., 0., 0.],
        [0., 0., 0.]])




   



```python
np.full((2,3),10)# 元组传入大小，10表示填充数值
```
array([[10, 10, 10],
        [10, 10, 10]])



    




```python
np.full((2,3), [1,2,3]) # 每行填入相同的列表
```
array([[1, 2, 3],
        [1, 2, 3]])



    



随机矩阵: `np.random`

|函数|含义|
|----|----|
|`np.random.rand`|0-1均匀分布的随机数组|
|`np.random.randn`|标准正态的随机数组|
|`np.random.randint`|随机整数组|
|`np.random.choice`|随机列表抽样|

## `np.random.rand`


```python
np.random.rand(3)# 生成服从0-1均匀分布的三个随机数
```

array([0.25659368, 0.37802498, 0.62494881])




```python
np.random.rand(3,3)# 生成服从0-1均匀分布的三个随机数
```
array([[0.64676496, 0.59502481, 0.61343668],
        [0.16019992, 0.49285208, 0.96761024],
        [0.94030055, 0.48943744, 0.1143115 ]])



    



对于服从区间 a 到 b 上的均匀分布可以如下生成：


```python
 a, b = 5, 15
```


```python
(b - a) * np.random.rand(3) + a
```

array([ 8.84499261, 10.21774591,  8.16028516])



**一般的，可以选择已有的库函数：**


```python
np.random.uniform(5, 15, 3)
```

array([8.86348101, 9.14266299, 8.60513876])



## `np.random.randn`

`randn` 生成了 $N(0,I)$ 的标准正态分布：


```python
np.random.randn(3)
```

array([ 1.41288442, -0.73967664, -0.23529916])




```python
np.random.randn(2, 2)
```
array([[ 0.85735525, -0.17674214],
        [-0.28607067,  1.49904315]])



    



对于服从$N(\mu, \sigma^2)$的一元正态分布则有


```python
mu,sigma=3,2.5
mu+np.random.randn(3)*sigma
```

array([-0.26367497,  1.87383756,  2.81701976])




```python
np.random.normal(3,2.5,3)
```

array([6.40062933, 3.28135583, 0.65048172])



`randint` 可以指定生成**随机整数**最大值（不包含）和维度大小：


```python
low, high, size = 5, 15, (2,2) # 生成5到14的随机整数
```


```python
np.random.randint(low, high, size)
```

array([[13, 14],
        [ 7, 10]])



`choice` 可以从给定的列表中，以一定概率和方式抽取结果，当不指定概率时为均匀采样，默认抽取方式为有放回抽样：


```python
my_list = ['a', 'b', 'c', 'd']
```


```python
np.random.choice(my_list, 2, replace=False, p=[0.1, 0.7, 0.1 ,0.1])
```

array(['a', 'b'], dtype='<U1')




```python
np.random.choice(my_list, (2,2), replace=True, p=[0.1, 0.7, 0.1 ,0.1])
```

array([['d', 'b'],
        ['b', 'b']], dtype='<U1')



![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-tyytLsj8-1608522933090)(attachment:image.png)\]](https://img-blog.csdnimg.cn/20201221120439843.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)



```python
np.random.choice(my_list, (3,3))
```

array([['d', 'c', 'c'],
        ['d', 'a', 'c'],
        ['d', 'b', 'b']], dtype='<U1')



当返回的元素个数与原列表相同时，不放回抽样等价于使用 `permutation` 函数，即打散原列表：


```python
np.random.permutation(my_list)
```


array(['d', 'b', 'c', 'a'], dtype='<U1')



最后，需要提到的是随机种子，它能够固定随机数的输出结果：


```python
np.random.seed(0)
```


```python
np.random.rand()
```

0.5488135039273248




```python
np.random.rand(0)
```


```python
np.random.rand()
```

0.5488135039273248




```python
np.random.rand()
```

0.6027633760716439



## 2. np数组的变形与合并

转置


```python
np.zeros((2,3)).T
```

array([[0., 0.],
        [0., 0.],
        [0., 0.]])


    



合并操作


```python
np.r_[np.zeros((2,3)),np.zeros((2,3))] #上下合并
```
array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])



    




```python
np.c_[np.zeros((2,3)),np.zeros((2,3))] #左右合并
```

array([[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])


    



一维数组和二维数组进行合并时，应当把其视作列向量，在长度匹配的情况下只能够使用左右合并的 c_ 操作：


```python
try:
    np.r_[np.array([0,0]),np.zeros((2,1))]
except Exception as e:
    Err_Msg=e
```


```python
Err_Msg
```


ValueError('all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)')



    


```python
np.r_[np.array([0,0]),np.zeros(2)]
```

array([0., 0., 0., 0.])




```python
np.c_[np.array([0,0]),np.zeros((2,3))]
```


array([[0., 0., 0., 0.],
         [0., 0., 0., 0.]])



    

维度变换: `reshape`


```python
target = np.arange(8).reshape(2,4)
```


```python
target
```
array([[0, 1, 2, 3],
           [4, 5, 6, 7]])




```python
target.reshape((4,2), order='C') # 按照行读取和填充
```




array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])




```python
target.reshape((4,2), order='F') # 按照列读取和填充
```




array([[0, 2],
           [4, 6],
           [1, 3],
           [5, 7]])



特别地，由于被调用数组的大小是确定的， reshape 允许有一个维度存在空缺，此时只需填充-1即可：


```python
target.reshape((4,-1))
```




array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])




```python
target = np.ones((3,1))
```


```python
target
```




array([[1.],
           [1.],
           [1.]])




```python
target.reshape(-1)
```




array([1., 1., 1.])



## 3. np数组的切片与索引

数组的切片模式支持使用 `slice` 类型的 `start:end:step` 切片，还可以直接传入列表指定某个维度的索引进行切片：


```python
target = np.arange(9).reshape(3,3)
```


```python
target
```




array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
target[:-1, [0,2]]
```




array([[0, 2],
           [3, 5]])




```python
target[0:1,:]
```




array([[0, 1, 2]])




```python
target[0:-1,[0]]
```




array([[0],
           [3]])



此外，还可以利用 `np.ix_` 在对应的维度上使用布尔索引，但此时不能使用 `slice` 切片：


```python
target[np.ix_([True, False, True], [True, False, True])]
```




array([[0, 2],
           [6, 8]])




```python
target[np.ix_([1,2], [True, False, True])]
```




array([[3, 5],
           [6, 8]])



当数组维度为1维时，可以直接进行布尔索引，而无需 `np.ix_ `：


```python
new = target.reshape(-1)
```


```python
new[new%2==0]
```




array([0, 2, 4, 6, 8])



## 常用函数

`where` 是一种条件函数，可以指定满足条件与不满足条件位置对应的填充值：


```python
a = np.array([-1,1,-1,0])
```


```python
np.where(a>0, a, 5) # 对应位置为True时填充a对应元素，否则填充5
```




array([5, 1, 5, 5])



`nonzero, argmax, argmin`

这三个函数返回的都是索引， `nonzero` 返回非零数的索引， `argmax, argmin` 分别返回最大和最小数的索引：


```python
a = np.array([-2,-5,0,1,3,-1])
```


```python
np.nonzero(a)
```




(array([0, 1, 3, 4, 5], dtype=int64),)




```python
a.argmax()
```




    4




```python
a.argmin()
```




1



any 指当序列**至少** 存在一个 True 或非零元素时返回 True ，否则返回 False

all 指当序列元素 **全为** True 或非零元素时返回 True ，否则返回 False


```python
a = np.array([0,1])
```


```python
a.any()
```




True




```python
a.all()
```




False



`cumprod, cumsum` 分别表示累乘和累加函数，返回同长度的数组， `diff` 表示和前一个元素做差，由于第一个元素为缺失值，因此在默认参数情况下，返回长度是原数组减1


```python
a = np.array([1,2,3])
```


```python
a.cumsum()
```




array([1, 3, 6], dtype=int32)




```python
a.cumprod()
```




array([1, 2, 6], dtype=int32)




```python
np.diff(a)
```




array([1, 1])



##  统计函数

常用的统计函数包括 `max, min, mean, median, std, var, sum, quantile` ，其中分位数计算是全局方法，因此不能通过 `array.quantile` 的方法调用：


```python
target = np.arange(5)
```


```python
target
```




array([0, 1, 2, 3, 4])




```python
 target.max()
```




4




```python
np.quantile(target, 0.15) # 0.5分位数
```




0.6



但是对于含有缺失值的数组，它们返回的结果也是缺失值，如果需要略过缺失值，必须使用 `nan*` 类型的函数，上述的几个统计函数都有对应的 `nan*` 函数。


```python
target = np.array([1, 2, np.nan])
```


```python
target
```




array([ 1.,  2., nan])




```python
target.max()
```




nan




```python
np.nanmax(target)
```




2.0




```python
np.nanquantile(target, 0.5)
```




1.5



对于协方差和相关系数分别可以利用 `cov, corrcoef `如下计算：


```python
target1 = np.array([1,3,5,9])
```


```python
target2 = np.array([1,5,3,-9])

```


```python
np.cov(target1, target2)
```




array([[ 11.66666667, -16.66666667],
           [-16.66666667,  38.66666667]])




```python
np.corrcoef(target1, target2)
```




array([[ 1.        , -0.78470603],
           [-0.78470603,  1.        ]])



最后，需要说明二维 `Numpy` 数组中统计函数的 `axis` 参数，它能够进行某一个维度下的统计特征计算，当 `axis=0` 时结果为列的统计指标，当 `axis=1`时结果为行的统计指标：


```python
target = np.arange(1,10).reshape(3,-1)
```


```python
target
```




array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
target.sum(0)
```




array([12, 15, 18])




```python
target.sum(1)
```




array([ 6, 15, 24])



## 广播机制

标量和数组的操作  
当一个标量和数组进行运算时，标量会自动把大小扩充为数组大小，之后进行逐元素操作：


```python
res = 3 * np.ones((2,2)) + 1
```


```python
res
```




array([[4., 4.],
           [4., 4.]])




```python
res = 1 / res
```


```python
res
```




array([[0.25, 0.25],
           [0.25, 0.25]])



二维数组之间的操作


```python
res = np.ones((3,2))
```


```python
res * np.array([[2,3]]) # 第二个数组扩充第一维度为3
```




array([[2., 3.],
           [2., 3.],
           [2., 3.]])




```python
res * np.array([[2],[3],[4]]) # 第二个数组扩充第二维度为2
```




array([[2., 2.],
           [3., 3.],
           [4., 4.]])




```python
res * np.array([[2]]) # 等价于两次扩充，第二个数组两个维度分别扩充为3和2
```




array([[2., 2.],
           [2., 2.],
           [2., 2.]])



一维数组与二维数组的操作


```python
np.ones(3) + np.ones((2,3))
```




array([[2., 2., 2.],
           [2., 2., 2.]])




```python
np.ones(3) + np.ones((2,1))
```




array([[2., 2., 2.],
           [2., 2., 2.]])




```python
np.ones(1) + np.ones((2,3))
```




array([[2., 2., 2.],
           [2., 2., 2.]])



## 向量与矩阵的计算

向量内积：`dot`

$$ a · b = \Sigma_i{a_ib_i}$$


```python
a = np.array([1,2,3])
```


```python
b = np.array([1,3,5])
```


```python
a.dot(b)
```




22



向量范数和矩阵范数: `np.linalg.norm`


```python
matrix_target =  np.arange(4).reshape(-1,2)
```


```python
matrix_target
```




array([[0, 1],
           [2, 3]])




```python
np.linalg.norm(matrix_target, 'fro')
```




3.7416573867739413




```python
np.linalg.norm(matrix_target, np.inf)
```




5.0




```python
np.linalg.norm(matrix_target, 2)
```




3.702459173643833




```python
vector_target =  np.arange(4)
```


```python
vector_target
```




array([0, 1, 2, 3])




```python
np.linalg.norm(vector_target, np.inf)
```




3.0




```python
np.linalg.norm(vector_target, 2)
```




3.7416573867739413




```python
np.linalg.norm(vector_target, 3)
```




3.3019272488946263



矩阵乘法


```python
a = np.arange(4).reshape(-1,2)
```


```python
b = np.arange(-4,0).reshape(-1,2)
```


```python
a
```




array([[0, 1],
           [2, 3]])




```python
b
```




array([[-4, -3],
           [-2, -1]])




```python
a@b
```




array([[ -2,  -1],
           [-14,  -9]])



## Ex1：利用列表推导式写矩阵乘法

## 方法1


```python
M1 = np.random.rand(2,3)
```


```python
M2 = np.random.rand(3,4)
```


```python
res = np.zeros((M1.shape[0],M2.shape[1]))
```


```python
res.shape
```




(2, 4)




```python
res
```




array([[0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
i=0
```


```python
def multifuc(i,j,k):
    res[i][j]+=M1[i][k] * M2[k][j]
    return res[i][j]
```


```python
%timeit -n 30 [ multifuc(i,j,k)  for i in range(M1.shape[0])  for j in range(M2.shape[1]) for k in range(M1.shape[1])]
```

83.3 µs ± 33.6 µs per loop (mean ± std. dev. of 7 runs, 30 loops each)
    


```python
(np.abs((M1@M2 - res) < 1e-15)).all()
```




True



## 方法2


```python
i=0
```


```python
sum([M1[i][k] * M2[k][j] for j in range(M2.shape[1]) for k in range(M1.shape[1])])
```




3.745920492921166




```python
%timeit -n 30 [sum([M1[i][k] * M2[k][j] for j in range(M2.shape[1]) for k in range(M1.shape[1])]) for i in range(M1.shape[0])  for j in range(M2.shape[1]) ]
```

203 µs ± 33.3 µs per loop (mean ± std. dev. of 7 runs, 30 loops each)
    


```python
(np.abs((M1@M2 - res) < 1e-15)).all()
```




True



很明显第一种方法会比datawhale官方的要少一个循环，虽然这样确实有点取巧

## Ex2：更新矩阵

### Ex2：更新矩阵
设矩阵 $A_{m×n}$ ，现在对 $A$ 中的每一个元素进行更新生成矩阵 $B$ ，更新方法是 $B_{ij}=A_{ij}\sum_{k=1}^n\frac{1}{A_{ik}}$ ，例如下面的矩阵为 $A$ ，则 $B_{2,2}=5\times(\frac{1}{4}+\frac{1}{5}+\frac{1}{6})=\frac{37}{12}$ ，请利用 `Numpy` 高效实现。
$$
 A=\left[
 \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix}
  \right] 
$$


```python
import numpy as np
```


```python
A=np.arange(1,10).reshape(3,-1)#start从0开始，默认不包含end
```


```python
A
```




array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
A_reverse = 1/A
```


```python
A_reverse
```




array([[1.        , 0.5       , 0.33333333],
           [0.25      , 0.2       , 0.16666667],
           [0.14285714, 0.125     , 0.11111111]])




```python
A_reverse.sum(axis=1).reshape(3,-1)
```




array([[1.83333333],
           [0.61666667],
           [0.37896825]])




```python
res =A*(A_reverse.sum(axis=1).reshape(3,-1))
```


```python
res
```




array([[1.83333333, 3.66666667, 5.5       ],
           [2.46666667, 3.08333333, 3.7       ],
           [2.65277778, 3.03174603, 3.41071429]])




```python
res.shape
```




(3, 3)



## Ex3：卡方统计量

设矩阵$A_{m\times n}$，记$B_{ij} = \frac{(\sum_{i=1}^mA_{ij})\times (\sum_{j=1}^nA_{ij})}{\sum_{i=1}^m\sum_{j=1}^nA_{ij}}$，定义卡方值如下：
$$\chi^2 = \sum_{i=1}^m\sum_{j=1}^n\frac{(A_{ij}-B_{ij})^2}{B_{ij}}$$
请利用`Numpy`对给定的矩阵$A$计算$\chi^2$ 

$(\sum_{i=1}^mA_{ij})\times (\sum_{j=1}^nA_{ij}) shape (8 \times 1) \times(1 \times 5)$ 


```python
np.random.seed(0)
```


```python
A = np.random.randint(10, 20, (8, 5))
```

通过分析我们可以看出其实矩阵$B$就是矩阵$A$通过对行和列求和再叉乘，然后除以所有元素的和


```python
A_rowsum=A.sum(axis=0).reshape(1,-1)
```


```python
A_columnsum=A.sum(axis=1).reshape(-1,1)
```


```python
A_columnsum.shape,A_rowsum.shape
```




((8, 1), (1, 5))




```python
A_rowsum*A_columnsum #有点离谱
#主要是*号对应数组用数组乘法，矩阵用矩阵乘法
```



array([[ 8160,  7548,  8772,  7888,  6868],
           [ 8760,  8103,  9417,  8468,  7373],
           [ 9600,  8880, 10320,  9280,  8080],
           [ 9480,  8769, 10191,  9164,  7979],
           [10200,  9435, 10965,  9860,  8585],
           [ 7320,  6771,  7869,  7076,  6161],
           [ 8040,  7437,  8643,  7772,  6767],
           [ 7680,  7104,  8256,  7424,  6464]])




```python
(A_rowsum*A_columnsum).shape
```




(8, 5)




```python
A_columnsum@A_rowsum #比较推荐这种
```




array([[ 8160,  7548,  8772,  7888,  6868],
           [ 8760,  8103,  9417,  8468,  7373],
           [ 9600,  8880, 10320,  9280,  8080],
           [ 9480,  8769, 10191,  9164,  7979],
           [10200,  9435, 10965,  9860,  8585],
           [ 7320,  6771,  7869,  7076,  6161],
           [ 8040,  7437,  8643,  7772,  6767],
           [ 7680,  7104,  8256,  7424,  6464]])




```python
B=(A_columnsum@A_rowsum)/A.sum()
```


```python
B.shape==A.shape
```




True




```python
res = ((A-B)**2/B).sum()
res
```




11.842696601945802




```python
np.random.seed(0)
A = np.random.randint(10, 20, (8, 5))
B = A.sum(0)*A.sum(1).reshape(-1, 1)/A.sum() #个人认为这样写其实会误导初学者
res = ((A-B)**2/B).sum()
res
```




11.842696601945802



## Ex4：改进矩阵计算的性能

### 原方法

![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-eDkWiRZe-1608522933099)(attachment:image.png)\]](https://img-blog.csdnimg.cn/20201221120104121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)



```python
import numpy as np
```


```python
np.random.seed(0)
m,n,p=100,80,50
B=np.random.randint(0,2,(m,p))
U=np.random.randint(0,2,(p,n))
Z=np.random.randint(0,2,(m,n))
```


```python
def solution(B=B, U=U, Z=Z):
    L_res = []
    for i in range(m):
        for j in range(n):
            norm_value = ((B[i]-U[:,j])**2).sum()
            L_res.append(norm_value*Z[i][j])
    return sum(L_res)
```


```python
solution(B, U, Z)
```




100566



![在这里插入图片描述](https://img-blog.csdnimg.cn/20201221120126404.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)


从上式可以看出，第一第二项分别为$B$的行平方和与$U$的列平方和，第三项是两倍的内积。因此，$Y$矩阵可以写为三个部分，第一个部分是$m×n$的全$1$矩阵每行乘以$B$对应行的行平方和，第二个部分是相同大小的全$1$矩阵每列乘以$U$对应列的列平方和，第三个部分恰为$B$矩阵与$U$矩阵乘积的两倍。从而结果如下：


```python
B[1].shape
```




(50,)




```python
U[:,1].shape
```




(50,)




```python
(((B**2).sum(axis=1).reshape(-1,1)+(U**2).sum(axis=0)-2*B@U)*Z).sum()
```




100566



## 连续整数的最大长度

输入一个整数的 `Numpy` 数组，返回其中严格递增连续整数子数组的最大长度。例如，输入 [1,2,5,6,7]，[5,6,7]为具有最大长度的递增连续整数子数组，因此输出3；输入[3,2,1,2,3,4,6]，[1,2,3,4]为具有最大长度的递增连续整数子数组，因此输出4。请充分利用 Numpy 的内置函数完成。（提示：考虑使用 `nonzero, diff` 函数）


```python
f = lambda x:np.diff(np.nonzero(np.r_[1,np.diff(x)!=1,1])).max()
f([1,2,5,6,7])
f([3,2,1,2,3,4,6])
```



4




```python
x=[1,2,5,6,7]
```


```python
np.diff(x)
```




array([1, 3, 1, 1])




```python
np.diff(x)!=1
```




array([False,  True, False, False])




```python
np.r_[1,np.diff(x)!=1,1]
```




array([1, 0, 1, 0, 0, 1], dtype=int32)




```python
np.nonzero(np.r_[1,np.diff(x)!=1,1])
```




(array([0, 2, 5], dtype=int64),)




```python
np.diff(np.nonzero(np.r_[1,np.diff(x)!=1,1])).max()
```




3




```python

```
1.842696601945802



## Ex4：改进矩阵计算的性能

### 原方法

[外链图片转存中...(img-eDkWiRZe-1608522933099)]


```python
import numpy as np
```


```python
np.random.seed(0)
m,n,p=100,80,50
B=np.random.randint(0,2,(m,p))
U=np.random.randint(0,2,(p,n))
Z=np.random.randint(0,2,(m,n))
```


```python
def solution(B=B, U=U, Z=Z):
    L_res = []
    for i in range(m):
        for j in range(n):
            norm_value = ((B[i]-U[:,j])**2).sum()
            L_res.append(norm_value*Z[i][j])
    return sum(L_res)
```


```python
solution(B, U, Z)
```




100566



改进方法：

令$Y_{ij} = \|B_i-U_j\|_2^2$，则$\displaystyle R=\sum_{i=1}^m\sum_{j=1}^n Y_{ij}Z_{ij}$，这在`Numpy`中可以用逐元素的乘法后求和实现，因此问题转化为了如何构造`Y`矩阵。

$$
\begin{split}Y_{ij} &= \|B_i-U_j\|_2^2\\
&=\sum_{k=1}^p(B_{ik}-U_{kj})^2\\
&=\sum_{k=1}^p B_{ik}^2+\sum_{k=1}^p U_{kj}^2-2\sum_{k=1}^p B_{ik}U_{kj}\\\end{split}
$$

从上式可以看出，第一第二项分别为$B$的行平方和与$U$的列平方和，第三项是两倍的内积。因此，$Y$矩阵可以写为三个部分，第一个部分是$m×n$的全$1$矩阵每行乘以$B$对应行的行平方和，第二个部分是相同大小的全$1$矩阵每列乘以$U$对应列的列平方和，第三个部分恰为$B$矩阵与$U$矩阵乘积的两倍。从而结果如下：


```python
B[1].shape
```




(50,)




```python
U[:,1].shape
```




(50,)




```python
(((B**2).sum(axis=1).reshape(-1,1)+(U**2).sum(axis=0)-2*B@U)*Z).sum()
```




100566



## 连续整数的最大长度

输入一个整数的 `Numpy` 数组，返回其中严格递增连续整数子数组的最大长度。例如，输入 [1,2,5,6,7]，[5,6,7]为具有最大长度的递增连续整数子数组，因此输出3；输入[3,2,1,2,3,4,6]，[1,2,3,4]为具有最大长度的递增连续整数子数组，因此输出4。请充分利用 Numpy 的内置函数完成。（提示：考虑使用 `nonzero, diff` 函数）


```python
f = lambda x:np.diff(np.nonzero(np.r_[1,np.diff(x)!=1,1])).max()
f([1,2,5,6,7])
f([3,2,1,2,3,4,6])
```




4




```python
x=[1,2,5,6,7]
```


```python
np.diff(x)
```




array([1, 3, 1, 1])




```python
np.diff(x)!=1
```



array([False,  True, False, False])




```python
np.r_[1,np.diff(x)!=1,1]
```




array([1, 0, 1, 0, 0, 1], dtype=int32)




```python
np.nonzero(np.r_[1,np.diff(x)!=1,1])
```




(array([0, 2, 5], dtype=int64),)




```python
np.diff(np.nonzero(np.r_[1,np.diff(x)!=1,1])).max()
```




3
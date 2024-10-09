title: PAT笔记
tags:
  - 数据结构
categories: []
author: whut ykh
date: 2021-10-16 14:19:00
---
# PAT笔记
这里记录了一下常用的保研机试，剑指offer和PAT题目分类解析，主要用来快速回顾和复习相关模板，以及数据结构相关的知识点。

## 字符串
- 在c++中处理字符串类型的题目时，我们一般使用`string`，有时候我们也使用`char[]`方式进行操作。
- `HH:MM:SS`可以直接通过字符串字典序排序
- 输入一个包含空格的字符串需要使用`getline(cin,s1)`
## STL
- `vector<int>`本省具备有字典序比较的方法，重载了`< == >`的运算符号
- `vector<int>::iterator iter=find(vec.begin(),vec.end(),target); if(iter==vec.end()) cout << "Not found" << endl;`

<!--more-->

## 高精度
- `int`的范围$-2 \times 10^9 - 2 \times 10^9$
- `long long` $-9 \times 10^{18} - 9 \times 10^{18}$
- 用`vector`按位存储
![在这里插入图片描述](https://img-blog.csdnimg.cn/5daf401d64574189a11ea27178441996.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 进制转换
- 其他进制化成10进制，采用秦九韶算法
![在这里插入图片描述](https://img-blog.csdnimg.cn/b70e15f5f88d44a681920d73cbe14f21.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

```cpp
typedef long long LL;
LL get(char c)
{
    if(c<='9') return c-'0';
    else return c-'a' + 10;
}

LL getnum(string a,LL r)
{
    LL res=0;
    for(int i=0;i<a.size();i++)
    {
        res = res * r + get(a[i]);
    }
    return res;
}
```

- 十进制转其他进制的方法，使用带余除法
![在这里插入图片描述](https://img-blog.csdnimg.cn/d9508f671e634bceb3732dfc41a08b60.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

```cpp
int get(char c)
{
    if(c<='9') return c-'0';
    else return c-'a' + 10;
}


char tochar(int c)
{
    if(c<=9) return c+'0';
    else return 'a' + c - 10;
}

//一个r进制数num转10进制
int numr_to10(string num,int r)
{
    int res = 0;
    for(int i=0;i<num.size();i++)
    {
        res = res * r + get(num[i]);
    }
    return res;
}

//一个10进制数num转r进制
string num10_tor(string num,int r)
{
    string res;
    int n = numr_to10(num,10); //先转成10进制整型
    while(n)
    {
        // cout<<tochar(n % r)<<endl;
        res = tochar(n % r) + res;
        n /= r;
    }
    return res;
}
// cout<<numr_to10("6a",16)<<" "<<num10_tor("15",16)<<endl;
```

## 判断质数
```cpp
//判断一个数是否为质数
bool is_prime(int n)
{
    if (n < 2) return false; // 1和0不是质数
    for(int i=2;i*i<=n;i++)
    {
        if(n % i == 0) return false; 
    }
    return true;
}
```
## 手写堆排序
堆是一个完全二叉树的结构，分为小根堆和大根堆两种结构。
- 小根堆的递归定义：小根堆的每个节点都小于他的左右孩子节点的值，树的根节点为最小值。
- 大根堆的递归定义：大根堆的每个节点都大于他的左右孩子节点的值，树的根节点为最大值。
在STL当中可以使用`prioirty_queue`来轻松实现大根堆和小根堆，但是只能实现前3个功能，有时候我们不得不自己实现一个手写的堆，同时这样也能让我们更理解堆排序的过程。
![在这里插入图片描述](https://img-blog.csdnimg.cn/c16a9e7d8c5a433084cd31cb996086e2.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

在AcWing基础课当中有两道经典例题，[AcWing 839. 模拟堆（这个复杂一点）](https://www.acwing.com/problem/content/841/)，[AcWing 838. 堆排序](https://www.acwing.com/problem/content/840/)
这里给出堆排序的模板级代码
```cpp
#include <iostream>
using namespace std;

const int N = 100010;

int heap[N],heapsize=0;

void down(int x)// 参数下标
{
    int p =x;
    if(2*x<=heapsize && heap[2*x]<heap[p]) p = 2*x;//左子树
    
    if(2*x+1<=heapsize && heap[2*x+1]<heap[p]) p=2*x+1;//右子树
    
    if(p!=x) 
    {
        swap(heap[p],heap[x]); //说明存在比父节点小的孩子节点
        down(p); //继续向下递归down
    }
}


void up(int x)// 参数下标
{
    while(x / 2 && heap[x] < heap[x/2]) //父节点比子节点大则交换
    {
        swap(heap[x],heap[x/2]);
        x >>= 1; // x = x/2
    }
}

int main()
{
    int n,m;
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &heap[i]);
    heapsize=n;
    
    // O(n)建堆
    for (int i = n / 2; i; i -- ) down(i);
    
    while (m -- )
    {
        printf("%d ",heap[1]); //最小值是小根堆的堆顶
        // 删除最小值，并重新建堆排序，从而获得倒数第二小的元素
        heap[1] = heap[heapsize];
        heapsize--;
        down(1);
    }
    return 0;
}
```
STL写法：`priority_queue`默认是大根堆，`less<int>`是对第一个参数的比较类，表示数字大的优先级越大，而`greater<int>`表示数字小的优先级越大，可以实现结构体运算符重载。
首先要引入头文件：`#include<queue>`
大根堆：
```cpp
priority_queue<int> q;
priority_queue<int, vector<int>, less<int> >q;
```
小根堆：

```cpp
priority_queue < int, vector<int>, greater<int> > q;
```

## 树
树是一种特殊的数据结构形式，在做题的过程当中，根据我的经验当题目需要使用树结构的时候主要有以下几种模式。
- **二叉树形式**，在二叉树模型下，我们可以根据题目建立出静态的树形结构，构建每个节点**左右孩子索引表**来建立树的结构同时实现对树的遍历。**如果已知或可以求得节点之间的关系，可以通过节点的度数或者访问标记找到根节点。**，当然也是可以通过邻接表的方式创建二叉树。
- 多叉树形式，多叉树形式其实又类似于**无向连通图**的概念，常通过创建**邻接表**或者**临接矩阵**的方式建立树，并实现进行树的遍历，也是可以根据节点关系求出根节点的。注意在临接表当中，边的数量一般大于节点数量的两倍即我们需要开票邻接表的边数空间为$M = 2 \times N + d$
- 森林，多连通块的方式，这种也是利用无向图的方式，以**邻接表**或者**临接矩阵**的方式构建树的结构，同时我们可以利用**并查集**的方式得到当前无向图中含有的连通块数量并找到根节点。

二叉树左右孩子索引表模型
```cpp
const int N = 100010;
int l[N],r[N]// 第i个节点的左孩子和右孩子的索引
bool has_father[N]; //建立树的时候判断一下当前节点有没有父节点，可用于寻找根节点

//初始化，-1表示子节点为空
memset(l,-1,sizeof l);
memset(r,-1,sizeof r);

// 查找根节点的过程
if(l[i]>=0) has_father[l[i]]=true;
if(r[i]>=0) has_father[r[i]]=true;
//查找根节点
int root = 0;
while(has_father[root]) root++;
```

二叉树的遍历过程（以先序遍历为例子）
```cpp
void dfs(int root)
{
	if(root==-1) return;
	cout<<root<<endl;
    if(l[root]>=0) dfs(l[root]);
    if(r[root]>=0) dfs(r[root]);
}
```

临接表模型
```cpp
const int N = 100010;
const int M = 2 * N + 10;
int h[N];//邻接表的N个节点头指针，h[i]表示以i为起点的，最新的一条边的编号
int e[M];// e[i] 表示第i条边的所指向的终点
int ne[M];// ne[i]表示与第i条边起点相同的下一条边的编号
int idx;// idx表示边的编号，每增加一条边就++

// 添加一条从a到b的边，如果是无向图，每次添加时要add(a,b)和add(b,a)
void add(int a,int b)
{
	e[idx] = b; // 第idx条边的终点为b
	ne[idx] = h[a]; // h[a] 和 第idx都是以a为起点的边，通过ne[idx]串联起来，找到上一条以a为起点的边h[a]
	h[a] = idx ++; //  更新当前以a为起点的边的最新编号
}

//初始化，-1表示节点为空
memset(h,-1,sizeof h);
```

临接表遍历过程方法1
```cpp
// x为起点，father为x的来源，防止节点遍历走回头路导致死循环
void dfs(int x,int father)
{
	cout<<x<<endl;
    for(int i = h[x];~i;i=ne[i]) // ~i就是i!=-1的意思
    {
        int to = e[i];
        if(to==father) continue;
        dfs(to,x);
    }
}
dfs(x,-1);
```

临接表遍历过程方法2
```cpp
const int N = 100010;
bool isvisited[N];
void dfs(int x)
{
	isvisited[x]=true;
	cout<<x<<endl;
    for(int i = h[x];~i;i=ne[i]) 
    {
        int to = e[i];
        if(isvisited[to]) continue;
        dfs(to);
    }
}
dfs(x);
```
## 树的深度
临接表模型：[AcWing1498. 最深的根](https://www.acwing.com/problem/content/1500/)
```cpp
int getdepth(int x,int father)
{
    // cout<<"father"<<father<<" node"<<x<<endl;
    int depth = 0;
    for(int i = h[x];~i;i=ne[i]) 
    {
        int to = e[i];
        if(to==father) continue;
        depth = max(depth,getdepth(to,x)+1);
    }
    return depth;
}
```
二叉树模型：[剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root==NULL) return 0;
        return max(maxDepth(root->left),maxDepth(root->right))+1;
    }
};
```
多叉树模型（该题也是求叶子节点个数的经典写法）：[AcWing 1476. 数叶子结点](https://www.acwing.com/problem/content/1478/)
```cpp
const int N = 100010;
int max_depth = 0;
int cnt[N];
void dfs(int x,int depth)
{
    //说明是叶子节点
    if(h[x]==-1)
    {
        cnt[depth]++;
        max_depth = max(max_depth,depth);
        return;
    }

    for(int i=h[x];~i;i=ne[i])
    {
        dfs(e[i],depth+1);
    }
}
dfs(root,0)
//输出每一层的叶子个数
for(int i=0;i<=max_depth;i++) cout<<" "<<cnt[i];
```

## 二叉搜索树
二叉搜索树 (BST) 递归定义为具有以下属性的二叉树：
- 若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值
- 若它的右子树不空，则右子树上所有结点的值均大于或等于它的根结点的值
- 它的左、右子树也分别为二叉搜索树

**二叉搜索树的中序遍历一定是有序的**




## 完全二叉树
完全二叉树 (CBT) 定义为除最深层外的其他层的结点数都达到最大个数，最深层的所有结点都连续集中在最左边的二叉树。
构造完全二叉树的方法，可以直接开辟一个一维数组利用左右孩子与根节点的下标映射关系。如果通过中序遍历的方式以单调递增的方式来赋值则构造出了一颗完全二叉搜索树。
![在这里插入图片描述](https://img-blog.csdnimg.cn/fd06bb725e7140f0b1349101b932a9fa.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
完全二叉树的赋值填充和构造过程（这里我们以中序遍历为例子）：
例题：[AcWing 1550. 完全二叉搜索树](https://www.acwing.com/problem/content/1552/)
```cpp
//中序遍历填充数据
int cnt; //记录已经赋值的节点下标
void dfs(int x) // 根节点为1-n
{
    if(2*x <=n) dfs(2*x);
    h[x] = a[cnt++];
    if(2*x+1<=n) dfs(2*x+1);
}

void dfs(int u, int& k)  // 中序遍历，k引用实现下标迁移
{
    if (u * 2 <= n) dfs(u * 2, k);
    tr[u] = w[k ++ ];
    if (u * 2 + 1 <= n) dfs(u * 2 + 1, k);
}
```
完全二叉树的节点个数规律：
- 具有n个结点的完全二叉树的深度为$\lfloor  log_2{n} \rfloor+ 1$
- 完全二叉树如果为满二叉树，且深度为$k$则总节点个数为$2^{k}-1$
- 完全二叉树的第$i(i  \geq 1)$层的节点数最大值为$2^{i-1}$
- 完全二叉树最后一层按从左到右的顺序进行编号，上面的层数皆为节点数的最大值，**因此不会出现左子树为空，右子树存在的节点**
- 根据完全二叉树的结构可知：**完全二叉树度为1的节点只能为1或者0**，则有当节点总数为$n$时，如果$n$为奇数，则$n_0 = (n+1)/2$，如果$n$为偶数，则$n_0 = n / 2$
> 关于最后一条性质的一些拓展
> **二叉树的重要性质：在任意一棵二叉树中，若叶子结点的个数为$n_0$，度为2的结点数为$n_2$，则$n_0=n_2+1$**
> 证明：
> 假设该二叉树总共有$n$个结点$(n=n_0+n_1+n_2)$，则该二叉树总共会有$n-1$条边，度为2的结点会延伸出两条边，度为1的结点会延伸出1条边。
> 则有$n - 1 = n_0+n_1+n_2- 1= 2 \times n_2 + n_1$
> 联立两式得到：$n_0=n_2+1$
> 拓展到完全二叉树，因为完全二叉树度为1的节点只有0个或者1个。即$n_1 = 0 或 1$
> 则节点总数$n=n_0+n_1+n_2 = 2 *n_0 + n_1 - 1$
> 由于节点个数必须为整数，因此可以得到以下结论：
> 当$n$为奇数时，必须使得$n_1=0$，则$n_0=(n + 1) / 2，n_2=n_0-1=(n + 1) / 2-1$
> 当$n$为偶数时，必须使得$n_1=1$，则$n_0=n / 2，n_2=n_0-1=n /2 -1$

例题（递归解法）：[leetcode 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)
```cpp
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int countNodes(TreeNode root) {
        return root==null ? 0:countNodes(root.left)+countNodes(root.right)+1;
    }
}
```

完全二叉树

## 二叉平衡树
#### AVL树
- AVL树是一种自平衡二叉搜索树。
- 在AVL树中，任何节点的两个子树的高度最多相差 1 个。
- 如果某个时间，某节点的两个子树之间的高度差超过 1，则将通过树旋转进行重新平衡以恢复此属性。
- AVL本质上还是维护一个二叉搜索树，所以不管如果旋转，其中序遍历依旧是不变的。
旋转法则：

![在这里插入图片描述](https://img-blog.csdnimg.cn/d38acda60f184d0987848cee7407709e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)AVL插入分为一下几种情况：
- LL型：新节点的插入位置在A的左孩子的左子树上，则右旋A
- RR型：新节点的插入位置在A的右孩子的右子树上，则左旋A
- LR型：新节点的插入位置在A的左孩子的右子树上，则左旋B，右旋A
- RL型：新节点的插入位置在A的右孩子的左子树上，则右旋B，左旋A
![在这里插入图片描述](https://img-blog.csdnimg.cn/9f339a4b86a84d42aafabfc3b35b5e8a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)

#### 红黑树
数据结构中有一类平衡的二叉搜索树，称为红黑树。
它具有以下 5 个属性：
- 节点是红色或黑色。
- 根节点是黑色。
- 所有叶子都是黑色。（叶子是 NULL节点）
- 每个红色节点的两个子节点都是黑色。
- 从任一节点到其每个叶子的所有路径都包含相同数目的黑色节点。
![在这里插入图片描述](https://img-blog.csdnimg.cn/19db0e5df72e4c66bcee2a26990e9b6d.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
## 图论相关
#### 并查集
经典例题：[AcWing 836. 合并集合](https://www.acwing.com/problem/content/838/)

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 100010;

int p[N];

int find(int x) // 查找x的祖先节点，并在回溯的过程当中进行路径压缩，将各节点直接指向根节点
{
    if(x!=p[x]) p[x] = find(p[x]); // x和p[x]不相等，则继续向上找父节点的父节点
    return p[x];
}

int main()
{
    int n;
    int m;
    scanf("%d%d", &n, &m);

    for(int i=1;i<=n;i++) 
        p[i]=i;

    while (m -- )
    {
        char op[2];
        int a,b;
        scanf("%s%d%d", op,&a,&b);
        int roota = find(a);
        int rootb = find(b);
        if(op[0]=='M')
        {

            if(roota == rootb) continue;
            p[roota] = rootb; // root merge
        }
        else
        {
            cout<< (roota==rootb ? "Yes":"No")<<endl;
        }

    }
    return 0;
}
```

#### dijstra算法
- 临接矩阵形式，适用于点的数量$N < 1000$的情形，朴素算法即可解决
- 邻接表形式，当$N>10000$，需要添加堆优化
一般来说堆优化版本的考试用的不多，这里就只介绍了朴素版本。
[Dijkstra求最短路 I](https://www.acwing.com/problem/content/851/)

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 510;
const int inf = 0x3f3f3f3f;
int n,m;
int g[N][N]; // 稠密图使用邻接矩阵
int dist[N]; // 存储距离
bool vis[N]; // 标志到该节点的距离是否已经被规整为最短距离

void dijkstra(int x)
{
    memset(dist, inf, sizeof dist);
    dist[x] = 0;

    for(int i=0;i<n;i++)//外层循环n次遍历每个节点
    {
        int t= -1;

        for(int j=1;j<=n;j++)
        {
            if(!vis[j]&&(t==-1 || dist[t]>dist[j])) t =j;
        }
        if(t==-1) break;
        vis[t]=true;

        for(int j=1;j<=n;j++)
        {
            if(!vis[j])
            {
                dist[j] = min(dist[j],dist[t]+g[t][j]);
            }
        }
    }

    if(dist[n]==inf) puts("-1");
    else cout<<dist[n]<<endl;

}


int main()
{
    scanf("%d%d", &n, &m);
    memset(g, inf, sizeof g);
    for(int i=0;i<m;i++)
    {
        int x,y,z;
        scanf("%d%d%d", &x, &y,&z);
        if(x==y) g[x][y]=0; // 自环
        g[x][y] = min(g[x][y],z); // 重边仅记录最小的边
    }

    dijkstra(1);
    return 0;
}
```

#### 最小生成树Prime
[AcWing 858.Prime算法求最小生成树](https://www.acwing.com/activity/content/code/content/1219581/)
```cpp
//这里填你的代码^^
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 510, INF = 0x3f3f3f3f;
int n,m;

int g[N][N]; //稠密图使用prim和邻接矩阵
int dist[N]; 
bool isvisited[N];

int prime(int x)
{
    memset(dist, 0x3f, sizeof dist);
    int res = 0;
    dist[x]=0;
    for(int i=0;i<n;i++)
    {
        int t=-1;
        for(int j=1;j<=n;j++)
            if(!isvisited[j] && (t==-1 || dist[t] > dist[j]))
                t= j;

        if(dist[t] == INF) return -1;
        //标记访问
        res += dist[t];
        isvisited[t]=true;

        //更新dist
        for(int j=1;j<=n;j++)
        {
            dist[j] = min(dist[j],g[t][j]); 
        }
    }
    return res;
}


int main()
{
    scanf("%d%d", &n, &m);
    memset(g, 0x3f, sizeof g);
    while (m -- )
    {
        int a,b,c;
        scanf("%d%d%d", &a, &b,&c);
        g[a][b] = g[b][a] = min(g[a][b],c); //无向图
    }


    int t = prime(1);

    if(t==-1)
        cout<<"impossible"<<endl;
    else
        cout<<t<<endl;

    return 0;
}
```
#### 最小生成树Kruskal
[AcWing859.Kruskal算法求最小生成树](https://www.acwing.com/problem/content/861/)
```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 100010, INF =0x3f3f3f3f;
const int M = 2*N;

int n,m;

struct Edge
{
    int x;
    int y;
    int w;
    bool operator < (const Edge & E) const
    {
        return w < E.w;
    }
}edge[M];

int p[N]; //并查集

int find(int x)//找祖宗节点
{
    if(x!=p[x]) p[x] = find(p[x]);
    return p[x];
}

int kruskal()
{
    int res = 0;
    int cnt=0;
    sort(edge,edge+m);
    for(int i=1;i<=n;i++) p[i]=i;//初始化并查集

    for(int i=0;i<m;i++)
    {
        int x = edge[i].x, y = edge[i].y, w = edge[i].w;

        int a = find(x);
        int b = find(y);
        //不是连通的
        if(a!=b)
        {
            p[b] = a;
            res += w;
            cnt++;
        }
    }
    //路径数量<n-1说明不连通
    if (cnt<n-1) return INF;
    return res;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 0; i < m; i ++ )
    {
        scanf("%d%d%d", &edge[i].x, &edge[i].y, &edge[i].w);
    }

    int t = kruskal();

    if(t == INF) cout<< "impossible"<<endl;
    else cout<<t<<endl;

    return 0;
}
```
#### 哈密顿图
- 通过图中所有顶点一次且仅一次的通路称为哈密顿通路。
- 通过图中所有顶点一次且仅一次的回路称为哈密顿回路。
- 具有哈密顿回路的图称为哈密顿图。
- 具有哈密顿通路而不具有哈密顿回路的图称为半哈密顿图


#### 欧拉图
- 通过图中所有边恰好一次且行遍所有顶点的通路称为欧拉通路。
- 通过图中所有边恰好一次且行遍所有顶点的回路称为欧拉回路。
- 具有欧拉回路的无向图或有向图称为欧拉图。
- 具有欧拉通路但不具有欧拉回路的无向图或有向图称为半欧拉图。
- **如果一个连通图的所有顶点的度数都为偶数，那么这个连通图具有欧拉回路，且这个图被称为欧拉图。**
- **如果一个连通图中有两个顶点的度数为奇数，其他顶点的度数为偶数，那么所有欧拉路径都从其中一个度数为奇数的顶点开始，并在另一个度数为奇数的顶点结束。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/9ce67d5dee3e4719aa4e38ca85de9564.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)


## 数学
## gcd
```cpp
LL gcd(LL a, LL b)  // 欧几里得算法
{
    return b ? gcd(b, a % b) : a;
}

```
## 1的个数(数位dp)
[ACWing1533.1的个数](https://www.acwing.com/problem/content/1535/)
[剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)
>给定一个数字 N，请你计算 1∼N 中一共出现了多少个数字 1。
例如，N=12 时，一共出现了 5 个数字 1，分别出现在 1,10,11,12 中。

解题思路：[相关视频链接](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/python3si-lu-dai-ma-10fen-zhong-jiang-qi-9btr/)
![在这里插入图片描述](https://img-blog.csdnimg.cn/49c9da0165364f10bde9df685f02be67.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0phY2tfX19F,size_16,color_FFFFFF,t_70)
```cpp
class Solution {
public:
    int countDigitOne(int n) {
        vector<int> num;
        while(n) num.push_back(n%10), n/=10;
        int res = 0;
        for(int i=num.size()-1;i>=0;i--)
        {
            int d = num[i];
            int left=0,right=0,power=1;
            for(int j=num.size()-1;j>i;j--) left = left * 10 + num[j];
            for(int j=i-1;j>=0;j--) right = right * 10 + num[j], power*=10;

            if(d==0) res += left*power;
            else if(d==1) res += left*power + right + 1;
            else res += (left+1) * power;
        }
        return res;
    }
};
```
---
title: java的String常用操作
date: 2020-09-09 20:02:38
tags: Java
author: whut ykh
---
##  java的String类常用操作
最近在准备蓝桥杯，补题目，顺便也是稍微缓一下暑假自闭，主要还是不想浪费300，顺便回顾一下java的基本语法，java的String类这一块感觉还是挺多操作的，字符串的不可变性也使得其操作和c++存在一些差别。这里需要推荐一个十分好的博客就是[廖雪峰的java教程](https://www.liaoxuefeng.com/wiki/1252599548343744),其实[菜鸟教程](https://www.runoob.com/java/java-string.html)也不错的。

Java字符串的一个重要特点就是字符串不可变。这种不可变性是通过内部的`private final char[]`字段，以及没有任何修改`char[]`的方法实现的。
<!-- more -->
### 创建字符串
创建字符串主要是两种常用方法吧，一个是new还有一个就是直接定义，毕竟已经是一个关键字了。
```java

public class Main {
    public static void main(String[] args) {
        String a="hello";
        a+="121";//Strin的+操作
        char[] helloArray = { 'r', 'u', 'n', 'o', 'o', 'b'};
        String helloString = new String(helloArray);
        System.out.println(a);
        System.out.println(helloString);
    }
}


```
当然如果需要对字符串转换成char数组的话也是封装好了函数的`toCharArray()`
```java
public class Main {
    public static void main(String[] args) {
        str="Hello";
        char[] lisc = str.toCharArray();
        String s = new String(lisc);
        System.out.println(s);
    }
}
```
### String的长度——length()
```java
public class Main {
    public static void main(String[] args) {
        String str = "abcde";
        int len = str.length();
        System.out.println("len="+len);
    }
}
```

### 字符串比较
当我们想要比较两个字符串是否相同时，要特别注意，**我们实际上是想比较字符串的内容是否相同**。必须使用`equals()`方法而不能用`==`。`==`比较的是两个变量是否指向同一个字符串对象。可以看一下下面两个结果，有什么不同。

```java
public class Main {
    public static void main(String[] args) {
        String s1 = "hello";
        String s2 = "HELLO".toLowerCase();
        System.out.println(s1 == s2);
        System.out.println(s1.equals(s2));
    }
}
```

这里还有一个用于比较java的字符串方法的`compareTo()`方法  

`compareTo()`方法用于两种方式的比较：
- 字符串与对象进行比较。
- 按字典顺序比较两个字符串。


他的返回值比较有趣，如果两者字符串相同的话会返回两者第一个不同字符的ASCII码的差值，如果两者的字符串长度不同的话
- 如果参数字符串等于此字符串，则返回值 0；
- 如果此字符串小于字符串参数，则返回一个小于 0 的值；
- 如果此字符串大于字符串参数，则返回一个大于 0 的值。

```java
int compareTo(Object o)
int compareTo(String anotherString)
```

下面是一个demo
```java
public class Test {
 
    public static void main(String args[]) {
        System.out.println("Hello World!");
        String str1 = "Strings";
        String str2 = "Strings";
        String str3 = "Strin1212";
        String str4 = "Stringr";

        int result = str1.compareTo( str2 );
        System.out.println(result);

        result = str2.compareTo( str3 );
        System.out.println(result);

        result = str3.compareTo( str1 );
        System.out.println(result);
    }
}
```

### String遍历——charAt() 方法
charAt() 方法用于返回指定索引处的字符。索引范围为从 0 到 length() - 1。

```java
public class Test {
    public static void main(String args[]) {
        String first="hello";
        for(int i=0;i<len1;i++)
            {
                char c1 = first.charAt(i);
            }
    }
}
```

如果已经变成了char[]了，不同于普通循环，你还能使用for each来循环,类似python的操作。
```java
char[] charlist={'a','b','c'};
for(char c: charlist)
{
    System.out.println(c);
}
```

### 去除首尾空白字符
其实用的不太多，具体可以看廖雪峰的教程，一般就是`trim()`方法,截去字符串两端的空格，但对于中间的空格不处理。

### 查找indexOf() 
这个菜鸟教程讲的挺细的可以看看。`indexOf()` 方法有以下四种形式：
- public int indexOf(int ch): 返回指定字符在字符串中第一次出现处的索引，如果此字符串中没有这样的字符，则返回 -1。

- public int indexOf(int ch, int fromIndex): 返回从 fromIndex 位置开始查找指定字符在字符串中第一次出现处的索引，如果此字符串中没有这样的字符，则返回 -1。

- int indexOf(String str): 返回指定字符在字符串中第一次出现处的索引，如果此字符串中没有这样的字符，则返回 -1。

- int indexOf(String str, int fromIndex): 返回从 fromIndex 位置开始查找指定字符在字符串中第一次出现处的索引，如果此字符串中没有这样的字符，则返回 -1。  

```java
public int indexOf(int ch )
public int indexOf(int ch, int fromIndex)
int indexOf(String str)
int indexOf(String str, int fromIndex)
```



`lastIndexOf()` 则差不多，进行反向搜索   
还有`startsWith(),endsWith()`

### 字串提取
`contains()`可以判断给定字串是否存在于原string当中

```java
String str = "abcde";
int index = str.indexOf(“bcd”); 
//判断是否包含指定字符串，包含则返回第一次出现该字符串的索引，不包含则返回-1
boolean b2 = str.contains("bcd");
//判断是否包含指定字符串，包含返回true，不包含返回false
```

substring()方法,索引从 0 开始。返回一个新字符串。  
```java
public String substring(int beginIndex)

public String substring(int beginIndex, int endIndex)
```

```java
public class Test {
    public static void main(String args[]) {
        String Str = new String("12345678908868");
 
        System.out.print("返回值 :" );
        System.out.println(Str.substring(4) );
 
        System.out.print("返回值 :" );
        System.out.println(Str.substring(4, 10) );
    }
}
```

### 替换子串replace()
要在字符串中替换子串，有两种方法即根据字符或字符串替换`replace()`
```java
String s = "hello";
s.replace('l', 'w'); 
s.replace("ll", "~~"); 
```

### 分割字符串split()
`split()`方法，里面可以用正则表达式  


```java
String s = "A,B,C,D";
String[] ss = s.split("\\,"); 
```

###  valueOf() 
做题的话感觉，主要用于char[]，其他数字类型的转换，当让也能用之前new的方法操作。

```java
public class Test {
    public static void main(String args[]) {
        double d = 1100.00;
        boolean b = true;
        long l = 1234567890;
        char[] arr = {'r', 'u', 'n', 'o', 'o', 'b' };

        System.out.println("返回值 : " + String.valueOf(d) );
        System.out.println("返回值 : " + String.valueOf(b) );
        System.out.println("返回值 : " + String.valueOf(l) );
        System.out.println("返回值 : " + String.valueOf(arr) );
    }
}

```
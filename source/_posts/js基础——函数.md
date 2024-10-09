title: js基础——函数
tags:
  - JavaScript
  - web基础
date: 2020-09-27 15:55:13
---
这一块关于闭包和箭头函数理解还不够，先提交一点记录一下，后期再补
<!--more-->
## 函数声明
一个函数定义（也称为函数声明，或函数语句）由一系列的function关键字组成，依次为：
1. 函数的名称。
2. 函数参数列表，包围在括号中并由逗号分隔。
3. 定义函数的 JavaScript 语句，用大括号`{}`括起来。


**当函数参数为基本类型时，则会采用值传递的方式，不会改变变量本身，而当你传递的是一个对象（即一个非原始值，例如`Array`或用户自定义的对象）作为参数的时候，而函数改变了这个对象的属性，这样的改变对函数外部是可见的**
```js
function myFunc(theObject) {
  theObject.make = "Toyota";
}

var mycar = {make: "Honda", model: "Accord", year: 1998};
var x, y;

x = mycar.make;     // x获取的值为 "Honda"

myFunc(mycar);
y = mycar.make;  
```

## 函数表达式——Function expressions
根据MDN的文档描述我大致理解了他所说的函数表达式用法和意义了，其实类似于c语言的函数指针和java匿名函数相结合的这种用法，这种用法的最大好处就是能够将函数作为一个参数或者变量，将其传递给其他的函数使用


## 箭头函数
箭头函数主要利用的是箭头表达式，箭头函数表达式的语法比函数表达式更简洁，箭头函数表达式更适用于那些本来需要匿名函数的地方，并且它不能用作构造函数。**箭头函数两大特点：更简短的函数并且不绑定this。**
箭头函数真的很灵活，我暂时还没法完全参透只是了解了一个大概，具体的用法可以去MDN上看官方文档[箭头函数](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Functions/Arrow_functions)

## 闭包
闭包是 JavaScript 中最强大的特性之一。JavaScript 允许函数嵌套，并且内部函数可以访问定义在外部函数中的所有变量和函数，以及外部函数能访问的所有变量和函数。


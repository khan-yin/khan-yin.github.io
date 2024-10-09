---
title: js基础——关于对象
date: 2020-10-02 16:03:30
tags:
  - JavaScript
  - web基础
---

## 对象与属性
同其他语言里面所描述的对象一样，在js里，一个对象就是一系列属性的集合，一个属性包含一个名和一个值。一个属性的值可以是函数，这种情况下属性也被称为方法。一个对象的属性可以被解释成一个附加到对象上的变量。对象有时也被叫作**关联数组**, 因为每个属性都有一个用于访问它的**字符串值**。
<!--more-->
属性的访问设置与修改主要是通过两种手段实现，一种是通过点号对变量进行调用，一种是通过方括号的方式访问，其中通过方括号的方式是一种动态判定法(属性名只有到运行时才能判定)。

一个对象的属性名可以是任何有效的 JavaScript 字符串，或者可以被转换为字符串的任何类型，**包括空字符串**。然而，一个属性的名称如果不是一个有效的 JavaScript 标识符（例如，一个由空格或连字符，或者以数字开头的属性名），**就只能通过方括号标记访问。**

方括号中的所有键都将转换为字符串类型，因为JavaScript中的对象只能使用`String`类型作为键类型,如果是`object`类型的话，也可以通过方括号直接添加属性，不过他添加属性的时候会调用`toString()`方法，并将其作为新的key值。

你可以在` for...in `语句中使用方括号标记以枚举一个对象的所有属性

**拓展**：从 ECMAScript 5 开始，有三种原生的方法用于列出或枚举对象的属性：
- **for...in循环** 
  该方法依次访问一个对象及其原型链中所有可枚举的属性。
- **Object.keys(o)**
  该方法返回对象 o 自身包含（不包括原型中）的所有可枚举属性的名称的数组。
- **Object.getOwnPropertyNames(o)**
  该方法返回对象 o 自身包含（不包括原型中）的所有属性(无论是否可枚举)的名称的数组。

## 创建新对象
JavaScript 拥有一系列预定义的对象，当然我们也可以自己创建对象，从  JavaScript 1.2 之后，你可以通过**对象初始化器**（Object Initializer）创建对象。或者你可以创建一个**构造函数**并使用该函数和`new`操作符初始化对象。

#### 使用对象初始化器
使用对象初始化器也被称作通过字面值创建对象，通过对象初始化器创建对象的语法如下：
```js
var obj = { property_1:   value_1,   // property_# 可以是一个标识符...
            2:            value_2,   // 或一个数字...
           ["property" +3]: value_3,  //  或一个可计算的key名... 
            // ...,
            "property n": value_n }; // 或一个字符串
```
这里 `obj`是新对象的名称，每一个 `property_i` 是一个标识符（可以是一个名称、数字或字符串字面量），并且每个 `value_i` 是一个其值将被赋予 `property_i` 的表达式。**obj 与赋值是可选的**；
```js
var myHonda = {color: "red", wheels: 4, engine: {cylinders: 4, size: 2.2}};//这里面的engine也是一个对象
```

#### 使用构造函数
使用构造函数实例化对象的过程分为两步：
1. 通过创建一个构造函数来定义对象的类型。首字母大写是非常普遍而且很恰当的惯用法。
2. 通过 `new` 创建对象实例。

#### 使用 Object.create 方法
对象也可以用 `Object.create()` 方法创建。该方法非常有用，因为它允许你为创建的对象选择一个原型对象，而不用定义构造函数。
```js
// Animal properties and method encapsulation
var Animal = {
  type: "Invertebrates", // 属性默认值
  displayType : function() {  // 用于显示type属性的方法
    console.log(this.type);
  }
}

// 创建一种新的动物——animal1 
var animal1 = Object.create(Animal);
animal1.displayType(); // Output:Invertebrates

// 创建一种新的动物——Fishes
var fish = Object.create(Animal);
fish.type = "Fishes";
fish.displayType(); // Output:Fishes
```

## 继承
所有的 JavaScript 对象至少继承于一个对象。**被继承的对象被称作原型，并且继承的属性可通过构造函数的`prototype`对象找到。**

## 为对象类型定义属性
你可以通过 prototype 属性为之前定义的对象类型增加属性。这为该类型的所有对象，而不是仅仅一个对象增加了一个属性。下面的代码为所有类型为 car 的对象增加了 color 属性，然后为对象 car1 的 color 属性赋值：
```js
Car.prototype.color = null;
car1.color = "black";
```

## 通过 this 引用对象
avaScript 有一个特殊的关键字 this，它可以在方法中使用以指代当前对象。

## 删除属性
你可以用`delete`操作符删除一个**不是继承而来**的属性。下面的例子说明如何删除一个属性：
```js
//Creates a new object, myobj, with two properties, a and b.
var myobj = new Object;
myobj.a = 5;
myobj.b = 12;

//Removes the a property, leaving myobj with only the b property.
delete myobj.a;
```

## 比较对象
**在 JavaScript 中 objects 是一种引用类型。两个独立声明的对象永远也不会相等，即使他们有相同的属性，只有在比较一个对象和这个对象的引用时，才会返回true.**
这边是官方给出的例子
```js
// 两个变量, 两个具有同样的属性、但不相同的对象
var fruit = {name: "apple"};
var fruitbear = {name: "apple"};

fruit == fruitbear // return false
fruit === fruitbear // return false
```
```js
// 两个变量, 同一个对象
var fruit = {name: "apple"};
var fruitbear = fruit;  // 将fruit的对象引用(reference)赋值给 fruitbear
                        // 也称为将fruitbear“指向”fruit对象
// fruit与fruitbear都指向同样的对象
fruit == fruitbear // return true
fruit === fruitbear // return true
```

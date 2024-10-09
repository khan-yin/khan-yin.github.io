title: js基础——数据类型与基本逻辑语句
tags:
  - JavaScript
  - web基础
categories: []
date: 2020-09-26 15:57:00
---
本文将记录学习JavaScript(包含有ES)语法基础,主要来自[MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript)的官方文档
<!--more-->
## 语法和数据类型
最新的 ECMAScript 标准定义了8种数据类型：
- 七种基本数据类型:
1. 布尔值（`Boolean`），有2个值分别是：true 和 false.
2. `null` ， 一个表明 null 值的特殊关键字。 JavaScript 是大小写敏感的，因此 `null` 与 `Null`、`NULL`或变体**完全不同**。
3. `undefined` ，和 null 一样是一个特殊的关键字，undefined 表示变量未定义时的属性。
4. 数字（`Number`），整数或浮点数，例如： 42 或者 3.14159。
5. 任意精度的整数 (`BigInt`) ，可以安全地存储和操作大整数，甚至可以超过数字的安全整数限制。
6. 字符串（`String`），字符串是一串表示文本值的字符序列，例如："Howdy" 。
7. 代表（`Symbol`） ( 在 ECMAScript 6 中新添加的类型).。一种实例是唯一且不可改变的数据类型。
- 以及对象（`Object`）。

### 运算细节
你可以使用 `undefined` 来判断一个变量是否已赋值
```js
var input;
if(input === undefined){
  doThis();
} else {
  doThat();
}
```
`undefined` 值在布尔类型环境中会被当作 `false`

数值类型环境中 `undefined` 值会被转换为 `NaN`

当你对一个 `null` 变量求值时，空值 `null` 在数值类型环境中会被当作`0`来对待，而布尔类型环境中会被当作 `false`

在包含的数字和字符串的表达式中使用加法运算符（+），JavaScript 会把数字转换成字符串
```js
x = "The answer is " + 42 // "The answer is 42"
y = 42 + " is the answer" // "42 is the answer"
```
在涉及其它运算符（译注：如下面的减号'-'）时，JavaScript语言不会把数字变为字符串
```js
"37" - 7 // 30
"37" + 7 // "377"
```



### 对象字面量
对象字面值是封闭在花括号对({})中的一个对象的零个或多个"属性名-值"对的（元素）列表。

对象属性名字可以是任意字符串，包括空串。如果对象属性名字不是合法的javascript标识符，它必须用""包裹。属性的名字不合法，那么便不能用.访问属性值，而是通过类数组标记("[]")访问和赋值。
```js
var unusualPropertyNames = {
  "": "An empty string",
  "!": "Bang!"
}
console.log(unusualPropertyNames."");   // 语法错误: Unexpected string
console.log(unusualPropertyNames[""]);  // An empty string
console.log(unusualPropertyNames.!);    // 语法错误: Unexpected token !
console.log(unusualPropertyNames["!"]); // Bang!
```

```js
var foo = {a: "alpha", 2: "two"};
console.log(foo.a);    // alpha
console.log(foo[2]);   // two
//console.log(foo.2);  // SyntaxError: missing ) after argument list
//console.log(foo[a]); // ReferenceError: a is not defined
console.log(foo["a"]); // alpha
console.log(foo["2"]); // two
```

ES2015新增模板字面量，直接打印出多行字符串。
```js
console.log(`Roses are red,
Violets are blue.
Sugar is sweet,
and so is foo.`)
```
## 流程控制与错误处理

### 判断语句的小细节

**其值不是undefined或null的任何对象（包括其值为false的布尔对象）在传递给条件语句时都将计算为true**

这里我们以**Boolean对象**为例子：  
如果需要，作为第一个参数传递的值将转换为布尔值。如果省略或值`0，-0，null，false，NaN，undefined`，或空字符串（`""`），该对象具有的初始值`false`。所有其他值，包括任何对象，空数组（`[]`）或字符串`"false"`，都会创建一个初始值为的对象`true`。


注意不要将基本类型中的布尔值 `true` 和 `false` 与值为 `true` 和 `false` 的 `Boolean` 对象弄混了,**不要在应该使用基本类型布尔值的地方使用 Boolean 对象**。

```js
let un;
//undifined
if (un){
    //这里的代码不会被执行
    console.log(un);
} 

un=null;
//null
if (un){
    //这里的代码不会被执行
    console.log(un);
} 

let b = new Boolean(false);

if (b){
    //这里的代码会被执行
    console.log(b.toString());
} 
let a= new Boolean("false");
if(a){
    //这里的代码会被执行
    console.log(a.toString());//会打印出false
}

let x = false;
if (x) {
    // 这里的代码不会执行
    console.log("false");
}
```
**正确使用途径**
不要用创建 `Boolean` 对象的方式将一个非布尔值转化成布尔值，直接将 `Boolean` 当做转换函数来使用即可，或者使用双重非（`!!`）运算符：
```js
var x = Boolean(expression);     // 推荐
var x = !!(expression);          // 推荐
var x = new Boolean(expression); // 不太好
```

### switch 语句
同java，c++

### 异常处理语句
你可以用 `throw` 语句抛出一个异常并且用 `try...catch` 语句捕获处理它。同Java语法差不多。


## 循环与迭代

### forEach
foreach里面不支持break,continue,
```js
const array1 = ['a', 'b', 'c'];//数组的类型其实也是对象
array1.forEach((element,index) => console.log(element,index));
```

### for...in
for...in 语句循环一个指定的变量来循环一个**对象**所有**可枚举的属性**。JavaScript 会为每一个不同的属性执行指定的语句。


### for...of
for...of 语句在**可迭代对象**（包括`Array、Map、Set、arguments` 等等）上创建了一个循环，对值的每一个独特属性调用一次迭代,**对象属于object，不可迭代**。
![对象不可迭代](./js基础/3.png)
**所以for of 则无法迭代js的object属性值**
```js
let arr = [3, 5, 7];
arr.foo = "hello";

for (let i in arr) {
   console.log(i,arr[i]); 
   // 输出 "0 3", "1 5", "2 7", "foo hello"
}

for (let i of arr) {
   console.log(i); // 输出 "3", "5", "7"
}

// 注意 for...of 的输出没有出现 "hello"
```
![好好理解](./js基础/2.png)
for … in循环将把foo包括在内了，但Array的length属性却不包括在内，所以length还是3.

![好好理解](./js基础/1.png)
这是MDN官方文档的解释，多看几遍就能理解，其实for in会把array作为一个对象来打迎打所有的属性，而for of则只对可迭代的对象进行遍历，也就是这里面的array了，这也能解释为什么foreach也只能打印数组元素了。

两者区别参考了博客[for…in和for…of的用法与区别1](https://blog.csdn.net/IUBKBK/article/details/90962430?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.edu_weight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.edu_weight)和[for…in和for…of的用法与区别2](https://blog.csdn.net/IUBKBK/article/details/90962430?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.edu_weight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.edu_weight)

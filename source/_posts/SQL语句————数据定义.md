---
title: SQL语句————数据定义
date: 2020-09-29 23:38:18
tags: 数据库系统
---
SQL又称结构化查询语句（Structed Query Language）是关系数据库的标准语言，也是一个通用的，功能极强的关系数据库语言。
SQL集**数据查询、数据操纵、数据定义、数据控制**功能于一体。
**目前没有一个数据库系统能支持SQL标准的所有概念和特性。但同时许多软件厂商对SQL基本命令集还进行了不同程度的扩充和修改，又可以支持标准以外的一些功能特性。**
<!--more-->
## 定义模式
在SQL中，模式定义语句如下：
```sql
CREATE SCHEMA <模式名> AUTHORIZATION <用户名>
```
如果没有指定<模式名>，那么<模式名>隐含为<用户名>
要创建模式，调用该命令的用户名必需拥有数据库管理员权限，或者获得了数据库管理员授权的CREATE SCHEMA的权限。

## 删除模式
```sql
DROP SCHEMA <模式名> <CASCADE|RESTRICT>
```
其中CASCADE|RESTRICT必须二选一，两者有不同的作用。
1. CASCADE，级联，表示在删除模式的同时把该模式中所有的数据库对象全部删除。
2. 选择了RESTRICT，限制，表示如果该模式中已经定义了下属的数据库对象，则拒绝该删除语句的执行。

## 基本表的定义与创建
```sql
CREATE TABLE <表名>(
    <列名><数据类型>,[列级完整性约束条件],
    <列名><数据类型>,[列级完整性约束条件],
    <列名><数据类型>,[列级完整性约束条件],
    ...
);
```

## 修改基本表
修改语句主要是通过ALERT TABLE来操作
```sql
ALTER TABLE <表名>
[ADD [COLUMN] <新列名><数据类型> [完整性约束]]
[ADD <表级完整性约束>]
[DROP [COLUMN] <列名> [CASCADE|RESTRICT] ]
[DROP CONSTRAINT <完整性约束名> [CASCADE|RESTRICT]]
[ALER COLUMN<列名><数据类型>];
```

## 删除表
```sql
DROP TABLE <表名> [RESTRICT|CASCADE]
```


## The Python environment
read-evaluate-loop

「读取-求值-输出」循环（英语：Read-Eval-Print
Loop，简称REPL）是一个简单的，交互式的编程环境。这个词常常用于指代一个Lisp的交互式开发环境，但也能指代命令行的模式和例如APL、BASIC、Clojure、F#、Haskell、J、Julia、Perl、PHP、Prolog、Python、R、Ruby、Scala、Smalltalk、Standard ML、Swift、Tcl、Javascript这样的程序语言所拥有的类似的编程环境。这也被称做交互式顶层构件（interactive toplevel）。


## Variables and expressions
To translate a real-world problem into one that can be solved by an algorithm,there are two interrelated tasks.

- Firstly.select the variables,-
- and secondly,find the expressions that relate to these variables.

Variables are labels attached to objects;they are not the object itself.

They are not containers for objects either.
A variable does not contain the object,rather it acts as a pointer or reference to an object.

Pyhton is a dynamically typed language.

Variable names can be bound to different values and types during program execution.

Variables, or more specifically the objects they point to, can change type depending on the values assigned to them.

### Variable scope

Each time a function executes,a new local namespace is created.This represents a local environment that contains the names of the parameters and variables that are assigned by the function.

To resolve a namespace when a function is called,the Python interpreter first searches the local namespace(that is ,the function itself) and if no match is found,it searches the global namespace.This global namespace is the module in which the function was defined.If the name is still not found,it searches the built-in namespace.Finally,if this fails then the interpreter raises a NameError ecception.

### Flow control and iteration


### Overview of data types and objects
**Python contains 12 built-in data types.**

These include four numeric types(int,float,complex,bool),four sequence types(str,list,tuple,range),one mapping type(dict),and two set types.

**All data types in Python are objects.**

Each object in Python has a type,a value,and an identity.The identity of an object acts as a pointer to the object's location in memory.

The type of an object,also known as the object's class,describes the object's internal representation as well as the methods and operations it supports.

Once an instance of an object is created,its identity and type cannot be changed.

We can get the identity of an object by using the built-in function id().This returns an identifying integer and on most systems this refers to its memory location, although you should not rely on this in any of your code.

Mutable object's such as lists can have their values changed.They have methods,such as insert() or append(),that change an objects value.

Immutable objects,such as  strings,cannot have their values changeed,so when we run their methods,they simply return a value rather than change the value of an underlying objet.

### Strings
Strings are immutable sequence objects,with each character representing an element in the sequence.

s.count(substring,[start,end])
Counts the occurrences of a substring with optional start and end parameters.

s.expandtabs([tabsize])
Replaces tabs with spaces.

s.find(substring,[start,end])
Returns the index of the first occurrence of a substring or returns -1 if the substring is not found.

s.isalnum()
Returns True if all characters are alphanumeric,returns False otherwise. # alphanumeric 字母

a.isalpha()
Returns True if all characters are alphabetic,returns False otherwise.

a.isdigit()
Returns True if all characters are digits,returns False otherwise.

s.join(t)
Joins the strings in sequence t.

s.replace(old,new [maxreplace])
Replaces old substring with new substring.

s.strip([characters])
Removes whitespace or optional characters.
Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）或字符序列。
注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
1、strip() 处理的时候，如果不带参数，默认是清除两边的空白符，例如：/n, /r, /t, ' ')。
2、strip() 带有参数的时候，这个参数可以理解一个要删除的字符的列表，是否会删除的前提是从字符串最开头和最结尾是不是包含要删除的字符，如果有就会继续处理，没有的话是不会删除中间的字符的。

s.split([separator],[maxsplit])
Splites a string separated by whitespace or an optional separator.Returns  a list.

'int' object has no attribute 'append'

### Lists
list(s)
Returns a list of the sequence s.

s.append(s)
Appends element x to the end of s.

s.extend(x)
Appends the list x to s.

s.count(x)
Counts the occurrences of x in s.

s.index(x,[start],[stop])
Returns the smallest index,i,where s[i]==x.Can include optional start and stop index for the search.

s.insert(i,e)
Insert x at index i.

s.pop(i)
Returns the element i and removes it from the list

s.remove(x)
Removes x from s.

s.reverse()
Reverses the order of s.

s.sort(key,[reverse])
Sorts s with optional key and reverse

nested 嵌套的
bracket operator 括号运算符。

### Functions as first class objects
In Python,it is not only data types that are treated as objects.Both functions and classes are what are known as first class objects,allowing them to be manipulated in the same ways as built-in data types.

By definition,first class objects are:
- Created at runtime
- Assigned as a variable or in data structure
- Passed as an argument to a function
- Returned as the result of a function

In Python,the term first class object is a bit of a misnomer since it implies some sort of hierarchy,whereas all Python objects are essentially first class.

Since user-defined functions are objects,we can do things such as include them in other objects,sunch as lists.

Functions can also be used as arguments for other functions.

## Higher order functions
Functions that take other functions as arguments, or that return functions, are called higher order functions.

Python3 contains two built-in higher order functions,filter() and map().

In earlier versions of Python,these functions returned lists; in Python3,they return an iterator,making them much more efficient.

The map() funciton provides an easy way to transform each item into an iterable object.

Note that both map and filter perform the identical function as to what can be achieved by list comprehensions.

更推荐使用list comprehensions

Note the difference between the list.sort() method and the sorted built-in function.

list.sort(), a method of the list object,sorts the existing instance of a list **without** copying it.This method changes the target object and returns None. It is an important convention in Python that functions or methods that change the object return None to make it clear that no new object was created and that the object itself was changed.

On the other hand,the sorted built-in function returns a new list.It actually accepts any iterable object as an argument,but it will always retrun a list.Both list sort and sorted take two optional keyword arguments as key.



## Recursive functions
In Python,we can implement a recursive function simply by calling it within its own function body.To stop a recursive function turning into an infinite loop, we need at least one argument that tests for a terminating case to end the recursion.This is sometimes called the base case.

It should be pointed out that recursion is different from iteration.

Although both involve repetition,iteration loops through a sequence of operations,whereas recursion repeatedly calls a function.Both need a selection statement to end.

Technically,recursion is a special case of iteration known as tail iteration,and it is usually always possible to convert an iterative function to a recursive function and vice versa.

The interesting thing about recursive functions is that they are able to describe an infinite object within a finite statement.

虽然两者都涉及重复，但迭代循环执行一系列操作，而递归重复调用一个函数。 两者都需要选择语句才能结束。 从技术上讲，递归是一种称为尾部迭代的迭代的特殊情况，通常总是可以将迭代函数转换为递归函数，反之亦然。 关于递归函数的有趣之处在于它们能够在有限语句中描述无限对象。

### Iteration:
~~~
def iterTest(low,high):
    while low<=high:
        print(low)
        low=low+1
~~~

### recursion:
~~~
def recurTest(low,high):
    if low<high:
        print(low)
        recuTest(low+1,high)
~~~

In general,iteration is more efficient;however,recursive functions are often easier to understand and write.Recursive functions are also useful for manipulating recursive data structures such as linked lists and trees,as we will see.

## Generators and co-routines

We can create functions that do not just return one result,but rather an entire sequence of results,by using the **yield** statement.These functions are called generators.

Python contains generator functions, which are an easy way to create iterators and they are especially useful as a replacement for unworkably long lists. **A generator yields items rather than build lists.**

Building a list to do this calculation takes significantly longer.The performance improvement as a result of using generators is because the values are **generated on demand**,rather than saved as a list in memory.*A calculation can begin before all the elements have been generated and elements are generated only when they are needed.*

In the preceding example, the sum method loads each number into memory when it is needed for the calculation. This is achieved by the generator object repeatedly calling the __next__() special method. Generators never return a value other than None.

生成器永远不会返回除None之外的值。

Typically,generator objects are used in for loops.

We can also create a **generator expression**,which,apart from replacing square brackets with parentheses,uses the same syntax and carries out the same operation as list comprehensions.Generator expressions,however,do not create a list,they create a generator object.This object does not create the data,but rater creates that data on demand.This means that generator objects do not support sequence methods sunch as append() and insert().You can,however,change a generator into a list using the list() function.
~~~
lst1 = [1,2,3,4]
gen1 = (10**i for i in lst1)
for x in gen1: print(x)
100
1000
10000
~~~


## Classes and object programming

Typically,classes are sets of funcitons,variables,and properties.

By organizing our programs around objects and data rather than actions and logic,we have a robust and flexible way to build complex applications.The actions and logic are still present of course,but by embodying them in objects,we have a way to encapsulate functionality,allowing objects to change in very specific ways.This makes our code less error-prone,easier to extend and maintain,and able to model real-world objects.

Classes are created in Python using the class statement.This defines a set of shared attributes associated with a collection of class instances.A class usually consists of a number of methods,class variables,and computed properties.It is important to understand that defining a class does not,by itself,create any instances of that class.To create an instance,a variable must be assined to a class.

The class body consists of a series of statements that execute during the class definition.

The functions defined inside a class are called instance methods.They apply some operations to the class instance by passing an instance of that class as the first argument.This argument is called self by convention,but it can be any legal identifier.

## Special methods
The methods that begin and end with two underscores are called special methods.

Apart from the following exception,special method,are generally called by the Python interpreter rather than the programmer;

for example,when we use the + operator,we actually invoking a call to __add__().

For example,rather than using my_object.__len__() we can use len(my_object) using len() on a string object is actually much faster because it returns the value representing the object's size in memory,rather than making a call to the object's __len__method.

The only special method we actually call inour programs,as common practice,is the **__init__** method,to invoke the initializer of the superclass in our own class definitions.

It is strongly **advised** *not to use the double underscore syntax* for your own objects because of potential current or future conflicts with Python's own special methods.

We may,however,want to implement special methods in custom objects,to give them some of the behavior of built-in types.
~~~
class my_class():
    def __init__(self, greet):
        self.greet = greet
    def __repr__(self):
        return 'a custom object (%r)' % (self.greet)
~~~

When we create an instance of this object and inspect it, we can see we get our customized string representation. Notice the use of the %r format placeholder to return the standard representation of the object. This is useful and best practice, because, in this case, it shows us that the greet object is a string indicated by the quotation marks:
~~~
a = my_class('giday')
a
a custom object('giday')
~~~
## Inheritance
It is possible to create a new class that modifies the behavior of an existing class through inheritance. This is done by passing the inherited class as an argument in the class definition. It is often used to modify the behavior of existing methods, for example:
~~~
class specialEmployee(Employee):
    def hours(self,numHours):
        self.owed += numHours * self.rate*2
        return("%.2f hours worked" % numHours)
~~~
For a subclass to define new class variables,it needs to define an __init__() method,as follows:
~~~
class specialEmployee(Employee):
    def __init__(self,name,rate,bonus):
        Employee.__init__(self,name,rate)#calls the base classes
        self.bonus = bonus
    def hours(self,numHours):
        self.owed += numHours * self.rate*2
        return("%.2f hours worked" % numHours)
~~~


Notice that the methods of the base class are not automatically invoked and it is necessary for the derived class to call them.We can test for class membership using the built-in function `isintance(obj1,obj2)`.This returns true if `obj1` belongs to the class of `obj2` or any class derived from `obj2`.

Within a class definition,it is assumed that all methods operate on the instance,but this is not a requirement.There are,however,other types of methods:`static methods` and `class methods`.


A static method is just an ordinary function that just happens to be defined in a class.

It *does not* perform any operations on the instance and it is defined using the `@staticmethod` class decorator.Static methods cannot access the attributes of an instance,so their most common **usage** is as a convenience to group utility functions together.
Class methods operate on the class itself,not the instance,in the same way that class variables are associated with the classes rather than instances of that class.They are defined using the `@classmethod` decorator,and are distinguished from instance methods in that the class is passed as the first argument.This is named `cls` by convention.
~~~
class Aexp(object):
    base = 2
    @classmethod
    def exp(cls,x):
        return(cls.base**x)
class Bexp(Aexp):
    base = 3
~~~
The class Bexp inherits from the Aexp class and changes the base class variable to 3. We canrun the parent class's exp() method as follows:
~~~
Bexp.exp(2)
9
~~~

Because a subclass inherits all the same fetures of its parent there is the potential for it to break inherited methods.Using class methods is a way to define exactly what methods are run.

## Data encapsulation and properties

Unless otherwise specified,all attributes and methods are accessible without restriction.This may cause problems when we are building object-oriented applications where we may want to hide the internal implementation of an object.This can lead to namespace conflicts between objects defined in derived classes with the base class.

To prevent this,the methods we define private attributes with **have a double underscore**,sunch as `__privateMethod()` .These method names are automatically changed `__Classname__privateMethod` to prevent name conflicts with methods defined in base classes.

Be aware that **this does not strictly hide private attributes**, rather it just provides a mechanism for **preventing name conflicts**.

It is recommended to use private attributes when using a class **property** to define mutable attributes.A property is a kind of attribute that rather than returning a stored value,computes its value when called.
~~~
class Bexp(Aexp):
    __base = 3
    def __exp(self):
    return(x**cls.base)
~~~
## Operations and expressions
There are number of operations that are common to all data types.For example, all data types, and generally all objects, can be tested for a truth value in some way.

#### The following are values that Python considers `False`:
- The None type
- False
- An integer, float, or complex zero
- An empty sequence or mapping
- An instance of a user-defined class that defines a __len__() or __bool__() method that returns zero or False

All other values are considered `True`.
## Boolean operations

Both the `and` operator and the `or` operator use "short-circuiting" when evaluating an expression.This means Python will only evaluate an operator if it needs to.

## Comparison and Arithmetic operators
For collection objects, these operators compare the number of elements and the equivalence operator `==`  returns True if each collection object is structurally equivalent, and the value of each element is identical.

## Membership, identity, and logical operations
Membershi operators(`in`,`not in`) test for variables in sequences,such as lists or strings do what you would expect,x in y returns True if a variable x is found in y.

The `is` operator compares object identity.

For example, the following snippet shows contrast equivalence with object identity:
~~~
x=[1,2,3];y=[1,2,3]
x==y #equivalence
True

x is y #object identity
False
x = y # assignment
x is y
True
~~~

## Built-in data types

Python data types can be divided into three categories:**numeric**,**sequence**,and **mapping**.

There is also the None object that represents a Null,or absence of a value.It should not be forgotten either that other objects such as classes,files,and exceptions can also properly be considered types;however,they will not be considered here.

Every value in Python has a data type. Unlike many programming languages, in Python you do not need to explicitly declare the type of a variable. Python keeps track of object types internally.

Category  | Name | Description
---|---|---
None  | None |  The null object.
Sequences | str| string of characters.
  Sequences | list |List of arbitrary objects. 
Sequences|Tuple|Group of arbitrary items.|
  Sequences| range |Creates a range of integers.
Mapping     |dict  |Dictionary of key-value pairs.
Mapping| set|   Mutable,unordered collection of unique items.
  Mapping|frozenset   |Immutable set
  
### None type

The None type is immutable and has one value, None. It is used to represent the absence of a value. It is returned by objects that do not explicitly return a value and evaluates to False in Boolean expressions. It is often used as the default value in optional arguments to allow the function to detect whether the caller has passed a value.

### Numeric Types
All numeric types, apart from bool, are signed and they are all immutable.

The integer type, int, represents whole numbers of unlimited range. Floating point numbers are represented by the native double precision floating point representation of the machine.

Complex numbers are represented by two floating point numbers. They are assigned using the j operator to signify the imaginary part of the complex number, for example:
`a = 2+3j`
 We can access the real and imaginary parts with `a.real` and `a.imag`, respectively.

### Representation error
It should be noted that the native double precision representation of floating point numbers leads to some unexpected results.
~~~
>>> 1-0.9
0.09999999999999998
>>> 1-0.9==0.1
False
~~~

This is a result of the fact that **most decimal fractions are not exactly representable as a binary fraction**,which is how most uderlying hardware represents floating point numbers.

For algorithms or applications where this may be an issue,Python provides a **decimal module**.

This module allows for the exact representation of decimal numbers and facilitates greater control properties such as rounding behavior,number of sigificant digits,and precision.

It defines two objects,a Decimal type,representing decimal numbers,and a Context type,representing various computational parameters such as precision,rounding,and error handling.

In addition, Decimal objects also have several methods for mathematical operations, such as natural exponents, x.exp(), natural logarithms, x.ln(), and base 10 logarithms, x.log10().

## Sequences
Lists and tuples are sequences of arbitrary objects, strings are sequences of characters. String, tuple, and range objects are immutable.

Note that for the immutable types, any operation will only return a value rather than actually change the value.

Method|Description
---|---
len(s)|Number of elements in s
min(s,[,default=obj,key=func])|The minimum value in s (alpabetically for strings)
max(s,[,default=obj,key=func])|Maximum value in s(alphabetically for strings)
sum(s,[,start=0])|The sum of elements (returns TypeError if s is not numeric)
all(s)|Returns True if all elements in `s` are True(that is,not `0`,`False`,or `Null`)
any(s)|Check whether any item in `s` is True

In addition, all sequences support the following operations:

Operation|Description
---|---
s+r|Concatenates two sequences of the same type
s*n|Make n copies of s, where n is an integer
v1,v2,...,vn=s|Unpacks n variables from s to v1,v2,and so on
s[i]|Indexing-returns element i of s
s[i:j:stride]|Slicing returns elements between i and j with optional stride
x in s|Returns True if element x is in s
x not in s|Returns true if element x is not in s

## Tuples
Tuples are immutable sequences of arbitrary objects.They are indexed by integers greater than zero.Tuples are hashable,which means we can sort lists of them and they can be used as keys to dictionaries.

Syntactically,tuples are just a comma-separated sequence of values;however,it is common practice to enclose them in parentheses:
`tpl= ('a', 'b', 'c')`

It is important to remember to use a trailing comma when creating a tuple with one element, for example:
 	`t = ('a',)`
 	
Without the trailing comma, this would be interpreted as a string.

We can also create a tuple using the built-in function tuple(). With no argument, this creates an empty tuple. If the argument to tuple() is a sequence then this creates a tuple of elements of that sequence, for example:
~~~
>>> tuple('sequence')
('s', 'e', 'q', 'u', 'e', 'n', 'c', 'e')
~~~

Most operators, such as those for slicing and indexing, work as they do on lists. However, because tuples are immutable, trying to modify an element of a tuple will give you a TypeError. We can compare tuples in the same way that we compare other sequences, using the `==`, `>` and `<` operators.

An important use of tuples is to allow us to assign more than one variable at a time by placing a tuple on the left-hand side of an assignment,for example:
~~~
>>> l=['one','two']
>>> x,y = l #assigns x and y to 'one' and 'two' respectively
~~~
We can actually use this multiple assignment to swap values in a tuple, for example:
~~~
x,y=y,x #x='two' and y = 'one'
~~~
A ValueError will be thrown if the number of values on each side of the assignment are not the same.

## Dictionaries

Dictionaries are arbitrary collections of objects indexed by numbers,strings,or other immutable objects.Dictionaries themselves are mutable;however,their index keys must be immutable.

Method |Description
---|---
len(d)|Number of items in d.
d.clear()|Removes all items from d
d.fromkeys(s,[,value])|Returns a new dictionary with keys from sequence s and values set to value.
d.get(k,[,v])|Returns d[k] if found,or else returns v,or None if v is not given.
d.items()|Returns a sequence of key:value pairs in d.
d.keys()|Returns a sequence of keys in d.
d.pop(k [,default])|Returns d[k] and removes it from d. If d[k] is not found,it returns default or raises KeyError.
d.popitem()|Removes a random key:value pair from d and returns it as a tuple.
d.setdefault(k [,v])|Returns d[k].If d[k] is not found,it returns v and sets d[k] to v.
d.update(b)|Adds all objects from b to d.
d.values()|Returns a sequence of values in d.

The relationship between the time an algorithm takes to run compared to the size of its input is often referred to as its time complexity.
算法运行的时间与其输入的大小之间的关系通常被称为其时间复杂度。

In contrast to the list object, when the in operator is applied to dictionaries, it uses a **hashing algorithm** and this has the effect of the increase in time for each lookup almost independent of the size of the dictionary. This makes dictionaries extremely useful as a way to work with large amounts of indexed data.

Notice when we print out the key:value pairs of the dictionary it does so in no particular order.This is not a problem since we use specified keys to look up each dictionary value rather than an ordered sequence of integers as is the case for strings and lists.

## Sorting dictionaries
The sorted() method has two optional arguments that are of interest: key and reverse.

The key argument has nothing to do with the dictionary keys, but rather is a way of passing a function to the sort algorithm to determine the sort order.

We use the `__getitem__` special method to sort the dictionary keys according to the dictionary values:
~~~
>>> d
{'a': 68, 'b': 1}
>>> sorted(list(d),key=d.__getitem__)
['b', 'a']
~~~
Essentially,what the preceding code doing is for every key in d to use the corresponding value to sort.We can also sort the values according to the sorted order of the dictionary keys.However,since dictionaries do not have a method to return a key by using its value,the equivalent of the `list.index` method for lists,using the optional key argument to do this is a little tricky. An alternative approach is to use a list comprehension,as the following example demonstrates:
~~~
>>> [value for (key,value) in sorted(d.items())]
[68, 1]
~~~

The sorted() method also has an optional reverse argument,and unsurprisingly,this does exactly what it says,reverses the order of the sorted list,for example:
~~~
>>> sorted(list(d),key=d.__getitem__,reverse=True)
['a', 'b']
~~~

## Dictionaries for text analysis
A common use of dictionaries is to count the occurrences of like items in a sequence; a typical example is counting the occurrences of words in a body of text.
~~~
def wordcount(fname):
    try:
        fhand=open(fname)
    except:
        print('File cannot be opened')
        exit()

    count=dict()
    for line in fhand:
        words=line.split()
        for word in words:
            if word not in count:
                count[word]=1
            else:
                count[word]+=1
    return(count)
~~~
This will retrun a dictionary with an element for each unique word in the text file.
Dictionary comprehensions work in an identical way to the list comprehensions.

## Sets
Sets are themselves mutable,we can add and remove items from them;however,the items themselves must be immutable.

An important distinction with sets is that they cannot contain duplicate items.

Sets are typically used to perform mathematical operations such as intersection, union, difference, and complement.

交并差补
intersection，union，difference，complement

Unlike sequence types,set types do not provide andy indexing or slicing operations.There are also no keys associated with values,as is the case with dictionaries.

There are two types of set objects in Python, the mutable set object and the immutable frozenset object. Sets are created using comma-separated values within curly braces.
By the way, we cannot create an empty set using a={}, because this will create a dictionary. To create an empty set, we write either a=set() or a=frozenset().
Method|Operators|Description
---|---|---
len(s)||Returns the number of elements in s.
s.copy()||Returns a shallow copy of s
s.difference(t)		|s-t-t2-...|Returns a set of all items in s but not in t
s.intersection(t)||Returns a set of all items in both t and s
s.isdisjoint(t)||Returns True if s and t have no items in common
s.issubset(t)|s<=t s<t(s!=t)|Returns True if all items in s are also in t
s.issuperset(t)| s>=t s>t(s!=t)|Returns True if all items in t are also in s
s.symmetric_difference(t)|s^t|Returns a set of all items that are in s or t,but not both
s.union(t)|s\|t1\|t2\|..|Returns a set of all items in s or t

The parameter `t` can be any Python object that supports iteration and all methods are available to both set and frozenset objects.
It is important to be aware that the operator versions of these methods require their arguments to be sets, whereas the methods themselves can accept any iterable type. For example, `s - [1,2,3]`, for any set s, will generate an unsupported operand type. Using the equivalent `s.difference([1,2,3])` will return a result.

Mutable set objects have additional methods
Method|Description
---|---
s.add(item)|Adds item to s.Has no effect if item is already present.
s.clear()|Remove all items from s
s.difference_update(t)|Removes all items in s that are also in t
s.discard(item)|Removes item from s.
s.intersection_update(t)|Removes all items from s that are not in the intersection of s and t.
s.pop()|Returns and removes an arbitrary item from s
s.remove(item)|Removes item from s
s.symmetric_difference_update(t)|Removes all items from s that are not in the symmetric difference of s ant t.
s.update(t)|Adds all the items in an iterable object t to s.

Notice that the set object does not care that its members are not all of the same type,as long as they are all immutable.
Hashable types all have a hash value that does not change throughout the lifetimeof the instance.**All built-in immutable types are  hashable**.**All built-in mutable types are not hashable**,so they cannot be used as elements of sets or keys to dictionaries.

## Immutable sets
Python has an immutable set type called frozenset.It works pretty much exactly like set apart from not allowing methods or operations that change values such as the `add()` or `clear()` methods.

There are several ways that this immutability can be useful. For example, since normal sets are mutable and therefore not hashable, they cannot be used as members of other sets. The frozenset, on the other hand, is immutable and therefore able to be used as a member of a set.
Also the immutable property of frozenset means we can use it for a key to a dictionary.

## Modules for data structures and algorithms
So far, we have looked at the built-in datatypes of strings, lists, sets, and dictionaries as well as the decimal and fractions modules. They are often described by the term abstract data types (ADTs).ADTs can be considered as mathematical specifications for the set of operations that can be performed on data.

## Collections
The collections module provides more specialized,high,performance alternatives for the built-in data types as well as a utility function to create named tuples.

Datatype or operation|Description
---|---
namedtuple()|Creates tuple subclasses with named fields.
deque|Lists with fast appends and pops either end.
ChainMap|Dictionary like class to create a single view of multiple mappings.
Counter |Dictionary subclass for counting hashable objects.
OrderedDict|Dictionary subclass that remembers the entry order.
defaultdict|Dictionary subclass that calls a function to supply missing vlaues.
UserDict </br> UserList</br>UserString|These three data types are simply wrappers for their underlying base classes.Their use has largely been supplanted by the ability to subclassses their respective base classes directly.Can be used to access the underlying object as an attribute.

## Deques
Double-ended queues,or deques,are list-like objects that support thread-safe,memory-effcient appends.

Deques are mutable and support some of the operations of lists,such as indexing.Deques can be assigned by index.However,we cannot directly slice deques.

The major advantage of deques over lists is that inserting items at the beginning of a deque is much faster than inserting items at the beginning of a list,although inserting items at the end of a deques is very slightly slower than the equivalent operation on a list.

Deques are thread,safe and can be serialized using the pickle module.

We can also use the rotate(n) method to move and rotate all items of n steps to the right for positive values of the integer n,or left for negative values of n the left,using positive integers as the argument
~~~
from collections import deque
>>> dq
deque(['w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g'])
>>> dq.rotate(2)
>>> dq
deque(['f', 'g', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e'])
>>> dq.rotate(2)
>>> dq
deque(['d', 'e', 'f', 'g', 'w', 'x', 'y', 'z', 'a', 'b', 'c'])
>>> 
~~~
a simple way to return a slice of a deque, as a list, which can be done as follows:
~~~
>>> import itertools
>>> list(itertools.islice(dq,3,9))
['z', 'a', 'b', 'c', 'd', 'e']
>>> 
~~~
The `itertools.islice` method works in the same way that slice works on a list,except rather than taking a list for an argument,it takes an iterable and returns slelected values,by start and stop indices,as a list.

`maxlen` a optional parameter that restricts the size of deque.
This makes  it ideally suited to a data structure known as a *circular buffer*.This is a fixed-size structure that is effectively connected end to end and they are typically used for buffering data streams.
~~~
>>> dq2 = deque([],maxlen=3)
>>> for i in range(6):
...     dq2.append(i)
...     print(dq2)
... 
deque([0], maxlen=3)
deque([0, 1], maxlen=3)
deque([0, 1, 2], maxlen=3)
deque([1, 2, 3], maxlen=3)
deque([2, 3, 4], maxlen=3)
deque([3, 4, 5], maxlen=3)
>>>
~~~
In this example,we are populating from the right and consuming from the left.Notice that once the buffer is full,the oldest values are consumed first,and values are replaced from the right.

## ChainMaps
`collection.chainmap` class provides a way to link a number of dictionaries,or other mappings,so that they can be treated as one object.

In addition,there is a maps attribute,a `new_child()` method,and a parents property.

The uderlying mappings for ChainMap objects are stored in a list and are accessible using the `maps[i]` attribute to retrieve `ith` dictionary.

ChainMaps are an ordered list of dictionaries.ChainMap is useful in applications where we are using a number of dictionaries containing related data.

The advantage of using ChainMaps, rather than just a dictionary, is that we **retain previously set values**. Adding a child context overrides values for the same key, but it does not remove it from the data structure. This can be useful for when we may need to keep a record of changes so that we can easily roll back to a previous setting.

We can retrieve and change any value in any of the dictionaries by providing the map() method with an appropriate index. This index represents a dictionary in the ChainMap. Also, we can retrieve the parent setting, that is, the default settings, by using the parents() method:
~~~
from collections import ChainMap
>>> cm2.maps[0]={'theme':'desert','showIndex':False}
>>> cm2
ChainMap({'theme': 'desert', 'showIndex': False}, {'theme': 'Default', 'language': 'eng', 'showIndex': True, 'showFooter': True})
>>> cm2['showIndex']
False
>>> cm2.parents
ChainMap({'theme': 'Default', 'language': 'eng', 'showIndex': True, 'showFooter': True})
>>>
~~~
## Counter objects
Counter is a subclass of a dictionary where each dictionary `key` is a hashable object and the associated value is an integer count of that object.
There are three ways to initialize a counter.

We can pass it any sequence object,a dictionary of `key:value` pairs,or a tuple of the format(`object = value,...`),for example:
~~~
>>> from collections import Counter
>>> c1 = Counter('anysequence')
>>> c2 = Counter({'a':1,'c':1,'e':3})
>>> c1
Counter({'e': 3, 'n': 2, 'a': 1, 'y': 1, 's': 1, 'q': 1, 'u': 1, 'c': 1})
>>> c2
Counter({'e': 3, 'a': 1, 'c': 1})
>>> 
~~~
We can also create an empty counter object and populate it by passing its update method an iterable or a dictionary, for example:
~~~
>>> ct = Counter()
>>> ct.update('abca')
>>> ct
Counter({'a': 2, 'b': 1, 'c': 1})
>>> ct.update({'a':3})
>>> ct
Counter({'a': 5, 'b': 1, 'c': 1})
>>>
~~~
Notice how the update method adds the counts rather than replacing them with new values. Once the counter is populated, we can access stored values in the same way we would for dictionaries, for example:
~~~
>>> ct
Counter({'a': 5, 'b': 1, 'c': 1})
>>> for item in ct:
...     print('%s:%d'%(item,ct[item]))
... 
a:5
b:1
c:1
~~~
The most notable difference between counter objects and dictionaries is that counter objects return a zero count for missing items rather than raising a key error,for example:
~~~
>>> ct['x']
0
>>>
~~~
We can create an iterator out of a Counter object by using its `elements()` method.This returns an iterator where counts below one are not included and the order is not guaranteed.
~~~
ct.update({'a':-3,'b':-2,'d':3,'e':2})
sorted(ct.elements())# returns a sorted list from the iterator
['d', 'd', 'd', 'e', 'e'] 
~~~
Two other Counter methods worth mentioning are `most_common()` and `subtract()` . 

The most common method takes a positive integer argument that determines the number of most common elements to return. Elements are returned as a list of (key ,value) tuples. The subtract method works exactly like update except instead of adding values, it subtracts them.
## Ordered dictionaries
When we test to see whether two dictionaries are equal, this equality is only based on their keys and values; however, with an OrderedDict , the insertion order is also considered An equality test between two `OrderedDicts` with the same keys and values but a different insertion order will return False :
~~~
>>> from collections import OrderedDict
>>> od1=OrderedDict()
>>> od1['one']=1
>>> od1['two']=2
>>> od2=OrderedDict()
>>> od2['two']=2
>>> od2['one']=1
>>> od1==od2
False
~~~
Similarly, when we add values from a list using update , the OrderedDict will retain the same order as the list. This is the order that is returned when we iterate the values, for example:
~~~
>>> kvs = [('three',3),('four',4),('five',5),('six',6)]
>>> od1.update(kvs)
>>> for k,v in od1.items():print(k,v)
... 
one 1
two 2
three 3
four 4
five 5
six 6
~~~

The OrderedDict is often used in conjunction with the sorted method to create a sorted dictionary.
~~~
>>> od3 = OrderedDict(sorted(od1.items(),key = lambda t:(4*t[1])-t[1]**2))
>>> od3.values()
odict_values([6, 5, 4, 1, 3, 2])
>>> od3
OrderedDict([('six', 6), ('five', 5), ('four', 4), ('one', 1), ('three', 3), ('two', 2)])
>>>
~~~

## defaultdict
The defaultdict object is a subclass of dict and therefore they share methods and operations. It acts as a convenient way to initialize dictionaries. With a dict , Python will throw a KeyError when attempting to access a key that is not already in the dictionary. The defaultdict overrides one method, __missing__(key) , and creates a new instance variable, default_factory . With defaultdict , rather than throw an error, it will run the function, supplied as the default_factory argument, which will generate a value. A simple use of defaultdict is to set default_factory to int and use it to quickly tally the counts of items in the dictionary, for example:
~~~
>>> def isprimary(c):
...     if(c=='red') or (c=='blue') or (c=='green'):
...             return True
...     else:
...             return False
... 
>>> from collections import defaultdict
>>> dd2=defaultdict(bool)
>>> dd2
defaultdict(<class 'bool'>, {})
>>> words=['blue','green','red','yellow']
>>> for word in words:dd2[word]=isprimary(word)
... 
>>> dd2
defaultdict(<class 'bool'>, {'blue': True, 'green': True, 'red': True, 'yellow': False})
~~~
## Named tuples
The namedtuple method returns a tuple-like object that has fields accessible with named indexes as well as the integer indexes of normal tuples.
~~~
>>> from collections import namedtuple
>>> space = namedtuple('space','x y z')
>>> space
<class '__main__.space'>
>>> s1 = space(x=2.0,y=4.0,z=10)#we can also use space(2.0,4.0,10)
>>> s1.x*s1.y*s1.z#calculates the volume
80.0
>>>
~~~

The namedtuple method take two optional Boolean arguments, verbose and rename . When verbose is set to True then the class definition is printed when it is built. This argument is depreciated in favor of using the __source attribute. When the rename argument is set to True then any invalid field names will be automatically replaced with positional arguments. As an example, we attempt to use def as a field name. This would normally generate an error, but since we have assigned rename to True , the Python interpreter allows this. However, when we attempt to look up the def value, we get a syntax error, since def is a reserved keyword. The illegal field name has been replaced by a field name created by adding an underscore to the positional value:
~~~
>>> space2 = namedtuple('space2','x def z',rename=True)
>>> s1 = space2(3,4,5)
>>> s.def
  File "<stdin>", line 1
    s.def
        ^
SyntaxError: invalid syntax
>>> s1
space2(x=3, _1=4, z=5)
>>>
~~~

In addition to the inherited tuple methods, the named tuple also defines three methods of its own, _make() , asdict() , and _replace . These methods begin with an underscore to prevent potential conflicts with field names. The _make() method takes an iterable as an argument and turns it into a named tuple object, for example:
~~~
>>> sl=[4,5,6]
>>> space._make(sl)
space(x=4, y=5, z=6)
>>>
~~~
The `_asdict` method returns an OrderedDict with the field names mapped to index keys and the values mapped to the dictionary values, for example:
~~~
>>> s1._asdict()
OrderedDict([('x', 3), ('_1', 4), ('z', 5)])
~~~
The _replace method returns a new instance of the tuple, replacing the specified values,
for example:
~~~
In[82]: s1._replace(x=7, z=9)
Out[82]: space2 (x=7, _l=4, z=9)
~~~
## Array
The array module defines a datatype array that is similar to the list datatype except for the constraint that their contents must be of a single type of the underlying representation, as is determined by the machine architecture or underlying C implementation.

The array objects support the following attributes and methods:
Attributed or method Description
Method|Description
---|---
a.typecode|The typecode character used to create the array.
a.itemsize|Size,in bytes of items stored in the array.
a.append(x)|Appends item x to the end of the array.
a.buffer_info()|Returns the memory location and length of the buffer userd to store the array.
a.byteswap()|Swaps the byte order of each item.Used for writing to a machine or file with a different byte order.
a.count(x)|Returns the number of occurrences of x in a .
a.extend(b)|Appends any iterable, b , to the end of array a .
a.frombytes(s)|Appends items from a string, s , as an array of machine values.
a.fromfile(f, n)|Reads n items, as machine values, from a file object, f , and appends them to a . Raises an EOFError if there are fewer than n items in n .
a.fromlist(l)|Appends items from list l .
a.fromunicode(s)|Extends a with unicode string s . Array a must be of type u or else ValueError is raised.
index(x)|Returns the first (smallest) index of item x.
a.insert(i, x)|Inserts item x before index i.
a.pop([i])|Removes and returns items with index i . Defaults to the last item (i = -1) if not specified.
a.remove(x)|Removes the first occurrence of item x .
a.reverse()|Reverses the order of items.
a.tobytes()|Convert the array to machine values and returns the bytes representation.
a.tofile(f)|Writes all items, as machine values, to file object f.
a.tolist()|Converts the array to a list.
a.tounicode()|Convert an array to unicode string. The array type must be 'u' or else a ValueError is raised.


Array objects support all the normal sequence operations such as indexing, slicing, concatenation, and multiplication.

因为我们对节省空间感兴趣，也就是说，我们正在处理大型数据集和有限的内存大小，所以我们通常在数组上执行就地操作，并且只在需要时创建副本。 通常，枚举用于对每个元素执行操作。 在下面的代码片段中，我们执行向数组中的每个项添加一个的简单操作：
应该注意的是，当对创建列表的数组（例如列表推导）执行操作时，首先使用数组的内存效率增益将被否定。 当我们需要创建一个新的数据对象时，解决方案是使用生成器表达式来执行操作，例如：
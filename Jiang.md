This is a brief supplementary notes from Jiang's introduction to Python at https://zhuanlan.zhihu.com/p/21332075

# Day 1:
## some good libraries:
* NumPy - used to handle large arrays
* SciPy - containing many useful scientific functions
* Matplotlib - used to plot
* Pandas - to analyze large datasets, in particular time series

## basic data structures:
* tuple  `b=(1,2.5,'data')`
* list   `c=[1,2.5,'data')`   
* dict   `d={'Name': 'Kobe','Country': 'US'}`
* set    `e=set(['u','d','ud','d','du'])`  
- set is an unordered collection object of unique objects
- can have multi-dimensional list, but not the same as matrix, cannot use matrix operation on it

## Python features: pass by address. 
For example:
```
x= [1, 2, 3, 4]
y = x
y[0] = 5
print(x)

x= [1,2,3,4]
z=x.copy()
z[0]=5
print(x)
```
The output will be `[5,2,3,4] , [1,2,3,4]`

## Reading/Writing files
use the built-in open() function. e.g. `file_object = open(filename, mode)` 

The mode can be: 'r' only be read; 'w' only writing; 'a' opens the file for appending; 'r+' both for reading and writing
(check the picture on the desktop for details)

* it is easy to forget closing the dile. So it is better to use the with statement:
```
with open("humpty.txt") as f:
    .......
```
This ensures that the file will be properly closed, even if an error occurs during its reading.

# Day 2:


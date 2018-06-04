This is a brief supplementary notes from Jiang's introduction to Python at https://zhuanlan.zhihu.com/p/21332075

# Day 1 (Foundamentals):
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

# Day 2 (Scientific computing):
## import modules example
```
import timeit

def fun1(x,y):
    return x**2+y**3

t_start=timeit.default_timer()
z=fun1(109.2,367.1)
t_end=timeit.default_timer()

cost=t_end-t_start
print('Time cost of fun1 is %f' %cost)  #Notice the %f  print format
```
## Numpy
Numpy provides a new data type: ndarray(N-dimensional Array)

The advantage of arrays than lists or tuples is that 1) they only store same-type objects 2) less memory 3) powerful extentions to indexing

Example:
```
In [1]: np.array([2, 3, 6, 7])
Out[1]: array([2, 3, 6, 7])
In [2]: np.array([2, 3, 6, 7.])
Out[2]: array([2., 3., 6., 7.]) #homogeneous
In [3]: np.array([2, 3, 6, 7+1j])
Out[3]: array([2.+0.j, 3.+0.j, 6.+0.j, 7.+1.j])
```


### create equal-spaced arrays
very useful in generating plots
```
#arange
np.arange(10,100,20,dtype=float) #start,stop,step,data type
#linspace
np.linspace(0.,2.5,5) #start,stop,how many intervals
```
### matrix by multi-D array
```
a=np.array([[1,2,3],[4,5,6]])
a.shape  #number of rows,columns etc.
a.ndim   #number of dimensions
a.size   #total number of elements
```
### shape changing
```
import numpy as np
a=np.arange(0,20,1)  #1-D
b=a.reshape((4,5))   #4 rows, 5 columns
c=a.reshape((20,1))  # 2 dimension
d=a.reshape((-1,4))  #-1: auto determine
#a.shape=(4,5)  #change the shape of a 
```
Size (N,),(N,1) and (1,N) are different! (1-D, 2-D(n rows), 2-D(n columns))

### other functions
```
np.dot(a,b)  #matrix multiply for high-D, inner product for 1-D
np.zeros(3)
np.zeros((2,2),complex)
np.ones((2,3))
np.random.rand(2,4)   #uniformly between 0&1
np.random.randn(2,4)  #standard normal(gaussian) with mean 0 and variance 1
```
### array slicing
indices in the format start:stop means from start, but not including stop.
#### 1-D
```
a=np.array([0,1,2,3,4])
a[1:3]   #1,2 since index start from 0
a[:3]    #0,1,2  if start omitted, by default is 0
a[1:]    #1,2,3,4  if stop omitted, by default is the array length
a[1:-1]  #1,2,3  until 1 from the end
# can also speficy step(after a second colon)
a[::2]  #0,2,4
a[1:4:2] #1,3
a[::-1] #4,3,2,1,0    use -1 to reverse an array
```
#### 2-D

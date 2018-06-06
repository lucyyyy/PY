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
same rule in 1-D, except for the trailing colons doesn't need to be given explicitly:
```
a=np.arange(12); a.shape=(3,4)
a[2,:]
a[2]  # This is the same as a[2,:]
```
### Copies and views
For a standard list, taking a slice creates its copy. (Changing slice would not change its original value)

For a Numpy array, taking a slice creates a view on the original array. Both arrays refers to the same memory, so would change simultaneously. To avoid this, just use .copy() method
```
a=np.arange(5)
b=a[2:].copy()
```
### Matrix multiplication
The * operator represents **elementwise multiplication**  
The dot() function is matrix multiplication, and it supports **matrix-vector multiplication**
```
A=np.array([[1,2],[3,4]])
x=np.array([10,20])
np.dot(A,x)  #50,110
np.dot(x,A)  #70,100
```
## SciPy
Basic structure:  
scipy.intergrate->integration and ordinary differential equations  
scipy.linalg -> linear algebra  
scipy.ndimage -> image processing  
scipy.optimize -> optimisation and root finding  
scipy.special -> special functions  
scipy.stats -> statistical functions  
To load a paticular module, use `from scipy import linalg`  

## sympy
Symbolic computation, can be useful for calculating explicit solutions to equations, integrations and so on.
```
import sympy as sy

x=sy.Symbol('x')
y=sy.Symbol('y')
a,b=sy.symbols('a b')

f=x**2+y**2-2*x*y+5
print(f)   #x**2 - 2*x*y + y**2 + 5

g=x**2+2-2*x+x**2-1  #auto simplify
print(g)   #2*x**2 - 2*x + 1
```
can have other uses such as **solve equations, integration, differentiation**
![use of symbol 1](http://mmbiz.qpic.cn/mmbiz/nliazs07woqkq6aoUyEKB0WlJqwfFpSzZj1dDKjaKwib4nxazKR4s4icOSZGjwUWqibKYM3IQvG92NEuaXhvYpict7Q/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)

![use of symbol 2]
(http://mmbiz.qpic.cn/mmbiz/nliazs07woqkq6aoUyEKB0WlJqwfFpSzZYENwgrc7Tb5tKGffXP5H1aibxo8H8y6O7ABib6l2EsgicnQZ1acxEED0Q/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)

![use of symbol 3](http://mmbiz.qpic.cn/mmbiz/nliazs07woqkq6aoUyEKB0WlJqwfFpSzZJ7T6lw1RncOMmqicrxDN3ibW48264WCI3OH5k5cs7GnkL2lvVJbiaqLAg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)

## Matplotlib
use it to plot. Add by ` from matplotlib import pyplot as plt` or `import matplotlib.pyplot as plt`

Application:
```
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,201)
y=x**0.5
plt.plot(x,y)

plt.figure(figsize=(3,3))
plt.plot(x,x**0.3,'r--')
plt.plot(x,x-1,'k--')
plt.plot(x,np.zeros_like(x),'k-')

# multiple plotting, legends,labels and title
plt.figure(figsize=(4,4))
for n in range(2,5):
    y=x**(1/n)
    plt.plot(x,y,label='x^(1/'+str(n)+')')
plt.legend(loc='best')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.xlim(-2,10)
plt.title('Multi-plot e.g. ', fontsize=18)

# Subplots
def pffcall(S,K):
    return np.maximum(S-K,0.0)
def pffput(S,K):
    return np.maximum(K-S,0.0)

S=np.linspace(50,151,100)
fig=plt.figure(figsize=(12,6))
sub1=fig.add_subplot(121) #col, row, num
sub1.set_title('Call',fontsize=18)
plt.plot(S,pffcall(S,100),'r-',lw=4)
plt.plot(S,np.zeros_like(S),'black',lw=1)
sub1.grid(True)
sub1.set_xlim([60,120])
sub1.set_ylim([-10,40])

sub2=fig.add_subplot(122)
sub2.set_title('Put',fontsize=18)
plt.plot(S,pffput(S,100),'r-',lw=4)
plt.plot(S,np.zeros_like(S),'black',lw=1)
sub2.grid(True)
sub2.set_xlim([60,120])
sub2.set_ylim([-10,40])

# Adding texts to plots
from scipy.stats import norm

def call(S,K=100,T=0.5,vol=0.6,r=0.05):
    d1=(np.log(S/K)+(r+0.5*vol**2)*T)/np.sqrt(T)/vol
    d2=(np.log(S/K)+(r-0.5*vol*2)*T)/np.sqrt(T)/vol
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

def delta(S,K=100,T=0.5,vol=0.6,r=0.05):
    d1=(np.log(S/K)+(r+0.5*vol**2)*T)/np.sqrt(T)/vol
    return norm.cdf(d1)

S=np.linspace(40,161,100)
fig=plt.figure(figsize=(7,6))
ax=fig.add_subplot(111)
plt.plot(S,(call(S)-call(100)),'r',lw=1)
plt.plot(100,0,'ro',lw=1)
plt.plot(S,np.zeros_like(S),'black',lw=1)
plt.plot(S,call(S)-delta(100)*S-(call(100)-delta(100)*100),'y',lw=1)

ax.annotate('$/Delta$ hedge',xy=(100,0),xytext=(110,-10),arrowprops= \
            dict(headwidth=3,width=0.5,facecolor='blacl',shrink=0.05))
ax.annotate('Original call',xy=(120,call(120)-call(100)),xytext= \
            (130,call(120)-call(100)),arrowprops=dict(headwidth=10, \
            width=3,facecolor='cyan',shrink=0.05))
plt.grid(True)
plt.xlim(40,160)
plt.xlabel('Stock price',fontsize=18)
plt.ylabel('Profits',fontsize=18)

# 3D plot of a function with 2 variables
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D

x,y=np.mgrid[-5:5:100j,-5:5:100j]
z=x**2+y**2
fig=plt.figure(figsize=(8,6))
ax=plt.axes(projection='3d')
surf=ax.plot_surface(x,y,z,rstride=1,cmap=cm.coolwarm,cstride=1,\
                     linewidth=0)
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.title('3D plot of $z=x^2+y^2$')
```
# Day 3 (Statistics)
## Stochastic and Monte Carlo
### Generate random mumber
```
import numpy.ramdom as npr

X=npr.standard_normal((5000))
Y=npr.normal(1,1,(5000))
Z=npr.uniform(-3,3,(5000))
W=npr.lognormal(0,1,(5000))
```
### Monte Carlo
MC is widely used in statistical mechanics, quantum physics, financial derivatives pricing and 
risk management.
#### An example in Option Pricing
```
from scipy.stats import norm

S=100;K=100;T=1
r=0.05;vol=0.5

I=10000 #MC paths
Z=npr.standard_normal(I)
ST=S*np.exp((r-0.5*vol**2)*T+vol*np.sqrt(T)*z)
V=np.mean(np.exp(-r*T)*np.maximum(ST-K,0))
print(V)
```
### Statistical application
Histogram plot by `.hist(bin=100,figsize=(8,6))` 

QQ-plot using library **statsmodel** 

Meachine learning library **scikit-learn** 

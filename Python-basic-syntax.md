# Python （basic syntax）
BASED ON  ---Python基础教程(crossin全60课) —-前93页

1. 变量命名规则：

     第一个字符必须是字母或者下划线“_”
     变量名称是对大小写敏感的，myname 和myName 不是同一个变量。 
     python 在定义一个变量时不需要给它限定类型。变量会根据赋给它的值，自动决定它的类
型。你也可以在程序中，改变它的值，于是也就改变了它的类型

4. 变量运算：

 a += 3 和 a=a+3 是一样的

5. for循环：
    for i in range(1,101):   #包括左边，不包括右边。所以输出的是1-100。 range（101）效果一致
        print i
6. 字符串：
- 引号的处理

内容带有单引号，就用双引号表示"It's good" 反之亦然

    ‘You are a "BAD" man’

python 中还有一种表示字符串的方法：三个引号（‘’‘）或者（"""）
在三个引号中，你可以方便地使用单引号和双引号，并且可以直接换行

    '''
    "What's your name?" I asked.
    "I'm Han Meimei."
    '''

或者： 转译字符 \

    ‘I\'m a \"good\" teacher’
    \n    #表示换行
    \t    #表示tab
- 类型转化和格式化

字符和数字不能直接用+相加，解决办法：

    print 'My age is' + str(18)    #类型转换，其他还有：int(x)， float(x) str(x) bool(x)

或

    num = 18
    print 'My age is' + str(num)

或

    num = 18
    print 'My age is %d' % num  #输出的时候，%d 会被%后面的值替换；
    print ‘Price is %f’ % 4.99  #想格式化的数值是小数，要用%f
    print ‘Price is %.2f’ % 4.99 #想保留两位小数，需要在f 前面加上条件：%.2f
    
    name = 'Crossin'
    print '%s is a good teacher.' % name  #可以用%s 来替换一段字符串
    
    print "%s's score is %d" % ('Mike', 87) #这种用()表示的一组数据在python 中被称为元组（tuple），是python 的一种基本数据结构
- 字符串的分割与连接
    sentence = 'I am an Englist sentence'
    sentence.split()  #默认是按照空白字符分割
    section.split('.') #也可指定分割符号，这时空白字符就可以得到保留了
    'aaa'.split('a') #将会得到['', '', '', '']，由四个空串组成的list。
    
    s = ';'
    li = ['apple', 'pear', 'orange']
    fruit = s.join(li)
    print fruit   #得到'apple;pear;orange'
    
    ''.join(['hello', 'world'])     #也可以用空串连接，得到'helloworld'
- 字符串同样可以索引，切片
    word = 'helloworld'
    for c in word:
      print c   #遍历
    
    print word[-2]  #索引，但是与list 不同的是，字符串不能能通过索引访问去更改其中的字符。
    print word[5:7] 
    newword = ','.join(word) 
7. 循环嵌套
    for i in range(0, 5):
        for j in range(0, 5):
            print '*',     #加逗号表示不换行
        print              #print 后面没有写任何东西，是起到换行的作用
8. bool类型

在python 中，以下数值会被认为是False：
为0 的数字，包括0，0.0
空字符串，包括''，""
表示空值的None
空集合，包括()，[]，{}
其他的值都认为是True。

9. 函数
    def sayHello():          #定义函数
      print 'hello world!'
    sayHello()
    
    def plus(num1, num2):    #含参，要注意提供的参数值的数量和类型需要跟函数定义中的一致
    print num1+num2

猜数字大小的小游戏，用到前面的概念：

    from random import randint
    num = randint(1, 100)
    print 'Guess what I think?'
    bingo = False
    while bingo == False:
      answer = input()
      bingo = isEqual(answer, num)


10. if elif else 语句
    if a == 1:
      print 'one'
    elif a == 2:
      print 'two'
    elif a == 3:
      print 'three'
    else:
      print 'too many'      #elif 可以没有，也可以有很多个；else 可以没有，如果有的话只能有一个，必须在条件语句的最后


11. list 列表
    l = [365, 'everyday', 0.618, True]   #数据类型kekebu可不同
    l[0]  #访问第一个元素 
    l[0] = 123 #修改
    l.append(1024)  #添加元素
    del l[0]  #删除
    l[-1]  #表示l 中的最后一个元素
    l[1:3]  #左包含，右不包含
    l[:3]  #如果不指定第一个数，切片就从列表第一个元素开始。
    l[1:]  #不指定第二个数，就一直到最后一个元素结束
    l[:]   #都不指定，则返回整个列表的一个拷贝
    l[1:-1]  #切片也可使用负数


12. 读文件
    f = file('data.txt')           #注意路径问题
    data = f.read()
    print data
    f.close()        #养成释放内存的习惯
    
    readline()     #每次读取一行，当前位置移到下一行 
    readlines()    #把内容按行读取至一个list 中


13. 写文件

python 默认是以只读模式（’r’）打开文件。如果想要写入内容，在打开文件的时候需要指定打开模式为写入。如果文件不存在，会自动创建文件。另外还有一种模式是'a'，appending。它也是一种写入模式，但你写入的内容不会覆盖之前的内容，而是添加到文件中。

    f = file('output.txt', 'w')    #或者用open() 用法同file一致
    f.write('a string you want to write')
    
    data = 'I will be in a file.\nSo cool!'
    out = open('output.txt', 'w')
    out.write(data)          #参数是字符串或字符串变量都可
    out.close()

一个计算总成绩的程序：（但因为字符编码的问题，中文字符没办法识别）

    f = file('data.txt')
    lines=f.readlines()
    f.close()
    results=[]
    for line in lines:
        data = line.split()
        sum = 0
        for score in data[1:]:
            sum += int(score)
        result = '%s\t: %d\n' % (data[0], sum)
        results.append(result)
    output = file('data.txt', 'w')
    output.writelines(results)
    output.close()


14. break和continue

while 循环 在条件不满足时 结束，for 循环 遍历完序列后 结束。想提前结束，就用break
break 是彻底地跳出循环，而continue 只是略过本次循环的余下内容，直接进入下一次循环。
但都是最内层的，外层还是继续。

![](https://d2mxuefqeaa7sj.cloudfront.net/s_77CF018C9572001EA80B2568C715C05873B53615AE48DD00024986234A0DC07C_1503390652164_image.png)

15. 异常处理

用这个结构可以防止整个程序因为一个错误而全部中断

    try:
      f = file('non-exist.txt')
      print 'File opened!'
      f.close()
    except:
      print 'File not exists.'
    print 'Done'    #无论如何，整个程序不会中断，最后的“Done”都会被输出


16. 字典 dictionary 

一种数据结构，基本格式是  d = {key1 : value1, key2 : value2 }  键和值
键只能是简单对象，list就不能作为键，不过可以作为值

    score = {'萧峰': 95,'段誉': 97,'虚竹': 89}
    print score['段誉'] #键值没有顺序，所以不能用索引访问
    
    for name in score:
    print score[name]    #存储的是键
    
    score['虚竹'] = 91  #改值
    score['慕容复'] = 88   #添加
    del score['萧峰']
    d = {}  #新建空字典


17. 模块 （工具箱一样）
    import random
    random.randint(1, 10)
    random.randchoic([1, 3, 5])
    dir(random)  #想知道random 有哪些函数和变量，可以用dir()方法
    
    from math import pi
    print pi   #只是用到random 中的某一个函数或变量
    
    from math import pi as math_pi
    print math_pi     #为了便于理解和避免冲突，你还可以给引入的方法换个名字

appendix：介绍random模块==

    from random import randint    #取随机数
    randint(5,10) #下限&上限 
    from random import choice     #random 的另一个方法，它的作用是从一个list 中随机挑选一个元素   
    direction = ['left', 'center', 'right']
    com = choice(direction)  

18. urllib2 模块：上网索取信息 
```
import urllib2
web = urllib2.urlopen('http 分割://www 分割.baidu.分割com')
content = web.read()
print content
```
实现查询某个城市的天气：
city.py 这个文件里有一个叫做city 的字典，它里面的key 是城市的名称，value 是对应的城市代码。不用把它copy 到自己的程序中，只要放在和你的代码同一路径下，用 from city import city 就可以引入city 这个字典。这里相当于用了一个自定义的模块，前一个“city”是模块名，也就是py 文件的名称，后一个“city”是模块中变量的名称。
```
cityname = raw_input('你想查哪个城市的天气？\n')
citycode = city.get(cityname)
if citycode:
url = ('http: // www .weather .com.cn/data/cityinfo/%s.html' % citycode)
content = urllib2.urlopen(url).read()
```
19. 如果提示编码不对，则在script最前面加上：
`# -*- coding: utf-8 -*-`
20. 面向对象：python是一个高度面向对象的语言，所有东西其实都是对象。
* 通过dir()方法可以查看一个类/变量的所有属性
* 调用类变量的方法是“对象.变量名”。你可以得到它的值，也可以改变它的值。
* 定义类方法不同于其他函数： 变量必须是self. 调用时不需再给self的值，直接“对象.方法名()”就行，self指调用的对象本身
* 初始化： __init__函数会在类被创建的时候自动调用，用来初始化类。它的参数，要在创建类的时候提供
例子：
```
class MyClass:
  name = 'Sam'

  def sayHi(self):
    print 'Hello %s' % self.name

mc = MyClass()
print mc.name
mc.name = 'Lily'
mc.sayHi()
```
21. 类的继承：
class  导出类/子类1（基本类/超类）:
pass  #表示不需要额外添加的功能

class 导出类/子类2（基本类/超类）:
def 同名函数来覆盖超类的方法 （同名函数中再调用原来超类的函数，就要给self的值）
e.g.
```
class Vehicle:
 def __init__(self, speed):
   self.speed = speed

 def drive(self, distance):
   print 'need %f hour(s)' % (distance / self.speed)

 class Bike(Vehicle):
 pass

 class Car(Vehicle):
  def __init__(self, speed, fuel):
   Vehicle.__init__(self, speed)
   self.fuel = fuel
```
22. bool and a or b 语句中，当bool 条件为真时，结果是a；当bool 条件为假时，结果是b。
    可用来简化 if-else结构（比如lambda语句中无法使用ifelse），不过其中的坑是： a本身为假值的时候，e.g. 0或"",就会输出b。解决办法： 把a换成[a]，b也换成[b],输出返回值列表的第一个值：
 ```
 a = ""
 b = "hell"
 c = (True and [a] or [b])[0]
 print c
 ```
23. 元组（tuple)也是一种序列，与list相似，它有和list 同样的索引、切片、遍历等操作。区别在于创建后无法修改。
用元组纪录返回值的例子：
```
def get_pos(n):
  return (n/2, n*2)
#取回有两种方法：
 x, y = get_pos(50)
 print x
 print y
 #或
 pos = get_pos(50)
 print pos[0]
 print pos[1]
```
24. 数学运算模块 math 很有用， import math
http://docs.python.org/2/library/math.html   查看官方完整文档
25. 正则表达式： 记录文本规则的代码
    模块   import re
    推荐一篇叫做《正则表达式30 分钟入门教程》的 文章
26. random 模块： 产生随机数 有很多用法

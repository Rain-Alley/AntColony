#蚁群算法解决TSP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import random
matplotlib.rcParams['font.family'] = 'STSong'

city_name = []
city_condition = []
"""
python打开文件，并把文件内容传给f,传出的内容以字符串表示，引入了with语句来自动帮我们调用close()方法，'r'为对文件的操作。
调用readline()可以每次读取一行内容，调用readlines()一次读取所有内容并按行返回list。每行都是一个字符串
要读取非UTF-8编码的文本文件，需要给open()函数传入encoding参数，例如，读取GBK编码的文件：
>>> f = open('E:\python\python\gbk.txt', 'r', encoding='gbk')
如：
f.read()
'1,41,94\n2,37,84\n3,53,67\n4,25,62\n5,7,64'
f.readline()
'1,41,94\n'
f.readlines()
['1,41,94\n',
 '2,37,84\n',
 '3,53,67\n',
 '4,25,62\n',
 '5,7,64\n']
"""
with open('30城市的坐标.txt','r',encoding='UTF-8') as f:
    lines = f.readlines()
#调用readlines()一次读取所有内容并按行返回list给lines
#for循环每次读取一行
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        city_name.append(line[0])
        city_condition.append([float(line[1]), float(line[2])])
city_condition = np.array(city_condition)#获取30城市坐标
"""
Python列表和Numpy数组的区别：
Numpy使用ndarray对象来处理多维数组，该对象是一个快速而灵活的大数据容器。
使用Python列表可以存储一维数组，通过列表的嵌套可以实现多维数组，
那么为什么还需要使用Numpy呢？Numpy是专门针对数组的操作和运算进行了设计，
所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，
数组越大，Numpy的优势就越明显。通常Numpy数组中的所有元素的类型都是相同的，
而Python列表中的元素类型是任意的，所以在通用性能方面Numpy数组不及Python列表，
但在科学计算中，可以省掉很多循环语句，代码使用方面比Python列表简单的多。
"""

"""
Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串,返回分割后的字符串列表。
str – 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
num – 分割次数。默认为 -1, 即分隔所有。
如：
>>> str="hello boy<[www.doiido.com]>byebye"
>>> str.split("[")[1].split("]")[0]     输出的是 [ 后的内容以及 ] 前的内容。[1]是对字符串分隔之后，取第二个字符
'www.doiido.com'
>>> str.split("[")[1].split("]")[0].split(".")      是先输出 [ 后的内容以及 ] 前的内容，然后通过 . 作为分隔符对字符串进行切片。
['www', 'doiido', 'com']
"""


#Distance距离矩阵
city_count = len(city_name)
Distance = np.zeros((city_count, city_count))
for i in range(city_count):
    for j in range(city_count):
        if i != j:
            Distance[i][j] = math.sqrt((city_condition[i][0] - city_condition[j][0]) ** 2 + (city_condition[i][1] - city_condition[j][1]) ** 2)
        else:
            Distance[i][j] = 100000

# 蚂蚁数量
AntCount = 100
# 城市数量
city_count = len(city_name)
# 信息素
alpha = 1  # 信息素重要程度因子
beta = 2  # 启发函数重要程度因子
rho = 0.1 #挥发速度
iter = 0  # 迭代初始值
MAX_iter = 200  # 最大迭代值
Q = 1
# 初始信息素矩阵，全是为1组成的矩阵
pheromonetable = np.ones((city_count, city_count))

# 候选集列表,存放100只蚂蚁的路径(一只蚂蚁一个路径),一共就Antcount个路径，一共是蚂蚁数量*31个城市数量
candidate = np.zeros((AntCount, city_count)).astype(int) 

# path_best存放的是相应的，每次迭代后的最优路径，每次迭代只有一个值
path_best = np.zeros((MAX_iter, city_count)) 

# 存放每次迭代的最优距离
distance_best = np.zeros( MAX_iter)
# 倒数矩阵
etable = 1.0 / Distance  

while iter <  MAX_iter:
    """
    路径创建
        
    这是numpy的切片操作，一般结构如num[a:b,c:d]，分析时以逗号为分隔符，
    逗号之前为要取的num行的下标范围(a到b-1)，逗号之后为要取的num列的下标范围(c到d-1)；
    前面是行索引，后面是列索引。
    如果是这种num[:b,c:d]，a的值未指定，那么a为最小值0；
    如果是这种num[a:,c:d]，b的值未指定，那么b为最大值；c、d的情况同理可得。
    所以重点就是看逗号，没逗号，就是看行了。冒号呢，就看成一维数组的形式啦。那逗号后面没有数，也就是不对列操作咯。
    如a = np.array([[1,2,3],[3,4,5],[4,5,6]])
    >>> a[1]   如果索引是一个数，则结果为元组的一个元素
        array([3, 4, 5])
    >>> a[1:]  如果索引带冒号，结果也是一个数组，没有逗号代表所有列
        array([[3, 4, 5],
                [4, 5, 6]])
    >>> a[1:,]
        array([[3, 4, 5],
                [4, 5, 6]])
    >>> a[:,1:3]  #第2列到第3列
        array([[2, 3],
                [4, 5],
                [5, 6]])
    >>> a[:,]  #对列也不操作，跟下面等价
        array([[1, 2, 3],
                [3, 4, 5],
                [4, 5, 6]])
    """
    # first：蚂蚁初始点选择
    if AntCount <= city_count:
    #np.random.permutation随机排列一个数组的
        candidate[:, 0] = np.random.permutation(range(city_count))[:AntCount]
    else:
        m =AntCount -city_count
        n =2
        candidate[:city_count, 0] = np.random.permutation(range(city_count))[:]
        while m >city_count:
            candidate[city_count*(n -1):city_count*n, 0] = np.random.permutation(range(city_count))[:]
            m = m -city_count
            n = n + 1
        candidate[city_count*(n-1):AntCount,0] = np.random.permutation(range(city_count))[:m]
    length = np.zeros(AntCount)#每次迭代的N个蚂蚁的距离值

    # second：选择下一个城市选择
    for i in range(AntCount):
        # 移除已经访问的第一个元素
        unvisit = list(range(city_count))  # 列表形式存储没有访问的城市编号
        visit = candidate[i, 0]  # 当前所在点,第i个蚂蚁在第一个城市
        unvisit.remove(visit)  # 在未访问的城市中移除当前开始的点
        for j in range(1, city_count):#访问剩下的city_count个城市，city_count次访问
            protrans = np.zeros(len(unvisit))#每次循环都更改当前没有访问的城市的转移概率矩阵1*30,1*29,1*28...
            # 下一城市的概率函数
            for k in range(len(unvisit)):
                # 计算当前城市到剩余城市的（信息素浓度^alpha）*（城市适应度的倒数）^beta
                # etable[visit][unvisit[k]],(alpha+1)是倒数分之一，pheromonetable[visit][unvisit[k]]是从本城市到k城市的信息素
                protrans[k] = np.power(pheromonetable[visit][unvisit[k]], alpha) * np.power(
                    etable[visit][unvisit[k]], (alpha + 1))

            # 累计概率，轮盘赌选择
            cumsumprobtrans = (protrans / sum(protrans)).cumsum()
            cumsumprobtrans -= np.random.rand()
            # 求出离随机数产生最近的索引值
            k = unvisit[list(cumsumprobtrans > 0).index(True)]
            # 下一个访问城市的索引值
            candidate[i, j] = k
            unvisit.remove(k)
            length[i] += Distance[visit][k]
            visit = k  # 更改出发点，继续选择下一个到达点
        length[i] += Distance[visit][candidate[i, 0]]#最后一个城市和第一个城市的距离值也要加进去

    """
    更新路径等参数
    """
    # 如果迭代次数为一次，那么无条件让初始值代替path_best,distance_best.
    if iter == 0:
        distance_best[iter] = length.min()
        path_best[iter] = candidate[length.argmin()].copy()
    else:
        # 如果当前的解没有之前的解好，那么当前最优还是为之前的那个值；并且用前一个路径替换为当前的最优路径
        if length.min() > distance_best[iter - 1]:
            distance_best[iter] = distance_best[iter - 1]
            path_best[iter] = path_best[iter - 1].copy()
        else:  # 当前解比之前的要好，替换当前解和路径
            distance_best[iter] = length.min()
            path_best[iter] = candidate[length.argmin()].copy()

    """
        信息素的更新
    """
    #信息素的增加量矩阵
    changepheromonetable = np.zeros((city_count, city_count))
    for i in range(AntCount):
        for j in range(city_count - 1):
            # 当前路径比如城市23之间的信息素的增量：1/当前蚂蚁行走的总距离的信息素
            changepheromonetable[candidate[i, j]][candidate[i][j + 1]] += Q / length[i]
            #Distance[candidate[i, j]][candidate[i, j + 1]]
        #最后一个城市和第一个城市的信息素增加量
        changepheromonetable[candidate[i, j + 1]][candidate[i, 0]] += Q / length[i]
    #信息素更新的公式：
    pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
    iter += 1

print("蚁群算法的最优路径",path_best[-1]+1)
print("迭代", MAX_iter,"次后","蚁群算法求得最优解",distance_best[-1])

# 路线图绘制
fig = plt.figure()
plt.title("Best roadmap")
x = []
y = []
path = []
for i in range(len(path_best[-1])):
    x.append(city_condition[int(path_best[-1][i])][0])
    y.append(city_condition[int(path_best[-1][i])][1])
    path.append(int(path_best[-1][i])+1)
x.append(x[0])
y.append(y[0])
path.append(path[0])
for i in range(len(x)):
    plt.annotate(path[i], xy=(x[i], y[i]), xytext=(x[i] + 0.3, y[i] + 0.3))
plt.plot(x, y,'-o')

# 距离迭代图
fig = plt.figure()
#plt.figure语()---在plt中绘制一张图片
plt.title("Distance iteration graph")#距离迭代图
plt.plot(range(1, len(distance_best) + 1), distance_best)
plt.xlabel("Number of iterations")#迭代次数
plt.ylabel("Distance value")#距离值
plt.show()
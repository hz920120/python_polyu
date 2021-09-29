# a = 100
# if a > 0:
#     print(a)
# else:
#     print(-a)

# print('''nihaonihao
#
# asdfadsf''')

# print(7 and 4)

# a = 1
# print(a)
# a = 'abc'
# print(a)
#
# print('Age:%d , Gender:%s' % (25, 'male'))

# classmates = ['huang', 'zhang', 'li']
# print(classmates)
# print(len(classmates))

# classmates.insert(0, 'meme')
# print(classmates)

# s = ['python', 'java', ['asp', 'php'], 'scheme']
# print(len(s[2]))


# names = ['huang', 'li', 'ma','zhang']
#
# for name in names:
#     print(name)


# s = 0
# for num in list(range(1, 101)):
#     s = s + num
# print(s)

# res = 0
# count = 100
# while count > 0:
#     res = count + res
#     count = count - 1
# print(res)


# d = {"m": 1, "b": 2, "c": "asdf"}
# for data in d:
#     if data is None:
#         print('ok')
#     else:
#         print("not ok")

# def my_abs(x):
#     if x >= 0:
#         return x
#     else:
#         return -x
#
#
# print(my_abs(-5))
#
# import math
#
#
# def qua(a, b, c):
#     res1 = (b * -1 + math.sqrt(b ** 2 - 4 * a * c)) / 2 * a
#     res2 = (b * -1 - math.sqrt(b ** 2 - 4 * a * c)) / 2 * a
#     return res1, res2
#
#
# x,y = qua(1, 5, 3)
# print(x, y)


# g = (x * x for x in range(1, 10))
# print(next(g))
# print(next(g))
# print(next(g))
# print(next(g))
# print(next(g))
import numpy as np

array1 = [[1., 2.], [3., 4.]]
array2 = np.prod(array1, axis=1)
print(array2)
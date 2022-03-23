from sympy import symbols, lambdify, exp, sin, cos
import numpy as np
import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt
from astropy.io import fits

"""
多高斯拟合 2021/09/16
"""


def gauss_2d_A(A):
    """
    二维多高斯模型
    :param A: [A0, x0, y0, s0_1,s0_2, theta_0, ..., An, xn, yn, sn_1, sn_2,theta_n]
    :return:
    多高斯模型的表达式
    """
    # A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, A1, x1, y1, s1_1, s1_2,theta_1])
    param_num = 6  # 一个二维高斯的参数个数(A0, x0, y0, s0_1,s0_2, theta_0)
    num = A.shape[0]  # 输入的参数个数
    num_j = num // param_num  # 对输入参数取整， 得到二维高斯的个数

    # 定义模型输入的的符号
    paras = []
    x = symbols('x')
    y = symbols('y')
    paras.append(x)
    paras.append(y)

    express1 = ''
    for i in range(num_j):
        temp = ' + A[%d*6+0] * exp(-((x - A[%d*6+1]) ** 2 * (cos(A[%d*6+5])**2 / (2 * A[%d*6+3]**2) + sin(A[%d*6+5])**2 / (2 * A[%d*6+4]**2)) \
        + (y - A[%d*6+2])**2 * (sin(A[%d*6+5])**2 / (2 * A[%d*6+3]**2) + cos(A[%d*6+5])**2 / (2 * A[%d*6+4]**2))\
        + (sin(2*A[%d*6+5]) / (2 * A[%d*6+4] ** 2) - sin(2*A[%d*6+5]) / (2 * A[%d*6+3] ** 2)) * (x - A[%d*6+1]) * (y - A[%d*6+2]) ))'\
               %(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i)

        express1 += temp
    express = express1[2:]
    express.replace(' ','')
    g = eval(express)  # <class 'sympy.core.mul.Mul'>

    g1 = lambdify(paras, g, 'numpy')
    return g1


def get_gauss_3d_func_by_paras(A):
    """
    三维多高斯模型
    根据传入的参数，实现动态确定多高斯模型的表达式
    :param A:
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, ..., An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3]
    :return:
        多高斯模型的函数表达式 <class 'function'>
    """
    # A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, A1, x1, y1, s1_1, s1_2,theta_1,v1, s1_3])
    param_num = 8  # 一个三维高斯的参数个数(A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3)
    num = A.shape[0]  # 输入的参数个数
    gauss_count = num // param_num  # 对输入参数取整， 得到三维高斯的个数

    paras = []
    x = symbols('x')
    y = symbols('y')
    v = symbols('v')
    paras.append(x)
    paras.append(y)
    paras.append(v)

    express1 = ''
    for gauss_i in range(gauss_count):
        temp = ' + A[%d*8+0] * exp(-((x - A[%d*8+1]) ** 2 * (cos(A[%d*8+5])**2 / (2 * A[%d*8+3]**2) + sin(A[%d*8+5])**2 / (2 * A[%d*8+4]**2)) + (y - A[%d*8+2])**2 * (sin(A[%d*8+5])**2 / (2 * A[%d*8+3]**2) + cos(A[%d*8+5])**2 / (2 * A[%d*8+4]**2)) + (sin(2*A[%d*8+5]) / (2 * A[%d*8+4] ** 2) - sin(2*A[%d*8+5]) / (2 * A[%d*8+3] ** 2)) * (x - A[%d*8+1]) * (y - A[%d*8+2]) + (v - A[%d*8+6]) ** 2 / (2 * A[%d*8+7]**2) ))' \
               % (gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i,
                  gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i)
        express1 += temp

    express = express1[2:]
    express = express.replace(' ', '')
    gauss_3d_str = eval(express)  # <class 'sympy.core.mul.Mul'>

    gauss_3d_func = lambdify(paras, gauss_3d_str, 'numpy')
    return gauss_3d_func


def fit_gauss_2d(X,Y,params):
    """
    :param X: 坐标系 (x,y)
    :param Y: 实际值 data(x,y)
    :param params: 模型的初始猜想值
    :return:
        返回模型的拟合值
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, ..., An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3]
    """
    power = 4
    gauss_2d_func = gauss_2d_A(params)
    gauss_2d_value = gauss_2d_func(X[:, 0], X[:, 1])
    weight = gauss_2d_value ** power / ((gauss_2d_value ** power).sum())  # 创建拟合的权重

    errorfunction = lambda p: np.ravel((gauss_2d_A(p)(X[:,0],X[:,1]) - Y)  * weight)
    p, success = optimize.leastsq(errorfunction, x0=params)
    param_num = 6  # 一个二维高斯的参数个数(A0, x0, y0, s0_1,s0_2, theta_0)
    num = params.shape[0]  # 输入猜想的参数个数
    num_j = num // param_num  # 对输入参数取整， 得到二维高斯的个数
    for i in range(num_j):
        p[i * param_num + 5] = (p[i * param_num + 5] / np.pi * 180) % 180
    return p


def errorfunc(p, X, Y):
    power = 1
    gauss_3d_func = get_gauss_3d_func_by_paras(params)
    gauss_3d_value = gauss_3d_func(X[:, 0], X[:, 1], X[:, 2])
    weight = gauss_3d_value ** power / ((gauss_3d_value ** power).sum())  #
    errorfunction = lambda p: np.ravel((get_gauss_3d_func_by_paras(p)(X[:, 0], X[:, 1], X[:, 2]) - Y) * weight)

    return errorfunction


def fit_gauss_3d(X, Y, params):
    """
    :param X: 坐标系 (x,y,v)
    :param Y: 实际值 data(x,y,v)
    :param params: 模型的初始猜想值
    :return:
        返回模型的拟合值
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, ..., An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3]
    """
    power = 4
    gauss_3d_func = get_gauss_3d_func_by_paras(params)
    gauss_3d_value = gauss_3d_func(X[:, 0], X[:, 1], X[:, 2])
    weight = gauss_3d_value ** power / ((gauss_3d_value ** power).sum()) # 创建拟合的权重
    # weight = np.ones_like(weight)
    errorfunction = lambda p: np.ravel((get_gauss_3d_func_by_paras(p)(X[:, 0], X[:, 1], X[:, 2]) - Y) * weight)
    p, success = optimize.leastsq(errorfunction, x0=params)
    # res_robust = least_squares(errorfunction,x0=params,loss='soft_l1',args=(X, Y))
    # print(res_robust)
    param_num = 8  # 一个二维高斯的参数个数(A0, x0, y0, s0_1,s0_2, theta_0)
    num = params.shape[0]  # 输入猜想的参数个数
    num_j = num // param_num  # 对输入参数取整， 得到二维高斯的个数
    for i in range(num_j):
        p[i*param_num + 5] = (p[i*param_num + 5] / np.pi * 180) % 180
    return p


def fit_gauss_3d_new(points_all, params):
    """
    :param points_all: [x, y, v, intensity]
    :param params: pandas.DataFrame
        LDC算法计算的3维高斯模型的初始猜想值
    :return:
        返回模型的拟合值
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, success_1
         ...,
         An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3， success_n]
         其中: theta为弧度值,success_n表示拟合是否成功
    """
    power = 1
    columns_name = [item for item in params.keys().values]
    # columns_name = ['A', 'x0', 'y0', 's1', 's2', 'theta', 'v0', 's3']
    params = params.values.flatten()
    # print(params.shape)
    # print(points_all.shape)
    gauss_3d_func = get_gauss_3d_func_by_paras(params)
    X = points_all['x_2'].values
    Y = points_all['y_1'].values
    V = points_all['v_0'].values
    Intensity = points_all['Intensity'].values
    gauss_3d_value = gauss_3d_func(X, Y, V)
    weight = gauss_3d_value ** power / ((gauss_3d_value ** power).sum()) # 创建拟合的权重
    # weight = np.ones_like(weight)
    errorfunction = lambda p: np.ravel((get_gauss_3d_func_by_paras(p)(X, Y, V) - Intensity) * weight)
    # params_fit, success = optimize.leastsq(errorfunction, x0=params, )
    low_ = []
    [low_.extend([0, 20, 20, 0, 0, 0, 0, 0]) for i in range(params.shape[0]//8)]
    up_ = []
    [up_.extend([20, 100, 100, 20, 20, 7, 2400, 40]) for i in range(params.shape[0] // 8)]
    res_robust = optimize.least_squares(errorfunction, x0=params, loss='soft_l1', bounds=[low_, up_])
    params_fit = res_robust.x
    success = res_robust['success']
    cost = res_robust['cost']
    param_num = 8  # 一个三维高斯的参数个数(A0, x0, y0, s0_1,s0_2, theta_0)
    num_j = params.shape[0] // param_num  # 对输入参数取整， 得到三维高斯的个数

    params_fit = params_fit.reshape([num_j, param_num])
    params_fit_df = pd.DataFrame(params_fit, columns=columns_name)
    # params_fit_df['theta'] = (params_fit_df['theta'] / np.pi * 180) % 180
    params_fit_df['success'] = success
    params_fit_df['cost'] = cost

    return params_fit_df


if __name__ == '__main__':

    # 对二维高斯云核进行拟合
    A0, x0, y0, s0_1,s0_2, theta_0, A1, x1, y1, s1_1, s1_2,theta_1 = 5,10,10, 2,3,0/180*np.pi, 9,20,20,2,4,45/180*np.pi
    A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, A1, x1, y1, s1_1, s1_2,theta_1, 10, 15,15,2,2,0])
    gauss_2d_11 = gauss_2d_A(A)

    Xin, Yin = np.mgrid[0:31, 0:31]
    X = np.vstack([Xin.flatten(), Yin.flatten()]).T
    Y = gauss_2d_11(X[:,0],X[:,1])

    data1 = Y.reshape(Xin.shape)
    plt.imshow(data1)
    plt.show()

    ind = np.where(Y > 0.5)[0]
    X2 = X[ind, ...]
    Y2 = Y[ind, ...]

    params = A - 1
    p = fit_gauss_2d(X,Y,params)
    print(A)
    print(p)

    print(p[5]/np.pi * 180)
    print(p[10]/np.pi * 180)

# 对三维高斯云核进行拟合
    print('fit 3d gauss' + '*'*30)
    A0, x0, y0, s0_3,s0_2, theta_0, v0, s0_1 = 5,10,12, 2,4,30/180*np.pi, 13,6
    A1, x1, y1, s1_3,s1_2, theta_1, v1, s1_1 = 8,18,21, 2,4,76/180*np.pi, 16,6
    A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3,A1, x1, y1, s1_1,s1_2, theta_1, v1, s1_3])
    gauss_3d_11 = get_gauss_3d_func_by_paras(A)

    Xin, Yin, Vin = np.mgrid[0:31, 0:41, 0:51]
    X = np.vstack([Vin.flatten(), Yin.flatten(), Xin.flatten()]).T

    Y = gauss_3d_11(X[:,0],X[:,1],X[:,2])
    Y1 = gauss_3d_11(X[:,2],X[:,1],X[:,1])

    a = Y - Y1
    # data1 = Y.reshape(Xin.shape)
    data1 = Y.reshape(Yin.shape)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for i, ax_item in enumerate([ax1, ax2, ax3]):
        ax_item.imshow(data1.sum(axis=i))
    plt.show()

    ind = np.where(Y > 0.5)[0]
    X2 = X[ind, ...]
    Y2 = Y[ind, ...]

    params = A - 1

    p = fit_gauss_3d(X,Y,params)
    print(p)
    print(A)

    # print(p[5]/np.pi * 180)
    # print(p[10]/np.pi * 180)

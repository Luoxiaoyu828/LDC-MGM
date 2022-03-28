import numpy as np
import time
import pandas as pd
from scipy import optimize
from scipy import integrate
from sympy import symbols, lambdify, exp, sin, cos

"""
多高斯拟合 2022/03/24
"""


def error_func(params, X, Y):
    power = 1
    gauss_3d_func = get_multi_gauss_func_by_params(params)
    gauss_3d_value = gauss_3d_func(X[:, 0], X[:, 1], X[:, 2])
    weight = gauss_3d_value ** power / ((gauss_3d_value ** power).sum())  #
    errorfunction = lambda p: np.ravel((get_multi_gauss_func_by_params(p)(X[:, 0], X[:, 1], X[:, 2]) - Y) * weight)

    return errorfunction


def get_params_bound(params_init, ndim=3):
    """
    :param ndim: 高斯模型的维数
    :param params_init: 1*m ndarray
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, ..., An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3]
        LDC算法计算的3维高斯模型的初始猜想值
    :return:
    """
    peak_range = 3
    peak_low = 5 * 0.23
    sigma_time = 1
    s_time = 1.5
    s_low = 1
    low_ = []
    up_ = []
    if ndim == 2:
        param_num = 6
        gauss_num = params_init.shape[0] // param_num
        params_init_mat = params_init.reshape([gauss_num, param_num])
        params_init_df = pd.DataFrame(params_init_mat,
                                      columns=['A0', 'x0', 'y0', 's0_1', 's0_2', 'theta_0'])
        A0 = params_init_df['A0'].values
        x0 = params_init_df['x0'].values
        y0 = params_init_df['y0'].values
        s0_1 = params_init_df['s0_1'].values
        s0_2 = params_init_df['s0_2'].values
        theta_0 = params_init_df['theta_0'].values

        A0_down = np.array([A0 - peak_range, peak_low * np.ones_like(A0)]).max(axis=0)
        A0_up = A0 + peak_range

        x0_down = x0 - sigma_time * s0_1
        x0_up = x0 + sigma_time * s0_1

        y0_down = y0 - sigma_time * s0_2
        y0_up = y0 + sigma_time * s0_2

        s0_1_down = np.array([s0_1 * s_time, s_low * np.ones_like(s0_1)]).max(axis=0)
        s0_1_up = s0_1 * (s_time - 1)

        s0_2_down = np.array([s0_2 * s_time, s_low * np.ones_like(s0_2)]).max(axis=0)
        s0_2_up = s0_2 * (s_time - 1)

        theta_0_down = np.zeros_like(theta_0)
        theta_0_up = np.zeros_like(theta_0) + 2 * np.pi

        low_.extend([A0_down, x0_down, y0_down, s0_1_down, s0_2_down, theta_0_down])
        up_.extend([A0_up, x0_up, y0_up, s0_1_up, s0_2_up, theta_0_up])
    elif ndim == 3:
        param_num = 8
        gauss_num = params_init.shape[0] // param_num
        params_init_mat = params_init.reshape([gauss_num, param_num])
        params_init_df = pd.DataFrame(params_init_mat,
                                      columns=['A0', 'x0', 'y0', 's0_1', 's0_2', 'theta_0', 'v0', 's0_3'])
        A0 = params_init_df['A0'].values
        x0 = params_init_df['x0'].values
        y0 = params_init_df['y0'].values
        s0_1 = params_init_df['s0_1'].values
        s0_2 = params_init_df['s0_2'].values
        theta_0 = params_init_df['theta_0'].values
        v0 = params_init_df['v0'].values
        s0_3 = params_init_df['s0_3'].values
        #
        A0_down = np.array([A0 - peak_range, peak_low * np.ones_like(A0)]).max(axis=0)
        A0_up = A0 + peak_range

        x0_down = x0 - sigma_time * s0_1
        x0_up = x0 + sigma_time * s0_1

        y0_down = y0 - sigma_time * s0_2
        y0_up = y0 + sigma_time * s0_2

        v0_down = v0 - sigma_time * s0_3
        v0_up = v0 + sigma_time * s0_3

        s0_1_down = np.array([s0_1 * (s_time - 1), s_low * np.ones_like(s0_1)]).min(axis=0)
        s0_1_up = s0_1 * s_time

        s0_2_down = np.array([s0_2 * (s_time - 1), s_low * np.ones_like(s0_2)]).min(axis=0)
        s0_2_up = s0_2 * s_time

        s0_3_down = np.array([s0_3 * (s_time - 1), s_low * np.ones_like(s0_3)]).min(axis=0)
        s0_3_up = s0_3 * s_time

        theta_0_down = np.zeros_like(theta_0) - 0.001
        theta_0_up = np.zeros_like(theta_0) + 2 * np.pi
        low_array = np.array([A0_down, x0_down, y0_down, s0_1_down, s0_2_down, theta_0_down, v0_down, s0_3_down]).T
        up_array = np.array([A0_up, x0_up, y0_up, s0_1_up, s0_2_up, theta_0_up, v0_up, s0_3_up]).T

        low_ = low_array.flatten()
        up_ = up_array.flatten()
    else:
        print('only fitting 2d or 3d gauss!')
        return
    return low_, up_


def get_multi_gauss_func_by_params(params_init, ndim=3):
    """
    根据传入的初始参数，构建2/3维多高斯模型的函数表达式
    根据传入的参数，实现动态确定多高斯模型的表达式

    :param ndim: 高斯模型的维数
    :param params_init: A: 1*m ndarray
        三维多高斯模型
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, ..., An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3]
        二维多高斯模型
    :param A: [A0, x0, y0, s0_1,s0_2, theta_0, ..., An, xn, yn, sn_1, sn_2,theta_n]
    :return:
        多高斯模型的函数表达式 <class 'function'>
    """
    gauss_3d_str = ' + A[%d*8+0] * exp(-((x - A[%d*8+1]) ** 2 * (cos(A[%d*8+5])**2 / (2 * A[%d*8+3]**2) + sin(A[%d*8+5])**2 / (2 * A[%d*8+4]**2)) + (y - A[%d*8+2])**2 * (sin(A[%d*8+5])**2 / (2 * A[%d*8+3]**2) + cos(A[%d*8+5])**2 / (2 * A[%d*8+4]**2)) + (sin(2*A[%d*8+5]) / (2 * A[%d*8+4] ** 2) - sin(2*A[%d*8+5]) / (2 * A[%d*8+3] ** 2)) * (x - A[%d*8+1]) * (y - A[%d*8+2]) + (v - A[%d*8+6]) ** 2 / (2 * A[%d*8+7]**2) ))'
    gauss_2d_str = ' + A[%d*6+0] * exp(-((x - A[%d*6+1]) ** 2 * (cos(A[%d*6+5])**2 / (2 * A[%d*6+3]**2) + sin(A[%d*6+5])**2 / (2 * A[%d*6+4]**2)) + (y - A[%d*6+2])**2 * (sin(A[%d*6+5])**2 / (2 * A[%d*6+3]**2) + cos(A[%d*6+5])**2 / (2 * A[%d*6+4]**2)) + (sin(2*A[%d*6+5]) / (2 * A[%d*6+4] ** 2) - sin(2*A[%d*6+5]) / (2 * A[%d*6+3] ** 2)) * (x - A[%d*6+1]) * (y - A[%d*6+2]) ))'

    A = params_init
    x, y, v = symbols('x'), symbols('y'), symbols('v')
    paras = [x, y, v]

    if ndim == 2:
        param_num = 6  # 一个2d高斯成分的参数个数 (A0, x0, y0, s0_1,s0_2, theta_0)
        gauss_str = gauss_2d_str
        paras = paras[:2]
    elif ndim == 3:
        param_num = 8  # 一个3d高斯成分的参数个数 (A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3)
        gauss_str = gauss_3d_str
    else:
        print('only fitting 2d or 3d gauss!')
        return
    gauss_num = params_init.shape[0] // param_num  # 高斯成分的个数
    express1 = ''
    for gauss_i in range(gauss_num):
        temp = gauss_str % (
            gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i,
            gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i, gauss_i)
        express1 += temp

    express = express1[2:]
    express = express.replace(' ', '')
    gauss_str_func = eval(express)  # <class 'sympy.core.mul.Mul'>

    gauss_multi_func = lambdify(paras, gauss_str_func, 'numpy')
    return gauss_multi_func


def fitting_multi_gauss_params(points_all, params_init, ndim=3):
    """
    根据数据点，初始化参数，对模型进行拟合，返回拟合参数

    :param ndim: 高斯模型的维数
    :param points_all: [x, y, v, intensity]
    :param params_init: 1*m ndarray
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, ..., An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3]
        LDC算法计算的3维高斯模型的初始猜想值
    :return:
        返回模型的拟合值
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, success_1, cost_1
         ...,
         An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3，success_n, cost_n]
         or
         返回模型的拟合值
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, success_1, cost_1
         ...,
        An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3，success_n, cost_n]

         其中: theta为弧度值,success_n表示拟合是否成功
    scipy.optimize.least_squares(fun, x0, jac='2-point', bounds=(- inf, inf), method='trf', ftol=1e-08, xtol=1e-08,
        gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={},
        jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})

         f_scalefloat, optional
            Value of soft margin between inlier and outlier residuals, default is 1.0.
            The loss function is evaluated as follows rho_(f**2) = C**2 * rho(f**2 / C**2),
            where C is f_scale, and rho is determined by loss parameter.
            This parameter has no effect with loss='linear', but for other loss values it is of crucial importance.
    """
    if params_init.ndim != 1:
        print('The shape of params_init must be 1*m.')
        return None
    power = 4
    gauss_multi_func = get_multi_gauss_func_by_params(params_init, ndim=ndim)
    low_ = []
    up_ = []
    if ndim == 2:
        columns_name = ['A', 'x0', 'y0', 's1', 's2', 'theta']
        param_num = 6
        gauss_num = params_init.shape[0] // param_num
        X = points_all['x_2'].values
        Y = points_all['y_1'].values
        gauss_multi_value = gauss_multi_func(X, Y)
        Intensity = points_all['Intensity'].values
        weight = gauss_multi_value ** power / ((gauss_multi_value ** power).sum())  # 创建拟合的权重
        errorfunc = lambda p: np.ravel((get_multi_gauss_func_by_params(p, ndim=2)(X, Y) - Intensity) * weight)
        [low_.extend([0, 20, 20, 0, 0, 0]) for _ in range(gauss_num)]
        [up_.extend([20, 100, 100, 20, 20, 7]) for _ in range(gauss_num)]
    elif ndim == 3:
        columns_name = ['A', 'x0', 'y0', 's1', 's2', 'theta', 'v0', 's3']
        param_num = 8
        gauss_num = params_init.shape[0] // param_num
        X = points_all['x_2'].values
        Y = points_all['y_1'].values
        V = points_all['v_0'].values
        gauss_multi_value = gauss_multi_func(X, Y, V)
        Intensity = points_all['Intensity'].values
        weight = gauss_multi_value ** power / ((gauss_multi_value ** power).sum())  # 创建拟合的权重
        errorfunc = lambda p: np.ravel((get_multi_gauss_func_by_params(p, ndim=3)(X, Y, V) - Intensity) * weight)
        low_, up_ = get_params_bound(params_init, ndim=3)
        # [low_.extend([0, 20, 20, 0, 0, 0, 0, 0]) for _ in range(gauss_num)]
        # [up_.extend([20, 100, 100, 20, 20, 7, 2400, 30]) for _ in range(gauss_num)]
    else:
        print('only fitting 2d or 3d gauss!')
        return

    gauss_num = params_init.shape[0] // param_num
    res_robust = optimize.least_squares(errorfunc, x0=params_init, loss='soft_l1', bounds=[low_, up_], f_scale=0.1)
    params_fit = res_robust.x
    success = res_robust['success']
    cost = res_robust['cost']

    params_fit = params_fit.reshape([gauss_num, param_num])
    params_fit_df = pd.DataFrame(params_fit, columns=columns_name)

    params_fit_df['success'] = success  # 拟合是否成功
    params_fit_df['cost'] = cost  # 拟合的代价函数

    return params_fit_df


def get_fit_outcat_df(params_fit_pf):
    """
    对拟合的结果进行整理得到拟合云核核表
    :param params_fit_pf:
        拟合数据结果，pandas.DataFrame
        返回模型的拟合值
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, success_1, cost_1
         ...,
         An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3，success_n, cost_n]
         or
         返回模型的拟合值
        [A0, x0, y0, s0_1,s0_2, theta_0, success_1, cost_1
         ...,
        An, xn, yn, sn_1, sn_2,theta_n, success_n, cost_n]
    :return:
        整理的拟合核表
    'ID': 分子云核编号
    'Peak1','Peak2', ['Peak3']: 峰值位置坐标
    'Cen1', 'Cen2', ['Cen3']: 质心位置坐标
    'Size1', 'Size2', ['Size3']: 云核的轴长[FWHM=2.3548*sigma]
    'theta': 云核在银经银纬面上的旋转角
    'Peak': 云核的峰值
    'Sum': 云核的总流量
    'success': 拟合是否成功
    'cost': 拟合函数和实际数据间的误差
    """
    if not isinstance(params_fit_pf, pd.core.frame.DataFrame):
        print('the params_fit_pf type must be pandas.DataFrame.')
        return

    clumps_num, params_num = params_fit_pf.shape  # 分子云核个数, 参数个数
    outcat_record = pd.DataFrame([])
    clumps_sum = np.zeros([clumps_num, 1])
    if params_num == 10:
        p_num = 8
        n_dim_ = 3
        Peak_item = ['Peak1', 'Peak2', 'Peak3']
        Cen_item = ['Cen1', 'Cen2', 'Cen3']
        Size_item = ['Size1', 'Size2', 'Size3']
        xyv_0 = ['x0', 'y0', 'v0']
        s_123 = ['s1', 's2', 's3']
        outcat_record_order = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3',
                               'theta', 'Peak', 'Sum', 'success', 'cost']
    elif params_num == 8:
        p_num = 6
        n_dim_ = 2
        Peak_item = ['Peak1', 'Peak2']
        Cen_item = ['Cen1', 'Cen2']
        Size_item = ['Size1', 'Size2']
        xyv_0 = ['x0', 'y0']
        s_123 = ['s1', 's2']
        outcat_record_order = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak', 'Sum',
                               'success', 'cost']
    else:
        print('only get 2d or 3d outcat!')
        return

    for i, item in enumerate(params_fit_pf.values):
        pfsc = item[: p_num]
        pfsc_1 = params_fit_pf.iloc[i]
        func = get_multi_gauss_func_by_params(pfsc, ndim=n_dim_)

        if params_num == 10:
            x_lim = [pfsc_1['x0'] - 10 * pfsc_1['s1'], pfsc_1['x0'] + 10 * pfsc_1['s1']]
            y_lim = [pfsc_1['y0'] - 10 * pfsc_1['s2'], pfsc_1['y0'] + 10 * pfsc_1['s2']]
            v_lim = [pfsc_1['v0'] - 10 * pfsc_1['s3'], pfsc_1['v0'] + 10 * pfsc_1['s3']]
            intergrate_bound = [x_lim, y_lim, v_lim]
        else:
            x_lim = [pfsc_1['x0'] - 10 * pfsc_1['s1'], pfsc_1['x0'] + 10 * pfsc_1['s1']]
            y_lim = [pfsc_1['y0'] - 10 * pfsc_1['s2'], pfsc_1['y0'] + 10 * pfsc_1['s2']]
            intergrate_bound = [x_lim, y_lim]

        integrate_result = integrate.nquad(func, intergrate_bound)
        clumps_sum[i, 0] = integrate_result[0]

    outcat_record['ID'] = np.array([i for i in range(clumps_num)])
    outcat_record[['Peak', 'success', 'cost']] = params_fit_pf[['A', 'success', 'cost']]
    outcat_record['theta'] = np.rad2deg(params_fit_pf['theta'].values) % 180    # 求得的角度需要用180取余
    outcat_record[['Sum']] = clumps_sum
    outcat_record[Peak_item] = params_fit_pf[xyv_0]
    outcat_record[Cen_item] = params_fit_pf[xyv_0]
    outcat_record[Size_item] = params_fit_pf[s_123] * 2.3548

    outcat_record = outcat_record[outcat_record_order]

    return outcat_record


def fitting_main(points_all, params_init, fit_outcat_path, ndim=3):
    """
    多高斯拟合的主函数，保存拟合的结果
    :param points_all: [m*4 pandas.DataFrame] [x,y,v,I]
    :param params_init: [1*m ndarray]
        [A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, ..., An, xn, yn, sn_1, sn_2,theta_n,vn, sn_3]
        LDC算法计算的3维高斯模型的初始猜想值
    :param fit_outcat_path: [str] 拟合核表的保存位置
    :param ndim: 高斯模型的维数
    :return:
        None
    """
    if fit_outcat_path.split('.')[-1] not in ['csv', 'txt']:
        print('the save file type must be one of *.csv and *.txt.')
        return

    if not isinstance(points_all, pd.core.frame.DataFrame):
        print('the points_all type must be pandas.DataFrame.')
        return

    if not isinstance(params_init, np.ndarray):
        print('the params_init type must be ndarray.')
        return

    print(time.ctime() + ': fitting ...')
    params_fit_df = fitting_multi_gauss_params(points_all, params_init, ndim=ndim)

    print(time.ctime() + ': fitting over, get outcat record.')
    outcat_record = get_fit_outcat_df(params_fit_df)

    if isinstance(outcat_record, pd.core.frame.DataFrame):
        outcat_record = outcat_record.round({'ID': 0, 'Peak1': 2, 'Peak2': 2, 'Peak3': 2, 'Cen1': 2, 'Cen2': 2, 'Cen3': 2,
                                     'Size1': 2, 'Size2': 2, 'Size3': 2, 'theta': 2, 'Peak': 2, 'Sum': 2})
        outcat_record.to_csv(fit_outcat_path, index=False, sep='\t')
        print(time.ctime() + ': outcat record saved success.\n')


if __name__ == '__main__':
    pass

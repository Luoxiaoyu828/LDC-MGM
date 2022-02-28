import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.stats
import math
from astropy.table import QTable
import os
import pandas as pd


# 根据真实数据拟合Peak和Sum
def fit_lognorm(data, xlabel, path):
    bins = 25
    uint = 'cm${}^{-3}$'
    eps_name = 'n_cm-3'
    # data = Sum
    data_len = data.shape[0]
    fit_par_lognorm = scipy.stats.lognorm.fit(data, floc=0)
    lognorm_dist_fitted = scipy.stats.lognorm(*fit_par_lognorm)
    t = np.linspace(data.min(), data.max(), 100)
    data_hist = np.histogram(data, bins)
    tex_fit = lognorm_dist_fitted.pdf(np.linspace(np.min(data), np.max(data), bins))
    correlation = scipy.stats.spearmanr(data_hist[0] / data_len, tex_fit)[0]
    plt.figure(figsize=(4, 4))
    plt.plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r', ls='-',
             label='Lognormal fit: $R^2$={}'.format(str(correlation)[:4]))
    # plt.hist(data,norm_hist=True, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel('N')
    plt.legend()
    png_path = os.path.join(path, '{}.png'.format(xlabel))
    plt.savefig(png_path)
    plt.close()
    return fit_par_lognorm, lognorm_dist_fitted


generate = lambda xyz, peak, x0, y0, v0, size1, size2, size3, theta: peak * np.exp(-(
        ((xyz[:, 0] - x0) ** 2) * (
        np.cos(theta) ** 2 / (2 * size1 ** 2) + np.sin(theta) ** 2 / (2 * size2 ** 2)) + (
                (xyz[:, 1] - y0) ** 2) * (
                np.sin(theta) ** 2 / (2 * size1 ** 2) + np.cos(theta) ** 2 / (2 * size2 ** 2)) + (
                xyz[:, 0] - x0) * (xyz[:, 1] - y0) * (
                2 * (-np.sin(2 * theta) / (4 * size1 ** 2) + np.sin(2 * theta) / (4 * size2 ** 2))) + (
                (xyz[:, 2] - v0) ** 2) / (2 * size3 ** 2)))


# 判断两个核之间是否可分
def is_separable_2(outcat1, outcat2):
    xy_distance = (np.sqrt((outcat1[:, 0] - outcat2[0, 0]) ** 2 + (outcat1[:, 1] - outcat2[0, 1]) ** 2))  # xy距离矩阵
    sigma_xy_distance = np.sqrt(outcat1[:, 3] ** 2 + outcat1[:, 4] ** 2) + np.sqrt(
        outcat2[0, 3] ** 2 + outcat2[0, 4] ** 2)  # 主轴次轴距离矩阵
    v_distance = (np.abs(outcat1[:, 2] - outcat2[0, 2]))  # v轴距离矩阵
    sigma_v_distance = outcat1[:, 5] + outcat2[0, 5]  # 速度轴距离矩阵
    func1_res = v_distance - sigma_v_distance
    func1_res[func1_res >= 0] = 0  # 可分
    func1_res[func1_res < 0] = 1  # 不可分
    func2_res = xy_distance - sigma_xy_distance
    func2_res[func2_res >= 0] = 0  # 可分
    func2_res[func2_res < 0] = 1  # 不可分
    result = np.zeros_like(func2_res)
    for i in range(func2_res.shape[0]):
        result[i] = func2_res[i] and func1_res[i]
    if np.count_nonzero(result):
        return False  # 不允许加入
    else:
        return True  # 允许加入


# 生成样本数据，其中Peak服从对数正态分布，Sum为不服从对数正态分布
def first_data_set(path, real_data_path, number=10000):
    real_data = fits.getdata(real_data_path)
    size_v, size_y, size_x = real_data.shape
    path1 = os.path.join(path, 'synthetic_data')
    if not os.path.exists(path1):
        os.makedirs(path1)
    colsName1 = ['ID', 'Peak', 'Size1', 'Size2', 'Size3', 'Sum']  # 样本数据抬头
    # 拟合出Sum和Peak对应的值
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    outcat = fits.getdata('./Generate/gauss_outcat_m16_13_ellipse.FIT')
    Peak = outcat['Peak']
    Sum = outcat['Sum']
    fit_par_lognorm_Sum, lognorm_dist_fitted_Sum = fit_lognorm(Sum, 'Total Flux', path1)
    fit_par_lognorm_Peak, lognorm_dist_fitted_Peak = fit_lognorm(Peak, 'Peak', path1)
    sigma = fit_par_lognorm_Peak[0]
    mu = fit_par_lognorm_Peak[2]
    peak_sample = np.random.lognormal(mean=np.log(mu), sigma=sigma, size=number)  # 生成服从对数正态分布的Peak

    # 生成n个样本数据
    core_excel_first = []
    x, y, z = np.mgrid[1:size_v + 1, 1:size_y + 1, 1:size_x + 1]
    xyz = np.column_stack([z.flat, y.flat, x.flat])
    for i in range(peak_sample.shape[0]):
        size1 = np.random.uniform(2, 4)
        size2 = np.random.uniform(2, 4)
        size3 = np.random.uniform(1, 7)
        theta = np.random.uniform(0, 180)
        x0 = size_x // 2
        y0 = size_y // 2
        v0 = size_v // 2
        if size1 < size2:
            size1, size2 = size2, size1  # size1做为主轴，size2做为次主轴
        res = generate(xyz, peak_sample[i], x0, y0, v0, size1, size2, size3, theta / 180 * np.pi)
        data = res.reshape((size_v, size_y, size_x))
        Sum = data.sum()  # 计算流量
        core_excel_first.append(
            [i + 1, peak_sample[i], size1 * 2.3548, size2 * 2.3548, size3 * 2.3548, Sum, theta])
    core_excel_first = np.array(core_excel_first)
    core_excel = QTable(core_excel_first[:, 0: 6], names=colsName1)
    core_excel_path = os.path.join(path1, 'core_table_first.txt')
    core_excel.write(core_excel_path, overwrite=True, format='ascii')  # 存储第一次样本数据的核表
    return core_excel_first, lognorm_dist_fitted_Sum, path1


# 从第一次的样本数据按照sum的对数正态分布取值
def second_data_set(core_excel_first, lognorm_dist_fitted_Sum, path, interval=200):
    # 根据Sum的期望值取出n个样本数据
    f = lambda x: 1
    colsName1 = ['ID', 'Peak', 'Size1', 'Size2', 'Size3', 'Sum']  # 核表抬头文件
    sample_num = core_excel_first.shape[0]  # 可以设为输入参数
    test_data = core_excel_first[np.argsort(core_excel_first[:, 5]), :]
    Sum = test_data[:, 5]
    Sum_min = test_data[:, 5].min()
    Sum_max = test_data[:, 5].max()
    res = np.array([[]])
    test_sum = test_data[:, 5]
    for i in range(math.ceil((Sum_max - Sum_min) / interval)):
        a = test_data[
            (test_sum >= Sum_min + i * interval) * (test_sum < Sum_min + (i + 1) * interval)]  # 划分区间，sum满足这个区间
        if a.size == 0:  # 若区间中没有值，跳到下一个区间
            continue
        port = lognorm_dist_fitted_Sum.expect(f, lb=Sum_min + i * interval,
                                              ub=Sum_min + (i + 1) * interval)  # 该区间的概率（数学期望）
        amount = math.ceil(port * sample_num)  # 在该区间应该取的个数
        row_rand_array = np.arange(a.shape[0])  # 随机取n行
        np.random.shuffle(row_rand_array)  # 随机取n行
        row_rand = a[row_rand_array[0: amount]]  # 随机取n行
        temp = row_rand.shape[0]
        while temp < amount:  # 重复取，若区间里面的个数小于amount
            row_rand1 = a[row_rand_array[0: amount - temp]]
            row_rand = np.concatenate((row_rand, row_rand1), axis=0)
            amount = amount - temp
            temp = row_rand1.shape[0]
        if res.size == 0:
            res = row_rand
        else:
            res = np.concatenate((res, row_rand), axis=0)  # 数组拼接
    temp1 = res[:, 5]
    plt.hist(temp1, 200)
    png_path = os.path.join(path, 'sample.png')
    plt.savefig(png_path)
    plt.close()
    core_sample = QTable(res[:, 0: 6], names=colsName1)
    core_sample_path = os.path.join(path, 'sample_data.txt')
    core_sample.write(core_sample_path, overwrite=True, format='ascii')  # 存储样本数据


def make_synthetic_clumps(n, real_data_path, path, core_sample_path=None, number=10000):
    real_data = fits.getdata(real_data_path)
    size_v, size_y, size_x = real_data.shape
    colsName2 = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
                 'Sum', 'Volume']  # 核表抬头
    x, y, z = np.mgrid[1:size_v + 1, 1:size_y + 1, 1:size_x + 1]
    xyz = np.column_stack([z.flat, y.flat, x.flat])
    n_datacube = 0  # 用于计数
    n_datacube_ = number // n  # 生成数据块的个数
    path2 = os.path.join(path, 'synthetic_clump_%03d' % n)  # 次级目录
    path3_model = os.path.join(path2, 'model')  # 底层目录，无噪声fits目录
    path3_outcat = os.path.join(path2, 'outcat')  # 核表目录
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3_model):
        os.makedirs(path3_model)
    if not os.path.exists(path3_outcat):
        os.makedirs(path3_outcat)
    if core_sample_path == None:
        core_sample_path = os.path.join(path, 'sample_data.txt')
    a = np.loadtxt(core_sample_path, skiprows=1)  # 读取样本数据
    fits_header = fits.open(real_data_path)[0].header  # 读取头文件
    angle = np.random.uniform(0, 180, size=[a.shape[0], 1])  # 随机生成旋转角度
    para = np.hstack((a, angle))  # 矩阵合并
    total_data = []
    while 1:
        row_rand_array = np.arange(para.shape[0])  # 随机取n行
        np.random.shuffle(row_rand_array)  # 随机取n行
        row_rand = para[row_rand_array[0: n]]  # 随机取n行
        new_coreTable = []  # 存储核表数据
        res_data = np.zeros_like(x, dtype=np.float64)  # 数据块
        is_join = True  # 用于第一次计算
        number_for = 0  # 记录加入失败的次数
        i = 0  # 当前数据块中核的个数
        while i < n:
            x0 = np.random.uniform(5, size_x - 5)  # 随机生成位置坐标
            y0 = np.random.uniform(5, size_y - 5)
            v0 = np.random.uniform(5, size_v - 5)
            if number_for == 2000:  # 若加入失败次数达到了2000次，则重新生成数据块
                n_datacube -= 1
                break
            if i != 0:  # 从数据块中加入第二个核开始，判断当前的核能否加入到数据块
                outcat1 = np.array(new_coreTable)[:, [1, 2, 3, 7, 8, 9]]  # 前i-1个核的相关数据
                outcat1[:, 3: 6] = outcat1[:, 3: 6] / 2.3548  # 当前核的相关数据
                outcat2 = np.array(
                    [[x0, y0, v0, row_rand[i, 2] / 2.3548, row_rand[i, 3] / 2.3548, row_rand[i, 4] / 2.3548]])
                is_join = is_separable_2(outcat1, outcat2)
            if is_join:  # 若第i个能加入，则执行
                res = generate(xyz, row_rand[i, 1], x0, y0, v0, row_rand[i, 2] / 2.3548,
                                          row_rand[i, 3] / 2.3548,
                                          row_rand[i, 4] / 2.3548, ((90 + row_rand[i, 6]) % 180) / 180 * np.pi)
                data = res.reshape((size_v, size_y, size_x))
                Sum1 = data.sum()
                res_data += data
                Volume = int(len(np.where(data > 0.01)[0]))  # 计算体积
                new_coreTable.append(
                    [i + 1, x0, y0, v0, x0, y0, v0, row_rand[i, 2], row_rand[i, 3], row_rand[i, 4], row_rand[i, 6],
                     row_rand[i, 1], Sum1, Volume])
                i += 1
            else:  # 若第i个不能加入，则重新生成
                number_for += 1
                continue
        new_coreTable = np.array(new_coreTable)  # list转化成ndarray
        res_data += real_data
        total_data.append(real_data)
        outcat_name = os.path.join(path3_outcat, 'synthetic_outcat_%03d.txt' % n_datacube)
        dataframe = pd.DataFrame(new_coreTable, columns=colsName2)
        dataframe = dataframe.round({'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Peak3': 3, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                     'Size1': 3, 'Size2': 3, 'Size3': 3, 'theta': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(outcat_name, sep='\t', index=False)

        data_hdu = fits.PrimaryHDU(res_data, header=fits_header)
        fits_name = os.path.join(path3_model, 'synthetic_model_%03d.fits' % n_datacube)
        fits.HDUList([data_hdu]).writeto(fits_name, overwrite=True)  # 无噪声数据存储

        n_datacube += 1
        if n_datacube == n_datacube_:
            break
    return total_data

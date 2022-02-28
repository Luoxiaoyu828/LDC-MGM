import numpy as np
from astropy.io import fits
import os
import pandas as pd


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


generate = lambda xyz, peak, x0, y0, v0, size1, size2, size3, theta: peak * np.exp(-(
        ((xyz[:, 0] - x0) ** 2) * (
        np.cos(theta) ** 2 / (2 * size1 ** 2) + np.sin(theta) ** 2 / (2 * size2 ** 2)) + (
                (xyz[:, 1] - y0) ** 2) * (
                np.sin(theta) ** 2 / (2 * size1 ** 2) + np.cos(theta) ** 2 / (2 * size2 ** 2)) + (
                xyz[:, 0] - x0) * (xyz[:, 1] - y0) * (
                2 * (-np.sin(2 * theta) / (4 * size1 ** 2) + np.sin(2 * theta) / (4 * size2 ** 2))) + (
                (xyz[:, 2] - v0) ** 2) / (2 * size3 ** 2)))


def make_clumps(n, path1, size_v=100, size_y=100, size_x=100, number=10000):
    colsName2 = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
                 'Sum', 'Volume']  # 核表抬头
    x, y, z = np.mgrid[1:size_v + 1, 1:size_y + 1, 1:size_x + 1]
    xyz = np.column_stack([z.flat, y.flat, x.flat])
    rms = 0.23  # 噪声
    n_datacube = 0  # 用于计数
    n_datacube_ = number // n  # 生成数据块的个数
    path2 = os.path.join(path1, 'n_clump_%03d' % n)  # 次级目录
    path3_model = os.path.join(path2, 'model')  # 底层目录，无噪声fits目录
    path3_out = os.path.join(path2, 'out')  # 有噪声fits目录
    path3_outcat = os.path.join(path2, 'outcat')  # 核表目录
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3_model):
        os.makedirs(path3_model)
    if not os.path.exists(path3_out):
        os.makedirs(path3_out)
    if not os.path.exists(path3_outcat):
        os.makedirs(path3_outcat)
    fits_header = fits.open('./Generate/no_data.fits')[0].header  # 读取头文件
    total_data = []
    while 1:
        new_coreTable = []  # 存储核表数据
        res_data = np.zeros_like(x, dtype=np.float64)  # 数据块
        is_join = True  # 用于第一次计算
        number_for = 0  # 记录加入失败的次数
        i = 0  # 当前数据块中核的个数
        while i < n:
            peak = np.random.uniform(0.46, 3)  # 随机生成peak值
            x0 = np.random.uniform(5, size_x - 5)  # 随机生成位置坐标
            y0 = np.random.uniform(5, size_y - 5)
            v0 = np.random.uniform(5, size_v - 5)
            size1 = np.random.uniform(2, 4)
            size2 = np.random.uniform(2, 4)
            size3 = np.random.uniform(1, 7)
            angle = np.random.uniform(0, 180)  # 随机生成旋转角度
            if size1 < size2:
                size1, size2 = size2, size1
            if number_for == 2000:  # 若加入失败次数达到了2000次，则重新生成数据块
                n_datacube -= 1
                break
            if i != 0:  # 从数据块中加入第二个核开始，判断当前的核能否加入到数据块
                outcat1 = np.array(new_coreTable)[:, [1, 2, 3, 7, 8, 9]]  # 前i-1个核的相关数据
                outcat1[:, 3: 6] = outcat1[:, 3: 6] / 2.3548  # 当前核的相关数据
                outcat2 = np.array(
                    [[x0, y0, v0, size1, size2, size3]])
                is_join = is_separable_2(outcat1, outcat2)
            if is_join:  # 若第i个能加入，则执行
                res = generate(xyz, peak, x0, y0, v0, size1, size2, size3, ((90 + angle) % 180) / 180 * np.pi)
                data = res.reshape((size_v, size_y, size_x))
                Sum1 = data.sum()
                res_data += data
                Volume = int(len(np.where(data > 0.01)[0]))  # 计算体积
                new_coreTable.append(
                    [i + 1, x0, y0, v0, x0, y0, v0, size1 * 2.3548, size2 * 2.3548, size3 * 2.3548, angle, peak, Sum1,
                     Volume])
                i += 1
            else:  # 若第i个不能加入，则重新生成
                number_for += 1
                continue
        new_coreTable = np.array(new_coreTable)  # list转化成ndarray
        dataframe = pd.DataFrame(new_coreTable, columns=colsName2)
        outcat_name = os.path.join(path3_outcat, 'gaussian_outcat_%03d.txt' % n_datacube)
        dataframe = dataframe.round({'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Peak3': 3, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                     'Size1': 3, 'Size2': 3, 'Size3': 3, 'theta': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(outcat_name, sep='\t', index=False)

        data_hdu = fits.PrimaryHDU(res_data, header=fits_header)
        fits_name = os.path.join(path3_model, 'gaussian_model_%03d.fits' % n_datacube)
        fits.HDUList([data_hdu]).writeto(fits_name, overwrite=True)  # 无噪声数据存储

        [data_x, data_y, data_v] = res_data.shape
        noise_data = rms * np.random.randn(data_x, data_y, data_v)  # 加入高斯噪声
        out_data = res_data + noise_data
        total_data.append(out_data)
        data_hdu = fits.PrimaryHDU(out_data, header=fits_header)
        fits_name = os.path.join(path3_out, 'gaussian_out_%03d.fits' % n_datacube)
        fits.HDUList([data_hdu]).writeto(fits_name, overwrite=True)  # 有噪声数据存储
        n_datacube += 1
        if n_datacube == n_datacube_:
            break
    return total_data

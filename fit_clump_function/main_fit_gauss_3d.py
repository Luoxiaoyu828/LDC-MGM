import numpy as np
import astropy.io.fits as fits
import os
from scipy import integrate
import pandas as pd
from fit_clump_function import sympy_fit_gauss, touch_clump
from multiprocessing import Pool


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_fit_outcat_record(p_fit_single_clump):
    # p_fit_single_clump[5] = (p_fit_single_clump[5] / np.pi * 180) % 180
    # 将弧度转换成角度, 并将其转换到0-180度

    param_num = 8  # 一个二维高斯的参数个数(A0, x0, y0, s0_1,s0_2, theta_0)
    num = p_fit_single_clump.shape[0]  # 拟合结果中参数个数
    num_j = num // param_num  # 对输入参数取整， 得到二维高斯的个数
    outcat_record = np.zeros([num_j, 14], np.float)

    for para_item in range(num_j):
        p_fit_single_clump_ = p_fit_single_clump[para_item * param_num: para_item * param_num + 8]

        func = sympy_fit_gauss.gauss_3d_A(p_fit_single_clump_)
        try:
            Sum_ = integrate.nquad(func, [[1, 120], [1, 120], [1, 120]])
        except Warning:
            Sum_ = -1
            print(p_fit_single_clump_)

        p_fit_single_clump_[3] = p_fit_single_clump_[3] * 2.3548
        p_fit_single_clump_[4] = p_fit_single_clump_[4] * 2.3548
        p_fit_single_clump_[7] = p_fit_single_clump_[7] * 2.3548

        outcat_record[para_item, 1:4] = p_fit_single_clump_[[1, 2, 6]]  # Peak1,2,3
        outcat_record[para_item, 4:7] = p_fit_single_clump_[[1, 2, 6]]  # Cen1,2,3
        outcat_record[para_item, 7] = p_fit_single_clump_[[3, 4]].max()  # Size1,2,3
        outcat_record[para_item, 8] = p_fit_single_clump_[[3, 4]].min()  # Size1,2,3
        outcat_record[para_item, 9] = p_fit_single_clump_[7]  # Size1,2,3
        outcat_record[para_item, 10] = p_fit_single_clump_[5]  # theta
        outcat_record[para_item, 11] = p_fit_single_clump_[0]  # peak value
        outcat_record[para_item, 12] = Sum_  # Sum
        outcat_record[para_item, 13] = 0  # Volume  没有求解 可用LDC中聚类的结果

    return outcat_record


def get_fit_outcat_record_2d(p_fit_single_clump):

    col_num = 11  # 核表参数的列数
    param_num = 6  # 一个二维高斯的参数个数(A0, x0, y0, s0_1,s0_2, theta_0)
    # p_fit_single_clump[5] = (p_fit_single_clump[5] / np.pi * 180) % 180
    # 将弧度转换成角度, 并将其转换到0-180度

    num = p_fit_single_clump.shape[0]  # 拟合结果中参数个数
    num_j = num // param_num  # 对输入参数取整， 得到二维高斯的个数
    outcat_record = np.zeros([num_j, col_num], np.float)

    for para_item in range(num_j):
        p_fit_single_clump_ = p_fit_single_clump[para_item * param_num: para_item * param_num + param_num]
        # print(p_fit_single_clump_)
        # [ 1.58 46.93 11.70  2.29  2.86 31.374  80.75  4.363]
        func = sympy_fit_gauss.gauss_2d_A(p_fit_single_clump_)
        try:
            Sum_ = integrate.nquad(func, [[1, 800], [1, 800]])
        except Warning:
            Sum_ = -1
            print(p_fit_single_clump_)

        p_fit_single_clump_[3] = p_fit_single_clump_[3] * 2.3548
        p_fit_single_clump_[4] = p_fit_single_clump_[4] * 2.3548

        outcat_record[para_item, 1:3] = p_fit_single_clump_[[1, 2]]  # Peak1,2
        outcat_record[para_item, 3:5] = p_fit_single_clump_[[1, 2]]  # Cen1,2
        outcat_record[para_item, 5] = p_fit_single_clump_[[3, 4]].max()  # Size1,2
        outcat_record[para_item, 6] = p_fit_single_clump_[[3, 4]].min()  # Size1,2

        outcat_record[para_item, 7] = p_fit_single_clump_[5]  # theta
        outcat_record[para_item, 8] = p_fit_single_clump_[0]  # peak value
        outcat_record[para_item, 9] = Sum_  # total flux
        outcat_record[para_item, 10] = 0  # Volume  没有求解 可用LDC中聚类的结果

    return outcat_record


def save_fit_result(out_fits_name, detected_mask_name, detected_outcat_name, fit_outcat_name):

    print('processing file->%s' % fit_outcat_name)
    mask = fits.getdata(detected_mask_name)
    data = fits.getdata(out_fits_name)
    # f_outcat = np.loadtxt(detected_outcat_name, skiprows=1)
    f_outcat = pd.read_csv(detected_outcat_name, sep='\t')

    touch_clump_record = touch_clump.connect_clump(f_outcat, mult=2)

    [data_x, data_y, data_v] = data.shape
    Xin, Yin, Vin = np.mgrid[1:data_x+1, 1:data_y+1, 1:data_v+1]
    X = np.vstack([Vin.flatten(), Yin.flatten(), Xin.flatten()]).T  # 坐标原点为1
    mask_flatten = mask.flatten()
    Y = data.flatten()
    fit_outcat = np.zeros([f_outcat.shape[0], 14])
    params_all = np.array([f_outcat['Peak'], f_outcat['Cen1'], f_outcat['Cen2'], f_outcat['Size1'] / 2.3548,
                           f_outcat['Size2'] / 2.3548, f_outcat['ID'], f_outcat['Cen3'], f_outcat['Size3'] / 2.3548]).T
    params_all[:, 5] = 0

    for i, item_tcr in enumerate(touch_clump_record):
        XX2 = np.array([0, 0, 0])
        YY2 = np.array([0])
        params = np.array([0])
        for id_clumps in item_tcr:
            ind = np.where(mask_flatten == id_clumps)[0]
            X2 = X[ind, :]
            Y2 = Y[ind]

            XX2 = np.vstack([XX2, X2])
            YY2 = np.hstack([YY2, Y2])

            params = np.hstack([params, params_all[id_clumps-1]])
        XX2 = XX2[1:, :]
        YY2 = YY2[1:]
        params = params[1:]
        p_fit_single_clump = sympy_fit_gauss.fit_gauss_3d(XX2, YY2, params)
        # 对于多高斯参数  需解析 p_fit_single_clump
        outcat_record = get_fit_outcat_record(p_fit_single_clump)
        for i, id_clumps in enumerate(item_tcr):
            # 这里有可能会存在多个核同时拟合的时候，输入核的编号和拟合结果中核编号不一致，从而和mask不匹配的情况
            outcat_record[i, 0] = id_clumps  # 对云核的编号进行赋值
        fit_outcat[item_tcr-1, :] = outcat_record

    table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
                   'Sum', 'Volume']

    dataframe = pd.DataFrame(fit_outcat, columns=table_title)
    dataframe = dataframe.round({'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Peak3': 3, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                 'Size1': 3, 'Size2': 3, 'Size3': 3, 'theta': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
    dataframe.to_csv(fit_outcat_name, sep='\t', index=False)


def fit_pool():
    work_path = 'test_data_zhou_again'
    for item in [10, 25, 100]:
        # item = 25
        work_path_item = os.path.join(work_path, 'n_clump_%03d' % item)

        detected_save = os.path.join(work_path_item, 'detected_result')
        detected_save_mask = os.path.join(detected_save, 'mask')
        detected_save_outcat = os.path.join(detected_save, 'outcat')
        fit_save_outcat = os.path.join(detected_save, 'fit_outcat_0929')
        match_result_save = os.path.join(work_path_item, 'match_result')
        Match_table = os.path.join(match_result_save, 'Match_table')

        create_folder(fit_save_outcat)
        file_list = os.listdir(os.path.join(work_path_item, 'out'))
        p = Pool(64)
        out_file_path = os.path.join(work_path_item, 'out')
        outcat_file_path = os.path.join(work_path_item, 'outcat')
        for item_clump in range(len(file_list)):
            # item_clump = 0
            out_fits_name = os.path.join(out_file_path, 'gaussian_out_%03d.fits' % item_clump)
            outcat_name = os.path.join(outcat_file_path, 'gaussian_outcat_%03d.txt' % item_clump)
            detected_outcat_name = os.path.join(detected_save_outcat, 'f_outcat_%03d.txt' % item_clump)
            detected_mask_name = os.path.join(detected_save_mask, 'f_mask_%03d.fits' % item_clump)
            Match_table_name = os.path.join(Match_table, 'Match_%03d.txt' % item_clump)
            fit_outcat_name = os.path.join(fit_save_outcat, 'Fit_%03d.txt' % item_clump)
            print('fitting the file: %s' % out_fits_name)

            p.apply_async(save_fit_result, args=(out_fits_name, detected_mask_name, detected_outcat_name, fit_outcat_name))
            # save_fit_result(data, mask, f_outcat, fit_outcat_name)

        p.close()
        p.join()


if __name__ == '__main__':
    # fit_pool()

    work_path = 'test_data_zhou_again'
    # for item in [25,100]:
    item = 25
    work_path_item = os.path.join(work_path, 'n_clump_%03d' % item)

    detected_save = os.path.join(work_path_item, 'detected_result')
    detected_save_mask = os.path.join(detected_save, 'mask')
    detected_save_outcat = os.path.join(detected_save, 'outcat')
    fit_save_outcat = os.path.join(detected_save, 'fit_outcat1')
    match_result_save = os.path.join(work_path_item, 'match_result')
    Match_table = os.path.join(match_result_save, 'Match_table')

    create_folder(fit_save_outcat)
    file_list = os.listdir(os.path.join(work_path_item, 'out'))

    out_file_path = os.path.join(work_path_item, 'out')
    outcat_file_path = os.path.join(work_path_item, 'outcat')
    # for item_clump in range(len(file_list)):
    item_clump = 0
    out_fits_name = os.path.join(out_file_path, 'gaussian_out_%03d.fits' % item_clump)
    outcat_name = os.path.join(outcat_file_path, 'gaussian_outcat_%03d.txt' % item_clump)
    detected_outcat_name = os.path.join(detected_save_outcat, 'f_outcat_%03d.txt' % item_clump)
    detected_mask_name = os.path.join(detected_save_mask, 'f_mask_%03d.fits' % item_clump)
    Match_table_name = os.path.join(Match_table, 'Match_%03d.txt' % item_clump)
    fit_outcat_name = os.path.join(fit_save_outcat, 'Fit_%03d.txt' % item_clump)
    print('fitting the file: %s' % out_fits_name)

    save_fit_result(out_fits_name, detected_mask_name, detected_outcat_name, fit_outcat_name)



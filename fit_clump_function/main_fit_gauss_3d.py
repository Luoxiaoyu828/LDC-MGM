import numpy as np
import astropy.io.fits as fits
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from fit_clump_function import multi_gauss_fitting, multi_gauss_fitting_new, touch_clump
from tools.ultil_lxy import create_folder, get_points_by_clumps_id, move_csv_png, restruct_fitting_outcat,\
    get_save_clumps_xyv
from tools.show_clumps import display_clumps_fitting
from DensityClust.locatDenClust2 import Data


def get_fit_outcat(origin_data_name, mask_name, outcat_name):
    """
    对LDC的检测结果进行3维高斯拟合，得到分子云核的参数
    :param origin_data_name: 原始数据
    :param mask_name: 检测得到的掩模
    :param outcat_name: 检测得到的核表
    :return:
        拟合核表 DataFrame格式
['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak', 'Sum', 'Volume']
    """
    print('processing file->%s' % origin_data_name)
    mask = fits.getdata(mask_name)
    data = fits.getdata(origin_data_name)
    f_outcat = pd.read_csv(outcat_name, sep='\t')

    touch_clump_record = touch_clump.connect_clump(f_outcat, mult=1)  # 得到相互重叠的云核
    print('The overlapping clumps are selected.')

    [data_x, data_y, data_v] = data.shape
    Xin, Yin, Vin = np.mgrid[1:data_x + 1, 1:data_y + 1, 1:data_v + 1]
    X = np.vstack([Vin.flatten(), Yin.flatten(), Xin.flatten()]).T  # 坐标原点为1
    mask_flatten = mask.flatten()
    Y = data.flatten()
    fit_outcat = np.zeros([f_outcat.shape[0], 14])
    params_init = np.array([f_outcat['Peak'], f_outcat['Cen1'], f_outcat['Cen2'], f_outcat['Size1'] / 2.3548,
                            f_outcat['Size2'] / 2.3548, f_outcat['ID'], f_outcat['Cen3'], f_outcat['Size3'] / 2.3548]).T
    params_init[:, 5] = 0  # 初始化的角度

    print('The initial parameters have finished')
    for i, item_tcr in enumerate(touch_clump_record):
        # item_tcr = touch_clump_record[0]
        XX2 = np.array([0, 0, 0])
        YY2 = np.array([0])
        params = np.array([0])
        print(time.ctime() + 'touch_clump %d/%d' % (i, len(touch_clump_record)))
        for id_clumps_index in item_tcr:
            id_clumps = int(f_outcat['ID'].values[id_clumps_index - 1])
            ind = np.where(mask_flatten == id_clumps)[0]
            X2 = X[ind, :]
            Y2 = Y[ind]

            XX2 = np.vstack([XX2, X2])
            YY2 = np.hstack([YY2, Y2])
            params = np.hstack([params, params_init[id_clumps_index - 1]])

        XX2 = XX2[1:, :]
        YY2 = YY2[1:]
        params = params[1:]
        p_fit_single_clump = multi_gauss_fitting.fit_gauss_3d(XX2, YY2, params)
        # 对于多高斯参数  需解析 p_fit_single_clump
        outcat_record = multi_gauss_fitting_new.get_fit_outcat_record(p_fit_single_clump)
        for i, id_clumps in enumerate(item_tcr):
            # 这里有可能会存在多个核同时拟合的时候，输入核的编号和拟合结果中核编号不一致，从而和mask不匹配的情况
            outcat_record[i, 0] = id_clumps  # 对云核的编号进行赋值
        fit_outcat[item_tcr - 1, :] = outcat_record

    table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
                   'Sum', 'Volume']

    dataframe = pd.DataFrame(fit_outcat, columns=table_title)
    dataframe = dataframe.round({'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Peak3': 3, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                 'Size1': 3, 'Size2': 3, 'Size3': 3, 'theta': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
    return dataframe


def fitting_LDC_clumps(points_path, outcat_name):
    """
    对LDC的检测结果进行3维高斯拟合，保存分子云核的参数
    :param points_path: 云核坐标点及强度文件所在的文件夹路径
    :param outcat_name: 检测得到的核表
    :return:
        拟合核表 DataFrame格式
    ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak', 'Sum']
    """
    print('processing file->%s' % outcat_name)
    f_outcat = pd.read_csv(outcat_name, sep=',')
    if f_outcat.shape[1]==1:
        f_outcat = pd.read_csv(outcat_name, sep='\t')

    csv_png_folder = outcat_name.replace('.csv', '_fitting')
    create_folder(csv_png_folder)

    # 得到相互重叠的云核(ID)
    touch_clump_record, _ = touch_clump.connect_clump_new(f_outcat, mult=0.9)
    print('The overlapping clumps are selected.')

    params_init_all = np.array([f_outcat['Peak'], f_outcat['Cen1'], f_outcat['Cen2'], f_outcat['Size1'] / 2.3548,
                            f_outcat['Size2'] / 2.3548, f_outcat['ID'], f_outcat['Cen3'], f_outcat['Size3'] / 2.3548]).T
    params_init_all[:, 5] = 0  # 初始化的角度
    print('The initial parameters (Initial guess) have finished')

    for i, item_tcr in enumerate(touch_clump_record):
        fit_outcat_name = os.path.join(csv_png_folder, 'fit_item%03d.csv' % i)
        fig_name = os.path.join(csv_png_folder, 'touch_clumps_%03d.png' % i)

        print(time.ctime() + 'touch_clump %d/%d' % (i, len(touch_clump_record)))
        clumps_id = f_outcat.iloc[item_tcr - 1]['ID'].values.astype(np.int64)
        points_all_df = get_points_by_clumps_id(clumps_id, points_path)

        params_init = params_init_all[item_tcr - 1].flatten()
        print('Solving a nonlinear least-squares problem to find parameters.')

        outcat_fitting = multi_gauss_fitting_new.fitting_main(points_all_df, params_init, clumps_id)

        if fit_outcat_name.split('.')[-1] not in ['csv', 'txt']:
            print('the save file type must be one of *.csv and *.txt.')
        else:
            outcat_fitting.to_csv(fit_outcat_name, sep='\t', index=False)

        pif_1 = outcat_fitting[['Cen1', 'Cen2', 'Cen3']] - points_all_df[['x_2', 'y_1', 'v_0']].values.min(axis=0)
        df_temp_1 = f_outcat.iloc[item_tcr - 1]
        display_clumps_fitting(pif_1, df_temp_1, points_all_df)

        fig = plt.gcf()
        fig.savefig(fig_name)
        plt.close(fig)

    move_csv_png(csv_png_folder)
    restruct_fitting_outcat(csv_png_folder)


def a_little_revise(outcat_name_loc):
    """
    1.1.5--->1.1.6 修复bug: clump_size[clump_ii, :] = 2.3548 * np.array([size_1[0], size_2[0], size_3[0])
    ---> clump_size[clump_ii, :] = 2.3548 * np.array([size_1[ind[0]], size_2[ind[0]], size_3[ind[0]]])
    在选择流量最大连通域中，对应的size也应跟着一起选择 而不是默认选择第一个
    R16和R2的结果，size: 第一个轴和第三个轴需要调换一下，基于wcs的核表需要转换成像素后再转换

    故在本次结果中需要做修改  后面的检测结果就不需要了
    """
    outcat_temp1 = pd.DataFrame([])
    outcat_temp = pd.read_csv(outcat_name_loc, sep='\t')
    if outcat_temp.shape[1]==1:
        outcat_temp = pd.read_csv(outcat_name_loc, sep=',')
    outcat_temp1[['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1',
                  'Size2', 'Size3', 'Peak', 'Sum', 'Volume']] = outcat_temp[
        ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size3',
         'Size2', 'Size1', 'Peak', 'Sum', 'Volume']]
    outcat_name_loc1 = outcat_name_loc.replace('.csv', '_revise.csv')
    outcat_temp1.to_csv(outcat_name_loc1, sep='\t', index=False)
    return outcat_name_loc1


def LDC_para_fit_Main(outcat_name_loc, points_path, origin_name, mask_name, re_outcat_path, mm_outcat):
    """

    """
    outcat_name_loc = a_little_revise(outcat_name_loc)

    # step 1: 准备拟合数据
    get_save_clumps_xyv(origin_name, mask_name, outcat_name_loc, points_path)

    # step 2: 进行拟合
    fitting_LDC_clumps(points_path, outcat_name_loc)

    # step 3: 将拟合核表整理成最终核表
    data_int = Data(origin_name)
    data_int.get_wcs()
    data_wcs = data_int.wcs
    re_outcat = pd.read_csv(re_outcat_path, sep='\t')
    mm_outcat_df = multi_gauss_fitting_new.exchange_pix2world(re_outcat, data_wcs)
    mm_outcat_df.to_csv(mm_outcat, sep='\t', index=False)


if __name__ == '__main__':
    outcat_name_loc = r'F:\Parameter_reduction\LDC\0155+030_L\LDC_auto_loc_outcat.csv'
    points_path = r'F:\Parameter_reduction\LDC\0155+030_L\0155+030_L_points'
    origin_name = r'F:\Parameter_reduction\LDC\0155+030_L\0155+030_L.fits'
    mask_name = r'F:\Parameter_reduction\LDC\0155+030_L\LDC_auto_mask.fits'
    re_outcat_path = r'F:\Parameter_reduction\LDC\0155+030_L\LDC_auto_loc_outcat_fitting\fitting_outcat.csv'
    mm_outcat = re_outcat_path.replace('fitting_outcat.csv', 'MWISP_outcat.csv')

    LDC_para_fit_Main(outcat_name_loc, points_path, origin_name, mask_name, re_outcat_path, mm_outcat)

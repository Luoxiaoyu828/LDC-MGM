import os
from astropy.coordinates import SkyCoord
from tools.show_clumps import make_plot_wcs_1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
from DensityClust.localDenClust2 import LocalDensityCluster as LDC
from DensityClust.localDenClust2 import Data
from t_match.match_6_ import match_simu_detect_new


split_list = [[0, 150, 0, 150], [0, 150, 90, 271], [0, 150, 210, 361],
              [90, 241, 0, 150], [90, 241, 90, 271], [90, 241, 210, 361]]


def split_cube_lxy(data_path, save_folder_all):

    split_list = [[0, 150, 0, 150], [0, 150, 90, 271], [0, 150, 210, 361],
                  [90, 241, 0, 150], [90, 241, 90, 271], [90, 241, 210, 361]]
    data_cube = SpectralCube.read(data_path)
    sub_cube_path_list = []
    for i, item in enumerate(split_list):
        sub_cude = data_cube[:, item[0]: item[1], item[2]: item[3]]
        # 更换文件路径
        sub_cube_path = os.path.join(save_folder_all, os.path.basename(data_path))
        sub_cube_path = sub_cube_path.replace('.fits', '_%02d.fits' % i)
        if not os.path.exists(sub_cube_path):
            sub_cude.write(sub_cube_path)
        sub_cube_path_list.append(sub_cube_path)
    return sub_cube_path_list


def get_outcat_i(outcat, i):
    outcat_split = [[0, 120, 0, 120], [0, 120, 30, 150], [0, 120, 30, 150],
                    [30, 150, 0, 120], [30, 150, 30, 150], [30, 150, 30, 150]]
    [cen2_min, cen2_max, cen1_min, cen1_max] = outcat_split[i]
    aa = outcat.loc[outcat['Cen1'] > cen1_min]
    aa = aa.loc[outcat['Cen1'] <= cen1_max]

    aa = aa.loc[outcat['Cen2'] > cen2_min]
    loc_outcat = aa.loc[outcat['Cen2'] <= cen2_max]
    return loc_outcat


def get_outcat_wcs_all(save_path, data_all_path):
    """
    对分块检测结果的整合
    :param save_path: LDC分块检测结果保存位置
    :param data_all_path: 原始待检测数据位置
    :return:
    """

    outcat_wcs_all = pd.DataFrame([])
    outcat_wcs_path = os.path.join(save_path, 'LDC_auto_outcat_wcs.csv')
    outcat_path = os.path.join(save_path, 'LDC_auto_outcat.csv')

    file_name = os.path.basename(data_all_path).replace('.fits', '')
    for i in range(6):
        # i = 1
        # outcat_i_path = r'test_data/synthetic/synthetic_model_0000_%02d/LDC_auto_outcat.csv' % i
        outcat_i_path = os.path.join(save_path, file_name + r'_%02d\LDC_auto_outcat.csv' % i)
        outcat_i = pd.read_csv(outcat_i_path, sep='\t')
        loc_outcat_i = get_outcat_i(outcat_i, i)

        origin_data_name = os.path.join(save_path, file_name + r'_%02d.fits' % i)
        data = Data(origin_data_name)
        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(loc_outcat_i)
        outcat_wcs_all = pd.concat([outcat_wcs_all, outcat_wcs], axis=0)

    outcat_wcs_all.to_csv(outcat_wcs_path, sep='\t', index=False)
    data = Data(data_path)
    ldc = LDC(data=data, para=None)
    data_wcs = ldc.data.wcs
    outcat_wcs_all = pd.read_csv(outcat_wcs_path, sep='\t')

    outcat_all = change_world2pix(outcat_wcs_all, data_wcs)
    outcat_all.to_csv(outcat_path, sep='\t', index=False)
    fig_name = os.path.join(save_path, 'LDC_auto_detect_result.png')
    make_plot_wcs_1(outcat_wcs_all, data_wcs, data.data_cube, fig_name=fig_name)


def make_plot_wcs_data_outcat(data_name, outcat_wcs_path):
    """
    将银经银纬核表绘制在数据块上
    :param data_name：分子云核数据块路径
    :param outcat_wcs_path：银经银纬核表路径
    """
    data = Data(data_name)
    outcat_wcs_all = pd.read_csv(outcat_wcs_path, sep='\t')
    data_wcs = data.wcs
    data_cube = data.data_cube
    make_plot_wcs_1(outcat_wcs_all, data_wcs, data_cube)


def make_plot_ij(match_outcat_path, col_names_g=None, col_names_s=None):
    """
    将匹配结果进行绘制
        :param match_outcat_path：匹配核表
        :param col_names_g：检测核表要比较的列名，如：['Cen1', 'Cen2', 'Cen3']
        :param col_names_s：仿真核表要比较的列名，如：['Cen1', 'Cen2', 'Cen3']
    return:
        绘制的1*3的图片
    """
    match_outcat = pd.read_csv(match_outcat_path, sep='\t')
    if col_names_s is None:
        col_names_s = ['Cen1', 'Cen2', 'Cen3']
    if col_names_g is None:
        col_names_g = ['Cen1', 'Cen2', 'Cen3']
    fig = plt.figure(figsize=[15, 4])
    ii = 1
    for c_n_g, c_n_s in zip(col_names_g, col_names_s):
        col_name_g = 'f_' + c_n_g
        col_name_s = 's_' + c_n_s
        data_g = match_outcat[col_name_g].values
        data_s = match_outcat[col_name_s].values
        ax = fig.add_subplot(1, 3, ii + 1)
        ax.scatter(data_g, data_s)
        ax.set_xlabel(col_name_g)
        ax.set_ylabel(col_name_s)
        ii += 1
    plt.show()


def compare_version(sop, dop, msp, s_cen=None, s_szie=None, g_cen=None):
    """
    将仿真核表和检测核表进行匹配，并把匹配结果的中心坐标散点图画出来.
    其他参数如要绘制，请调用make_plot_ij()函数
        :param sop：仿真核表路径(也可以是其他作为标准的核表)
        :param dop：检测核表路径
        :param msp：匹配核表保存路径
        :param s_cen：仿真核表的质心列名，默认为：['Cen1', 'Cen2', 'Cen3']
        :param s_cen：仿真核表的轴长列名，默认为：['Size1', 'Size2', 'Size3']
        :param g_cen：检测核表的质心列名，默认为：['Cen1', 'Cen2', 'Cen3']
    return:
        None
    """

    if g_cen is None:
        g_cen = ['Cen1', 'Cen2', 'Cen3']
    if s_szie is None:
        s_szie = ['Size1', 'Size2', 'Size3']
    if s_cen is None:
        s_cen = ['Cen1', 'Cen2', 'Cen3']
    match_cfg = match_simu_detect_new(sop, dop, msp, s_cen=s_cen, s_szie=s_szie, g_cen=g_cen)

    mop = match_cfg['Match_table_name']
    make_plot_ij(mop, col_names_g=g_cen, col_names_s=s_cen)


def change_world2pix(outcat, data_wcs):
    """
    将算法检测的结果(像素单位)转换到天空坐标系上去
    :return:
    outcat_wcs
    ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume']
    -->3d

     ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
     -->2d
    """
    # outcat_record = self.result.outcat_record
    table_title = outcat.keys()
    if outcat is None:
        return None
    else:
        if 'Cen3' not in table_title:
            # 2d result
            peak1, peak2 = data_wcs.all_world2pix(outcat['Peak1'], outcat['Peak2'], 1)
            cen1, cen2 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1)
            size1, size2 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30])

            clump_Peak = np.column_stack([peak1, peak2])
            clump_Cen = np.column_stack([cen1, cen2])
            clustSize = np.column_stack([size1, size2])
            clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

            id_clumps = []  # MWSIP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
            for item_l, item_b in zip(cen1, cen2):
                str_l = 'MWSIP' + ('%.03f' % item_l).rjust(7, '0')
                if item_b < 0:
                    str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
                else:
                    str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
                id_clumps.append(str_l + str_b)
            id_clumps = np.array(id_clumps)

        elif 'Cen3' in table_title:
            # 3d result
            peak1, peak2, peak3 = data_wcs.all_world2pix(outcat['Peak1'], outcat['Peak2'], outcat['Peak3']*1000, 1)
            cen1, cen2, cen3 = data_wcs.all_world2pix(outcat['Cen1'], outcat['Cen2'], outcat['Cen3']*1000, 1)
            size1, size2, size3 = np.array([outcat['Size1'] / 30, outcat['Size2'] / 30, outcat['Size3'] / 0.166])
            clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

            clump_Peak = np.column_stack([peak1, peak2, peak3])
            clump_Cen = np.column_stack([cen1, cen2, cen3])
            clustSize = np.column_stack([size1, size2, size3])
            id_clumps = outcat['ID']
        else:
            print('outcat_record columns name are: ' % table_title)
            return None

        outcat = np.column_stack(
            (id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))
        outcat = pd.DataFrame(outcat, columns=table_title)
        return outcat


def change_world2pix_fit(outcat, data_wcs):
    """
    将算法检测的结果(像素单位)转换到天空坐标系上去
    :return:
    outcat_wcs
    ['ID', 'Galactic_Longitude', 'Galactic_Latitude', 'Velocity',
       'Size_major', 'Size_minor', 'Size_velocity', 'Theta', 'Peak', 'Flux',
       'Flux_SNR', 'Peak_SNR', 'Success', 'Cost']
    -->3d

     ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
     -->2d
    """
    # outcat_record = self.result.outcat_record
    table_title = outcat.keys()
    Theta = outcat['Theta'].values
    Flux_SNR_Cost = outcat[['Flux_SNR', 'Peak_SNR', 'Success', 'Cost']]
    if outcat is None:
        return None
    else:
        if 'Velocity' not in table_title:
            # 2d result
            peak1, peak2 = data_wcs.all_world2pix(outcat['Peak1'], outcat['Peak2'], 1)
            cen1, cen2 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1)
            size1, size2 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30])

            clump_Peak = np.column_stack([peak1, peak2])
            clump_Cen = np.column_stack([cen1, cen2])
            clustSize = np.column_stack([size1, size2])
            clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

            id_clumps = []  # MWSIP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
            for item_l, item_b in zip(cen1, cen2):
                str_l = 'MWSIP' + ('%.03f' % item_l).rjust(7, '0')
                if item_b < 0:
                    str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
                else:
                    str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
                id_clumps.append(str_l + str_b)
            id_clumps = np.array(id_clumps)

        elif 'Velocity' in table_title:
            # 3d result
            cen1, cen2, cen3 = data_wcs.all_world2pix(outcat['Galactic_Longitude'], outcat['Galactic_Latitude'],
                                                      outcat['Velocity'] * 1000, 1)
            size1, size2, size3 = np.array(
                [outcat['Size_major'] / 30, outcat['Size_minor'] / 30, outcat['Size_velocity'] / 0.166])
            clustPeak, clustSum = np.array([outcat['Peak'], outcat['Flux']])

            clump_Cen = np.column_stack([cen1, cen2, cen3])
            clustSize = np.column_stack([size1, size2, size3])
            id_clumps = outcat['ID']
        else:
            print('outcat_record columns name are: ' % table_title)
            return None

        outcat = np.column_stack(
            (id_clumps, clump_Cen, clustSize, Theta, clustPeak, clustSum, Flux_SNR_Cost))
        outcat = pd.DataFrame(outcat, columns=table_title)
        return outcat


if __name__ == '__main__':
    data_path = r'D:\LDC\test_data\R2_R16_region\0145-005_L.fits'
    save_path = r'D:\LDC\test_data\R2_R16_region\0145-005_L13_noise_2_rho_3'
    outcat_all = get_outcat_wcs_all(save_path, data_path)
    outcat_all.to_csv(os.path.join(save_path, 'LDC_auto_outcat.csv'), index=False)


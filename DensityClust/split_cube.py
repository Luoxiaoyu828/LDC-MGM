import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
from tools.show_clumps import make_plot_wcs_1
from DensityClust.localDenClust2 import LocalDensityCluster as LDC
from DensityClust.localDenClust2 import Data
from t_match.match_6_ import match_simu_detect


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
    outcat_all = pd.DataFrame([])
    for i in range(6):
        # i = 1
        # outcat_i_path = r'test_data/synthetic/synthetic_model_0000_%02d/LDC_auto_outcat.csv' % i
        outcat_i_path = os.path.join(save_path, r'0145-005_L_%02d\LDC_auto_outcat.csv' % i)
        outcat_i = pd.read_csv(outcat_i_path, sep='\t')
        loc_outcat_i = get_outcat_i(outcat_i, i)

        origin_data_name = os.path.join(save_path, r'0145-005_L_%02d.fits' % i)
        data = Data(origin_data_name)
        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(loc_outcat_i)
        outcat_wcs_all = pd.concat([outcat_wcs_all, outcat_wcs], axis=0)
        outcat_all = pd.concat([outcat_all, loc_outcat_i], axis=0)
    data = Data(data_all_path)
    ldc = LDC(data=data, para=None)
    data_wcs = ldc.data.wcs
    data_cube = ldc.data.data_cube
    make_plot_wcs_1(outcat_wcs_all, data_wcs, data_cube)
    # fig = plt.gcf()
    # fig_name = os.path.join(save_path, 'LDC_auto_detect_result1.png')
    # fig.savefig(fig_name, bbox_inches='tight')
    # plt.close(fig=fig)


def make_data_outcat_wcs(data_name, outcat_wcs_all_path):
    data = Data(data_name)
    outcat_wcs_all = pd.read_csv(outcat_wcs_all_path, sep='\t')
    ldc = LDC(data=data, para=None)
    data_wcs = ldc.data.wcs
    data_cube = ldc.data.data_cube
    make_plot_wcs_1(outcat_wcs_all, data_wcs, data_cube)


def make_plot_ij(match_outcat, col_names=['Cen1', 'Cen2', 'Cen3']):

    fig = plt.figure(figsize=[15,4])
    for ii, cen in enumerate(col_names):
        plt_xy = []
        xy_label = []
        for fs in ['f_', 's_']:
            col_name = fs + cen
            xy_label.append(col_name)
            plt_xy.append(match_outcat[col_name].values)
        ax = fig.add_subplot(1,3,ii+1)
        ax.scatter(plt_xy[0], plt_xy[1])
        ax.set_xlabel(xy_label[0])
        ax.set_ylabel(xy_label[1])
    plt.show()


def compare_version(sop, dop, msp):
    """
    sop: 整体检测核表路径
    dop：分块检测核表路径
    msp：匹配结果保存路径
    """
    # sop = r'test_data/no_split_outcat/LDC_auto_outcat.csv'
    # dop = r'test_data/no_split_outcat/LDC_auto_outcat_pinjie.csv'
    # sy_op = r'test_data/synthetic/synthetic_outcat_0000.csv'
    # msp = r'test_data/no_split_outcat/match_sy'
    match_simu_detect(simulated_outcat_path=sop, detected_outcat_path=dop, match_save_path=msp)

    mop = msp + '/Match_table/Match_outcat.txt'
    match_outcat = pd.read_csv(mop, sep='\t')
    make_plot_ij(match_outcat, col_names=['Size1', 'Size2', 'Size3'])
    make_plot_ij(match_outcat, col_names=['Peak', 'Sum', 'Volume'])
    make_plot_ij(match_outcat, col_names=['Cen1', 'Cen2', 'Cen3'])


def change_pix2world(outcat, data_wcs):
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


if __name__ == '__main__':
    data_path = r'D:\Users\Administrator\Desktop\zhanr\MyProject_2\0145-005_L_4_6_True\0145-005_L.fits'
    save_path = r'D:\Users\Administrator\Desktop\zhanr\MyProject_2\0145-005_L_4_6_True'
    get_outcat_wcs_all(save_path, data_path)

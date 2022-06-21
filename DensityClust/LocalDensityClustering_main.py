import os
import astropy.io.fits as fits
import shutil
import pandas as pd
from DensityClust.localDenClust2 import LocalDensityCluster as LDC
from tools.show_clumps import deal_data
import split_cube
# from DensityClust.locatDenClust3 import Data, Param, LocalDensityCluster
from DensityClust.localDenClust2 import Data, Param, LocalDensityCluster


def ldc_base(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name, fig_name):
    ldc = LocalDensityCluster(data=data, para=para)
    ldc.detect()
    ldc.save_detect_log(detect_log)

    ldc.result.save_outcat(outcat_name, loc=0)
    ldc.result.save_outcat_wcs(outcat_wcs_name, loc=0)

    ldc.result.save_outcat(loc_outcat_name, loc=1)
    ldc.result.save_outcat_wcs(loc_outcat_wcs_name, loc=1)

    ldc.result.save_mask(mask_name)
    ldc.result.make_plot_wcs_1(fig_name)
    print(ldc.data.data_path + ' has finished!')


def ldc_base_split(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name, fig_name):
    ldc = LocalDensityCluster(data=data, para=para)
    ldc.detect()
    ldc.save_detect_log(detect_log)
    # outcat_split = [[0, 120, 0, 120], [30, 120, 0, 120], [30, 150, 0, 120],
    #                 [30, 150, 0, 120], [30, 150, 0, 120], [30, 150, 0, 120]]
    # outcat = ldc.result.outcat
    # outcat_i = pd.read_csv(r'test_data/synthetic/synthetic_model_0000_00/LDC_auto_outcat.csv', sep='\t')

    ldc.result.save_outcat(outcat_name, loc=0)
    ldc.result.save_outcat_wcs(outcat_wcs_name, loc=0)

    ldc.result.save_outcat(loc_outcat_name, loc=1)
    ldc.result.save_outcat_wcs(loc_outcat_wcs_name, loc=1)

    ldc.result.save_mask(mask_name)
    ldc.result.make_plot_wcs_1(fig_name)
    print(ldc.data.data_path + ' has finished!')


def localDenCluster(data_name, para, mask_name=None, outcat_name=None, outcat_wcs_name=None, loc_outcat_name=None,
                    loc_outcat_wcs_name=None, detect_log=None, fig_name='', paras_set=None):
    """
    LDC algorithm
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数，dict
        para.rho_min: Minimum density [5*rms]
        para.delta_min: Minimum delta [4]
        para.v_min: Minimum volume [27]
        para.noise: The noise level of the data, used for data truncation calculation [2*rms]
        para.dc: auto
    :param mask_name: 掩模数据的保存路径(str) [*.fits]
    :param outcat_name: 基于像素单位的核表保存路径(str) [*.csv]
    :param outcat_wcs_name: 基于wcs的核表保存路径(str) [*.csv]
    :param loc_outcat_name: 基于像素单位的局部区域核表保存路径(str) [*.csv]
    :param loc_outcat_wcs_name: 基于wcs的局部区域核表保存路径(str) [*.csv]
    :param detect_log: 检测中的信息保存文件(str) [*.txt]
    :param flags: 代码调用还是软件界面调用，默认为True(代码调用)
    :return:
    """
    data = Data(data_name)
    para.set_rms_by_data(data)

    if paras_set is not None:
        para.set_para(paras_set)

    ldc_base(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name,
             fig_name)


def localDenCluster_split_mode(data_name, para, save_folder_all=None, paras_set=None):
    """
    LDC algorithm
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数，dict
        para.rho_min: Minimum density [5*rms]
        para.delta_min: Minimum delta [4]
        para.v_min: Minimum volume [27]
        para.noise: The noise level of the data, used for data truncation calculation [2*rms]
        para.dc: auto
    :param mask_name: 掩模数据的保存路径(str) [*.fits]
    :param outcat_name: 基于像素单位的核表保存路径(str) [*.csv]
    :param outcat_wcs_name: 基于wcs的核表保存路径(str) [*.csv]
    :param loc_outcat_name: 基于像素单位的局部区域核表保存路径(str) [*.csv]
    :param loc_outcat_wcs_name: 基于wcs的局部区域核表保存路径(str) [*.csv]
    :param detect_log: 检测中的信息保存文件(str) [*.txt]
    :param flags: 代码调用还是软件界面调用，默认为True(代码调用)
    :return:
    """
    # data = Data(data_name)
    if save_folder_all is None:
        save_folder_all = data_name.replace('.fits', '')
        os.makedirs(save_folder_all, exist_ok=True)

    sub_cube_path_list = split_cube.split_cube_lxy(data_name)
    outcat_wcs_all = pd.DataFrame([])

    for ii, data_name in enumerate(sub_cube_path_list):

        save_folder = data_name.replace('.fits', '')
        os.makedirs(save_folder, exist_ok=True)

        mask_name = os.path.join(save_folder, 'LDC_auto_mask.fits')
        outcat_name = os.path.join(save_folder, 'LDC_auto_outcat.csv')
        outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_outcat_wcs.csv')
        loc_outcat_name = os.path.join(save_folder, 'LDC_auto_loc_outcat.csv')
        loc_outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_loc_outcat_wcs.csv')
        detect_log = os.path.join(save_folder, 'LDC_auto_detect_log.txt')
        fig_name = os.path.join(save_folder, 'LDC_auto_detect_result.png')

        data = Data(data_name)
        para.set_rms_by_data(data)

        if paras_set is not None:
            para.set_para(paras_set)

        ldc_base_split(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name,
                       mask_name, fig_name)

        # 处理局部块的核表
        outcat_i = pd.read_csv(outcat_name, sep='\t')
        loc_outcat_i = split_cube.get_outcat_i(outcat_i, ii)

        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(loc_outcat_i)
        outcat_wcs_all = pd.concat([outcat_wcs_all, outcat_wcs], axis=0)

        shutil.copy(detect_log, os.path.join(save_folder_all, 'LDC_auto_detect_log_%02d.txt' % ii))

    outcat_all_name = os.path.join(save_folder_all, 'LDC_auto_outcat.csv' % ii)
    outcat_wcs_all.to_csv(outcat_all_name, index=False)


def LDC_main(data_name, para, save_folder=None, split=False):
    """
    LDC算法入口，对指定的数据进行检测，将结果保存到指定位置
    data_name：数据文件[*.fits]
    para: LDC 算法的参数设置
    save_folder：检测结果保存路径
    """
    if not split:
        if save_folder is None:
            save_folder = data_name.replace('.fits', '')
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        mask_name = os.path.join(save_folder, 'LDC_auto_mask.fits')
        outcat_name = os.path.join(save_folder, 'LDC_auto_outcat.csv')
        outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_outcat_wcs.csv')

        loc_outcat_name = os.path.join(save_folder, 'LDC_auto_loc_outcat.csv')
        loc_outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_loc_outcat_wcs.csv')

        detect_log = os.path.join(save_folder, 'LDC_auto_detect_log.txt')
        fig_name = os.path.join(save_folder, 'LDC_auto_detect_result.png')
        localDenCluster(data_name, para, mask_name=mask_name, outcat_name=outcat_name, outcat_wcs_name=outcat_wcs_name,
                        loc_outcat_name=loc_outcat_name, loc_outcat_wcs_name=loc_outcat_wcs_name, detect_log=detect_log,
                        fig_name=fig_name)
    else:
        localDenCluster_split_mode(data_name, para)



if __name__ == '__main__':
    data_name = r'D:\LDC\test_data\synthetic\synthetic_model_0000.fits'
    para = Param(delta_min=4, gradmin=0.01, v_min=27, noise_times=3, rms_times=6)
    save_folder = r'D:\LDC\test_data\synthetic\synthetic_model'
    LDC_main(data_name, para, save_folder, split=True)

    # deal_data(save_folder, os.path.join(save_folder, 'ei'))
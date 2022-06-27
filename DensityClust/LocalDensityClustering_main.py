import os
import shutil
import pandas as pd
from DensityClust.localDenClust2 import LocalDensityCluster as LDC
from DensityClust import split_cube
from tools.show_clumps import make_plot_wcs_1
from DensityClust.localDenClust2 import Data, Param


def ldc_base(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name, fig_name):
    """
    :param mask_name: 掩模数据的保存路径(str) [*.fits]
    :param outcat_name: 基于像素单位的核表保存路径(str) [*.csv]
    :param outcat_wcs_name: 基于wcs的核表保存路径(str) [*.csv]
    :param loc_outcat_name: 基于像素单位的局部区域核表保存路径(str) [*.csv]
    :param loc_outcat_wcs_name: 基于wcs的局部区域核表保存路径(str) [*.csv]
    :param detect_log: 检测中的信息保存文件(str) [*.txt]
    :param fig_name: 检测结果可视化图片的路径(str) [*.png]
    """

    ldc = LDC(data=data, para=para)
    ldc.detect()
    ldc.save_detect_log(detect_log)

    ldc.result.save_outcat(outcat_name, loc=0)
    ldc.result.save_outcat_wcs(outcat_wcs_name, loc=0)

    if loc_outcat_name is not None:
        ldc.result.save_outcat(loc_outcat_name, loc=1)

    if loc_outcat_wcs_name is not None:
        ldc.result.save_outcat_wcs(loc_outcat_wcs_name, loc=1)

    ldc.result.save_mask(mask_name)
    ldc.result.make_plot_wcs_1(fig_name)
    print(ldc.data.data_path + ' has finished!')


def localDenCluster(data_name, para, save_folder, save_loc):
    """
    LDC algorithm
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数，dict
        para.rho_min: Minimum density [5*rms]
        para.delta_min: Minimum delta [4]
        para.v_min: Minimum volume [27]
        para.noise: The noise level of the data, used for data truncation calculation [2*rms]
        para.dc: auto
    :param save_folder：检测结果保存路径，如果是分块模式，中间结果也会保存
    :param save_loc: 是否保存检测的局部核表，默认为False(不保存)
    :return:
    结果保存的路径列表
    """
    ldc_cfg = {}
    mask_name = os.path.join(save_folder, 'LDC_auto_mask.fits')
    outcat_name = os.path.join(save_folder, 'LDC_auto_outcat.csv')
    outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_outcat_wcs.csv')
    detect_log = os.path.join(save_folder, 'LDC_auto_detect_log.txt')
    fig_name = os.path.join(save_folder, 'LDC_auto_detect_result.png')
    ldc_cfg['mask_name'] = mask_name
    ldc_cfg['outcat_name'] = outcat_name
    ldc_cfg['outcat_wcs_name'] = outcat_wcs_name
    ldc_cfg['detect_log'] = detect_log
    ldc_cfg['fig_name'] = fig_name
    if save_loc:
        loc_outcat_name = os.path.join(save_folder, 'LDC_auto_loc_outcat.csv')
        loc_outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_loc_outcat_wcs.csv')
        ldc_cfg['loc_outcat_name'] = loc_outcat_name
        ldc_cfg['loc_outcat_wcs_name'] = loc_outcat_wcs_name
    else:
        loc_outcat_name = None
        loc_outcat_wcs_name = None

    data = Data(data_name)
    para.set_rms_by_data(data)

    ldc_base(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name,
             fig_name)
    return ldc_cfg


def localDenCluster_split_mode(data_name, para, save_folder_all, save_loc):
    """
    LDC algorithm
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数，dict
        para.rho_min: Minimum density [5*rms]
        para.delta_min: Minimum delta [4]
        para.v_min: Minimum volume [27]
        para.noise: The noise level of the data, used for data truncation calculation [2*rms]
        para.dc: auto
    :param save_folder_all：检测结果保存路径，如果是分块模式，中间结果也会保存
    :param save_loc: 是否保存检测的局部核表，默认为False(不保存)
    :return:
    """
    outcat_wcs_all_name = os.path.join(save_folder_all, 'LDC_auto_outcat_wcs.csv')      # 拼接的银经银纬核表路径
    outcat_all_name = os.path.join(save_folder_all, 'LDC_auto_outcat.csv')      # 拼接的像素核表路径
    fig_name = os.path.join(save_folder_all, 'LDC_auto_detect_result.png')

    sub_cube_path_list = split_cube.split_cube_lxy(data_name, save_folder_all)
    outcat_wcs_all = pd.DataFrame([])  # 拼接的银经银纬核表

    for ii, sub_data_name in enumerate(sub_cube_path_list):
        sub_save_folder = sub_data_name.replace('.fits', '')
        os.makedirs(sub_save_folder, exist_ok=True)
        ldc_cfg = localDenCluster(sub_data_name, para, sub_save_folder, save_loc)

        # 处理局部块的核表
        outcat_name = ldc_cfg['outcat_name']
        detect_log = ldc_cfg['detect_log']
        outcat_i = pd.read_csv(outcat_name, sep='\t')
        loc_outcat_i = split_cube.get_outcat_i(outcat_i, ii)  # 取局部数据块的局部核表，根据块的位置，局部区域会有不同
        data = Data(sub_data_name)
        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(loc_outcat_i)
        shutil.copy(detect_log, os.path.join(save_folder_all, 'LDC_auto_detect_log_%02d.txt' % ii))

        outcat_wcs_all = pd.concat([outcat_wcs_all, outcat_wcs], axis=0)

    outcat_wcs_all.to_csv(outcat_wcs_all_name, sep='\t', index=False)   # 保存整合的银经银纬的核表
    # 保存整合的像素的核表及绘制检测云核的位置
    data = Data(data_name)
    ldc = LDC(data=data, para=None)
    data_wcs = ldc.data.wcs
    outcat_wcs_all = pd.read_csv(outcat_wcs_all_name, sep='\t')
    outcat_all = split_cube.change_world2pix(outcat_wcs_all, data_wcs)
    outcat_all.to_csv(outcat_all_name, sep='\t', index=False)
    make_plot_wcs_1(outcat_wcs_all, data_wcs, data.data_cube, fig_name=fig_name)


def LDC_main(data_name, para, save_folder=None, split=False, save_loc=False):
    """
    LDC算法入口，对指定的数据进行检测，将结果保存到指定位置
    :param data_name：数据文件[*.fits]
    :param para: LDC 算法的参数设置
    :param save_folder：检测结果保存路径，如果是分块模式，中间结果也会保存
    :param split: 是否采用分块检测再拼接策略，默认为False(不分块)
    :param save_loc: 是否保存检测的局部核表，默认为False(不保存)
    """
    if save_folder is None:
        save_folder = data_name.replace('.fits', '')
    os.makedirs(save_folder, exist_ok=True)

    if split:
        localDenCluster_split_mode(data_name, para, save_folder_all=save_folder, save_loc=save_loc)
    else:
        localDenCluster(data_name, para, save_folder=save_folder, save_loc=save_loc)


if __name__ == '__main__':
    data_name = r'D:\LDC\test_data\R2_R16_region\0145-005_L.fits'
    para = Param(delta_min=4, gradmin=0.01, v_min=27, noise_times=21, rms_times=25)
    save_folder = r'D:\LDC\test_data\R2_R16_region\0145-005_L13_noise_21_rho_25_126'
    LDC_main(data_name, para, save_folder, split=True, save_loc=False)

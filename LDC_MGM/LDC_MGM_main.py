import os
import shutil
import pandas as pd
from DensityClust.localDenClust2 import LocalDensityCluster as LDC
from DensityClust import split_cube
from tools import show_clumps
from DensityClust.localDenClust2 import Data, Param
from DensityClust.LocalDensityClustering_main import localDenCluster
from fit_clump_function import main_fit_gauss_3d as mgm


def LDC_MGM_split_mode(data_name, para, save_folder_all, save_mgm_png):
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
    sub_cube_path_list = split_cube.split_cube_lxy(data_name, save_folder_all)
    outcat_wcs_all = pd.DataFrame([])
    outcat_fit_wcs_all = pd.DataFrame([])

    outcat_wcs_all_name = os.path.join(save_folder_all, 'LDC_auto_outcat_wcs.csv')
    outcat_all_name = os.path.join(save_folder_all, 'LDC_auto_outcat.csv')
    outcat_fit_wcs_all_name = os.path.join(save_folder_all, 'MWISP_outcat_wcs.csv')
    outcat_fit_all_name = os.path.join(save_folder_all, 'MWISP_outcat.csv')
    fig_name = os.path.join(save_folder_all, 'LDC_auto_detect_result.png')
    fig_name = fig_name.replace('.png', '_fit.png')

    for ii, sub_data_name in enumerate(sub_cube_path_list):
        sub_save_folder = sub_data_name.replace('.fits', '')
        os.makedirs(sub_save_folder, exist_ok=True)
        ldc_cfg = localDenCluster(sub_data_name, para, sub_save_folder)

        # 处理局部块的核表
        outcat_name = ldc_cfg['outcat_name']
        detect_log = ldc_cfg['detect_log']
        outcat_i = pd.read_csv(outcat_name, sep='\t')
        loc_outcat_i = split_cube.get_outcat_i(outcat_i, ii)  # 取局部数据块的局部核表，根据块的位置，局部区域会有不同
        data = Data(data_path=sub_data_name)
        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(loc_outcat_i)
        shutil.copy(detect_log, os.path.join(save_folder_all, 'LDC_auto_detect_log_%02d.txt' % ii))
        outcat_wcs_all = pd.concat([outcat_wcs_all, outcat_wcs], axis=0)

        loc_outcat_i_name = outcat_name.replace('.csv', '_loc.csv')
        loc_outcat_i.to_csv(loc_outcat_i_name, sep='\t', index=False)
        # 进行MGM
        mask_name = ldc_cfg['mask_name']
        mgm.MGM_main(loc_outcat_i_name, sub_data_name, mask_name, sub_save_folder, save_mgm_png)
        outcat_fit_wcs_path = os.path.join(sub_save_folder, 'MWISP_outcat.csv')
        outcat_fit_wcs = pd.read_csv(outcat_fit_wcs_path, sep='\t')

        fig_name_fit = os.path.join(sub_save_folder, 'LDC_auto_detect_result_fit.png')
        data = Data(data_path=sub_data_name)
        data_wcs = data.wcs
        show_clumps.make_plot_wcs_1(outcat_fit_wcs, data_wcs, data.data_cube, fig_name=fig_name_fit)
        outcat_fit_wcs_all = pd.concat([outcat_fit_wcs_all, outcat_fit_wcs], axis=0)

    # 保存整合的银经银纬的核表(LDC, MGM的核表)
    outcat_wcs_all.to_csv(outcat_wcs_all_name, sep='\t', index=False)
    outcat_fit_wcs_all.to_csv(outcat_fit_wcs_all_name, sep='\t', index=False)

    # 保存整合的像素的核表及绘制检测云核的位置
    data = Data(data_path=data_name)
    ldc = LDC(data=data, para=None)
    data_wcs = ldc.data.wcs
    outcat_wcs_all = pd.read_csv(outcat_wcs_all_name, sep='\t')
    outcat_all = split_cube.change_world2pix(outcat_wcs_all, data_wcs)
    outcat_all.to_csv(outcat_all_name, sep='\t', index=False)
    show_clumps.make_plot_wcs_1(outcat_wcs_all, data_wcs, data.data_cube, fig_name=fig_name)

    outcat_fit_wcs_all = pd.read_csv(outcat_fit_wcs_all_name, sep='\t')
    outcat_fit_all = split_cube.change_world2pix_fit(outcat_fit_wcs_all, data_wcs)
    outcat_fit_all.to_csv(outcat_fit_all_name, sep='\t', index=False)

    show_clumps.make_plot_wcs_1(outcat_fit_wcs_all, data_wcs, data.data_cube, fig_name=fig_name)


def LDC_MGM_main(data_name, para, save_folder=None, split=False, save_mgm_png=False, thresh_num=1):
    """
    LDC_MGM算法入口，对指定的数据进行检测，将结果保存到指定位置
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数, dict
            para.rho_min: Minimum density [5*rms]
            para.delta_min: Minimum delta [4]
            para.v_min: Minimum volume [25, 5]
            para.noise: The noise level of the data, used for data truncation calculation [2*rms]
            para.dc: auto
    :param save_folder: 检测结果保存路径，如果是分块模式，中间结果也会保存
    :param save_loc: 是否保存检测的局部核表，默认为False(不保存)
    split: 是否采用分块检测再拼接策略
    save_mgm_png: 是否保存MGM的中间结果图片，默认为False: 不保存

    Usage:

    data_name = r'*******.fits'
    para = Param(delta_min=4, gradmin=0.01, v_min=[25, 5], noise_times=5, rms_times=2, rms_key='RMS')
    para.rm_touch_edge = False
    save_folder = r'########'
    save_mgm_png = False
    LDC_MGM_main(data_name, para, save_folder, split=False, save_mgm_png=save_mgm_png)

    """
    if save_folder is None:
        save_folder = data_name.replace('.fits', '')
    os.makedirs(save_folder, exist_ok=True)

    if split:
        LDC_MGM_split_mode(data_name, para, save_folder_all=save_folder, save_mgm_png=save_mgm_png)
    else:
        ldc_cfg = localDenCluster(data_name, para, save_folder)
        outcat_name = ldc_cfg['outcat_name']
        mask_name = ldc_cfg['mask_name']
        data_name = ldc_cfg['data_path']

        mgm.MGM_main(outcat_name, data_name, mask_name, save_folder, para=para, thresh_num=thresh_num, save_png=save_mgm_png)


if __name__ == '__main__':
    num = 2
    data_name = r'D:\LDC\test_data\synthetic\synthetic_model_000%d.fits' % num
    para = Param(delta_min=4, gradmin=0.01, v_min=[25, 5], noise_times=5, rms_times=2, rms_key='RMS')
    para.rm_touch_edge = False
    save_folder = r'D:\LDC\test_data\R2_R16_region\0145-005_L13_noise_2_rho_5_128_deb1'
    save_mgm_png = False
    LDC_MGM_main(data_name, para, save_folder, split=False, save_mgm_png=save_mgm_png)

import numpy as np
import os
import time
import pandas as pd
from tqdm import tqdm

from fit_clump_function import multi_gauss_fitting_new, touch_clump
from tools.ultil_lxy import create_folder, get_points_by_clumps_id, move_csv_png, restruct_fitting_outcat,\
    get_save_clumps_xyv
from tools.show_clumps import display_clumps_fitting
from DensityClust.locatDenClust2 import Data


def fitting_LDC_clumps(points_path, outcat_name, data_rms, ldc_mgm_path=None, fitting_outcat_path=None):
    """
    对LDC的检测结果进行3维高斯拟合，保存分子云核的参数
    拟合核表 DataFrame格式
    ['ID', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Theta', 'Peak', 'Sum', 'Flux_SNR', 'Peak_SNR',
     'Success', 'Cost']

    :param points_path:
        云核坐标点及强度文件所在的文件夹路径
    :param outcat_name:
        算法检测得到的核表(像素级)
    :param ldc_mgm_path:
        拟合结果保存文件夹，默认位置为检测核表的同级目录，命名为：核表+_fitting
    :param fitting_outcat_path:
        特殊指定的拟合核表保存位置，默认为ldc_mgm_path路径下，命名为Fitting_outcat.csv
    :return:
        fitting_outcat_path: 拟合核表文件保存路径
    """
    log_file = points_path.replace('points', 'LDC_MGM_log.txt')
    time_st = time.time()
    file = open(log_file, 'a+')
    fwhm_sigma = 2.3548
    print('Data information:\nprocessing file->%s' % outcat_name, file=file)
    print('='*20 + '\n', file=file)
    f_outcat = pd.read_csv(outcat_name, sep=',')
    if f_outcat.shape[1] == 1:
        f_outcat = pd.read_csv(outcat_name, sep='\t')

    if ldc_mgm_path is None:
        create_folder(outcat_name.replace('.csv', '_fitting'))

    # 得到相互重叠的云核(ID)
    touch_clump_record, _ = touch_clump.connect_clump_new(f_outcat, mult=0.9)
    print('Fitting process:', file=file)
    print('The overlapping clumps are selected.', file=file)

    params_init_all = np.array([f_outcat['Peak'], f_outcat['Cen1'], f_outcat['Cen2'], f_outcat['Size1'] / fwhm_sigma,
                                f_outcat['Size2'] / fwhm_sigma, f_outcat['ID'], f_outcat['Cen3'],
                                f_outcat['Size3'] / fwhm_sigma]).T
    params_init_all[:, 5] = 0  # 初始化的角度
    print('The initial parameters (Initial guess) have finished.', file=file)
    touch_record_i = 0
    for item_tcr in tqdm(touch_clump_record):

        fit_outcat_name = os.path.join(ldc_mgm_path, 'fit_item%03d.csv' % touch_record_i)
        fig_name = os.path.join(ldc_mgm_path, 'touch_clumps_%03d.png' % touch_record_i)
        print(time.ctime() + '-->touch_clump %d/%d have %d clump[s].' % (
                touch_record_i, len(touch_clump_record), len(item_tcr)), file=file)
        touch_record_i += 1
        clumps_id = f_outcat.iloc[item_tcr - 1]['ID'].values.astype(np.int64)

        points_all_df = get_points_by_clumps_id(clumps_id, points_path)
        params_init = params_init_all[item_tcr - 1].flatten()

        outcat_fitting = multi_gauss_fitting_new.fitting_main(points_all_df, params_init, clumps_id, data_rms)
        if fit_outcat_name.split('.')[-1] not in ['csv', 'txt']:
            print('the save file type must be one of *.csv and *.txt.')
        else:
            outcat_fitting.to_csv(fit_outcat_name, sep='\t', index=False)

        pif_1 = outcat_fitting[['Cen1', 'Cen2', 'Cen3']] - points_all_df[['x_2', 'y_1', 'v_0']].values.min(axis=0)
        df_temp_1 = f_outcat.iloc[item_tcr - 1]
        display_clumps_fitting(pif_1, df_temp_1, points_all_df, fig_name)

    print('=' * 20 + '\n', file=file)
    move_csv_png(ldc_mgm_path)
    fitting_outcat_path = restruct_fitting_outcat(ldc_mgm_path, fitting_outcat_path)
    # 将分开拟合的核表整合在一起，保存在ldc_mgm_path下，命名为fitting_outcat.csv
    time_end = time.time()

    print('Fitting information:\nFitting clumps used %.2f seconds.' % (time_end - time_st), file=file)
    file.close()
    return fitting_outcat_path


def MGM_main(outcat_name_loc, origin_name, mask_name, save_path):
    """
    outcat_name_loc: LDC 检测核表
    origin_name: 检测的原始数据
    mask_name: LDC检测得到的mask
    save_path: 拟合结果保存位置
    """
    # 初始化对应文件保存路径
    if not os.path.exists(outcat_name_loc):
        raise FileExistsError('\n' + outcat_name_loc + ' not exists.')

    if not os.path.exists(origin_name):
        raise FileExistsError('\n' + origin_name + ' not exists.')

    if not os.path.exists(mask_name):
        raise FileExistsError('\n' + mask_name + ' not exists.')

    create_folder(save_path)
    points_path = create_folder(os.path.join(save_path, 'points'))
    ldc_mgm_path = create_folder(os.path.join(save_path, 'LDC_MGM_outcat'))
    MWISP_outcat_path = os.path.join(save_path, 'MWISP_outcat.csv')

    # step 1: 准备拟合数据
    get_save_clumps_xyv(origin_name, mask_name, outcat_name_loc, points_path)

    # step 2: 进行拟合并保存拟合像素级核表
    data_int = Data(origin_name)
    data_rms = data_int.rms
    fitting_outcat_path = fitting_LDC_clumps(points_path, outcat_name_loc, data_rms, ldc_mgm_path)

    # step 3: 将拟合核表整理成最终核表并保存
    data_int.get_wcs()
    data_wcs = data_int.wcs
    fitting_outcat = pd.read_csv(fitting_outcat_path, sep='\t')
    MWISP_outcat_df = multi_gauss_fitting_new.exchange_pix2world(fitting_outcat, data_wcs)
    MWISP_outcat_df = MWISP_outcat_df.round(
        {'Galactic_Longitude': 4, 'Galactic_Latitude': 4, 'Velocity': 2, 'Size_major': 0, 'Size_minor': 0,
         'Size_velocity': 2, 'Peak': 1, 'Flux': 1, 'Flux_SNR': 1, 'Peak_SNR': 1, 'theta': 0})
    MWISP_outcat_df.to_csv(MWISP_outcat_path, sep='\t', index=False)


if __name__ == '__main__':
    outcat_name_loc = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L/LDC_auto_loc_outcat.csv'
    origin_name = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L\0155+030_L.fits'
    mask_name = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L\LDC_auto_mask.fits'
    save_path = 'fitting_result7'

    MGM_main(outcat_name_loc, origin_name, mask_name, save_path)

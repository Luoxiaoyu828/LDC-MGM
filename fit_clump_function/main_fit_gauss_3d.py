import numpy as np
import os
import time
import pandas as pd
from tqdm import tqdm
import threading
from fit_clump_function import multi_gauss_fitting_new, touch_clump
from tools.ultil_lxy import create_folder, get_points_by_clumps_id, move_csv_png, restruct_fitting_outcat,\
    get_save_clumps_xyv
from tools import show_clumps
from DensityClust.localDenClust2 import Data, Param


def fitting_LDC_clumps(points_path, outcat_name, data_rms, thresh_num, save_png=False, ldc_mgm_path=None,
                       fitting_outcat_path=None):
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
    touch_clump_record = touch_clump.connect_clump_new(f_outcat, mult=0.8)
    print('Fitting process:', file=file)
    print('The overlapping clumps are selected.', file=file)

    if 'Cen3' in f_outcat.columns:
        params_init_all = np.array([f_outcat['Peak'], f_outcat['Cen1'], f_outcat['Cen2'], f_outcat['Size1'] / fwhm_sigma,
                                    f_outcat['Size2'] / fwhm_sigma, f_outcat['ID'], f_outcat['Cen3'],
                                    f_outcat['Size3'] / fwhm_sigma]).T
    else:
        params_init_all = np.array(
            [f_outcat['Peak'], f_outcat['Cen1'], f_outcat['Cen2'], f_outcat['Size1'] / fwhm_sigma,
             f_outcat['Size2'] / fwhm_sigma, f_outcat['ID']]).T
    params_init_all[:, 5] = 0  # 初始化的角度
    print('The initial parameters (Initial guess) have finished.', file=file)
    if thresh_num > 1:
        clump_num = f_outcat.shape[0]
        tsk = []
        touch_num = len(touch_clump_record)
        deal_num = clump_num // thresh_num
        st = 0
        for thre_i in range(thresh_num):
            temp_num = 0
            for item_i, item in enumerate(touch_clump_record[st:]):
                temp_num += item.shape[0]

                if thre_i == thresh_num - 1:
                    end_ = touch_num
                if temp_num > deal_num:
                    end_ = item_i + 1 + st
                    break

            touch_i = st
            if thre_i == thresh_num - 1:
                end_ = len(touch_clump_record)
            touch_clump_record_ = touch_clump_record[st: end_]
            t1 = threading.Thread(target=fitting_threading,
                                  args=(touch_clump_record_, ldc_mgm_path, file, points_path, params_init_all, data_rms,
                                        save_png, f_outcat, touch_i))
            tsk.append(t1)
            st = end_

        for t in tsk:
            t.start()
        for t in tsk:
            t.join()
    elif thresh_num == 1:
        touch_i = 0
        fitting_threading(touch_clump_record, ldc_mgm_path, file, points_path, params_init_all, data_rms, save_png,
                          f_outcat, touch_i)

    print('=' * 20 + '\n', file=file)
    move_csv_png(ldc_mgm_path)
    fitting_outcat_path = restruct_fitting_outcat(ldc_mgm_path, fitting_outcat_path)
    # 将分开拟合的核表整合在一起，保存在ldc_mgm_path下，命名为fitting_outcat.csv
    time_end = time.time()
    print('Fitting information:\nFitting clumps used %.2f seconds.' % (time_end - time_st), file=file)
    file.close()
    return fitting_outcat_path


def fitting_threading(touch_clump_record, ldc_mgm_path, file, points_path, params_init_all, data_rms, save_png,
                          f_outcat, touch_i_):
    touch_i = touch_i_
    for item_tcr in tqdm(touch_clump_record):

        fit_outcat_name = os.path.join(ldc_mgm_path, 'fit_item%03d.csv' % touch_i)
        fig_name = os.path.join(ldc_mgm_path, 'touch_clumps_%03d.png' % touch_i)
        print(time.ctime() + '-->touch_clump %d/%d have %d clump[s].' % (
            touch_i, len(touch_clump_record), len(item_tcr)), file=file)
        touch_i += 1

        clumps_id = f_outcat.iloc[item_tcr - 1]['ID'].values.astype(np.int64)
        points_all_df = get_points_by_clumps_id(clumps_id, points_path)
        params_init = params_init_all[item_tcr - 1].flatten()
        if 'v_0' in points_all_df.columns:
            ndim = 3
        else:
            ndim = 2
        outcat_fitting = multi_gauss_fitting_new.fitting_main(points_all_df, params_init, clumps_id, data_rms, ndim=ndim)
        if fit_outcat_name.split('.')[-1] not in ['csv', 'txt']:
            print('the save file type must be one of *.csv and *.txt.')
        else:
            outcat_fitting.to_csv(fit_outcat_name, sep='\t', index=False)

        if save_png:
            pif_1 = outcat_fitting[['Cen1', 'Cen2', 'Cen3']] - points_all_df[['x_2', 'y_1', 'v_0']].values.min(axis=0)
            df_temp_1 = f_outcat.iloc[item_tcr - 1]
            show_clumps.display_clumps_fitting(pif_1, df_temp_1, points_all_df, fig_name)


def MGM_main(outcat_name_pix, origin_name, mask_name, save_path, para=None, thresh_num=6, save_png=False):
    """
    outcat_name_loc: LDC 检测像素核表
    origin_name: 检测的原始数据
    mask_name: LDC检测得到的mask
    save_path: 拟合结果保存位置
    save_png: 是否保存拟合结果的图片，默认为False：不保存
    """
    # 初始化对应文件保存路径
    if not os.path.exists(outcat_name_pix):
        raise FileExistsError('\n' + outcat_name_pix + ' not exists.')

    if not os.path.exists(origin_name):
        raise FileExistsError('\n' + origin_name + ' not exists.')

    if not os.path.exists(mask_name):
        raise FileExistsError('\n' + mask_name + ' not exists.')

    f_outcat = pd.read_csv(outcat_name_pix, sep='\t')
    if f_outcat.shape[1] == 1:
        f_outcat = pd.read_csv(outcat_name_pix, sep=',')

    if f_outcat.shape[0] == 0:
        return
    else:
        create_folder(save_path)
        points_path = create_folder(os.path.join(save_path, 'points'))
        ldc_mgm_path = create_folder(os.path.join(save_path, 'LDC_MGM_outcat'))
        MWISP_outcat_path = os.path.join(save_path, 'MGM_MWISP_outcat.csv')

        # step 1: 准备拟合数据
        get_save_clumps_xyv(origin_name, mask_name, outcat_name_pix, points_path)

        # step 2: 进行拟合并保存拟合像素级核表
        data_int = Data(data_path=origin_name)
        data_int.calc_background_rms(rms_key=para.rms_key, data_rms_path=para.data_rms_path, rms=para.rms)
        data_rms = data_int.rms
        fitting_outcat_path = fitting_LDC_clumps(points_path, outcat_name_pix, data_rms, thresh_num, save_png, ldc_mgm_path)

        # step 3: 将拟合核表整理成最终核表并保存
        data_int.get_wcs()
        data_wcs = data_int.wcs
        resolution = data_int.res
        fitting_outcat = pd.read_csv(fitting_outcat_path, sep='\t')
        MWISP_outcat = multi_gauss_fitting_new.exchange_pix2world(fitting_outcat, data_wcs, resolution)

        MWISP_outcat.to_csv(MWISP_outcat_path, sep='\t', index=False)

        aa = pd.read_csv(MWISP_outcat_path, sep='\t')
        MWISP_outcat = aa.round(
            {'Galactic_Longitude': 3, 'Galactic_Latitude': 3, 'Velocity': 2, 'Size_major': 0, 'Size_minor': 0,
             'Size_velocity': 2, 'Peak': 1, 'Flux': 1, 'Flux_SNR': 1, 'Peak_SNR': 1, 'Theta': 0})
        if os.path.exists(MWISP_outcat_path):
            os.remove(MWISP_outcat_path)
        MWISP_outcat.to_csv(MWISP_outcat_path, sep='\t', index=False)

        fig_name_fit = os.path.join(save_path, 'LDC_auto_detect_result_fit.png')
        show_clumps.make_plot_wcs_1(MWISP_outcat, data_wcs, data_int.data_cube, fig_name=fig_name_fit)


if __name__ == '__main__':
    outcat_name_loc = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L/LDC_auto_loc_outcat.csv'
    origin_name = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L\0155+030_L.fits'
    mask_name = r'F:\Parameter_reduction\LDC\0170+010_L/MGM_problem_cell/0155+030_L\LDC_auto_mask.fits'
    save_path = 'fitting_result1'

    MGM_main(outcat_name_loc, origin_name, mask_name, save_path)

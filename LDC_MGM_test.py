import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DensityClust.locatDenClust2 import Data
from DensityClust import LocalDensityClustering_main as LDC
from fit_clump_function import multi_gauss_fitting, multi_gauss_fitting_new
from tools.ultil_lxy import get_datacube_by_points

from fit_clump_function import main_fit_gauss_3d as mgm
from tools.ultil_lxy import create_folder
import os


def fitting_func():
    origin_data_name = r'0170+010_L\0170+010_L.fits'
    mask_name = r'0170+010_L\LDC_auto_mask.fits'
    outcat_name = r'0170+010_L\LDC_auto_loc_outcat.csv'
    fit_outcat_name = r'0170+010_L\LDC_auto_loc_outcat_fitting\fitting_outcat.csv'
    save_path = origin_data_name.replace('.fits', '_points')

    file_tem = []

    ldc_outcat = pd.read_csv(outcat_name, sep=',')
    fit_outcat = pd.read_csv(fit_outcat_name, sep='\t')
    ii = 2

    clump_id = ldc_outcat.iloc[ii]['ID']
    points_name_list = [
        r'F:\Parameter_reduction\LDC\0170+010_L\0170+010_L_points\clump_id_xyz_intensity_%04d.csv' % clump_id]
    data = get_datacube_by_points(points_name_list)
    clump_inf_fit = fit_outcat.iloc[ii][['ID', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum']]

    clmup_inf_ldc = ldc_outcat.iloc[np.argwhere(ldc_outcat['ID'].values == clump_id)[0]][
        ['ID', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum']]
    title_str = ''
    for item in clmup_inf_ldc.keys():
        temp = item + ': %6.2f, ' % clmup_inf_ldc[item]
        title_str += temp

    title_str_1 = ''
    for item in clump_inf_fit.keys():
        temp = item + ': %6.2f, ' % clump_inf_fit[item]
        title_str_1 += temp

    title = title_str + '\n\n' + title_str_1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))

    im0 = ax1.imshow(data.sum(0), origin='lower')  # x银纬,y银经
    im1 = ax2.imshow(data.sum(1), origin='lower')  # x银纬，y速度
    im2 = ax3.imshow(data.sum(2), origin='lower')  # x银经，y速度
    # plt.text(-20,10,title,fontdict={'fontsize': 12})

    plt.show()

    points = pd.read_csv(points_name_list[0])
    f_outcat = ldc_outcat
    params_init = np.array([f_outcat['Peak'], f_outcat['Cen1'], f_outcat['Cen2'], f_outcat['Size1'] / 2.3548,
                            f_outcat['Size2'] / 2.3548, f_outcat['ID'], f_outcat['Cen3'], f_outcat['Size3'] / 2.3548]).T
    params_init[:, 5] = 0  # 初始化的角度
    params = params_init[ii,:].reshape([1,8])
    columns_name = ['A', 'x0', 'y0', 's1', 's2', 'theta', 'v0', 's3']
    params = pd.DataFrame(params, columns=columns_name)
    params_fit_pf = multi_gauss_fitting.get_multi_gauss_params(points, params)
    outcat_record = multi_gauss_fitting_new.get_fit_outcat_df(params_fit_pf, data_rms)

    data_int = Data(origin_data_name)
    data_int.get_wcs()
    data_wcs = data_int.wcs
    mm_outcat = multi_gauss_fitting_new.exchange_pix2world(fit_outcat, data_wcs)


if __name__ == '__main__':
    data_name = r'0170+010_L/error_data_for_fitting/data/gaussian_out_000.fits' # 待检测的云核
    save_folder = r'0170+010_L/error_data_for_fitting/data/gaussian_out_000'  # 检测结果保存的位置

    LDC.LDC_fast(data_name, save_folder)

    outcat_name_loc = r'0170+010_L/error_data_for_fitting/data/gaussian_out_000/LDC_auto_outcat.csv' # LDC核表
    origin_name = data_name
    mask_name = r'0170+010_L/error_data_for_fitting/data/gaussian_out_000/LDC_auto_mask.fits' #　LDC Mask
    save_path = '0170+010_L/error_data_for_fitting/data/fitting_result1' # 参数还原保存位置

    mgm.LDC_para_fit_Main(outcat_name_loc, origin_name, mask_name, save_path)






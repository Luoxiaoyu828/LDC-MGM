import math
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from skimage import measure

from LDC.DensityClust.LocalDensityClustering_main import localDenCluster
from LDC.DensityClust.localDenClust2 import Param, Data
from LDC.Generate.fits_header import Header
from LDC.fit_clump_function.multi_gauss_fitting_new import fitting_main
from LDC.tools.ultil_lxy import get_save_clumps_xyv
from LDC.DensityClust.clustring_subfunc import get_area_v_len, get_clump_angle


def get_rms(data, mask):
    """
    计算’mask‘区域外的‘rms’
    :param data: 星云数据
    :param mask:  掩膜区域
    :return: rms
    """
    x, y = np.where(mask != 0)
    data[x, y] = 0
    rms = np.sum(data[:, :] ** 2) / (data.size - x.size)
    return rms ** 0.5


def GaussianBlur(data, ksize, sigma):
    """
    高斯滤波, 目前只完成 2d-Gaussian fittering
    :param data: 待滤波数据
    :param ksize: 高斯核尺寸, 必须为单数
    :param sigma: 高斯核sigma
    :return: 滤波结果
    """
    result = np.zeros_like(data)

    if np.ndim(data) == 2:
        # 生成2维高斯滤波器
        gaussian_filter = np.zeros([ksize, ksize])
        for x in range(ksize):
            a = x - (ksize - 1) / 2
            for y in range(ksize):
                b = y - (ksize - 1) / 2
                gaussian_filter[x, y] = (math.exp(-(a * a + b * b) / (2 * sigma * sigma))) / (
                        2 * math.pi * sigma * sigma)
        filter_sum = np.sum(gaussian_filter)
        gaussian_filter = gaussian_filter / filter_sum

        # 进行滤波
        x, y = data.shape
        a = int((ksize - 1) / 2)
        data_ = np.zeros([x + ksize, y + ksize])
        data_[a:x + a, a:y + a] = data

        for i in range(x):
            for j in range(y):
                result[i, j] = np.sum(data_[i:i + ksize, j:j + ksize] * gaussian_filter)

    if np.ndim(data) == 3:
        # 生成3维高斯滤波器
        gaussian_filter = np.zeros([ksize, ksize, ksize])

    return result


def extra_record(data, label_data, para):
    """
    将 single_scale 细节 LDC 检测结果在 all_scale 上重新计算
    :param data: all_scale 原始数据, “需要转换为 Data 类”
    :param label_data: single_scale 细节 LDC 检测结果中的 mask 数据
    :param para: single_scale 在 LDC 检测中的参数设置
    :return: 返回重新计算的核表, pd.DataFrame 类型
    """
    dim = data.data_cube.ndim
    if dim == 3:
        table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak',
                       'Sum', 'Volume', 'Angle']
    else:
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume', 'Angle']
    label_data_all = np.zeros_like(label_data, np.int32)
    if label_data.max() == 0:
        LDC_outcat = pd.DataFrame([], columns=table_title)
    else:
        props = measure.regionprops_table(label_image=label_data, intensity_image=data.data_cube,
                                          properties=['weighted_centroid', 'area', 'mean_intensity',
                                                      'weighted_moments_central', 'max_intensity',
                                                      'image_intensity', 'bbox'],
                                          extra_properties=(get_area_v_len, get_clump_angle))
        id_temp = 1
        area_v_idx = []
        props_bbox = np.array([props['bbox-%d' % bbox_i] for bbox_i in range(dim * 2)], np.int32)
        for props_i, props_item in enumerate(props['get_area_v_len']):
            g_l_area = props_item[0]
            v_len = props_item[1]
            label_image = props_item[2]
            props_bbox_i = props_bbox[:, props_i]
            if g_l_area >= para.v_min[0] and v_len >= para.v_min[1]:
                label_temp = np.zeros_like(label_data_all, np.int32)
                if dim == 3:
                    label_temp[props_bbox_i[0]: props_bbox_i[3], props_bbox_i[1]: props_bbox_i[4],
                    props_bbox_i[2]: props_bbox_i[5]] = id_temp * label_image
                else:
                    label_temp[props_bbox_i[0]: props_bbox_i[2],
                    props_bbox_i[1]: props_bbox_i[3]] = id_temp * label_image

                label_data_all += label_temp
                id_temp += 1
                area_v_idx.append(props_i)
        area_v_idx = np.array(area_v_idx, np.int32)

        clump_Angle = props['get_clump_angle-0'][area_v_idx]
        image_intensity = props['image_intensity'][area_v_idx]
        max_intensity = props['max_intensity'][area_v_idx]
        bbox = np.array([props['bbox-%d' % item][area_v_idx] for item in range(dim)])
        Peak123 = np.zeros([dim, area_v_idx.shape[0]])
        for ps_i, item in enumerate(max_intensity):
            max_idx = np.argwhere(image_intensity[ps_i] == item)[0]
            peak123 = max_idx + bbox[:, ps_i]
            Peak123[:, ps_i] = peak123.T
        if dim == 3:
            clump_Cen = np.array(
                [props['weighted_centroid-2'][area_v_idx], props['weighted_centroid-1'][area_v_idx],
                 props['weighted_centroid-0'][area_v_idx]])
            size_3 = (props['weighted_moments_central-0-0-2'][area_v_idx] / props['weighted_moments_central-0-0-0'][
                area_v_idx]) ** 0.5
            size_2 = (props['weighted_moments_central-0-2-0'][area_v_idx] / props['weighted_moments_central-0-0-0'][
                area_v_idx]) ** 0.5
            size_1 = (props['weighted_moments_central-2-0-0'][area_v_idx] / props['weighted_moments_central-0-0-0'][
                area_v_idx]) ** 0.5
            clump_Size = 2.3548 * np.array([size_3, size_2, size_1])
        elif dim == 2:
            clump_Cen = np.array(
                [props['weighted_centroid-1'][area_v_idx], props['weighted_centroid-0'][area_v_idx]])
            size_2 = (props['weighted_moments_central-0-2'][area_v_idx] / props['weighted_moments_central-0-0'][
                area_v_idx]) ** 0.5
            size_1 = (props['weighted_moments_central-2-0'][area_v_idx] / props['weighted_moments_central-0-0'][
                area_v_idx]) ** 0.5
            clump_Size = 2.3548 * np.array([size_2, size_1])
        else:
            clump_Cen = None
            clump_Size = None
            print('Only 2D and 3D are supported!')

        clump_Volume = props['area'][area_v_idx]
        clump_Peak = props['max_intensity'][area_v_idx]
        clump_Sum = clump_Volume * props['mean_intensity'][area_v_idx]
        clump_Peak123 = Peak123 + 1
        clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
        id_clumps = np.array([item + 1 for item in range(len(area_v_idx))], np.int32)
        id_clumps = id_clumps.reshape([id_clumps.shape[0], 1])

        LDC_outcat = np.column_stack((id_clumps, clump_Peak123.T[:, ::-1], clump_Cen.T, clump_Size.T, clump_Peak.T,
                                      clump_Sum.T, clump_Volume.T, clump_Angle.T))

        LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)

    return LDC_outcat


def Mult_LDC_main(data_name, para, save_folder=None, save_loc=False, sigma=1):
    """
    基于多尺度分析的局部密度聚类检测算法
    :param data_name: 待检测数据路径
    :param para: all_scale LDC 检测参数
    :param save_folder: 保存路径, 默认以 data_name 文件名命名
    :param save_loc: LDC检测的局部核表是否保存, 默认 False
    """
    if save_folder is None:
        save_folder = data_name.replace('.fits', '')
    os.makedirs(save_folder, exist_ok=True)

    # 全尺度LDC检测, 检测结果存储在all_scale目录下
    save_allscale = os.path.join(save_folder, 'all_scale')
    os.makedirs(save_allscale, exist_ok=True)
    all_ldc = localDenCluster(data_name=data_name, para=para, save_folder=save_allscale, save_loc=save_loc)

    # 滤波提取细节信息, 存储在文件single_scal/Detailed_infomation.fits中
    save_singlescale = os.path.join(save_folder, 'single_scale')
    os.makedirs(save_singlescale, exist_ok=True)
    src = fits.getdata(data_name)
    ksize = 5
    # sigma = sigma
    blur = GaussianBlur(src, ksize=ksize, sigma=sigma)
    delta = src - blur

    header = Header(dim=2, size=np.shape(delta), rms=0.23, history_info=None,
                    information=None)
    fits_header = header.write_header()
    data_hdu = fits.PrimaryHDU(delta, header=fits_header)
    save_detailed = os.path.join(save_singlescale, f'Detailed_infomation.fits')
    fits.HDUList([data_hdu]).writeto(save_detailed, overwrite=True)

    # 计算细节数据中的噪声水平
    all_mask = fits.getdata(all_ldc['mask_name'])
    single_rms = get_rms(delta, all_mask)

    # 对single_scale进行LDC，获取质心位置
    para_1 = Param(delta_min=3, gradmin=0.01, v_min=[4, 3], noise_times=2, rms_times=3, res=[30, 30, 0.166],
                   dc=0.01, data_rms_path='', rms=single_rms)
    single_ldc = localDenCluster(data_name=save_detailed, para=para_1, save_folder=save_singlescale, save_loc=save_loc)

    # 判断单尺度上是否检测到数据
    all_outcat = pd.read_csv(all_ldc['outcat_name'], sep='\t')
    single_outcat = pd.read_csv(single_ldc['outcat_name'], sep='\t')
    if single_outcat.shape[0] == 0:
        print('未在单尺度细节上检测出有效成分！')
        return all_outcat, single_outcat

    # all_scale掩膜数据 --> (x,y,E), 并存储在 points\clump_id_xyz_intensity_****.csv 中, 作为GMMs的数据集.
    save_points = os.path.join(save_folder, 'points')
    os.makedirs(save_points, exist_ok=True)
    get_save_clumps_xyv(all_ldc['data_path'], all_ldc['mask_name'], all_ldc['outcat_name'],
                        save_points)

    # 读取合并所有的点阵数据集（x,y,E）
    points_num = pd.read_csv(all_ldc['outcat_name'], sep='\t')['ID'].values.astype(np.int64)
    points_all_df = pd.DataFrame()
    for iiii in points_num:
        points_all_df_ = pd.read_csv(save_points + '/clump_id_xyz_intensity_%04d.csv' % iiii)
        points_all_df = pd.concat([points_all_df, points_all_df_])

    # GMMs拟合的基本参数
    data_int = Data(data_path=all_ldc['data_path'])
    data_int.calc_background_rms(rms_key=para.rms_key, data_rms_path=para.data_rms_path, rms=para.rms)
    data_rms = data_int.rms

    # 利用single_scale上的掩膜, 在all_scale上还原形状参数作为GMMs初值.
    single_mask = fits.getdata(single_ldc['mask_name'])
    new_outcat = extra_record(data_int, single_mask, para_1)

    params_init_all = np.array(
        [new_outcat['Peak'], new_outcat['Cen1'], new_outcat['Cen2'], new_outcat['Size1'],
         new_outcat['Size2'], new_outcat['ID']]).T  # 使用ID列占位角度参数
    params_init_all[:, 5] = 0  # 添加角度参数

    # ！ 此处反而降低了精度
    # 对部分偏差较大的初值进行重置
    # all_outcat = pd.read_csv(all_ldc['outcat_name'], sep='\t')
    # size1_sum = new_outcat['Size1'].values.sum()
    # params_init_all[:, 3] = all_outcat['Size1'].values * new_outcat['Size1'].values / size1_sum
    # size2_sum = new_outcat['Size2'].values.sum()
    # params_init_all[:, 4] = all_outcat['Size2'].values * new_outcat['Size2'].values / size2_sum

    # 将参数初值改写为一维向量形式
    params_init = params_init_all.flatten()
    clumps_id = new_outcat['ID'].values.astype(np.int64)

    outcat_record = fitting_main(points_all_df, params_init, clumps_id, data_rms, ndim=2)
    outcat_record.to_csv(os.path.join(save_folder, "fitting_result.csv"),
                         columns=['ID', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Theta', 'Peak'], index=False)

    return all_outcat, outcat_record


if __name__ == '__main__':
    para = Param(delta_min=1, gradmin=0.01, v_min=[9, 3], noise_times=5, rms_times=2, res=[30, 30, 0.166],
                 dc=0.3, data_rms_path='', rms=0.23)
    data_name = r'D:\A1-毕业论文\0-执行代码\演示实验1\Simulation_2d\simulation_2d.fits'
    save_folder = r'D:\A1-毕业论文\0-执行代码\演示结果1'
    Mult_LDC_main(data_name, para, save_folder)

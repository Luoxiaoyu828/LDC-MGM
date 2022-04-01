import os
import numpy as np
from astropy.io import fits
import pandas as pd


def create_folder(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
        print(path + 'created successfully!')


def get_save_clumps_xyv(origin_data_name, mask_name, outcat_name, save_path):
    """
    将云核的坐标及对应的强度整理保存为.csv文件
    :param origin_data_name: 原始数据
    :param mask_name: 检测得到的掩模
    :param outcat_name: 检测得到的核表
    :param save_path: 坐标保存的文件夹
    :return:
        DataFarame [ x_2, y_1 , v_0, Intensity]
    """
    data = fits.getdata(origin_data_name)
    mask = fits.getdata(mask_name)
    f_outcat = pd.read_csv(outcat_name, sep='\t')

    [data_x, data_y, data_v] = data.shape
    Xin, Yin, Vin = np.mgrid[1:data_x + 1, 1:data_y + 1, 1:data_v + 1]
    X = np.vstack([Vin.flatten(), Yin.flatten(), Xin.flatten()]).T  # 坐标原点为1
    mask_flatten = mask.flatten()
    Y = data.flatten()
    clumps_id = f_outcat['ID'].values.astype(np.int64)
    for id_clumps_item in clumps_id:
        clump_item_df = pd.DataFrame([])
        ind = np.where(mask_flatten == id_clumps_item)[0]

        clump_item_df[['x_2', 'y_1', 'v_0']] = X[ind, :]
        clump_item_df['Intensity'] = Y[ind]
        clump_item_name = os.path.join(save_path, 'clump_id_xyz_intensity_%04d.csv' % id_clumps_item)

        clump_item_df.to_csv(clump_item_name, index=False)


def get_data_points(points_all_df):
    """

    :param points_all_df: [x,y,v,I]
    :return: data_cube
    """
    pif = points_all_df[['x_2', 'y_1', 'v_0']].values
    pif_1 = pif - pif.min(axis=0)
    # pif_1 = pif

    data = np.zeros(pif_1.max(axis=0) + 1)
    for i, pif_ in enumerate(pif_1):
        # print(pif_)
        data[pif_[0], pif_[1], pif_[2]] = points_all_df['Intensity'].values[i]

    return data


def get_datacube_by_points(points_name_list):
    """
    根据云核的点和强度，返回局部立方体
    :param points_name: 云核点坐标及强度文件名列表
    :return:
        np.array (n*m*k)
    """
    points_all = pd.DataFrame()
    for points_name in points_name_list:
        points = pd.read_csv(points_name)
        points_all = pd.concat([points_all, points], axis=0)

    data = get_data_points(points_all)
    return data


if __name__ == '__main__':
    pass
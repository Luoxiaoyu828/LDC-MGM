import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import threading

from distrib.DensityClust.localDenClust2 import Data
from distrib.DensityClust.localDenClust2 import LocalDensityCluster as LDC


def make_plot_wcs_1(outcat_wcs, data_wcs, data_cube):
    """
    在积分图上绘制检测结果
    当没有检测到云核时，只画积分图
    """
    markersize = 8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = 'True'
    plt.rcParams['ytick.right'] = 'True'
    plt.rcParams['xtick.color'] = 'red'
    plt.rcParams['ytick.color'] = 'red'

    fig = plt.figure(figsize=(10, 8.5), dpi=100)

    axes0 = fig.add_axes([0.15, 0.1, 0.7, 0.82], projection=data_wcs.celestial)
    axes0.set_xticks([])
    axes0.set_yticks([])
    if data_cube.ndim == 3:
        im0 = axes0.imshow(data_cube.sum(axis=0))
    else:
        im0 = axes0.imshow(data_cube)
    if outcat_wcs.values.shape[0] > 0:
        outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Cen1'].values, b=outcat_wcs['Cen2'].values,
                                unit="deg")
        axes0.plot_coord(outcat_wcs_c, 'r*', markersize=markersize)

    axes0.set_xlabel("Galactic Longitude", fontsize=12)
    axes0.set_ylabel("Galactic Latitude", fontsize=12)
    # axes0.set_title(title, fontsize=12)
    pos = axes0.get_position()
    pad = 0.01
    width = 0.02
    axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

    cbar = fig.colorbar(im0, cax=axes1)
    cbar.set_label('K m s${}^{-1}$')
    plt.show()


def show_outcat_data(data_name, outcat_name):
    """
    将核表中的质心点在云核数据的银经银纬积分图上绘制
    :param data_name: 原始数据文件
    :param outcat_name 云核核表数据,可以是像素坐标也可以是wcs坐标

     data_name = r'F:\Parameter_reduction\LDC\0170+010_L\0170+010_L.fits'
     outcat_name = r'F:\Parameter_reduction\LDC\0170+010_L\LDC_auto_loc_outcat.csv'
    """
    data = Data(data_path=data_name)
    outcat = pd.read_csv(outcat_name, sep=',')
    if 'Cen1' in outcat.keys():
        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(outcat)
    else:
        outcat_wcs = outcat

    data_wcs = data.wcs
    data_cube = data.data_cube
    make_plot_wcs_1(outcat_wcs, data_wcs, data_cube)


def create_folder(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
        print(path + ' created successfully!')
    return path


def get_save_clumps_xyv(origin_data_name, mask_name, outcat_name, save_path, thresh_num=4):
    """
    将云核的坐标及对应的强度整理保存为.csv文件
    :param origin_data_name: 原始数据
    :param mask_name: 检测得到的掩模
    :param outcat_name: 检测得到的核表
    :param save_path: 坐标保存的文件夹
    :param thresh_num: 执行过程中的多线程个数, 默认为4个线程
    :return:
        None
    """
    create_folder(save_path)
    data_instance = Data(data_path=origin_data_name)
    data = data_instance.data_cube
    mask = fits.getdata(mask_name)
    f_outcat = pd.read_csv(outcat_name, sep='\t')
    if f_outcat.shape[1] == 1:
        f_outcat = pd.read_csv(outcat_name, sep=',')
    mask_flatten = mask.flatten()
    data_dim = data.ndim
    Y = data.flatten()
    if data_dim == 3:

        [data_x, data_y, data_v] = data.shape
        Xin, Yin, Vin = np.mgrid[1:data_x + 1, 1:data_y + 1, 1:data_v + 1]
        X = np.vstack([Vin.flatten(), Yin.flatten(), Xin.flatten()]).T  # 坐标原点为1
    elif data_dim == 2:
        [data_x, data_y] = data.shape
        Xin, Yin = np.mgrid[1:data_x + 1, 1:data_y + 1]
        X = np.vstack([Yin.flatten(), Xin.flatten()]).T  # 坐标原点为1
    else:
        raise ValueError('Only 2D and 3D are supported!!!')

    clumps_id = f_outcat['ID'].values.astype(np.int64)

    tsk = []
    deal_num = clumps_id.shape[0] // thresh_num
    for thre_i in range(thresh_num):
        st = thre_i * deal_num
        end_ = (thre_i + 1) * deal_num

        if thre_i == thresh_num - 1:
            end_ = clumps_id.shape[0]
        clumps_id_st_end_ = clumps_id[st: end_]
        t1 = threading.Thread(target=save_point_csv,
                              args=(save_path, mask_flatten, Y, X, clumps_id_st_end_))
        tsk.append(t1)

    for t in tsk:
        t.start()
    for t in tsk:
        t.join()


def save_point_csv(save_path, mask_flatten, Y, X, clumps_id_st_end_):
    """
    将指定的云核的坐标及对应的强度整理保存为.csv文件，多线程会调用

    :param save_path: 坐标文件保存的文件夹
    :param mask_flatten: 检测得到的掩模数据拉直后的结果
    :param Y: 原始数据强度值拉直后的结果
    :param X: 原始数据的坐标
    :param clumps_id_st_end_: 要执行的云核ID号的起始编号
    :return:
        None
    """
    for id_clumps_item in clumps_id_st_end_:
        clump_item_name = os.path.join(save_path, 'clump_id_xyz_intensity_%04d.csv' % id_clumps_item)

        clump_item_df = pd.DataFrame([])
        ind = np.where(mask_flatten == id_clumps_item)[0]
        if X.shape[1] == 3:
            clump_item_df[['x_2', 'y_1', 'v_0']] = X[ind, :]
        elif X.shape[1] == 2:
            clump_item_df[['x_2', 'y_1']] = X[ind, :]
        else:
            raise ValueError('only supporting 2D or 3D')
        clump_item_df['Intensity'] = Y[ind]

        clump_item_df.to_csv(clump_item_name, index=False)


def get_data_points(points_all_df):
    """
    利用云核坐标点及强度的csv文件数据，将数据重构成局部data_cube
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
    根据云核的点和强度，返回局部立方体, 可将多个云核拼接成一个局部的data_cube，主要用于显示相互重叠的云核
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


def get_points_by_clumps_id(clumps_id, points_path):
    """
    根据给定的云核id,将这些云核的坐标及强度返回
    :param clumps_id 云核的id [1,2,3]
    :param points_path 保存csv的路径
    """
    points_all_df = pd.DataFrame([])

    for id_clumps_index in clumps_id:
        clump_id_path = os.path.join(points_path, 'clump_id_xyz_intensity_%04d.csv' % id_clumps_index)
        xyv_intensity = pd.read_csv(clump_id_path)
        points_all_df = pd.concat([points_all_df, xyv_intensity], axis=0)
    points_all_df = points_all_df.dropna()   # 删除存在nan的行数据
    return points_all_df


def move_csv_png(csv_png_folder):
    """
    对MGM拟合的结果进行整理
    """
    csv_path = [os.path.join(csv_png_folder, item) for item in os.listdir(csv_png_folder) if item.endswith('.csv')]
    png_path = [os.path.join(csv_png_folder, item) for item in os.listdir(csv_png_folder) if item.endswith('.png')]
    csv_folder = os.path.join(csv_png_folder, 'csv')
    png_folder = os.path.join(csv_png_folder, 'png')
    create_folder(csv_folder)
    create_folder(png_folder)

    csv_path_ob = [os.path.join(csv_folder, item) for item in os.listdir(csv_png_folder) if item.endswith('.csv')]
    png_path_ob = [os.path.join(png_folder, item) for item in os.listdir(csv_png_folder) if item.endswith('.png')]
    for item_or, item_ob in zip(csv_path, csv_path_ob):
        if os.path.exists(item_or):
            shutil.move(item_or, item_ob)

    for item_or, item_ob in zip(png_path, png_path_ob):
        if os.path.exists(item_or):
            shutil.move(item_or, item_ob)


def restruct_fitting_outcat(csv_png_folder, fitting_outcat_path=None):
    """
    将所有拟合结果拼接成一个核表,并根据ID排序
    """
    csv_folder = os.path.join(csv_png_folder, 'csv')
    csv_path_ob = [os.path.join(csv_folder, item) for item in os.listdir(csv_folder) if item[-7:-4].isalnum()]
    outcat_df = pd.DataFrame([])
    for item in csv_path_ob:
        outcat_item = pd.read_csv(item, sep='\t')
        if outcat_item.shape[1] == 1:
            outcat_item = pd.read_csv(item, sep=',')
        outcat_df = pd.concat([outcat_df, outcat_item], axis=0)

    fitting_outcat = outcat_df.sort_values(by='ID')
    if fitting_outcat_path is None:
        fitting_outcat_path = os.path.join(csv_png_folder, os.path.pardir)
        fitting_outcat_path = os.path.abspath(fitting_outcat_path)
        fitting_outcat_path = os.path.join(fitting_outcat_path, 'Fitting_outcat.csv')

    fitting_outcat.to_csv(fitting_outcat_path, sep='\t', index=False)
    return fitting_outcat_path


def display_data(data):
    if data.ndim == 3:
        fig = plt.figure(figsize=(15, 8))

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.imshow(data.sum(0), origin='lower')  # x银纬,y银经
        ax2.imshow(data.sum(1), origin='lower')  # x银纬，y速度
        ax3.imshow(data.sum(2), origin='lower')  # x银经，y速度
        plt.show()
        return fig, (ax1, ax2, ax3)


def display_clumps_fitting(pif_1, pif_2, data):

    fig, (ax, ax1, ax2) = display_data(data)
    ax.scatter(pif_1['Cen3'], pif_1['Cen2'], c='r')
    ax1.scatter(pif_1['Cen3'], pif_1['Cen1'], c='r')
    ax2.scatter(pif_1['Cen2'], pif_1['Cen1'], c='r', label='fitting')

    # pif_2 = df_temp_1[['Cen1', 'Cen2', 'Cen3']] - points_all_df[['x_2', 'y_1', 'v_0']].values.min(axis=0)
    ax.scatter(pif_2['Cen3'], pif_2['Cen2'], c='k')
    ax1.scatter(pif_2['Cen3'], pif_2['Cen1'], c='k')
    ax2.scatter(pif_2['Cen2'], pif_2['Cen1'], c='k', label='ldc')

    ax2.legend(loc='best', framealpha=0.5)
    p_1_1, p_1_2, p_2_1, p_2_2 = pif_1['Cen3'].values, pif_2['Cen3'].values, pif_1['Cen2'].values, pif_2[
        'Cen2'].values
    for ii in range(p_1_1.shape[0]):
        ax.plot([p_1_1[ii], p_1_2[ii]], [p_2_1[ii], p_2_2[ii]], 'r')

    p_1_1, p_1_2, p_2_1, p_2_2 = pif_1['Cen3'].values, pif_2['Cen3'].values, pif_1['Cen1'].values, pif_2[
        'Cen1'].values
    for ii in range(p_1_1.shape[0]):
        ax1.plot([p_1_1[ii], p_1_2[ii]], [p_2_1[ii], p_2_2[ii]], 'r')

    p_1_1, p_1_2, p_2_1, p_2_2 = pif_1['Cen2'].values, pif_2['Cen2'].values, pif_1['Cen1'].values, pif_2[
        'Cen1'].values
    for ii in range(p_1_1.shape[0]):
        ax2.plot([p_1_1[ii], p_1_2[ii]], [p_2_1[ii], p_2_2[ii]], 'r')


if __name__ == '__main__':
    outcat_name = r'F:\Parameter_reduction\LDC\0170+010_L\LDC_auto_loc_outcat_fitting\fitting_outcat.csv'
    outcat_name_1 = r'F:\Parameter_reduction\LDC\0170+010_L\LDC_auto_loc_outcat.csv'
    origin_data_name = r'F:\Parameter_reduction\LDC\0170+010_L\0170+010_L.fits'
    data = Data(data_path=origin_data_name)

    data_wcs = data.wcs
    data_cube = data.data_cube
    pif_1 = pd.read_csv(outcat_name, sep='\t')
    pif_2 = pd.read_csv(outcat_name_1, sep=',')
    points_name_list = [os.path.join(r'F:\Parameter_reduction\LDC\0170+010_L\0170+010_L_points', item) for item in
                        os.listdir(r'F:\Parameter_reduction\LDC\0170+010_L\0170+010_L_points')]
    data = get_datacube_by_points(points_name_list)
    points_all = pd.DataFrame()
    for points_name in points_name_list:
        points = pd.read_csv(points_name)
        points_all = pd.concat([points_all, points], axis=0)
    pif_2 = pif_2[['Cen1', 'Cen2', 'Cen3']] - points_all[['x_2', 'y_1', 'v_0']].values.min(axis=0)
    pif_1 = pif_1[['Cen1', 'Cen2', 'Cen3']] - points_all[['x_2', 'y_1', 'v_0']].values.min(axis=0)
    display_clumps_fitting(pif_1, pif_2, data)

import os
import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from DensityClust.locatDenClust2 import DetectResult, Data
from DensityClust.locatDenClust2 import LocalDensityCluster as LDC


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
    create_folder(save_path)

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


def get_points_by_clumps_id(clumps_id, points_path):
    points_all_df = pd.DataFrame([])

    for id_clumps_index in clumps_id:
        clump_id_path = os.path.join(points_path, 'clump_id_xyz_intensity_%04d.csv' % id_clumps_index)
        xyv_intensity = pd.read_csv(clump_id_path)
        points_all_df = pd.concat([points_all_df, xyv_intensity], axis=0)

    return points_all_df


def data_points_prepare():
    origin_data_name = r'0170+010_L\0170+010_L.fits'
    mask_name = r'0170+010_L\LDC_auto_mask.fits'
    outcat_name = r'0170+010_L\LDC_auto_outcat.csv'
    save_path = r'0170+010_L\0170+010_L_all_points'
    create_folder(save_path)
    get_save_clumps_xyv(origin_data_name, mask_name, outcat_name, save_path)


def move_csv_png(csv_png_folder):
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


def restruct_fitting_outcat(csv_png_folder):
    csv_folder = os.path.join(csv_png_folder, 'csv')
    csv_path_ob = [os.path.join(csv_folder, item) for item in os.listdir(csv_folder)]
    outcat_df = pd.DataFrame([])
    for item in csv_path_ob:
        outcat_item = pd.read_csv(item, sep='\t')
        outcat_df = pd.concat([outcat_df, outcat_item], axis=0)

    fitting_outcat = outcat_df.sort_values(by='ID')
    fitting_outcat.to_csv(os.path.join(csv_png_folder, 'fitting_outcat.csv'), sep='\t', index=False)



def pix2wcs_show_result():
    origin_data_name = r'F:\Parameter_reduction\LDC\0170+010_L\0170+010_L.fits'
    mask_name = r'0170+010_L\LDC_auto_mask.fits'
    outcat_name = r'F:\Parameter_reduction\LDC\0170+010_L\LDC_auto_loc_outcat_fitting\fitting_outcat.csv'
    outcat_name_1 = r'F:\Parameter_reduction\LDC\0170+010_L\LDC_auto_loc_outcat_wcs.csv'
    save_path = r'0170+010_L\0170+010_L_all_points'

    data = Data(origin_data_name)
    ldc = LDC(data=data,para=None)
    ldc.result.outcat = pd.read_csv(outcat_name_1, sep='\t')
    outcat = ldc.result.outcat
    outcat_new = outcat[['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1',
       'Size2', 'Size3', 'Peak', 'Sum']]
    outcat_new['Volume'] = outcat['cost']
    ldc.result.outcat_wcs = ldc.change_pix2world(outcat_new)
    # ldc.result.data.get_wcs()
    # detect_result = DetectResult()
    # detect_result.outcat_wcs = ldc.result.outcat_wcs
    data_wcs = data.wcs
    data_cube = data.data_cube
    outcat_wcs = ldc.result.outcat_wcs
    make_plot_wcs_1(outcat_wcs, data_wcs, data_cube)


def display_data(data):
    if data.ndim == 3:
        fig = plt.figure(figsize=(15, 8))

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        im0 = ax1.imshow(data.sum(0), origin='lower')  # x银纬,y银经
        im1 = ax2.imshow(data.sum(1), origin='lower')  # x银纬，y速度
        im2 = ax3.imshow(data.sum(2), origin='lower')  # x银经，y速度
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
    data = Data(origin_data_name)

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

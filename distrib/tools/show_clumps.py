import astropy.io.fits as fits
import pandas as pd
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import os
import numpy as np
from astropy.coordinates import SkyCoord
from distrib.tools.ultil_lxy import get_data_points


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.color'] = 'red'
plt.rcParams['ytick.color'] = 'red'


def get_wcs(data_name):
    """
    得到wcs信息
    :param data_name: fits文件
    :return:
    data_wcs
    """
    data_header = fits.getheader(data_name)
    keys = data_header.keys()
    key = [k for k in keys if k.endswith('4')]
    try:
        [data_header.remove(k) for k in key]
        data_header.remove('VELREF')
    except KeyError:
        pass
    data_wcs = WCS(data_header)

    return data_wcs


def make_plot_11(data_mask_plot, spectrum_max, spectrum_mean, Cen3, Peak3, title_name, save_path):

    bottom_123 = 0.66
    pad = 0.005
    width = 0.01
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_axes([0.1, bottom_123, 0.28, 0.28])
    ax2 = fig.add_axes([0.4, bottom_123, 0.28, 0.28])
    ax3 = fig.add_axes([0.7, bottom_123, 0.28, 0.28])
    ax4 = fig.add_axes([0.1, 0.33, 0.8, 0.28])

    im0 = ax1.imshow(data_mask_plot.sum(0), origin='lower')  # x银纬,y银经
    im1 = ax2.imshow(data_mask_plot.sum(1), origin='lower')  # x银纬，y速度
    im2 = ax3.imshow(data_mask_plot.sum(2), origin='lower')  # x银经，y速度

    ax1.set_xlabel('latitude', fontsize=8)
    ax1.set_ylabel('Longitude', fontsize=8)
    ax2.set_xlabel('Latitude', fontsize=8)
    ax2.set_ylabel('Velocity', fontsize=8)
    ax2.set_title(title_name, fontsize=12)
    ax3.set_xlabel('Longitude', fontsize=8)
    ax3.set_ylabel('Velocity', fontsize=8)

    pos = ax1.get_position()
    axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])
    cbar = fig.colorbar(im0, cax=axes1)
    # cbar.set_label('K m s${}^{-1}$')

    pos = ax2.get_position()
    axes2 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])
    cbar = fig.colorbar(im1, cax=axes2)

    pos = ax3.get_position()
    axes3 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])
    cbar = fig.colorbar(im2, cax=axes3)
    # plt.xticks([]), plt.yticks([])

    line3, = ax4.plot(spectrum_max)
    line4, = ax4.plot([Peak3, Peak3], [spectrum_max.min(), spectrum_max.max() * 1.2], 'b--')
    ax4.set_ylabel('K [T]')
    line3.set_label('Maximum spectrum')
    line4.set_label('Peak3')
    ax4.legend()

    ax5 = fig.add_axes([0.1, 0.05, 0.8, 0.28])
    line1,  = ax5.plot(spectrum_mean)
    line2,  = ax5.plot([Cen3, Cen3], [spectrum_mean.min(), spectrum_mean.max() * 1.2], 'b--')
    line1.set_label('Average spectrum')
    line2.set_label('Cen3')
    ax5.legend()
    ax5.set_xlabel('Velocity [pixels]')
    ax5.set_ylabel('K [T]')

    save_fig_path = os.path.join(save_path, '%s.png') % title_name
    fig.savefig(save_fig_path)
    plt.close()


def make_plot_wcs(data_name, loc_outcat_wcs_name, outcat_wcs_name):
    # data_name = r'R2_data\data_9\0180-005\0180-005_L.fits'
    fits_path = data_name.replace('.fits', '')
    title = fits_path.split('\\')[-1]
    # loc_outcat_wcs_name = os.path.join(fits_path, 'LDC_loc_outcat_wcs.txt')
    # outcat_wcs_name = os.path.join(fits_path, 'LDC_outcat_wcs.txt')
    fig_name = os.path.join(fits_path, title + '.png')

    outcat_wcs = pd.read_csv(outcat_wcs_name, sep='\t')
    loc_outcat_wcs = pd.read_csv(loc_outcat_wcs_name, sep='\t')

    wcs = get_wcs(data_name)
    data_cube = fits.getdata(data_name)

    loc_outcat_wcs_c = SkyCoord(frame="galactic", l=loc_outcat_wcs['Cen1'].values, b=loc_outcat_wcs['Cen2'].values,
                                unit="deg")
    outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Cen1'].values, b=outcat_wcs['Cen2'].values, unit="deg")

    fig = plt.figure(figsize=(5, 4.25), dpi=100)

    axes0 = fig.add_axes([0.15, 0.1, 0.7, 0.82], projection=wcs.celestial)
    axes0.set_xticks([])
    axes0.set_yticks([])
    im0 = axes0.imshow(data_cube.sum(axis=0))
    axes0.plot_coord(outcat_wcs_c, 'r*', markersize=2.5)
    axes0.plot([30, 30], [30, 90], 'r')
    axes0.plot([90, 30], [30, 30], 'r')
    axes0.plot([90, 90], [30, 90], 'r')
    axes0.plot([90, 30], [90, 90], 'r')

    axes0.set_xlabel("Galactic Longutide", fontsize=12)
    axes0.set_ylabel("Galactic Latitude", fontsize=12)
    axes0.set_title(title, fontsize=12)
    pos = axes0.get_position()
    pad = 0.01
    width = 0.02
    axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

    cbar = fig.colorbar(im0, cax=axes1)
    cbar.set_label('K m s${}^{-1}$')


def display_data(data):
    if data.ndim == 3:
        fig = plt.figure(figsize=(15, 8))

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.imshow(data.sum(0), origin='lower')  # x银纬,y银经
        ax1.set_xlabel('$b$')
        ax1.set_ylabel('$l$')
        ax2.imshow(data.sum(1), origin='lower')  # x银纬，y速度
        ax2.set_xlabel('$b$')
        ax2.set_ylabel('$v$')
        ax3.imshow(data.sum(2), origin='lower')  # x银经，y速度
        ax3.set_xlabel('$l$')
        ax3.set_ylabel('$v$')
        return fig, (ax1, ax2, ax3)


def display_clumps_fitting(pif_1, df_temp_1, points_all_df, fig_name):
    data = get_data_points(points_all_df)
    fig, (ax, ax1, ax2) = display_data(data)
    ax.scatter(pif_1['Cen3'], pif_1['Cen2'], c='r')
    ax1.scatter(pif_1['Cen3'], pif_1['Cen1'], c='r')
    ax2.scatter(pif_1['Cen2'], pif_1['Cen1'], c='r', label='fitting')

    pif_2 = df_temp_1[['Cen1', 'Cen2', 'Cen3']] - points_all_df[['x_2', 'y_1', 'v_0']].values.min(axis=0)
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

    fig.savefig(fig_name)
    plt.close(fig)


def make_plot_wcs_1(outcat_wcs, data_wcs, data_cube, fig_name=''):
    """
    在积分图上绘制检测结果
    当没有检测到云核时，只画积分图
    """
    markersize = 2
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
        if 'Cen1' in outcat_wcs.keys():
            outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Cen1'].values, b=outcat_wcs['Cen2'].values,
                                    unit="deg")
        else:
            outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Galactic_Longitude'].values,
                                    b=outcat_wcs['Galactic_Latitude'].values, unit="deg")
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
    if fig_name == '':
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig=fig)


def deal_data(path_outcat_pix, path_outcat_wcs, path_mask, path_data, path_save_fig, v_len=30):
    """

    对检测的结果，将每个云核进行可视化
    处理数据并调用 make_plot_11()画图

    :param path_detect: 一个Cell检测结果保存的路径
    :param path_save_fig: 对云核可视化的保存路径
    :return:
    """

    """
    :param path_outcat: 像素核表
    :param path_outcat_wcs: 经纬度核表
    :param path_data: 数据
    :param path_mask: 掩膜
    :param path_save_fig: 图片保存路径
    :param v_len: 谱线速度轴的1/2（自己定义）
    :return:
    """
    if not os.path.exists(path_save_fig):
        os.mkdir(path_save_fig)

    table_LDC = pd.read_csv(path_outcat_pix, sep='\t')  # 像素核表
    if table_LDC.shape[1] == 1:
        table_LDC = pd.read_csv(path_outcat_pix, sep=',')
    table_LDC_wcs = pd.read_csv(path_outcat_wcs, sep='\t')  # 经纬度核表
    if table_LDC_wcs.shape[1] == 1:
        table_LDC_wcs = pd.read_csv(path_outcat_wcs, sep=',')

    data = fits.getdata(path_data)  # 数据
    mask = fits.getdata(path_mask)  # 掩膜
    [size_v, size_y, size_x] = mask.shape

    expend = 1
    for item_i, item in enumerate(table_LDC['ID']):
        item = int(item)
        peak1 = table_LDC['Peak1'][item_i]
        peak2 = table_LDC['Peak2'][item_i]
        peak3 = table_LDC['Peak3'][item_i]
        cen3 = table_LDC['Cen3'][item_i]
        title_name = table_LDC_wcs['ID'][item_i]

        mask1 = mask.copy()
        mask1[mask1 != item] = 0
        clump_item = data * mask1 / item
        [a, b, c] = np.where(clump_item != 0)
        aa = np.array([[a.min() - expend, 0], [a.max() + 1 + expend, size_v]])
        bb = np.array([[b.min() - expend, 0], [b.max() + 1 + expend, size_y]])
        cc = np.array([[c.min() - expend, 0], [c.max() + 1 + expend, size_x]])
        clump_item_loc = clump_item[aa[0].max(): aa[1].min(), bb[0].max(): bb[1].min(), cc[0].max(): cc[1].min()]

        v_range = np.array([[int(peak3 - v_len), 0], [int(peak3 + v_len), size_v - 1]])
        data_i_max = np.zeros(v_len * 2)
        data_i_mean_ = np.zeros(v_len * 2)

        data_i_max[:v_range[1].min() - v_range[0].max()] = data[v_range[0].max(): v_range[1].min(), int(peak2) - 1, int(peak1) - 1]

        data_i_mean = clump_item[v_range[0].max(): v_range[1].min(), bb[0].max(): bb[1].min(), cc[0].max(): cc[1].min()]
        for i in range(v_range[1].min() - v_range[0].max()):
            data_i_mean_[i] = data_i_mean[i, ...].mean()

        loc_cen3 = cen3 - v_range[0].max()-1
        loc_peak3 = peak3 - v_range[0].max()-1

        make_plot_11(clump_item_loc, data_i_max, data_i_mean_, loc_cen3, loc_peak3, title_name, path_save_fig)


def ldc_pic_ver(path_detect='', path_save_fig='', loc=False):

    if loc == True:
        path_outcat = os.path.join(path_detect, 'LDC_loc_outcat.csv')
        path_outcat_wcs = os.path.join(path_detect, 'LDC_loc_outcat_wcs.csv')
    else:
        path_outcat = os.path.join(path_detect, 'LDC_outcat.csv')
        path_outcat_wcs = os.path.join(path_detect, 'LDC_outcat_wcs.csv')

    path_mask = os.path.join(path_detect, 'LDC_mask.fits')
    detect_log = os.path.join(path_detect, 'LDC_detect_log.txt')
    f = open(detect_log)
    file_name = [item for item in f.readlines() if item.count('fits')]
    f.close()
    file_name = file_name[0].split(': ')[-1][:-1]
    path_data = file_name
    deal_data(path_outcat, path_outcat_wcs, path_mask, path_data, path_save_fig, v_len=30)


def plot_average_spectr(data):
    '''
    平均谱SpectralCube read data
    data = SpectralCube.read(path).with_spectral_unit(u.km/u.s)
    '''

    plt.figure(dpi=300)
    plt.step(data.spectral_axis.value, np.nanmean(
        np.nanmean(data, axis=1), axis=1), c='black')
    plt.grid(linestyle='--')  # 加个网格
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Average $T_\mathregular{MB}$ (k)')
    plt.show()


def plot_lv_diagram(data):
    '''
    画l-v图SpectralCube read data
    data = SpectralCube.read(path).with_spectral_unit(u.km/u.s)
    '''

    datalv = np.nanmean(data, axis=1)
    plt.figure(dpi=300)
    extent = [data.longitude_extrema[1].value, data.longitude_extrema[0].value,
              data.spectral_extrema[0].value, data.spectral_extrema[1].value]
    im = plt.imshow(datalv.data, origin='lower', extent=extent, aspect='auto')
    # origin表示图像像素起始位置
    # imshow aspect调整长宽比
    cb = plt.colorbar(pad=0.01, mappable=im)
    cb.set_label("Brightness Temperature [K]")

    plt.gca().set_ylabel("Velocity [km/s]")
    plt.gca().set_xlabel("Galactic longitude [deg]")

    plt.tight_layout()
    plt.show()


def plot_bv_diagram(data):
    '''
    画b-v图SpectralCube read data
    data = SpectralCube.read(path).with_spectral_unit(u.km/u.s)
    '''
    datalv = np.nanmean(data, axis=2)
    plt.figure(dpi=300)
    extent = [data.latitude_extrema[1].value, data.latitude_extrema[0].value,
              data.spectral_extrema[0].value, data.spectral_extrema[1].value]
    im = plt.imshow(datalv.data, origin='lower', extent=extent, aspect='auto')
    # origin表示图像像素起始位置
    # imshow aspect调整长宽比
    cb = plt.colorbar(pad=0.01, mappable=im)
    cb.set_label("Brightness Temperature [K]")

    plt.gca().set_ylabel("Velocity [km/s]")
    plt.gca().set_xlabel("Galactic latitude [deg]")

    plt.tight_layout()
    plt.show()

    """
    # Usage:----------------------------------------------
    
    # path = '/home/data/clumps_share/MWISP/R2_R16_Region/0145-005_L.fits'
    # path = '/home/data/clumps_share/MWISP/G100+00/luoxy/1025+030_L.fits'
    path = '/home/data/clumps_share/MWISP/R2_R16_Region/1860+000_L.fits'
    data = SpectralCube.read(path).with_spectral_unit(u.km/u.s)
    print(data)
    plot_average_spectr(data)
    plot_lv_diagram(data)
    plot_bv_diagram(data)
    """


if __name__ == '__main__':
    pass




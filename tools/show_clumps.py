import astropy.io.fits as fits
import pandas as pd
import matplotlib.pyplot as plt
from astropy import wcs
import os
from astropy.coordinates import SkyCoord
from tools.ultil_lxy import get_data_points


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
    [data_header.remove(k) for k in key]
    data_header.remove('VELREF')
    data_wcs = wcs.WCS(data_header)
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
        im0 = ax1.imshow(data.sum(0), origin='lower')  # x银纬,y银经
        im1 = ax2.imshow(data.sum(1), origin='lower')  # x银纬，y速度
        im2 = ax3.imshow(data.sum(2), origin='lower')  # x银经，y速度
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


if __name__ == '__main__':
    pass




from matplotlib import pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
import os
from DensityClust.clustring_subfunc import *
from DensityClust.clustring_subfunc import get_wcs


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = 'True'
plt.rcParams['ytick.right'] = 'True'
plt.rcParams['xtick.color'] = 'red'
plt.rcParams['ytick.color'] = 'red'


def make_plot_wcs_1(data_name, outcat_wcs_name):
    """
    在积分图上绘制检测结果
    :param data_name: 3d data cube fits
    :param outcat_wcs_name: detection outcat
    :return:
    """
    # sc = SkyCoord(1 * u.deg, 2 * u.deg, radial_velocity=20 * u.km / u.s)
    fits_path = data_name.replace('.fits', '')
    title = fits_path.split('\\')[-1]

    outcat_wcs = pd.read_csv(outcat_wcs_name, sep='\t')
    wcs = get_wcs(data_name)
    data_cube = fits.getdata(data_name)

    fig = plt.figure(figsize=(10, 8.5), dpi=100)

    axes0 = fig.add_axes([0.15, 0.1, 0.7, 0.82], projection=wcs.celestial)
    axes0.set_xticks([])
    axes0.set_yticks([])
    im0 = axes0.imshow(data_cube.sum(axis=0))
    if outcat_wcs.values.shape[0] > 0:
        outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Cen1'].values, b=outcat_wcs['Cen2'].values, unit="deg")
        axes0.plot_coord(outcat_wcs_c, 'r*', markersize=5)

    axes0.set_xlabel("Galactic Longitude", fontsize=12)
    axes0.set_ylabel("Galactic Latitude", fontsize=12)
    axes0.set_title(title, fontsize=12)
    pos = axes0.get_position()
    pad = 0.01
    width = 0.02
    axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

    cbar = fig.colorbar(im0, cax=axes1)
    cbar.set_label('K m s${}^{-1}$')


def make_plot(outcat_name, data, lable_num=False):
    """
    将检测结果标记在原始图像上(3d data-->三个积分图),可以选择把序号标记在上面
    :param outcat_name:
    :param data:
    :param lable_num:
    :return:
    """
    outcat = pd.read_csv(outcat_name, sep='\t')
    ID = outcat['ID'].values
    Sum = outcat['Sum'].values
    if data.ndim == 2:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
        ax0.imshow(data, cmap='gray')
        if outcat is not None:
            Cen1 = outcat['Cen1'] - 1
            Cen2 = outcat['Cen2'] - 1
            Peak1 = outcat['Peak1'] - 1
            Peak2 = outcat['Peak2'] - 1

            ax0.plot(Cen1, Cen2, '.', color='red')
            ax0.plot(Peak1, Peak2, '*', color='green')
            if lable_num:  
                for i in range(outcat.shape[0]):       
                    ax0.text(Cen1[i], Cen2[i], '%d:%.2f' % (ID[i], Sum[i]), color='r')
                    ax0.text(Cen1[i], Cen2[i], '%d' % (ID[i]), color='r')

        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.show()

    elif data.ndim == 3:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 6))
        ax0.imshow(data.sum(axis=0), cmap='gray')
        ax1.imshow(data.sum(axis=1), cmap='gray')
        ax2.imshow(data.sum(axis=2), cmap='gray')
        if outcat is not None:
            Cen1 = outcat['Cen1'] - 1
            Cen2 = outcat['Cen2'] - 1
            Cen3 = outcat['Cen3'] - 1

            ax0.scatter(Cen1, Cen2, marker='.', c='red', s=8)
            ax1.scatter(Cen1, Cen3, marker='*', c='red', s=8)
            ax2.scatter(Cen2, Cen3, marker='^', c='red', s=8)

            if lable_num:
                for i in range(outcat.shape[0]):
                    ax0.text(Cen1[i], Cen2[i], '%d' % ID[i], color='g')
                    ax1.text(Cen1[i], Cen3[i], '%d' % ID[i], color='g')
                    ax2.text(Cen2[i], Cen3[i], '%d' % ID[i], color='g')

        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.show()


def make_tri_plot(data):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 6))
    ax0.imshow(data.sum(axis=0), cmap='gray',)
    ax1.imshow(data.sum(axis=1), cmap='gray')
    ax2.imshow(data.sum(axis=2), cmap='gray')

    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def make_two_plot(data, outcat):
    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    ax0.imshow(data, cmap='gray')
    ax0.plot(outcat[:, 3] - 1, outcat[:, 4] - 1, 'r*')
    ax0.plot(outcat[:, 1] - 1, outcat[:, 2] - 1, 'go')

    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_match():
    match_txt = r'test_data\2d_simulated_clump\gaussian_out_360\match\Match_table\Match_LDC.txt'
    match = pd.read_csv(match_txt, sep='\t')
    plt.subplot(3, 2, 1)
    plt.plot(match['s_Cen1'].values, match['f_Cen1'].values, '.')
    plt.subplot(3, 2, 2)
    plt.plot(match['s_Cen2'].values, match['f_Cen2'].values, '.')
    plt.subplot(3, 2, 3)
    plt.plot(match['s_Size1'].values, match['f_Size1'].values, '.')
    plt.subplot(3, 2, 4)
    plt.plot(match['s_Size2'].values, match['f_Size2'].values, '.')
    plt.subplot(3, 2, 5)
    plt.plot(match['s_Peak'].values, match['f_Peak'].values, '.')
    plt.subplot(3, 2, 6)
    plt.plot(match['s_Sum'].values, match['f_Sum'].values, '.')
    plt.show()


if __name__ == '__main__':
        pass
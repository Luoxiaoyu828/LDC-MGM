import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
import numpy as np
import astropy.io.fits as fits
import os
import pandas as pd
import astropy.units as u
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from show_clumps import get_wcs

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.color'] = 'red'
plt.rcParams['ytick.color'] = 'red'


def get_clump_name(item_l, item_b, item_v, id_prefix):
    str_l = id_prefix + ('%.03f' % item_l).rjust(7, '0')
    if item_b < 0:
        str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
    else:
        str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
    if item_v < 0:
        str_v = '-' + ('%.03f' % abs(item_v)).rjust(6, '0')
    else:
        str_v = '+' + ('%.03f' % abs(item_v)).rjust(6, '0')
    id_clumps = str_l + str_b + str_v
    return id_clumps


def make_plot_11(data_mask_plot, spectrum_max, spectrum_mean, parm, extent, header_hdu, save_path):

    bottom_123 = 0.66
    pad = 0.005
    width = 0.01
    index0 = np.unravel_index(np.argmax(data_mask_plot[0]), data_mask_plot[0].shape)  # lb积分图---projection方式

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax1 = fig.add_axes([3 / 50, bottom_123, 0.28, 0.28], projection=header_hdu[0, parm['pix_cen2'] - index0[0] - 1:,
                                                                    parm['pix_cen1'] - len(data_mask_plot[0][0]) +
                                                                    index0[1]:])
    ax2 = fig.add_axes([9/25, bottom_123, 0.28, 0.28])
    ax3 = fig.add_axes([33/50, bottom_123, 0.28, 0.28])
    ax4 = fig.add_axes([0.1, 0.33, 0.8, 0.28])
    ax5 = fig.add_axes([0.1, 0.05, 0.8, 0.28])

    v1 = extent[0][3] - extent[0][2]
    l1 = extent[0][0] - extent[0][1]
    v2 = extent[1][3] - extent[1][2]
    l2 = extent[1][0] - extent[1][1]

    im0 = ax1.imshow(data_mask_plot[0], origin='lower')  # x银纬,y银经
    im1 = ax2.imshow(data_mask_plot[1], origin='lower', extent=extent[0], aspect=abs(l1/v1))  # x银纬，y速度
    im2 = ax3.imshow(data_mask_plot[2], extent=extent[1], aspect=abs(l2/v2), origin='lower')  # x银经，y速度
    lon = ax1.coords[0]
    lat = ax1.coords[1]
    lon.set_ticklabel(exclude_overlapping=True)
    lat.set_ticklabel(exclude_overlapping=True)  # 防止刻度重叠

    xmajorFormatter = FormatStrFormatter('%1.2f')
    ax2.xaxis.set_major_formatter(xmajorFormatter)  # 刻度调整
    ax3.xaxis.set_major_formatter(xmajorFormatter)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(4))

    ax1.set_ylabel('Galactic_Latitude', fontsize=8)
    ax1.set_xlabel('Galactic_Longitude', fontsize=8)
    ax2.set_xlabel('Galactic_Longitude [deg]', fontsize=8)
    ax2.set_ylabel('Velocity [km/s]', fontsize=8)
    ax2.set_title(parm['title_name'], fontsize=12)
    ax3.set_xlabel('Galactic_Latitude [deg]', fontsize=8)
    ax3.set_ylabel('Velocity [km/s]', fontsize=8)
    formatter = mpl.ticker.StrMethodFormatter('{x:.0f}')  # colorbar设置整数

    pos1 = ax1.get_position()
    axes1 = fig.add_axes([pos1.xmax + pad, pos1.ymin, width, 1 * (pos1.ymax - pos1.ymin)])
    cbar = fig.colorbar(im0, cax=axes1, format=formatter)
    # cbar.set_label('K m s${}^{-1}$')

    # pos2 = ax2.bbox._bbox
    pos2 = ax2.get_position()
    axes2 = fig.add_axes([pos2.xmax + pad, pos2.ymin, width, 1 * (pos2.ymax - pos2.ymin)])
    cbar = fig.colorbar(im1, cax=axes2, format=formatter)

    pos3 = ax3.get_position()
    axes3 = fig.add_axes([pos3.xmax + pad, pos3.ymin, width, 1 * (pos3.ymax - pos3.ymin)])
    cbar = fig.colorbar(im2, cax=axes3, format=formatter)

    line3, = ax4.plot(parm['v_wcs_range'], spectrum_max)
    line4, = ax4.plot([parm['wcs_peak3'], parm['wcs_peak3']], [parm['a_min'], parm['a_max'] * 1.2], 'b--')
    ax4.fill_between(parm['fill_gray'], parm['a_min'], parm['a_max'] * 1.2, facecolor='gray', alpha=0.3)
    ax4.set_xlim(min(parm['v_wcs_range']), max(parm['v_wcs_range']))
    ax4.set_ylabel('T [K]')
    line3.set_label('Maximum spectrum')
    line4.set_label('Peak3')
    ax4.legend()

    line1,  = ax5.plot(parm['v_wcs_range'], spectrum_mean)
    line2,  = ax5.plot([parm['wcs_cen3'], parm['wcs_cen3']], [parm['b_min'], parm['b_max'] * 1.2], 'b--')
    ax5.fill_between(parm['fill_gray'], parm['b_min'], parm['b_max'] * 1.2, facecolor='gray', alpha=0.3)
    ax5.set_xlim(min(parm['v_wcs_range']), max(parm['v_wcs_range']))
    line1.set_label('Average spectrum')
    line2.set_label('Cen3')
    ax5.legend()
    ax5.set_xlabel('Velocity [km/s]')
    ax5.set_ylabel('T [K]')

    save_fig_path = os.path.join(save_path, '%s.png') % parm['title_name']
    # plt.tight_layout()
    fig.savefig(save_fig_path)
    # plt.show()
    plt.close()


# 证认画图
def deal_data(path_data, path_outcat, path_outcat_wcs, path_mask, path_save_fig, use_outcat=False, v_len=30, nn=1.5,
              v_resolution=0.166, km_number=1000, id_prefix='MWISP'):

    """
    made in zjj's hand

    :param path_data: fits文件
    :param path_outcat: 像素核表
    :param path_outcat_wcs: WCS核表
    :param path_mask: mask文件
    :param path_save_fig: 保存路径
    :param use_outcat: Default: False, eg: False---图显示mask区域
    :param v_len: Default: 30
    :param nn: Default: 1.5
    :param v_resolution: 速度分辨率 Default: 0.166
    :param km_number: WCS核表Cen3单位：m/s---km_number=1000, WCS核表Cen3单位：km/s---km_number=1
    :return:
    """

    os.makedirs(path_save_fig, exist_ok=True)

    data_wcs = get_wcs(path_data)
    data = SpectralCube.read(path_data).with_spectral_unit(u.km / u.s)
    table_outcat = pd.read_csv(path_outcat, sep='\t')  # 像素核表
    table_wcs = pd.read_csv(path_outcat_wcs, sep='\t')  # 经纬度核表, sep='\t'

    if table_outcat.shape[1] == 1:
        table_outcat = pd.read_csv(path_outcat)  # 像素核表
        table_wcs = pd.read_csv(path_outcat_wcs)

    data1 = fits.getdata(path_data)  # 数据
    mask = fits.getdata(path_mask)  # 掩膜
    data1[np.isnan(data1)] = -9999

    [size_v, size_y, size_x] = mask.shape

    expend = 1
    pix_number = 120

    for item_i, item in enumerate(table_outcat['ID']):
        item = int(item)

        peak1 = table_outcat['Peak1'][item_i]
        peak2 = table_outcat['Peak2'][item_i]
        peak3 = table_outcat['Peak3'][item_i]
        peak3_wcs = table_wcs['Peak3'][item_i]

        cen1 = table_outcat['Cen1'][item_i]
        cen2 = table_outcat['Cen2'][item_i]
        cen3 = table_outcat['Cen3'][item_i]
        cen1_wcs = table_wcs['Cen1'][item_i]
        cen2_wcs = table_wcs['Cen2'][item_i]
        cen3_wcs = table_wcs['Cen3'][item_i]

        # MWISP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
        title_name = get_clump_name(cen1_wcs, cen2_wcs, cen3_wcs, id_prefix)
        mask1 = mask.copy()
        mask1[mask1 != item] = 0
        mask1 = mask1 / item
        clump_item = data1 * mask1
        [a, b, c] = np.where(clump_item != 0)

        if use_outcat:
            # 核表中的位置是以1为坐标原点，python数组的索引是从0开始，索引Cen统一减去1
            size1 = table_outcat['Size1'][item_i]
            size2 = table_outcat['Size2'][item_i]
            size_12 = (size1 + size2) / 2
            aa = np.array([[cen3 - 1 - nn * size_12, 0], [cen3 - 1 + nn * size_12 + 1, size_v]], np.int64)
            bb = np.array([[cen2 - 1 - nn * size_12, 0], [cen2 - 1 + nn * size_12 + 1, size_y]], np.int64)
            cc = np.array([[cen1 - 1 - nn * size1, 0], [cen1 - 1 + nn * size1 + 1, size_x]], np.int64)
        else:
            # mask区域
            aa = np.array([[a.min() - expend, 0], [a.max() + 1 + expend, size_v]])
            bb = np.array([[b.min() - expend, 0], [b.max() + 1 + expend, size_y]])
            cc = np.array([[c.min() - expend, 0], [c.max() + 1 + expend, size_x]])

        clump_item_loc = clump_item[aa[0].max(): aa[1].min(), bb[0].max(): bb[1].min()
                                    , cc[0].max(): cc[1].min()]  # 经纬度积分图
        if np.any(clump_item_loc == -9999):
            print('%s data contain Nan!' % title_name)
            continue
        else:
            # 调整积分图坐标轴extend
            cen3_st = cen3_wcs / km_number - (cen3 - aa[0].max()) * v_resolution
            cen3_end = cen3_wcs / km_number + (aa[1].min() - cen3) * v_resolution
            cen2_st = cen2_wcs - (cen2 - bb[0].max()) / pix_number
            cen2_end = cen2_wcs + (bb[1].min() - cen2) / pix_number
            cen1_st = cen1_wcs - (cen1 - cc[0].max()) / pix_number
            cen1_end = cen1_wcs + (cc[1].min() - cen1) / pix_number

            v1_size, l1_size, b1_size = clump_item_loc.shape
            lv = abs(v1_size - l1_size)
            bv = abs(v1_size - b1_size)
            lv_clump = np.zeros((max(v1_size, l1_size), max(v1_size, l1_size)))
            bv_clump = np.zeros((max(v1_size, b1_size), max(v1_size, b1_size)))
            lv_2 = lv//2
            bv_2 = bv//2
            lv_lv = lv-lv_2
            bv_bv = bv-bv_2

            if v1_size >= l1_size:
                lv_clump[:, lv_2:v1_size - lv_lv] = clump_item_loc.sum(2)
                v_st = max(data.spectral_extrema[0].value, cen3_st)
                v_end = min(data.spectral_extrema[1].value, cen3_end)
                l_st = max(data.longitude_extrema[0].value, cen1_st) - lv_lv / pix_number
                l_end = min(data.longitude_extrema[1].value, cen1_end) + lv_2 / pix_number
            else:
                lv_clump[lv_2:l1_size - lv_lv, :] = clump_item_loc.sum(2)
                l_st = max(data.longitude_extrema[0].value, cen1_st)
                l_end = min(data.longitude_extrema[1].value, cen1_end)
                v_st = max(data.spectral_extrema[0].value, cen3_st) - lv_lv * v_resolution
                v_end = min(data.spectral_extrema[1].value, cen3_end) + lv_2 * v_resolution

            if v1_size >= b1_size:
                bv_clump[:, bv_2:v1_size - bv_bv] = clump_item_loc.sum(1)
                v_st1 = max(data.spectral_extrema[0].value, cen3_st)
                v_end1 = min(data.spectral_extrema[1].value, cen3_end)
                b_st = max(data.latitude_extrema[0].value, cen2_st) - bv_bv / pix_number
                b_end = min(data.latitude_extrema[1].value, cen2_end) + bv_2 / pix_number
            else:
                bv_clump[bv_2:b1_size - bv_bv, :] = clump_item_loc.sum(1)
                b_end = min(data.latitude_extrema[1].value, cen2_end)
                b_st = max(data.latitude_extrema[0].value, cen2_st)
                v_st1 = max(data.spectral_extrema[0].value, cen3_st) - bv_bv * v_resolution
                v_end1 = min(data.spectral_extrema[1].value, cen3_end) + bv_2 * v_resolution

            extent_lv = [l_end, l_st, v_st, v_end]
            extent_bv = [b_end, b_st, v_st1, v_end1]
            extent = [extent_lv, extent_bv]

            # lb
            lb_clump = np.zeros((max(b1_size, l1_size), max(b1_size, l1_size)))
            lb = abs(b1_size - l1_size)
            lb_2 = lb // 2
            lb_lb = lb - lb_2
            if b1_size >= l1_size:
                lb_clump[lb_2:b1_size - lb_lb, :] = clump_item_loc.sum(0)
            else:
                lb_clump[:, lb_2:l1_size - lb_lb] = clump_item_loc.sum(0)

            # 平均谱数据处理
            mask2 = mask1.copy()
            area = mask2.sum(1).sum(1)
            mean_index = np.where(area != 0)[0]
            area_index = np.argmax(area)
            mask2[[i for i in range(0, len(mask2.sum(1).sum(1)))]] = mask2[area_index]
            mask2_ = np.ones_like(mask2)
            mask2 = mask2_ * mask2[area_index]
            clump_item1 = data1 * mask2

            # peak3与v_len对应----[v_len - peak3, v_len + size_v-peak3]
            # 转为wcs范围-----[peak3_wcs - peak3*v_resolution,
            v_wcs_range = [i for i in np.arange(peak3_wcs / km_number - v_len * v_resolution,
                                                peak3_wcs / km_number + v_len * v_resolution - 0.06, v_resolution)]  # 坐标轴范围

            v_range = np.array([[int(peak3 - v_len - 1), 0], [int(peak3 + v_len - 1), size_v]])

            data_i_max = np.zeros(v_len * 2)
            data_i_max1 = data1[v_range[0].max(): v_range[1].min(), int(peak2) - 1, int(peak1) - 1]

            data_i_mean_ = np.zeros(v_len * 2)
            data_i_mean = clump_item1[v_range[0].max(): v_range[1].min(), bb[0].max(): bb[1].min()
                                      , cc[0].max(): cc[1].min()]
            data_i_mean1 = data_i_mean.mean(1).mean(1)  # 强度平均
            if peak3 < v_len:
                data_i_max[int(v_len - peak3 + 1):len(data_i_max1)+int(v_len - peak3 + 1)] = data_i_max1
                data_i_mean_[int(v_len - peak3 + 1):len(data_i_max1)+int(v_len - peak3 + 1)] = data_i_mean1
            else:
                data_i_max[:len(data_i_max1)] = data_i_max1
                data_i_mean_[:len(data_i_max1)] = data_i_mean1

            a_min = min(data_i_max)  # 虚线
            a_max = max(data_i_max)
            b_min = min(data_i_mean_)
            b_max = max(data_i_mean_)

            data_i_max = [x if x != 0 else None for x in data_i_max]
            data_i_mean_ = [x if x != 0 else None for x in data_i_mean_]
            mask_max_min = mean_index.shape[0] // 2 + 1
            loc_cen3 = cen3_wcs / km_number  # 虚线坐标
            loc_peak3 = peak3_wcs / km_number
            min_index = loc_cen3 - mask_max_min * v_resolution     # 阴影范围
            max_index = loc_cen3 + (mask_max_min+1) * v_resolution
            fill_x = [i for i in np.arange(min_index, max_index - 0.06, v_resolution)]

            parm = {'wcs_cen3': loc_cen3, 'wcs_peak3': loc_peak3, 'pix_cen1': cen1, 'pix_cen2': cen2
                    , 'v_wcs_range': v_wcs_range, 'fill_gray': fill_x, 'a_min': a_min, 'a_max': a_max
                    , 'b_min': b_min, 'b_max': b_max, 'title_name': title_name}
            data_all = [lb_clump, lv_clump, bv_clump]

            make_plot_11(data_all, data_i_max, data_i_mean_, parm, extent, data_wcs, path_save_fig)


if __name__ == '__main__':

    path_detect1 = r'D:\OneDrive_lxy\OneDrive - ctgu.edu.cn\LDC_MGM-main\data/debug data/mgm_bug/test3'
    path_data1 = r'D:\OneDrive_lxy\OneDrive - ctgu.edu.cn\LDC_MGM-main\data/debug data/mgm_bug/15.684-0.29_0.8_U.fits'
    path_save_fig = r'D:\OneDrive_lxy\OneDrive - ctgu.edu.cn\LDC_MGM-main\data/debug data/mgm_bug/test3/fig2'
    alg_name = 'LDC'
    path_outcat = os.path.join(path_detect1, alg_name + '_outcat.csv')
    path_outcat_wcs = os.path.join(path_detect1, alg_name + '_outcat_wcs.csv')
    path_mask = os.path.join(path_detect1, alg_name + '_mask.fits')
    deal_data(path_outcat=path_outcat, path_outcat_wcs=path_outcat_wcs, path_mask=path_mask
              , path_save_fig=path_save_fig, path_data=path_data1, km_number=1000)
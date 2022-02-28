import numpy as np
import time
from astropy import wcs
from tabulate import tabulate
import astropy.io.fits as fits
import pandas as pd


def setdiff_nd(a1, a2):
    """
    python 使用numpy求二维数组的差集
    :param a1:
    :param a2:
    :return:
    """
    # a1 = index_value
    # a2 = np.data_cube([point_ii_xy])
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])

    a3 = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    return a3


def get_xyz(data):
    """
    :param data: 3D data
    :return: 3D data coordinates
    第1,2,3维数字依次递增

     :param data: 2D data
    :return: 2D data coordinates
    第1,2维数字依次递增

    """
    nim = data.ndim
    if nim == 3:
        size_x, size_y, size_z = data.shape
        x_arange = np.arange(1, size_x+1)
        y_arange = np.arange(1, size_y+1)
        z_arange = np.arange(1, size_z+1)
        [xx, yy, zz] = np.meshgrid(x_arange, y_arange, z_arange, indexing='ij')
        xyz = np.column_stack([zz.flatten(), yy.flatten(), xx.flatten()])

    else:
        size_x, size_y = data.shape
        x_arange = np.arange(1, size_x + 1)
        y_arange = np.arange(1, size_y + 1)
        [xx, yy] = np.meshgrid(x_arange, y_arange, indexing='ij')
        xyz = np.column_stack([yy.flatten(), xx.flatten()])
    return xyz


def kc_coord_3d(point_ii_xy, xm, ym, zm, r):
    """
    :param point_ii_xy: 当前点坐标(x,y,z)
    :param xm: size_x
    :param ym: size_y
    :param zm: size_z
    :param r: 2 * r + 1
    :return:
    返回delta_ii_xy点r邻域的点坐标
    """
    it = point_ii_xy[0]
    jt = point_ii_xy[1]
    kt = point_ii_xy[2]

    xyz_min = np.array([[1, it - r], [1, jt - r], [1, kt - r]])
    xyz_min = xyz_min.max(axis=1)

    xyz_max = np.array([[xm, it + r], [ym, jt + r], [zm, kt + r]])
    xyz_max = xyz_max.min(axis=1)

    x_arange = np.arange(xyz_min[0], xyz_max[0] + 1)
    y_arange = np.arange(xyz_min[1], xyz_max[1] + 1)
    v_arange = np.arange(xyz_min[2], xyz_max[2] + 1)

    [p_k, p_i, p_j] = np.meshgrid(x_arange, y_arange, v_arange, indexing='ij')
    Index_value = np.column_stack([p_k.flatten(), p_i.flatten(), p_j.flatten()])
    Index_value = setdiff_nd(Index_value, np.array([point_ii_xy]))

    ordrho_jj = np.matmul(Index_value - 1, np.array([[1], [xm], [ym * xm]]))
    ordrho_jj.reshape([1, ordrho_jj.shape[0]])

    return ordrho_jj[:, 0], Index_value


def kc_coord_2d(point_ii_xy, xm, ym, r):
    """
    :param point_ii_xy: 当前点坐标(x,y)
    :param xm: size_x
    :param ym: size_y
    :param r: 2 * r + 1
    :return:
    返回point_ii_xy点r邻域的点坐标
    """
    it = point_ii_xy[0]
    jt = point_ii_xy[1]

    xyz_min = np.array([[1, it - r], [1, jt - r]])
    xyz_min = xyz_min.max(axis=1)

    xyz_max = np.array([[xm, it + r], [ym, jt + r]])
    xyz_max = xyz_max.min(axis=1)

    x_arrange = np.arange(xyz_min[0], xyz_max[0] + 1)
    y_arrange = np.arange(xyz_min[1], xyz_max[1] + 1)

    [p_k, p_i] = np.meshgrid(x_arrange, y_arrange, indexing='ij')
    Index_value = np.column_stack([p_k.flatten(), p_i.flatten()])
    Index_value = setdiff_nd(Index_value, np.array([point_ii_xy]))

    return Index_value


def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)


def save_outcat(outcat_name, outcat):
    """
    # 保存LDC检测的直接结果，即单位为像素
    :param outcat_name: 核表的路径
    :param outcat: 核表数据
    :return:
    """

    outcat_colums = outcat.shape[1]
    pd.DataFrame.to_fwf = to_fwf
    if outcat_colums == 10:
        # 2d result
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
        dataframe = pd.DataFrame(outcat, columns=table_title)
        dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                     'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(outcat_name, sep='\t', index=False)
        # dataframe.to_fwf(outcat_name)
    elif outcat_colums == 13:
        # 3d result
        table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum',
                       'Volume']
        dataframe = pd.DataFrame(outcat, columns=table_title)
        dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                     'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(outcat_name, sep='\t', index=False)
        # dataframe.to_fwf(outcat_name)

    elif outcat_colums == 11:
        # fitting 2d data result
        fit_outcat_name = outcat_name
        fit_outcat = outcat
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                       'Sum', 'Volume']
        dataframe = pd.DataFrame(fit_outcat, columns=table_title)
        dataframe = dataframe.round(
            {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
             'Sum': 3, 'Volume': 3})
        dataframe.to_csv(fit_outcat_name, sep='\t', index=False)
        # dataframe.to_fwf(fit_outcat_name)
    else:
        print('outcat columns is %d' % outcat_colums)


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
    data_wcs = wcs.WCS(data_header)

    return data_wcs


def change_pix2word(data_wcs, outcat):
    """
    将算法检测的结果(像素单位)转换到天空坐标系上去
    :param data_wcs: 头文件得到的wcs
    :param outcat: 算法检测核表
    :return:
    outcat_wcs
    ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume'] -->3d
     ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume']-->2d
    """
    outcat_column = outcat.shape[1]

    if outcat_column == 10:
        # 2d result
        peak1, peak2 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], 1)
        clump_Peak = np.column_stack([peak1, peak2])
        cen1, cen2 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1)
        clump_Cen = np.column_stack([cen1, cen2])
        size1, size2 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30])
        clustSize = np.column_stack([size1, size2])
        clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])
        id_clumps = []  # MWSIP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
        for item_l, item_b in zip(cen1, cen2):
            str_l = 'MWSIP' + ('%.03f' % item_l).rjust(7, '0')
            if item_b < 0:
                str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
            else:
                str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
            id_clumps.append(str_l + str_b)
        id_clumps = np.array(id_clumps)
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']

    elif outcat_column == 13:
        # 3d result
        peak1, peak2, peak3 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], outcat['Peak3'], 1)
        clump_Peak = np.column_stack([peak1, peak2, peak3 / 1000])
        cen1, cen2, cen3 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1)
        clump_Cen = np.column_stack([cen1, cen2, cen3 / 1000])
        size1, size2, size3 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30, outcat['Size3'] * 0.166])
        clustSize = np.column_stack([size1, size2, size3])
        clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

        id_clumps = []  # G017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
        for item_l, item_b, item_v in zip(cen1, cen2, cen3 / 1000):
            str_l = 'MWISP' + ('%.03f' % item_l).rjust(7, '0')
            if item_b < 0:
                str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
            else:
                str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
            if item_v < 0:
                str_v = '-' + ('%.03f' % abs(item_v)).rjust(6, '0')
            else:
                str_v = '+' + ('%.03f' % abs(item_v)).rjust(6, '0')
            id_clumps.append(str_l + str_b + str_v)
        id_clumps = np.array(id_clumps)
        table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3',
                       'Peak', 'Sum', 'Volume']
    else:
        print('outcat columns is %d' % outcat_column)
        return None

    outcat_wcs = np.column_stack((id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))
    outcat_wcs = pd.DataFrame(outcat_wcs, columns=table_title)
    return outcat_wcs


def change_word2pix(data_wcs, outcat_wcs):
    """
    将算法检测的天空坐标系结果转换到像素单位上去
    :param data_wcs: 头文件得到的wcs
    :param outcat_wcs: outcat_wcs
    :return:
    outcat
    ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume'] -->3d
     ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume']-->2d
    """
    outcat_column = outcat_wcs.shape[1]

    if outcat_column == 10:
        # 2d result
        peak1, peak2 = data_wcs.all_world2pix(outcat_wcs['Peak1'], outcat_wcs['Peak2'], 1)
        clump_Peak = np.column_stack([peak1, peak2])
        cen1, cen2 = data_wcs.all_world2pix(outcat_wcs['Cen1'], outcat_wcs['Cen2'], 1)
        clump_Cen = np.column_stack([cen1, cen2])
        size1, size2 = np.array([outcat_wcs['Size1'] / 30, outcat_wcs['Size2'] / 30])
        clustSize = np.column_stack([size1, size2])
        clustPeak, clustSum, clustVolume = np.array([outcat_wcs['Peak'], outcat_wcs['Sum'], outcat_wcs['Volume']])

        id_clumps = np.array([item + 1 for item in range(outcat_wcs.shape[0])])
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']

    elif outcat_column == 13:
        # 3d result
        peak1, peak2, peak3 = data_wcs.all_world2pix(outcat_wcs['Peak1'], outcat_wcs['Peak2'],
                                                     outcat_wcs['Peak3'] * 1000, 1)
        clump_Peak = np.column_stack([peak1, peak2, peak3])
        cen1, cen2, cen3 = data_wcs.all_world2pix(outcat_wcs['Cen1'], outcat_wcs['Cen2'], outcat_wcs['Cen3'] * 1000, 1)
        clump_Cen = np.column_stack([cen1, cen2, cen3])
        size1, size2, size3 = np.array([outcat_wcs['Size1'] / 30, outcat_wcs['Size2'] / 30, outcat_wcs['Size3'] / 0.166])
        clustSize = np.column_stack([size1, size2, size3])
        clustPeak, clustSum, clustVolume = np.array([outcat_wcs['Peak'], outcat_wcs['Sum'], outcat_wcs['Volume']])

        id_clumps = np.array([item + 1 for item in range(outcat_wcs.shape[0])])
        table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3',
                       'Peak', 'Sum', 'Volume']
    else:
        print('outcat columns is %d' % outcat_column)
        return None

    outcat = np.column_stack((id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))
    outcat = pd.DataFrame(outcat, columns=table_title)
    return outcat


def get_outcat_local(outcat):
    """
    返回局部区域的检测结果：
    原始图为120*120  局部区域为30-->90, 30-->90 左开右闭
    :param outcat:
    :return:
    """
    # outcat = pd.read_csv(txt_name, sep='\t')
    cen1_min = 30
    cen1_max = 90
    cen2_min = 30
    cen2_max = 90
    aa = outcat.loc[outcat['Cen1'] > cen1_min]
    aa = aa.loc[outcat['Cen1'] <= cen1_max]
    aa = aa.loc[outcat['Cen2'] > cen2_min]
    aa = aa.loc[outcat['Cen2'] <= cen2_max]
    return aa


def my_print(str_='', vosbe_=False):
    if vosbe_:
        print(str_)


if __name__ == '__main__':
    xm, ym, zm = 100, 80, 120
    r = 4
    delta_ii_xy = np.array([43, 22, 109])
    t0 = time.time()
    index, index_value = kc_coord_3d(delta_ii_xy, xm, ym, zm, r)
    t1 = time.time()
    print((t1-t0) * 10000000)
    delta_ii_xy = np.array([43, 22])
    aa1 = kc_coord_2d(delta_ii_xy, xm, ym, r)
    xx = get_xyz(np.zeros([50, 50]))

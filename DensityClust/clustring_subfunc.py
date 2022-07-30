import numpy as np
from scipy import ndimage
from astropy import wcs
from tabulate import tabulate
import astropy.io.fits as fits
import pandas as pd
from scipy.spatial import kdtree as kdt
import tqdm
import threading


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
        print('outcat_record columns is %d' % outcat_colums)


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
        print('outcat_record columns is %d' % outcat_column)
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
    outcat_record
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
        print('outcat_record columns is %d' % outcat_column)
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
    # outcat_record = pd.read_csv(txt_name, sep='\t')
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


def get_feature(rho, rho_Ind, kdt_xx, num, r):
    """
    1.3.0版本中计算参数的函数
    """
    INN = np.zeros_like(rho, np.int32)
    delta = np.zeros_like(rho, np.float32)
    grad = np.zeros_like(rho, np.float32) - 1
    i = 0
    for rho_i_idx in tqdm.tqdm(rho_Ind[1: num]):
        rho_i_idx_up = rho_Ind[: i + 1]
        rho_i_point = kdt_xx.data[rho_i_idx]
        ball_idx = kdt_xx.query_ball_point(rho_i_point, r=r, workers=4)
        ball_idx.remove(rho_i_idx)
        inter_sec = np.intersect1d(rho_i_idx_up, np.array(ball_idx))
        if inter_sec.shape[0] > 0:
            kdt_i_idx_up = kdt.KDTree(kdt_xx.data[inter_sec])
            gloab_idx = inter_sec
        else:
            kdt_i_idx_up = kdt.KDTree(kdt_xx.data[rho_i_idx_up])
            gloab_idx = rho_i_idx_up
        near_dis, near_idx = kdt_i_idx_up.query(rho_i_point, k=1, workers=4)
        INN[rho_i_idx] = gloab_idx[near_idx]
        delta[rho_i_idx] = near_dis
        grad[rho_i_idx] = (rho[gloab_idx[near_idx]] - rho[rho_i_idx]) / near_dis
        i += 1
    delta[rho_Ind[0]] = delta.max()
    return INN, delta, grad


def get_feature_thre(rho, rho_Ind, kdt_xx, r, INN, delta, grad, st, end_, ii, kk, thresh_num):
    """
    采用多线程计算
    """
    i = ii
    k = 1
    rho_Ind_thre = rho_Ind[st: end_]
    for rho_i_idx in rho_Ind_thre:
        rho_i_idx_up = rho_Ind[: i + 1]
        rho_i_point = kdt_xx.data[rho_i_idx]
        ball_idx = kdt_xx.query_ball_point(rho_i_point, r=r, workers=4)
        ball_idx.remove(rho_i_idx)
        inter_sec = np.intersect1d(rho_i_idx_up, np.array(ball_idx))
        if inter_sec.shape[0] > 0:
            kdt_i_idx_up = kdt.KDTree(kdt_xx.data[inter_sec])
            gloab_idx = inter_sec
        else:
            kdt_i_idx_up = kdt.KDTree(kdt_xx.data[rho_i_idx_up])
            gloab_idx = rho_i_idx_up
            k += 1

        near_dis, near_idx = kdt_i_idx_up.query(rho_i_point, k=1, workers=4)
        INN[rho_i_idx] = gloab_idx[near_idx]
        delta[rho_i_idx] = near_dis
        grad[rho_i_idx] = (rho[gloab_idx[near_idx]] - rho[rho_i_idx]) / near_dis
        i += 1
    print('threading number is %d/%d, glob %d' % (kk, thresh_num, k))


def get_feature_loc_thre(rho_up, rho_Ind_loc, idx_rho_loc_glob, kd_tree_loc, r, INN, delta, grad, st, end_, ii, kk):
    """
    采用多线程计算
    """
    INN_loc = np.zeros_like(rho_Ind_loc, np.int32)
    delta_loc = np.zeros_like(rho_Ind_loc, np.float32)
    grad_loc = np.zeros_like(rho_Ind_loc, np.float32) - 1
    i = ii
    k = 0
    rho_ind_loc_temp = rho_Ind_loc[st: end_]
    for rho_i_idx in rho_ind_loc_temp:
        rho_i_idx_up = rho_Ind_loc[: i + 1]
        rho_i_point = kd_tree_loc.data[rho_i_idx]
        ball_idx = kd_tree_loc.query_ball_point(rho_i_point, r=r, workers=1)
        ball_idx.remove(rho_i_idx)
        inter_sec = np.intersect1d(rho_i_idx_up, np.array(ball_idx))
        if inter_sec.shape[0] > 0:
            kdt_i_idx_up = kdt.KDTree(kd_tree_loc.data[inter_sec])
            gloab_idx = inter_sec
        else:
            kdt_i_idx_up = kdt.KDTree(kd_tree_loc.data[rho_i_idx_up])
            gloab_idx = rho_i_idx_up
            k += 1
        near_dis, near_idx = kdt_i_idx_up.query(rho_i_point, k=1, workers=1)   # 寻找最近的
        INN_loc[rho_i_idx] = idx_rho_loc_glob[gloab_idx[near_idx]]
        delta_loc[rho_i_idx] = near_dis
        grad_loc[rho_i_idx] = (rho_up[gloab_idx[near_idx]] - rho_up[rho_i_idx]) / near_dis
        i += 1

    # INN[idx_rho_loc_glob] = INN_loc
    INN[idx_rho_loc_glob[rho_ind_loc_temp]] = INN_loc[rho_ind_loc_temp]
    delta[idx_rho_loc_glob[rho_ind_loc_temp]] = delta_loc[rho_ind_loc_temp]
    grad[idx_rho_loc_glob[rho_ind_loc_temp]] = grad_loc[rho_ind_loc_temp]
    print('threading number is %d, glob %d' % (kk, k))


def get_feature_loc(rho_up, rho_Ind_loc, idx_rho_loc_glob, kd_tree_loc, r, ND_num, ND):
    """
    构建局部kdtree,进行聚类
    """
    INN_loc = np.zeros(ND_num, np.int32)
    delta_loc = np.zeros(ND_num, np.float32)
    grad_loc = np.zeros(ND_num, np.float32) - 1

    INN = np.zeros(ND, np.int32)
    delta = np.zeros(ND, np.float32)
    grad = np.zeros(ND, np.float32) - 1

    i = 0
    k = 0
    for rho_i_idx in tqdm.tqdm(rho_Ind_loc[1:]):
        rho_i_idx_up = rho_Ind_loc[: i + 1]
        rho_i_point = kd_tree_loc.data[rho_i_idx]
        ball_idx = kd_tree_loc.query_ball_point(rho_i_point, r=r, workers=1)
        ball_idx.remove(rho_i_idx)
        inter_sec = np.intersect1d(rho_i_idx_up, np.array(ball_idx))
        if inter_sec.shape[0] > 0:
            kdt_i_idx_up = kdt.KDTree(kd_tree_loc.data[inter_sec])
            gloab_idx = inter_sec
        else:
            kdt_i_idx_up = kdt.KDTree(kd_tree_loc.data[rho_i_idx_up])
            gloab_idx = rho_i_idx_up
            k += 1
        near_dis, near_idx = kdt_i_idx_up.query(rho_i_point, k=1, workers=1)   # 寻找最近的

        INN_loc[rho_i_idx] = idx_rho_loc_glob[gloab_idx[near_idx]]
        delta_loc[rho_i_idx] = near_dis
        grad_loc[rho_i_idx] = (rho_up[gloab_idx[near_idx]] - rho_up[rho_i_idx]) / near_dis
        i += 1
    delta_loc[rho_Ind_loc[0]] = delta_loc.max()

    INN[idx_rho_loc_glob] = INN_loc
    delta[idx_rho_loc_glob] = delta_loc
    grad[idx_rho_loc_glob] = grad_loc
    print('threading number is %d' % k)
    return INN, delta, grad


def get_feature_loc_threshing(rho_up, rho_Ind_loc, idx_rho_loc_glob, kd_tree_loc, r, ND_num, ND, thresh_num=5):
    """
    多线程调用函数
    """
    INN = np.zeros(ND, np.int32)
    delta = np.zeros(ND, np.float32)
    grad = np.zeros(ND, np.float32) - 1
    tsk = []
    deal_num = ND_num // thresh_num
    for thre_i in range(thresh_num):
        st = thre_i * deal_num
        end_ = (thre_i + 1) * deal_num
        if thre_i == 0:
            st = 1
        if thre_i == thresh_num - 1:
            end_ = ND_num
        csum_i = st - 1
        t1 = threading.Thread(target=get_feature_loc_thre,
                              args=(
                              rho_up, rho_Ind_loc, idx_rho_loc_glob, kd_tree_loc, r, INN, delta, grad, st, end_, csum_i,
                              thre_i))
        tsk.append(t1)
    for t in tsk:
        t.start()
    for t in tsk:
        t.join()
    delta[idx_rho_loc_glob[rho_Ind_loc[0]]] = delta.max()
    return INN, delta, grad


def get_feature_threshing(rho, rho_Ind, kdt_xx, ND_num, r, thresh_num=5):
    """
    构建全局kdtree,用多线程计算
    """
    INN = np.zeros_like(rho_Ind, np.int32)
    delta = np.zeros_like(rho_Ind, np.float32)
    grad = np.zeros_like(rho_Ind, np.float32) - 1

    tsk = []
    deal_num = ND_num // thresh_num
    for thre_i in range(thresh_num):
        st = thre_i * deal_num
        end_ = (thre_i + 1) * deal_num

        if thre_i == 0:
            st = 1
        if thre_i == thresh_num - 1:
            end_ = ND_num
        csum_i = st - 1
        t1 = threading.Thread(target=get_feature_thre,
                              args=(rho, rho_Ind, kdt_xx, r, INN, delta, grad, st, end_, csum_i, thre_i, thresh_num))
        tsk.append(t1)
    for t in tsk:
        t.start()
    for t in tsk:
        t.join()
    delta[rho_Ind[0]] = delta.max()
    return INN, delta, grad


def get_feature_new(rho, rho_sort_idx, kdt_xx_loc, num, r):
    INN = np.zeros_like(rho, np.int32)
    delta = np.zeros_like(rho, np.float32)
    grad = np.zeros_like(rho, np.float32) - 1
    i = 0
    for rho_i_idx in tqdm.tqdm(rho_sort_idx[1: num]):
        rho_i_idx_up = rho_sort_idx[: i+1]
        rho_i_point = kdt_xx_loc.data[rho_i_idx]
        ball_idx = kdt_xx_loc.query_ball_point(rho_i_point, r=r, workers=4)
        ball_idx.remove(rho_i_idx)
        inter_sec = np.intersect1d(rho_i_idx_up, np.array(ball_idx))
        if inter_sec.shape[0] > 0:
            kdt_i_idx_up = kdt.KDTree(kdt_xx_loc.data[inter_sec])
            gloab_idx = inter_sec
        else:
            kdt_i_idx_up = kdt.KDTree(kdt_xx_loc.data[rho_i_idx_up])
            gloab_idx = rho_i_idx_up
        near_dis, near_idx = kdt_i_idx_up.query(rho_i_point, k=1, workers=4)
        INN[rho_i_idx] = gloab_idx[near_idx]
        delta[rho_i_idx] = near_dis
        grad[rho_i_idx] = (rho[gloab_idx[near_idx]] - rho[rho_i_idx]) / near_dis
        i += 1
    delta[rho_sort_idx[0]] = delta.max()
    return INN, delta, grad


def assignation(rho, delta, delta_min, rho_min, v_min, rho_Ind, INN):
    """
    根据INN将点分配到不同的类别中去，并用统一的数字编号

    rho: 对数据的密度估计[ND * 1]，对data_cube直接拉直
    delta: 每个点的距离
    delta_min: 算法参数，距离最小值[4]
    rho_min: 算法参数，类中心密度最小值[3*rms]
    v_min: 算法参数，体积最小值[27]
    rho_Ind: 对rho降序排序的索引rho[rho_Ind[0]]为最大值
    INN: 两个密度点的联系 % index of nearest neighbor with higher density
        INN[i] = j 表示比第i个点大的点中最近的点为第j个点
    """
    clust_index = np.intersect1d(np.where(rho > rho_min), np.where(delta > delta_min))
    clusterInd = -1 * np.ones_like(rho, np.int32)
    clusterInd[clust_index] = np.arange(clust_index.shape[0]) + 1

    for i in range(rho.shape[0]):
        ordrho_i = rho_Ind[i]
        if clusterInd[ordrho_i] == -1:  # not centroid
            clusterInd[ordrho_i] = clusterInd[INN[ordrho_i]]

    clusterInd_ = np.zeros_like(clusterInd)
    ii = 0
    for i in range(1, clusterInd.max() + 1, 1):
        volume_idx = np.where(clusterInd == i)[0]
        if volume_idx.shape[0] > v_min:
            ii += 1
            clusterInd_[volume_idx] = ii
    return clusterInd_


def divide_boundary_by_grad(clusterInd, rho, grad, gradmin, v_min, data_shape):
    """
    根据梯度确定边界，
    clusterInd: 聚类编号[ND * 1],同一个类用一个数字标记
    rho: 对数据的密度估计[ND * 1]，对data_cube直接拉直
    grad: 每个点的梯度
    gradmin: 算法参数，梯度最小值[0.01]
    v_min: 算法参数，体积最小值[27]
    data_shape: 处理的数据块尺寸[m*n*k]
    """
    clump_id = 0
    clumpInd = np.zeros_like(clusterInd, np.int32)
    for cluster_i in tqdm.tqdm(range(1, clusterInd.max() + 1, 1)):
        index_cluster_i = np.where(clusterInd == cluster_i)[0]
        clump_rho = rho[index_cluster_i]
        clump_grad = grad[index_cluster_i]
        rho_max_min = clump_rho.max() - clump_rho.min()

        clump_grad_i = clump_grad / rho_max_min
        index_grad = np.where(clump_grad_i > gradmin)[0]
        rho_cc_mean = clump_rho[index_grad].mean()

        index_cc_rho = np.where(clump_rho > rho_cc_mean)[0]
        index_cluster = np.union1d(index_cc_rho, index_grad)
        if index_cluster.shape[0] > v_min:
            clump_id += 1
            idx_cluster = index_cluster_i[index_cluster]

            # clumpInd = np.zeros_like(clusterInd, np.int32)
            clumpInd[idx_cluster] = clump_id
            # label_data_ = clumpInd.reshape(data_shape)
            # label_data_ = ndimage.binary_fill_holes(label_data_).astype(np.int32)
            # label_data += label_data_
        label_data = clumpInd.reshape(data_shape)
    return label_data


def divide_boundary_by_grad_fill(clusterInd, rho, grad, gradmin, v_min, data_shape):
    """
    根据梯度确定边界，填充空洞
    clusterInd: 聚类编号[ND * 1],同一个类用一个数字标记
    rho: 对数据的密度估计[ND * 1]，对data_cube直接拉直
    grad: 每个点的梯度
    gradmin: 算法参数，梯度最小值[0.01]
    v_min: 算法参数，体积最小值[27]
    data_shape: 处理的数据块尺寸[m*n*k]
    """
    clump_id = 0
    label_data = np.zeros(data_shape, np.in32)
    for cluster_i in tqdm.tqdm(range(1, clusterInd.max() + 1, 1)):
        index_cluster_i = np.where(clusterInd == cluster_i)[0]
        clump_rho = rho[index_cluster_i]
        clump_grad = grad[index_cluster_i]
        rho_max_min = clump_rho.max() - clump_rho.min()

        clump_grad_i = clump_grad / rho_max_min
        index_grad = np.where(clump_grad_i > gradmin)[0]
        rho_cc_mean = clump_rho[index_grad].mean()

        index_cc_rho = np.where(clump_rho > rho_cc_mean)[0]
        index_cluster = np.union1d(index_cc_rho, index_grad)
        if index_cluster.shape[0] > v_min:
            clump_id += 1
            idx_cluster = index_cluster_i[index_cluster]
            clumpInd = np.zeros_like(clusterInd, np.int32)
            clumpInd[idx_cluster] = clump_id
            label_data_ = clumpInd.reshape(data_shape)
            label_data_ = ndimage.binary_fill_holes(label_data_).astype(np.int32)
            label_data += label_data_
    return label_data


if __name__ == '__main__':
    # data = np.array([1.1, 1.5, 2.31, 1.8, 1.4, 1.3, 0.8, 0.85, 0.9, 1.3, 1.5, 1.6, 1.8, 1.9, 2.3, 1.7, 1.3, 1.2, 0.8])
    # # data = np.array([1.1, 1.5, 2.31, 1.8, 1.4, 1.3, 0.8, 0.85])
    # # data = np.concatenate([data, data, data, data])
    # # data[21] = 2.31 - 0.001
    # # data[40] = 2.31 - 0.002
    # # data[59] = 2.31 - 0.003
    # x = np.arange(1, data.shape[0]+1, 1).reshape([data.shape[0], 1]) - 1
    # rho = data
    # rho_Ind = np.argsort(-rho)
    # kdt_xx = kdt.KDTree(x)
    # r = 6
    # noise = 0.9
    # idx_rho_loc_glob = np.where(rho > noise)[0]
    # rho_up = rho[idx_rho_loc_glob]
    # ND_num = rho_up.shape[0]
    # ND = rho.shape[0]
    # rho_Ind_loc = np.argsort(-rho_up)
    #
    # aa = x[idx_rho_loc_glob]
    #
    # kd_tree_loc = kdt.KDTree(aa)
    #
    # INN, delta, grad = get_feature(rho, rho_Ind, kdt_xx, ND_num, r)
    # #
    # INN1, delta1, grad1 = get_feature_loc(rho_up, rho_Ind_loc, idx_rho_loc_glob, kd_tree_loc, r, ND_num, ND)
    # #
    # # INN2_, delta2_, grad2_ = get_feature_threshing(rho, rho_Ind, kdt_xx, num, r)
    #
    # INN3, delta3, grad3 = get_feature_loc_threshing(rho_up, rho_Ind_loc, idx_rho_loc_glob, kd_tree_loc, r, ND_num, ND,
    #                                                 thresh_num=2)
    # #
    # # print('*'*10)
    # print(INN)
    # print(INN1)
    # print(INN3)
    # IIII = np.vstack([INN,INN3]).T
    # print(np.vstack([INN,INN]))
    # # print(INN2_)
    # print(np.abs(INN - INN3).sum())
    # print(np.abs(grad - grad3).sum())
    # print(np.abs(delta - delta3).sum())
    #
    # plt.figure()
    # plt.plot(INN, INN3, '.')
    # plt.show()
    pass

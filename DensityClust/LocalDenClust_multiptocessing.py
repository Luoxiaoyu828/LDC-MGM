import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy import wcs
import pandas as pd
import os
import numpy as np
from skimage import filters
import time
from skimage import measure
from scipy import ndimage
import matplotlib.pyplot as plt
from threading import Thread
from multiprocessing import Pool
from time import sleep, ctime
from DensityClust.clustring_subfunc import \
    kc_coord_3d, kc_coord_2d, get_xyz
"""
在计算距离和梯度的时候，采用了多线程
"""

class Data:
    def __init__(self, data_name):
        self.data_name = data_name
        self.data = None
        self.wcs = None
        self.size_x = 0
        self.size_y = 0
        self.size_z = 0
        self.ND = 0
        self.get_data_inf()

    def get_data_inf(self):
        data = fits.getdata(data_name)
        # self.wcs = self.get_wcs()
        size_x, size_y, size_z = data.shape
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.data = data
        self.ND = size_x * size_y * size_z

    def get_wcs(self):
        """
        得到wcs信息
        :return:
        data_wcs
        """
        data_header = fits.getheader(self.data_name)
        keys = data_header.keys()
        key = [k for k in keys if k.endswith('4')]
        [data_header.remove(k) for k in key]
        try:
            data_header.remove('VELREF')
        except:
            pass
        data_wcs = wcs.WCS(data_header)
        return data_wcs


class LocDenCluster:
    
    def __init__(self, para, data_name):
        """
        根据决策图得到聚类中心和聚类中心个数
        :param para:
            para.rho_min: Minimum density
            para.delta_min: Minimum delta
            para.v_min: Minimum volume
            para.noise: The noise level of the data, used for data truncation calculation
            para.sigma: Standard deviation of Gaussian filtering
        """
        self.out = None
        self.outcat = None
        self.mask = None
        self.gradmin = para["gradmin"]
        self.rhomin = para["rho_min"]
        self.deltamin = para["delta_min"]
        self.v_min = para["v_min"]
        self.rms = para["noise"]
        self.dc = para['dc']
        self.is_plot = para['is_plot']
        self.Data = Data(data_name)
        ND = self.Data.ND
        self.Gradient = np.zeros(ND, np.float)
        self.IndNearNeigh = np.zeros(ND, np.int)
        self.delta = np.zeros(ND, np.float)

    def summary(self):
        table_title = ['rho_min', 'delta_min', 'v_min', 'gradmin', 'noise', 'dc']
        para = np.array([[self.rhomin, self.deltamin, self.v_min, self.gradmin, self.rms, self.dc]])
        para_pd = pd.DataFrame(para, columns=table_title)
        # print(para_pd)
        return para_pd

    def change_pix2word(self):
        """
        将算法检测的结果(像素单位)转换到天空坐标系上去
        :return:
        outcat_wcs
        ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume']
        -->3d

         ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
         -->2d
        """
        outcat = self.outcat
        if outcat is None:
            return
        else:
            outcat_column = outcat.shape[1]
            data_wcs = self.Data.wcs
            if outcat_column == 10:
                # 2d result
                peak1, peak2 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], 1)
                cen1, cen2 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1)
                size1, size2 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30])

                clump_Peak = np.column_stack([peak1, peak2])
                clump_Cen = np.column_stack([cen1, cen2])
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
                cen1, cen2, cen3 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1)
                size1, size2, size3 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30, outcat['Size3'] * 0.166])
                clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

                clump_Peak = np.column_stack([peak1, peak2, peak3 / 1000])
                clump_Cen = np.column_stack([cen1, cen2, cen3 / 1000])
                clustSize = np.column_stack([size1, size2, size3])
                id_clumps = []  # MWISP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
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

    def densityCluster_3d(self):

        data = self.Data.data

        k1 = 1  # 第1次计算点的邻域大小
        k2 = np.ceil(self.deltamin).astype(np.int)  # 第2次计算点的邻域大小
        xx = get_xyz(data)  # xx: 3D data coordinates  坐标原点是 1
        dim = data.ndim
        size_x, size_y, size_z = data.shape
        maxed = size_x + size_y + size_z
        ND = size_x * size_y * size_z

        # Initialize the return result: mask and out
        mask = np.zeros_like(data, dtype=np.int)
        out = np.zeros_like(data, dtype=np.float)

        data_filter = filters.gaussian(data, self.dc)
        rho = data_filter.flatten()
        rho_Ind = np.argsort(-rho)
        rho_sorted = rho[rho_Ind]

        delta, IndNearNeigh, Gradient = np.zeros(ND, np.float), np.zeros(ND, np.int), np.zeros(ND, np.float)
        delta[rho_Ind[0]] = np.sqrt(size_x ** 2 + size_y ** 2 + size_z ** 2)

        # delta 记录距离，
        # IndNearNeigh 记录：两个密度点的联系 % index of nearest neighbor with higher density
        IndNearNeigh[rho_Ind[0]] = rho_Ind[0]
        t0_ = time.time()

        # calculating delta and Gradient
        for ii in range(1, ND):
            # 密度降序排序后，即密度第ii大的索引(在rho中)
            ordrho_ii = rho_Ind[ii]
            rho_ii = rho_sorted[ii]  # 第ii大的密度值
            if rho_ii >= self.rms:
                delta[ordrho_ii] = maxed
                point_ii_xy = xx[ordrho_ii, :]
                get_value = True  # 判断是否需要在大循环中继续执行，默认需要，一旦在小循环中赋值成功，就不在大循环中运行
                idex, bt = kc_coord_3d(point_ii_xy, size_z, size_y, size_x, k1)
                for ordrho_jj, item in zip(idex, bt):
                    rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                    dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                    gradient = (rho_jj - rho_ii) / dist_i_j
                    if dist_i_j <= delta[ordrho_ii] and gradient >= 0:
                        delta[ordrho_ii] = dist_i_j
                        Gradient[ordrho_ii] = gradient
                        IndNearNeigh[ordrho_ii] = ordrho_jj
                        get_value = False

                if get_value:
                    # 表明，在(2 * k1 + 1) * (2 * k1 + 1) * (2 * k1 + 1)的邻域中没有找到比该点高，距离最近的点，则在更大的邻域中搜索
                    idex, bt = kc_coord_3d(point_ii_xy, size_z, size_y, size_x, k2)
                    for ordrho_jj, item in zip(idex, bt):
                        rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                        dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                        gradient = (rho_jj - rho_ii) / dist_i_j
                        if dist_i_j <= delta[ordrho_ii] and gradient >= 0:
                            delta[ordrho_ii] = dist_i_j
                            Gradient[ordrho_ii] = gradient
                            IndNearNeigh[ordrho_ii] = ordrho_jj
                            get_value = False

                if get_value:
                    delta[ordrho_ii] = k2 + 0.0001
                    Gradient[ordrho_ii] = -1
                    IndNearNeigh[ordrho_ii] = ND
            else:
                IndNearNeigh[ordrho_ii] = ND

        delta_sorted = np.sort(-delta) * -1
        delta[rho_Ind[0]] = delta_sorted[1]
        t1_ = time.time()
        print('delata, rho and Gradient are calculated, using %.2f seconds' % (t1_ - t0_))
        # 根据密度和距离来确定类中心
        clusterInd = -1 * np.ones(ND + 1)
        clust_index = np.intersect1d(np.where(rho > self.rhomin), np.where(delta > self.deltamin))

        clust_num = len(clust_index)
        # icl是用来记录第i个类中心在xx中的索引值
        icl = np.zeros(clust_num, dtype=int)
        n_clump = 0
        for ii in range(clust_num):
            i = clust_index[ii]
            icl[n_clump] = i
            n_clump += 1
            clusterInd[i] = n_clump
        # assignation 将其他非类中心分配到离它最近的类中心中去
        # clusterInd = -1 表示该点不是类的中心点，属于其他点，等待被分配到某个类中去
        # 类的中心点的梯度Gradient被指定为 - 1
        if self.is_plot == 1:
            pass
        for i in range(ND):
            ordrho_i = rho_Ind[i]
            if clusterInd[ordrho_i] == -1:  # not centroid
                clusterInd[ordrho_i] = clusterInd[IndNearNeigh[ordrho_i]]
            else:
                Gradient[ordrho_i] = -1  # 将类中心点的梯度设置为-1

        clump_volume = np.zeros(n_clump)
        for i in range(n_clump):
            clump_volume[i] = clusterInd.tolist().count(i + 1)

        # centInd [类中心点在xx坐标下的索引值，类中心在centInd的索引值: 代表类别编号]
        centInd = []
        for i, item in enumerate(clump_volume):
            if item >= self.v_min:
                centInd.append([icl[i], i])
        centInd = np.array(centInd, np.int)
        mask_grad = np.where(Gradient > self.gradmin)[0]

        # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
        n_clump = centInd.shape[0]
        clump_sum, clump_volume, clump_peak = np.zeros([n_clump, 1]), np.zeros([n_clump, 1]), np.zeros([n_clump, 1])
        clump_Cen, clump_size = np.zeros([n_clump, dim]), np.zeros([n_clump, dim])
        clump_Peak = np.zeros([n_clump, dim], np.int)
        clump_ii = 0

        for i, item in enumerate(centInd):
            rho_cluster_i = np.zeros(ND)
            index_cluster_i = np.where(clusterInd == (item[1] + 1))[0]  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
            index_cc = np.intersect1d(mask_grad, index_cluster_i)
            rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
            rho_cc_mean = rho[index_cc].mean() * 0.2
            index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
            index_cluster_rho = np.union1d(index_cc, index_cc_rho)

            cl_1_index_ = xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
            # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
            clustNum = cl_1_index_.shape[0]

            cl_i = np.zeros(data.shape, np.int)
            for item_ in cl_1_index_:
                cl_i[item_[2], item_[1], item_[0]] = 1
            # 形态学处理
            # cl_i = morphology.closing(cl_i)  # 做开闭运算会对相邻两个云核的掩膜有影响
            L = ndimage.binary_fill_holes(cl_i).astype(int)
            L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
            STATS = measure.regionprops(L)

            Ar_sum = []
            for region in STATS:
                coords = region.coords  # 经过验证，坐标原点为0
                temp = 0
                for item_coord in coords:
                    temp += data[item_coord[0], item_coord[1], item_coord[2]]
                Ar_sum.append(temp)
            Ar = np.array(Ar_sum)
            ind = np.where(Ar == Ar.max())[0]
            L[L != ind[0] + 1] = 0
            cl_i = L / (ind[0] + 1)
            coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标
            if coords.shape[0] > self.v_min:
                coords = coords[:, [2, 1, 0]]
                clump_i_ = np.zeros(coords.shape[0])
                for j, item_coord in enumerate(coords):
                    clump_i_[j] = data[item_coord[2], item_coord[1], item_coord[0]]

                clustsum = clump_i_.sum() + 0.0001  # 加一个0.0001 防止分母为0
                clump_Cen[clump_ii, :] = np.matmul(clump_i_, coords) / clustsum
                clump_volume[clump_ii, 0] = clustNum
                clump_sum[clump_ii, 0] = clustsum

                x_i = coords - clump_Cen[clump_ii, :]
                clump_size[clump_ii, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                                           - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
                clump_i = data * cl_i
                out = out + clump_i
                mask = mask + cl_i * (clump_ii + 1)
                clump_peak[clump_ii, 0] = clump_i.max()
                clump_Peak[clump_ii, [2, 1, 0]] = np.argwhere(clump_i == clump_i.max())[0]
                clump_ii += 1
            else:
                pass

        clump_Peak = clump_Peak + 1
        clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
        id_clumps = np.array([item + 1 for item in range(n_clump)], np.int).T
        id_clumps = id_clumps.reshape([n_clump, 1])

        LDC_outcat = np.column_stack(
            (id_clumps, clump_Peak, clump_Cen, clump_size, clump_peak, clump_sum, clump_volume))
        LDC_outcat = LDC_outcat[:clump_ii, :]
        table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak',
                       'Sum', 'Volume']
        LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)

        self.outcat = LDC_outcat
        self.mask = mask
        self.out = out

    def get_delta(self, rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end):
        # print(ND_start, ND_end)
        print('---开始---', ND_start, '时间', ctime())

        for ii in range(ND_start, ND_end, 1):
            # 密度降序排序后，即密度第ii大的索引(在rho中)
            ordrho_ii = rho_Ind[ii]
            rho_ii = rho_sorted[ii]  # 第ii大的密度值
            delta_ordrho_ii, Gradient_ordrho_ii, IndNearNeigh_ordrho_ii = 0, 0, 0
            if rho_ii >= self.rms:
                delta_ordrho_ii = maxed
                point_ii_xy = xx[ordrho_ii, :]
                get_value = True  # 判断是否需要在大循环中继续执行，默认需要，一旦在小循环中赋值成功，就不在大循环中运行
                idex, bt = kc_coord_3d(point_ii_xy, size_z, size_y, size_x, k1)
                for ordrho_jj, item in zip(idex, bt):
                    rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                    dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                    gradient = (rho_jj - rho_ii) / dist_i_j
                    if dist_i_j <= delta_ordrho_ii and gradient >= 0:
                        delta_ordrho_ii = dist_i_j
                        Gradient_ordrho_ii = gradient
                        IndNearNeigh_ordrho_ii = ordrho_jj
                        get_value = False

                if get_value:
                    # 表明，在(2 * k1 + 1) * (2 * k1 + 1) * (2 * k1 + 1)的邻域中没有找到比该点高，距离最近的点，则在更大的邻域中搜索
                    idex, bt = kc_coord_3d(point_ii_xy, size_z, size_y, size_x, k2)
                    for ordrho_jj, item in zip(idex, bt):
                        rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                        dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                        gradient = (rho_jj - rho_ii) / dist_i_j
                        if dist_i_j <= delta_ordrho_ii and gradient >= 0:
                            delta_ordrho_ii = dist_i_j
                            Gradient_ordrho_ii = gradient
                            IndNearNeigh_ordrho_ii = ordrho_jj
                            get_value = False

                if get_value:
                    delta_ordrho_ii = k2 + 0.0001
                    Gradient_ordrho_ii = -1
                    IndNearNeigh_ordrho_ii = ND
            else:
                IndNearNeigh_ordrho_ii = ND
            # print(delta_ordrho_ii)
            self.delta[ordrho_ii] = delta_ordrho_ii
            self.Gradient[ordrho_ii] = Gradient_ordrho_ii
            self.IndNearNeigh[ordrho_ii] = IndNearNeigh_ordrho_ii
        print('***结束***', ND_start, '时间', ctime())
        print(self.delta.max())
        print(self.Gradient.max())
        print(self.IndNearNeigh.max())

    def densityCluster_3d_multi(self):

        data = self.Data.data

        k1 = 1  # 第1次计算点的邻域大小
        k2 = np.ceil(self.deltamin).astype(np.int)  # 第2次计算点的邻域大小
        xx = get_xyz(data)  # xx: 3D data coordinates  坐标原点是 1
        dim = data.ndim
        size_x, size_y, size_z = data.shape
        maxed = size_x + size_y + size_z
        ND = size_x * size_y * size_z

        # Initialize the return result: mask and out
        mask = np.zeros_like(data, dtype=np.int)
        out = np.zeros_like(data, dtype=np.float)

        data_filter = filters.gaussian(data, self.dc)
        rho = data_filter.flatten()
        rho_Ind = np.argsort(-rho)
        rho_sorted = rho[rho_Ind]

        self.delta[rho_Ind[0]] = np.sqrt(size_x ** 2 + size_y ** 2 + size_z ** 2)

        # delta 记录距离，
        # IndNearNeigh 记录：两个密度点的联系 % index of nearest neighbor with higher density
        self.IndNearNeigh[rho_Ind[0]] = rho_Ind[0]
        t0_ = time.time()

        # calculating delta and Gradient



        # p = Pool(count)
        # for i_count in range(count):
        #     ND_start = 1 + i_count*ittt
        #     ND_end = 1 + (i_count + 1) * ittt
        #     if i_count == count-1:
        #         ND_end = ND
        #     p.apply_async(self.get_delta, args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end))
            # detect_single(data_ij_name, para)
        # p.apply_async(self.get_delta,
        #               args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, 1, ND))
        # self.get_delta(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, 1, ND)
        # p.close()
        # p.join()

        count = 4
        ittt = int(ND / count)
        ts = []
        for i_count in range(count):
            ND_start = 1 + i_count*ittt
            ND_end = 1 + (i_count + 1) * ittt
            if i_count == count-1:
                ND_end = ND
            t = Thread(target=self.get_delta, args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end))
            ts.append(t)
        [i.start() for i in ts]
        [i.join() for i in ts]

        print(self.delta.max())
        print(self.Gradient.max())
        print(self.IndNearNeigh.max())

        #     p = Pool(count)
        #     for data_ij_name in data_ij_name_list:
        #         p.apply_async(detect_single, args=(data_ij_name, para))
        #         # detect_single(data_ij_name, para)
        #     p.close()
        #     p.join()

        # t.join()
        # for ii in range(1, ND):
        #     # 密度降序排序后，即密度第ii大的索引(在rho中)
        #     ordrho_ii = rho_Ind[ii]
        #     rho_ii = rho_sorted[ii]  # 第ii大的密度值
        # t = Thread(target=self.get_delta, args=(ordrho_ii, rho_ii, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND))
        # t.start()
        # t.join()
        # ts.append(t)
        # ts = []
        # count = 2
        # # for i_count in range(count):
        # ND_start, ND_end = 1, int(ND/2)
        # t = Thread(target=self.get_delta, args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end))
        # t.start()
        # t.join()

        # t1 = Thread(target=self.get_delta, args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, 1000000))
        # t2 = Thread(target=self.get_delta, args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, 1000000, 2000000))
        # t3 = Thread(target=self.get_delta,
        #             args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, 2000000, ND))
        #
        # # 启动线程运行
        # t1.start()
        # t2.start()
        # t3.start()
        #
        # # 等待所有线程执行完毕
        # t1.join()  # join() 等待线程终止，要不然一直挂起
        # t2.join()
        # t3.join()
        delta_sorted = np.sort(-self.delta) * -1
        self.delta[rho_Ind[0]] = delta_sorted[1]
        t1_ = time.time()
        print('delata, rho and Gradient are calculated, using %.2f seconds' % (t1_ - t0_))
        # 根据密度和距离来确定类中心
        clusterInd = -1 * np.ones(ND + 1)
        clust_index = np.intersect1d(np.where(rho > self.rhomin), np.where(self.delta > self.deltamin))

        clust_num = len(clust_index)
        # icl是用来记录第i个类中心在xx中的索引值
        icl = np.zeros(clust_num, dtype=int)
        n_clump = 0
        for ii in range(clust_num):
            i = clust_index[ii]
            icl[n_clump] = i
            n_clump += 1
            clusterInd[i] = n_clump
        # assignation 将其他非类中心分配到离它最近的类中心中去
        # clusterInd = -1 表示该点不是类的中心点，属于其他点，等待被分配到某个类中去
        # 类的中心点的梯度Gradient被指定为 - 1
        if self.is_plot == 1:
            pass
        for i in range(ND):
            ordrho_i = rho_Ind[i]
            if clusterInd[ordrho_i] == -1:  # not centroid
                clusterInd[ordrho_i] = clusterInd[self.IndNearNeigh[ordrho_i]]
            else:
                self.Gradient[ordrho_i] = -1  # 将类中心点的梯度设置为-1

        clump_volume = np.zeros(n_clump)
        for i in range(n_clump):
            clump_volume[i] = clusterInd.tolist().count(i + 1)

        # centInd [类中心点在xx坐标下的索引值，类中心在centInd的索引值: 代表类别编号]
        centInd = []
        for i, item in enumerate(clump_volume):
            if item >= self.v_min:
                centInd.append([icl[i], i])
        centInd = np.array(centInd, np.int)
        mask_grad = np.where(self.Gradient > self.gradmin)[0]

        # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
        n_clump = centInd.shape[0]
        clump_sum, clump_volume, clump_peak = np.zeros([n_clump, 1]), np.zeros([n_clump, 1]), np.zeros([n_clump, 1])
        clump_Cen, clump_size = np.zeros([n_clump, dim]), np.zeros([n_clump, dim])
        clump_Peak = np.zeros([n_clump, dim], np.int)
        clump_ii = 0

        for i, item in enumerate(centInd):
            rho_cluster_i = np.zeros(ND)
            index_cluster_i = np.where(clusterInd == (item[1] + 1))[0]  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
            index_cc = np.intersect1d(mask_grad, index_cluster_i)
            rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
            rho_cc_mean = rho[index_cc].mean() * 0.2
            index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
            index_cluster_rho = np.union1d(index_cc, index_cc_rho)

            cl_1_index_ = xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
            # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
            clustNum = cl_1_index_.shape[0]

            cl_i = np.zeros(data.shape, np.int)
            for item_ in cl_1_index_:
                cl_i[item_[2], item_[1], item_[0]] = 1
            # 形态学处理
            # cl_i = morphology.closing(cl_i)  # 做开闭运算会对相邻两个云核的掩膜有影响
            L = ndimage.binary_fill_holes(cl_i).astype(int)
            L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
            STATS = measure.regionprops(L)

            Ar_sum = []
            for region in STATS:
                coords = region.coords  # 经过验证，坐标原点为0
                temp = 0
                for item_coord in coords:
                    temp += data[item_coord[0], item_coord[1], item_coord[2]]
                Ar_sum.append(temp)
            Ar = np.array(Ar_sum)
            ind = np.where(Ar == Ar.max())[0]
            L[L != ind[0] + 1] = 0
            cl_i = L / (ind[0] + 1)
            coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标
            if coords.shape[0] > self.v_min:
                coords = coords[:, [2, 1, 0]]
                clump_i_ = np.zeros(coords.shape[0])
                for j, item_coord in enumerate(coords):
                    clump_i_[j] = data[item_coord[2], item_coord[1], item_coord[0]]

                clustsum = clump_i_.sum() + 0.0001  # 加一个0.0001 防止分母为0
                clump_Cen[clump_ii, :] = np.matmul(clump_i_, coords) / clustsum
                clump_volume[clump_ii, 0] = clustNum
                clump_sum[clump_ii, 0] = clustsum

                x_i = coords - clump_Cen[clump_ii, :]
                clump_size[clump_ii, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                                           - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
                clump_i = data * cl_i
                out = out + clump_i
                mask = mask + cl_i * (clump_ii + 1)
                clump_peak[clump_ii, 0] = clump_i.max()
                clump_Peak[clump_ii, [2, 1, 0]] = np.argwhere(clump_i == clump_i.max())[0]
                clump_ii += 1
            else:
                pass

        clump_Peak = clump_Peak + 1
        clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
        id_clumps = np.array([item + 1 for item in range(n_clump)], np.int).T
        id_clumps = id_clumps.reshape([n_clump, 1])

        LDC_outcat = np.column_stack(
            (id_clumps, clump_Peak, clump_Cen, clump_size, clump_peak, clump_sum, clump_volume))
        LDC_outcat = LDC_outcat[:clump_ii, :]
        table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak',
                       'Sum', 'Volume']
        LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)

        self.outcat = LDC_outcat
        self.mask = mask
        self.out = out

    def densityCluster_2d(self):
        """
        根据决策图得到聚类中心和聚类中心个数
        """
        data = self.Data.data
        k = 1  # 计算点的邻域大小
        k2 = np.ceil(self.deltamin).astype(np.int)  # 第2次计算点的邻域大小
        xx = get_xyz(data)  # xx: 2D data coordinates  坐标原点是 1
        dim = data.ndim
        mask = np.zeros_like(data, dtype=np.int)
        out = np.zeros_like(data, dtype=np.float)
        data_filter = filters.gaussian(data, self.dc)
        size_x, size_y = data.shape
        rho = data_filter.flatten()
        rho_Ind = np.argsort(-rho)
        rho_sorted = rho[rho_Ind]
        maxd = size_x + size_y
        ND = len(rho)

        # delta 记录距离， # IndNearNeigh 记录：两个密度点的联系 % index of nearest neighbor with higher density
        delta, IndNearNeigh, Gradient = np.zeros(ND, np.float), np.zeros(ND, np.int), np.zeros(ND, np.float)

        delta[rho_Ind[0]] = np.sqrt(size_x ** 2 + size_y ** 2)
        IndNearNeigh[rho_Ind[0]] = rho_Ind[0]

        t0 = time.time()
        # 计算 delta, Gradient
        for ii in range(1, ND):
            # 密度降序排序后，即密度第ii大的索引（在rho中）
            ordrho_ii = rho_Ind[ii]
            rho_ii = rho_sorted[ii]  # 第ii大的密度值
            if rho_ii >= self.rms:
                delta[ordrho_ii] = maxd
                point_ii_xy = xx[ordrho_ii, :]
                get_value = True  # 判断是否需要在大循环中继续执行，默认需要，一旦在小循环中赋值成功，就不在大循环中运行

                bt = kc_coord_2d(point_ii_xy, size_y, size_x, k)
                for item in bt:
                    rho_jj = data_filter[item[1] - 1, item[0] - 1]
                    dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                    gradient = (rho_jj - rho_ii) / dist_i_j

                    if dist_i_j <= delta[ordrho_ii] and gradient >= 0:
                        delta[ordrho_ii] = dist_i_j
                        Gradient[ordrho_ii] = gradient
                        IndNearNeigh[ordrho_ii] = (item[1] - 1) * size_y + item[0] - 1
                        get_value = False

                if get_value:  # 表明在小领域中没有找到比该点高，距离最近的点，则进行更大领域的搜索
                    bt = kc_coord_2d(point_ii_xy, size_y, size_x, k2)
                    for item in bt:
                        rho_jj = data_filter[item[1] - 1, item[0] - 1]
                        dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                        gradient = (rho_jj - rho_ii) / dist_i_j

                        if dist_i_j <= delta[ordrho_ii] and gradient >= 0:
                            delta[ordrho_ii] = dist_i_j
                            Gradient[ordrho_ii] = gradient
                            IndNearNeigh[ordrho_ii] = (item[1] - 1) * size_y + item[0] - 1
                            get_value = False
                if get_value:
                    delta[ordrho_ii] = k2 + 0.0001
                    Gradient[ordrho_ii] = -1
                    IndNearNeigh[ordrho_ii] = ND
            else:
                IndNearNeigh[ordrho_ii] = ND

        delta_sorted = np.sort(-delta) * (-1)
        delta[rho_Ind[0]] = delta_sorted[1]
        t1 = time.time()
        print('delata, rho and Gradient are calculated, using %.2f seconds' % (t1 - t0))

        # 根据密度和距离来确定类中心
        NCLUST = 0
        clustInd = -1 * np.ones(ND + 1)
        clust_index = np.intersect1d(np.where(rho > self.rhomin), np.where(delta > self.deltamin))

        clust_num = clust_index.shape[0]
        print(clust_num)

        # icl是用来记录第i个类中心在xx中的索引值
        icl = np.zeros(clust_num, dtype=int)
        for ii in range(0, clust_num):
            i = clust_index[ii]
            icl[NCLUST] = i
            NCLUST += 1
            clustInd[i] = NCLUST

        # assignation
        # 将其他非类中心分配到离它最近的类中心中去
        # clustInd = -1
        # 表示该点不是类的中心点，属于其他点，等待被分配到某个类中去
        # 类的中心点的梯度Gradient被指定为 - 1
        if self.is_plot == 1:
            plt.scatter(rho, delta, marker='.')
            plt.show()

        for i in range(ND):
            ordrho_i = rho_Ind[i]
            if clustInd[ordrho_i] == -1:  # not centroid
                clustInd[ordrho_i] = clustInd[IndNearNeigh[ordrho_i]]
            else:
                Gradient[ordrho_i] = -1  # 将类中心点的梯度设置为-1

        clustVolume = np.zeros(NCLUST)
        for i in range(NCLUST):
            clustVolume[i] = clustInd.tolist().count(i + 1)

        # % centInd [类中心点在xx坐标下的索引值，
        # 类中心在centInd的索引值: 代表类别编号]
        centInd = []
        for i, item in enumerate(clustVolume):
            if item >= self.v_min:
                centInd.append([icl[i], i])
        centInd = np.array(centInd, np.int)

        mask_grad = np.where(Gradient > self.gradmin)[0]

        # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
        NCLUST = centInd.shape[0]
        clustSum, clustVolume, clustPeak = np.zeros([NCLUST, 1]), np.zeros([NCLUST, 1]), np.zeros([NCLUST, 1])
        clump_Cen, clustSize = np.zeros([NCLUST, dim]), np.zeros([NCLUST, dim])
        clump_Peak = np.zeros([NCLUST, dim], np.int)
        clump_ii = 0
        for i, item in enumerate(centInd):
            # centInd[i, 1] --> item[1] 表示第i个类中心的编号
            rho_clust_i = np.zeros(ND)
            index_clust_i = np.where(clustInd == (item[1] + 1))[0]
            index_cc = np.intersect1d(mask_grad, index_clust_i)
            rho_clust_i[index_clust_i] = rho[index_clust_i]
            if len(index_cc) > 0:
                rho_cc_mean = rho[index_cc].mean() * 0.2
            else:
                rho_cc_mean = self.rms
            index_cc_rho = np.where(rho_clust_i > rho_cc_mean)[0]
            index_clust_rho = np.union1d(index_cc, index_cc_rho)

            cl_1_index_ = xx[index_clust_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
            # clustInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
            cl_i = np.zeros(data.shape, np.int)
            for j, item_ in enumerate(cl_1_index_):
                cl_i[item_[1], item_[0]] = 1
            # 形态学处理
            # cl_i = morphology.closing(cl_i)  # 做开闭运算会对相邻两个云核的掩膜有影响
            L = ndimage.binary_fill_holes(cl_i).astype(int)
            L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.

            STATS = measure.regionprops(L)

            Ar_sum = []
            for region in STATS:
                coords = region.coords  # 经过验证，坐标原点为0
                coords = coords[:, [1, 0]]
                temp = 0
                for item_coord in coords:
                    temp += data[item_coord[1], item_coord[0]]
                Ar_sum.append(temp)
            Ar = np.array(Ar_sum)
            ind = np.where(Ar == Ar.max())[0]
            L[L != ind[0] + 1] = 0
            cl_i = L / (ind[0] + 1)
            coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标
            clustNum = coords.shape[0]
            if clustNum > self.v_min:
                coords = coords[:, [1, 0]]
                clump_i_ = np.zeros(coords.shape[0])
                for j, item_coord in enumerate(coords):
                    clump_i_[j] = data[item_coord[1], item_coord[0]]

                clustsum = sum(clump_i_) + 0.0001  # 加一个0.0001 防止分母为0
                clump_Cen[clump_ii, :] = np.matmul(clump_i_, coords) / clustsum
                clustVolume[clump_ii, 0] = clustNum
                clustSum[clump_ii, 0] = clustsum

                x_i = coords - clump_Cen[clump_ii, :]
                clustSize[clump_ii, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                                          - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
                clump_i = data * cl_i
                out = out + clump_i
                mask = mask + cl_i * (clump_ii + 1)
                clustPeak[clump_ii, 0] = clump_i.max()
                clump_Peak[clump_ii, [1, 0]] = np.argwhere(clump_i == clump_i.max())[0]
                clump_ii += 1
            else:
                pass
        clump_Peak = clump_Peak + 1
        clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
        id_clumps = np.array([item + 1 for item in range(NCLUST)], np.int).T
        id_clumps = id_clumps.reshape([NCLUST, 1])

        LDC_outcat = np.column_stack((id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))

        LDC_outcat = LDC_outcat[:clump_ii, :]
        table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
        LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)

        self.outcat = LDC_outcat
        self.mask = mask
        self.out = out

    def save_outcat_wcs(self, outcat_wcs_name):
        """
        # 保存LDC检测的直接结果，即单位为像素
        :return:
        """
        outcat_wcs = self.change_pix2word()
        outcat_colums = outcat_wcs.shape[1]
        if outcat_colums == 10:
            # 2d result
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
            dataframe = pd.DataFrame(outcat_wcs, columns=table_title)
            dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 13:
            # 3d result
            dataframe = outcat_wcs.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                         'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 11:
            # fitting 2d data result
            fit_outcat = outcat_wcs
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                           'Sum', 'Volume']
            dataframe = pd.DataFrame(fit_outcat, columns=table_title)
            dataframe = dataframe.round(
                {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
                 'Sum': 3, 'Volume': 3})
        else:
            print('outcat columns is %d' % outcat_colums)
            return

        dataframe.to_csv(outcat_wcs_name, sep='\t', index=False)

    def save_outcat(self, outcat_name):
        """
        # 保存LDC检测的直接结果，即单位为像素
        :param outcat_name: 核表的路径
        :return:
        """
        outcat = self.outcat
        outcat_colums = outcat.shape[1]
        if outcat_colums == 10:
            # 2d result
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
            dataframe = pd.DataFrame(outcat, columns=table_title)
            dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 13:
            # 3d result
            dataframe = outcat.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                         'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 11:
            # fitting 2d data result
            fit_outcat = outcat
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                           'Sum', 'Volume']
            dataframe = pd.DataFrame(fit_outcat, columns=table_title)
            dataframe = dataframe.round(
                {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
                 'Sum': 3, 'Volume': 3})
        else:
            print('outcat columns is %d' % outcat_colums)
            return

        dataframe.to_csv(outcat_name, sep='\t', index=False)

    def make_plot_wcs_1(self):
        """
        在积分图上绘制检测结果
        """
        outcat_wcs = self.change_pix2word()
        if outcat_wcs is None:
            return
        else:
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = 'True'
            plt.rcParams['ytick.right'] = 'True'
            plt.rcParams['xtick.color'] = 'red'
            plt.rcParams['ytick.color'] = 'red'
            # data_name = r'R2_data\data_9\0180-005\0180-005_L.fits'
            fits_path = self.Data.data_name.replace('.fits', '')
            title = fits_path.split('\\')[-1]
            fig_name = os.path.join(fits_path, title + '.png')


            wcs = self.Data.wcs
            data_cube = fits.getdata(data_name)
            outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Cen1'].values, b=outcat_wcs['Cen2'].values, unit="deg")

            fig = plt.figure(figsize=(10, 8.5), dpi=100)

            axes0 = fig.add_axes([0.15, 0.1, 0.7, 0.82], projection=wcs.celestial)
            axes0.set_xticks([])
            axes0.set_yticks([])
            im0 = axes0.imshow(data_cube.sum(axis=0))
            axes0.plot_coord(outcat_wcs_c, 'r*', markersize=2.5)
            axes0.set_xlabel("Galactic Longutide", fontsize=12)
            axes0.set_ylabel("Galactic Latitude", fontsize=12)
            axes0.set_title(title, fontsize=12)

            pos = axes0.get_position()
            pad = 0.01
            width = 0.02
            axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

            cbar = fig.colorbar(im0, cax=axes1)
            cbar.set_label('K m s${}^{-1}$')
            plt.show()


if __name__ == '__main__':
    data_name = r'F:\LDC_python\detection\R2_data\data_9\data_9\0175+000\0175+000_L.fits'
    para = {"gradmin": 0.01, "rho_min": 1.9, "delta_min": 4, "v_min": 27, "noise": 0.46, "dc": 0.6, "is_plot": 0}
    ldc = LocDenCluster(para=para, data_name=data_name)
    ldc.densityCluster_3d_multi()
    ldc.summary()
    # ldc.make_plot_wcs_1()
    outcat = ldc.outcat


    def func(name, sec):
        print('---开始---', name, '时间', ctime())
        sleep(sec)
        print('***结束***', name, '时间', ctime())
        return 2*sec


    # 创建 Thread 实例
    t1 = Thread(target=func, args=('第一个线程', 1))
    t2 = Thread(target=func, args=('第二个线程', 2))

    # 启动线程运行
    t1.start()
    t2.start()

    # 等待所有线程执行完毕
    t1.join()  # join() 等待线程终止，要不然一直挂起
    t2.join()


    def run_threading(target, args, count):
        """
        :param target: 目标函数
        :param args: 函数参数
        :param count: 线程数量
        """
        ts = []
        for i in range(count):
            t = Thread(target=target, args=args)
            ts.append(t)
        [i.start() for i in ts]
        [i.join() for i in ts]



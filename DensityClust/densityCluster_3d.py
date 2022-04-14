from skimage import filters
import numpy as np
from skimage import measure, morphology
from scipy import ndimage
import time
from time import sleep, ctime
from threading import Thread
import matplotlib.pyplot as plt

from DensityClust.clustring_subfunc import \
    kc_coord_3d, kc_coord_2d, get_xyz


def densityCluster_3d(data, para):
    """
    根据决策图得到聚类中心和聚类中心个数
    :param data: 3D data
    :param para:
        para.rho_min: Minimum density
        para.delta_min: Minimum delta
        para.v_min: Minimum volume
        para.noise: The noise level of the data, used for data truncation calculation
        para.sigma: Standard deviation of Gaussian filtering
    :return:
        NCLUST: number of clusters
        centInd:  centroid index vector
    """
    # 参数初始化
    gradmin = para["gradmin"]
    rhomin = para["rho_min"]
    deltamin = para["delta_min"]
    v_min = para["v_min"]
    rms = para["noise"]
    dc = para['dc']
    is_plot = para['is_plot']

    k1 = 1  # 第1次计算点的邻域大小
    k2 = np.ceil(deltamin).astype(np.int)   # 第2次计算点的邻域大小
    xx = get_xyz(data)  # xx: 3D data coordinates  坐标原点是 1
    dim = data.ndim
    size_x, size_y, size_z = data.shape
    maxed = size_x + size_y + size_z
    ND = size_x * size_y * size_z

    # Initialize the return result: mask and out
    mask = np.zeros_like(data, dtype=np.int)
    out = np.zeros_like(data, dtype=np.float)
    if dc > 0:
        data_filter = filters.gaussian(data, dc)
    elif dc == 0:
        data_filter = data
    else:
        data_filter = data

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
        rho_ii = rho_sorted[ii]   # 第ii大的密度值
        if rho_ii >= rms:
            delta_ordrho_ii = maxed
            Gradient_ordrho_ii = 0
            IndNearNeigh_ordrho_ii = 0
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

            delta[ordrho_ii] = delta_ordrho_ii
            Gradient[ordrho_ii] = Gradient_ordrho_ii
            IndNearNeigh[ordrho_ii] = IndNearNeigh_ordrho_ii
        else:
            IndNearNeigh[ordrho_ii] = ND

    delta_sorted = np.sort(-1 * delta) * -1
    delta[rho_Ind[0]] = delta_sorted[1]
    t1_ = time.time()
    print('delata, rho and Gradient are calculated, using %.2f seconds' % (t1_ - t0_))
    t0_ = time.time()
    # 根据密度和距离来确定类中心
    clusterInd = -1 * np.ones(ND + 1)
    clust_index = np.intersect1d(np.where(rho > rhomin), np.where(delta > deltamin))

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
    if is_plot == 1:
        pass
    for i in range(ND):
        ordrho_i = rho_Ind[i]
        if clusterInd[ordrho_i] == -1:    # not centroid
            clusterInd[ordrho_i] = clusterInd[IndNearNeigh[ordrho_i]]
        else:
            Gradient[ordrho_i] = -1  # 将类中心点的梯度设置为-1

    clump_volume = np.zeros(n_clump)
    for i in range(n_clump):
        clump_volume[i] = np.where(clusterInd == (i + 1))[0].shape[0]
    # centInd [类中心点在xx坐标下的索引值，类中心在centInd的索引值: 代表类别编号]
    centInd = []
    for i, item in enumerate(clump_volume):
        if item >= v_min:
            centInd.append([icl[i], i])
    centInd = np.array(centInd, np.int)

    # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
    n_clump = centInd.shape[0]
    clump_sum, clump_volume, clump_peak = np.zeros([n_clump, 1]), np.zeros([n_clump, 1]), np.zeros([n_clump, 1])
    clump_Cen, clump_size = np.zeros([n_clump, dim]), np.zeros([n_clump, dim])
    clump_Peak = np.zeros([n_clump, dim], np.int)
    clump_ii = 0

    for i, item_cent in enumerate(centInd):
        rho_cluster_i = np.zeros(ND)
        index_cluster_i = np.where(clusterInd == (item_cent[1] + 1))[0]   # centInd[i, 1] --> item[1] 表示第i个类中心的编号
        clump_rho = rho[index_cluster_i]
        rho_max_min = clump_rho.max() - clump_rho.min()
        Gradient_ = Gradient.copy()
        grad_clump_i = Gradient_ / rho_max_min
        mask_grad = np.where(grad_clump_i > gradmin)[0]
        index_cc = np.intersect1d(mask_grad, index_cluster_i)
        rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
        rho_cc_mean = rho[index_cc].mean()
        index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
        index_cluster_rho = np.union1d(index_cc, index_cc_rho)

        cl_1_index_ = xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
        # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
        clustNum = cl_1_index_.shape[0]

        cl_i = np.zeros(data.shape, np.int)
        for j, item in enumerate(cl_1_index_):
            cl_i[item[2], item[1], item[0]] = 1
        # 形态学处理
        L = ndimage.binary_fill_holes(cl_i).astype(int)
        L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
        STATS = measure.regionprops(L)

        Ar_sum = []
        for region in STATS:
            coords = region.coords  # 经过验证，坐标原点为0
            temp = 0
            for j, item in enumerate(coords):
                temp += data[item[0], item[1], item[2]]
            Ar_sum.append(temp)
        Ar = np.array(Ar_sum)
        ind = np.where(Ar == Ar.max())[0]
        L[L != ind[0] + 1] = 0
        cl_i = L / (ind[0] + 1)
        coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标
        if coords.shape[0] > v_min:
            coords = coords[:, [2, 1, 0]]
            clump_i_ = np.zeros(coords.shape[0])
            for j, item in enumerate(coords):
                clump_i_[j] = data[item[2], item[1], item[0]]

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

    LDC_outcat = np.column_stack((id_clumps, clump_Peak, clump_Cen, clump_size, clump_peak, clump_sum, clump_volume))
    LDC_outcat = LDC_outcat[:clump_ii, :]
    t1_ = time.time()
    print('Outcats are calculated, using %.2f seconds' % (t1_ - t0_))
    return LDC_outcat, mask, out


def densityCluster_2d(data, para):
    """
    根据决策图得到聚类中心和聚类中心个数
    :param data: 2D data
    :param para:
        para.rho_min: Minimum density
        para.delta_min: Minimum delta
        para.v_min: Minimum volume
        para.noise: The noise level of the data, used for data truncation calculation
        para.dc: Standard deviation of Gaussian filtering

    :return:
        NCLUST: number of clusters
        centInd:  centroid index vector
    """
    # 参数初始化
    gradmin = para["gradmin"]
    rhomin = para["rho_min"]
    deltamin = para["delta_min"]
    v_min = para["v_min"]
    rms = para["noise"]
    sigma = para['dc']
    is_plot = para['is_plot']
    k = 1   # 计算点的邻域大小
    k2 = np.ceil(deltamin).astype(np.int)  # 第2次计算点的邻域大小
    xx = get_xyz(data)  # xx: 2D data coordinates  坐标原点是 1
    dim = data.ndim
    mask = np.zeros_like(data, dtype=np.int)
    out = np.zeros_like(data, dtype=np.float)
    data_filter = filters.gaussian(data, sigma)
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
        rho_ii = rho_sorted[ii]   # 第ii大的密度值
        if rho_ii >= rms:
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

            if get_value:   # 表明在小领域中没有找到比该点高，距离最近的点，则进行更大领域的搜索
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
    print('delata, rho and Gradient are calculated, using %.2f seconds' % (t1-t0))

    # 根据密度和距离来确定类中心
    NCLUST = 0
    clustInd = -1 * np.ones(ND + 1)
    clust_index = np.intersect1d(np.where(rho > rhomin), np.where(delta > deltamin))

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
    if is_plot == 1:
        plt.scatter(rho, delta, marker='.')
        plt.show()

    for i in range(ND):
        ordrho_i = rho_Ind[i]
        if clustInd[ordrho_i] == -1:    # not centroid
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
        if item >= v_min:
            centInd.append([icl[i], i])
    centInd = np.array(centInd, np.int)

    mask_grad = np.where(Gradient > gradmin)[0]

    # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
    NCLUST = centInd.shape[0]
    clustSum, clustVolume, clustPeak = np.zeros([NCLUST, 1]), np.zeros([NCLUST, 1]), np.zeros([NCLUST, 1])
    clump_Cen, clustSize = np.zeros([NCLUST, dim]), np.zeros([NCLUST, dim])
    clump_Peak = np.zeros([NCLUST, dim], np.int)
    clump_ii = 0
    for i, item in enumerate(centInd):   # centInd[i, 1] --> item[1] 表示第i个类中心的编号
        rho_clust_i = np.zeros(ND)
        index_clust_i = np.where(clustInd == (item[1] + 1))[0]
        index_cc = np.intersect1d(mask_grad, index_clust_i)
        rho_clust_i[index_clust_i] = rho[index_clust_i]
        if len(index_cc) > 0:
            rho_cc_mean = rho[index_cc].mean() * 0.2
        else:
            rho_cc_mean = rms
        index_cc_rho = np.where(rho_clust_i > rho_cc_mean)[0]
        index_clust_rho = np.union1d(index_cc, index_cc_rho)

        cl_1_index_ = xx[index_clust_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
        # clustInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
        cl_i = np.zeros(data.shape, np.int)
        for j, item in enumerate(cl_1_index_):
            cl_i[item[1], item[0]] = 1
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
            for j, item in enumerate(coords):
                temp += data[item[1], item[0]]
            Ar_sum.append(temp)
        Ar = np.array(Ar_sum)
        ind = np.where(Ar == Ar.max())[0]
        L[L != ind[0] + 1] = 0
        cl_i = L / (ind[0] + 1)
        coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标
        clustNum = coords.shape[0]
        if clustNum > v_min:
            coords = coords[:, [1, 0]]
            clump_i_ = np.zeros(coords.shape[0])
            for j, item in enumerate(coords):
                clump_i_[j] = data[item[1], item[0]]

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
    return LDC_outcat, mask, out


def get_delta(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end, rms):

    Gradient = np.zeros([ND_end - ND_start], np.float)
    IndNearNeigh = np.zeros([ND_end - ND_start], np.int) + ND
    delta = np.zeros([ND_end - ND_start], np.float)
    print('---start---%d--time--%s' % (ND_start, ctime()))
    item_i = 0
    for ii in range(ND_start, ND_end, 1):
        # 密度降序排序后，即密度第ii大的索引(在rho中)
        ordrho_ii = rho_Ind[ii]
        rho_ii = rho_sorted[ii]  # 第ii大的密度值
        delta_ordrho_ii, Gradient_ordrho_ii, IndNearNeigh_ordrho_ii = 0, 0, ND
        if rho_ii >= rms:
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
            # return [delta, Gradient, IndNearNeigh, ND_start, ND_end]
        delta[item_i] = delta_ordrho_ii
        Gradient[item_i] = Gradient_ordrho_ii
        IndNearNeigh[item_i] = IndNearNeigh_ordrho_ii
        item_i += 1
    print('---end---%d--time--%s' % (ND_start, ctime()))

    return [delta, Gradient, IndNearNeigh, ND_start, ND_end]


def densityCluster_3d_multi(data, para):
    # -*- coding: utf-8 -*-
    from concurrent.futures import ThreadPoolExecutor
    import time

    """
    根据决策图得到聚类中心和聚类中心个数
    :param data: 3D data
    :param para:
        para.rho_min: Minimum density
        para.delta_min: Minimum delta
        para.v_min: Minimum volume
        para.noise: The noise level of the data, used for data truncation calculation
        para.sigma: Standard deviation of Gaussian filtering
    :return:
        NCLUST: number of clusters
        centInd:  centroid index vector
    """
    # 参数初始化
    gradmin = para["gradmin"]
    rhomin = para["rho_min"]
    deltamin = para["delta_min"]
    v_min = para["v_min"]
    rms = para["noise"]
    dc = para['dc']
    is_plot = para['is_plot']

    k1 = 1  # 第1次计算点的邻域大小
    k2 = np.ceil(deltamin).astype(np.int)   # 第2次计算点的邻域大小
    xx = get_xyz(data)  # xx: 3D data coordinates  坐标原点是 1
    dim = data.ndim
    size_x, size_y, size_z = data.shape
    maxed = size_x + size_y + size_z
    ND = size_x * size_y * size_z

    # Initialize the return result: mask and out
    mask = np.zeros_like(data, dtype=np.int)
    out = np.zeros_like(data, dtype=np.float)

    data_filter = filters.gaussian(data, dc)
    rho = data_filter.flatten()
    rho_Ind = np.argsort(-rho)
    rho_sorted = rho[rho_Ind]

    Gradient = np.zeros(ND, np.float)
    IndNearNeigh = np.zeros(ND, np.int) + ND
    delta = np.zeros(ND, np.float)

    delta[rho_Ind[0]] = np.sqrt(size_x ** 2 + size_y ** 2 + size_z ** 2)

    # delta 记录距离，
    # IndNearNeigh 记录：两个密度点的联系 % index of nearest neighbor with higher density
    IndNearNeigh[rho_Ind[0]] = rho_Ind[0]
    t0_ = time.time()
    rho_sorted_ = rho_sorted.copy()
    rho_sorted_[rho_sorted_ < rms] = rho_sorted_.max()
    indx = np.where(rho_sorted_ == rho_sorted_.min())[0]
    count = 6
    pool = ThreadPoolExecutor(max_workers=count)
    item_all = indx[0]
    ittt = int(item_all / (count))
    ts = []
    for i_count in range(count):
        ND_start = 1 + i_count * ittt
        ND_end = 1 + (i_count + 1) * ittt

        if i_count == count - 1:
            ND_end = item_all

        future1 = pool.submit(get_delta, rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end, rms)
        ts.append(future1)
    # 创建包含2个线程的线程池

    [item.done() for item in ts]
    #
    for item_feature in ts:
        [delta_, Gradient_, IndNearNeigh_, ND_start, ND_end] = item_feature.result()
        ordrho_ii = [rho_Ind[ii] for ii in range(ND_start, ND_end, 1)]
        ii = [ii - ND_start for ii in range(ND_start, ND_end, 1)]
        delta[ordrho_ii] = delta_[ii]
        Gradient[ordrho_ii] = Gradient_[ii]
        IndNearNeigh[ordrho_ii] = IndNearNeigh_[ii]

    pool.shutdown()
    # [delta_, Gradient_, IndNearNeigh_] = feature1.result()
    # for ii in range(1, 1600000, 1):
    #     ordrho_ii = rho_Ind[ii]
    #
    #     delta[ordrho_ii] = delta_[ii - 1]
    #     Gradient[ordrho_ii] = Gradient_[ii - 1]
    #     IndNearNeigh[ordrho_ii] = IndNearNeigh_[ii - 1]
    #
    # 向线程池提交一个任务, 20和10会作为action_a/b()方法的参数
    # future1 = pool.submit(get_delta, rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, 1, 16000, noise)
    # future2 = pool.submit(get_delta, rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, 16000, ND, noise)

    # 判断future1线程是否结束---返回False表示该线程未结束，True表示该线程已经结束
    # print("future1线程的状态：" + str(future1.done()))  # 此时future1线程已结束
    # # 判断future2线程是否结束
    # print("future2线程的状态：" + str(future2.done()))  # 此时future2线程未结束，因为休眠了3秒

    # 查看future1代表的任务返回的结果，如果线程未运行完毕，会暂时阻塞，等待线程运行完毕后再执行、输出；
    # print(future1.result())  # 此处会直接输出
    # 查看future2代表的任务返回的结果
    # print(future2.result())  # 此处会等待3秒，因为方法中休眠了3秒
    # [delta_, Gradient_, IndNearNeigh_, ND_start, ND_end] = future1.result()
    # ordrho_ii = [rho_Ind[ii] for ii in range(ND_start, ND_end, 1)]
    # ii = [ii - ND_start for ii in range(ND_start, ND_end, 1)]
    # delta[ordrho_ii] = delta_[ii]
    # Gradient[ordrho_ii] = Gradient_[ii]
    # IndNearNeigh[ordrho_ii] = IndNearNeigh_[ii]
    #
    # [delta_, Gradient_, IndNearNeigh_, ND_start, ND_end] = future2.result()
    #
    # ordrho_ii = [rho_Ind[ii] for ii in range(ND_start, ND_end, 1)]
    # ii = [ii - ND_start for ii in range(ND_start, ND_end, 1)]
    # delta[ordrho_ii] = delta_[ii]
    # Gradient[ordrho_ii] = Gradient_[ii]
    # IndNearNeigh[ordrho_ii] = IndNearNeigh_[ii]
    # # 关闭线程池
    # pool.shutdown()
    # calculating delta and Gradient
    # count = 4
    # ittt = int(ND / count)
    # ts = []
    # for i_count in range(count):
    #     ND_start = 1 + i_count * ittt
    #     ND_end = 1 + (i_count + 1) * ittt
    #     if i_count == count - 1:
    #         ND_end = ND
    #     t = Thread(target=get_delta,
    #                args=(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end, noise))
    #     ts.append(t)
    # [i.start() for i in ts]
    # [i.join() for i in ts]
    # get_delta(rho_Ind, rho_sorted, xx, maxed, rho, size_z, size_y, size_x, k1, k2, ND, ND_start, ND_end, noise)

    delta_sorted = np.sort(-delta) * -1
    delta[rho_Ind[0]] = delta_sorted[1]
    t1_ = time.time()
    print('delata, rho and Gradient are calculated, using %.2f seconds' % (t1_ - t0_))
     # 根据密度和距离来确定类中心
    clusterInd = -1 * np.ones(ND + 1)
    clust_index = np.intersect1d(np.where(rho > rhomin), np.where(delta > deltamin))

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
    if is_plot == 1:
        pass
    for i in range(ND):
        ordrho_i = rho_Ind[i]
        if clusterInd[ordrho_i] == -1:    # not centroid
            clusterInd[ordrho_i] = clusterInd[IndNearNeigh[ordrho_i]]
        else:
            Gradient[ordrho_i] = -1  # 将类中心点的梯度设置为-1

    clump_volume = np.zeros(n_clump)
    for i in range(n_clump):
        clump_volume[i] = clusterInd.tolist().count(i + 1)

    # centInd [类中心点在xx坐标下的索引值，类中心在centInd的索引值: 代表类别编号]
    centInd = []
    for i, item in enumerate(clump_volume):
        if item >= v_min:
            centInd.append([icl[i], i])
    centInd = np.array(centInd, np.int)
    mask_grad = np.where(Gradient > gradmin)[0]

    # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
    n_clump = centInd.shape[0]
    clump_sum, clump_volume, clump_peak = np.zeros([n_clump, 1]), np.zeros([n_clump, 1]), np.zeros([n_clump, 1])
    clump_Cen, clump_size = np.zeros([n_clump, dim]), np.zeros([n_clump, dim])
    clump_Peak = np.zeros([n_clump, dim], np.int)
    clump_ii = 0

    for i, item in enumerate(centInd):
        rho_cluster_i = np.zeros(ND)
        index_cluster_i = np.where(clusterInd == (item[1] + 1))[0]   # centInd[i, 1] --> item[1] 表示第i个类中心的编号
        index_cc = np.intersect1d(mask_grad, index_cluster_i)
        rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
        rho_cc_mean = rho[index_cc].mean() * 0.2
        index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
        index_cluster_rho = np.union1d(index_cc, index_cc_rho)

        cl_1_index_ = xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
        # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
        clustNum = cl_1_index_.shape[0]

        cl_i = np.zeros(data.shape, np.int)
        for j, item in enumerate(cl_1_index_):
            cl_i[item[2], item[1], item[0]] = 1
        # 形态学处理
        # cl_i = morphology.closing(cl_i)  # 做开闭运算会对相邻两个云核的掩膜有影响
        L = ndimage.binary_fill_holes(cl_i).astype(int)
        L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
        STATS = measure.regionprops(L)

        Ar_sum = []
        for region in STATS:
            coords = region.coords  # 经过验证，坐标原点为0
            temp = 0
            for j, item in enumerate(coords):
                temp += data[item[0], item[1], item[2]]
            Ar_sum.append(temp)
        Ar = np.array(Ar_sum)
        ind = np.where(Ar == Ar.max())[0]
        L[L != ind[0] + 1] = 0
        cl_i = L / (ind[0] + 1)
        coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标
        if coords.shape[0] > v_min:
            coords = coords[:, [2, 1, 0]]
            clump_i_ = np.zeros(coords.shape[0])
            for j, item in enumerate(coords):
                clump_i_[j] = data[item[2], item[1], item[0]]

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

    LDC_outcat = np.column_stack((id_clumps, clump_Peak, clump_Cen, clump_size, clump_peak, clump_sum, clump_volume))
    LDC_outcat = LDC_outcat[:clump_ii, :]
    return LDC_outcat, mask, out


if __name__ == '__main__':
    pass

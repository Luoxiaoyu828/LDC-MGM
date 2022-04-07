import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
"""
% 判断两个云核是否重叠，返回1或者0
% mult表示倍数
% 判断条件：
% 两个云核中心点间的距离d1
% 两个云核的轴长(sigma)之和的长度对 d2
% d1 > mult* d2 --> 0  没有重叠  否则为重叠

start_ind = 5;
clump_1 = outcat_i(start_ind:start_ind+5);
clump_2 = outcat_j(start_ind:start_ind+5);

distance_cen = ((clump_cen_i - clump_cen_j) ** 2).sum()**0.5 # % 两个云核中心点间的距离
distance_size = (clump_size_i**2).sum()**0.5 + (clump_size_j**2).sum()**0.5 # % 轴长之和构成的长度

if distance_cen > distance_size * mult
    touch_ = 0;
else
    touch_ = 1;
end

    #####################
    根据邻接矩阵生成连通的树，确定哪些云核相互重叠
   connected_components(csgraph, directed=True, connection='weak',
                         return_labels=True)
    
    Analyze the connected components of a sparse graph
    
    .. versionadded:: 0.11.0
    
    Parameters
    ----------
    csgraph : array_like or sparse matrix
        The N x N matrix representing the compressed sparse graph.  The input
        csgraph will be converted to csr format for the calculation.
    directed : bool, optional
        If True (default), then operate on a directed graph: only
        move from point i to point j along paths csgraph[i, j].
        If False, then find the shortest path on an undirected graph: the
        algorithm can progress from point i to j along csgraph[i, j] or
        csgraph[j, i].
    connection : str, optional
        ['weak'|'strong'].  For directed graphs, the type of connection to
        use.  Nodes i and j are strongly connected if a path exists both
        from i to j and from j to i. A directed graph is weakly connected
        if replacing all of its directed edges with undirected edges produces
        a connected (undirected) graph. If directed == False, this keyword
        is not referenced.
    return_labels : bool, optional
        If True (default), then return the labels for each of the connected
        components.
    
    Returns
    -------
    n_components: int
        The number of connected components.
    labels: ndarray
        The length-N array of labels of the connected components.
    
    References
    ----------
    .. [1] D. J. Pearce, "An Improved Algorithm for Finding the Strongly
           Connected Components of a Directed Graph", Technical Report, 2005
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import connected_components
    
    >>> graph = [
    ... [ 0, 1 , 1, 0 , 0 ],
    ... [ 0, 0 , 1 , 0 ,0 ],
    ... [ 0, 0, 0, 0, 0],
    ... [0, 0 , 0, 0, 1],
    ... [0, 0, 0, 0, 0]
    ... ]
    >>> graph = csr_matrix(graph)
    >>> print(graph)
      (0, 1)    1
      (0, 2)    1
      (1, 2)    1
      (3, 4)    1
    
    >>> n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    >>> n_components
    2
    >>> labels
    array([0, 0, 0, 1, 1], dtype=int32)

"""


# def touch_clump(outcat_i, outcat_j, mult):
#     if len(outcat_i) > 10:
#         start_ind = 4
#         clump_i = outcat_i[start_ind:start_ind + 6]
#         clump_j = outcat_j[start_ind:start_ind + 6]
#
#         distance_cen = np.sqrt(((clump_i[0:3] - clump_j[0:3]) ** 2).sum()) # % 两个云核中心点间的距离
#         distance_size = np.sqrt(((clump_i[3:6] + clump_j[3:6]) ** 2).sum()) # % 轴长之和构成的长度
#     else:
#         start_ind = 3
#         clump_i = outcat_i[start_ind:start_ind + 4]
#         clump_j = outcat_j[start_ind:start_ind + 4]
#
#         distance_cen = np.sqrt(((clump_i[0:2] - clump_j[0:2]) ** 2).sum())  # % 两个云核中心点间的距离
#         distance_size = np.sqrt(((clump_i[2:4] + clump_j[2:4]) ** 2).sum())  # % 轴长之和构成的长度
#     if distance_cen > (distance_size * mult):
#         touch_ = 0 # 代表没有重叠
#     else:
#         touch_ = 1
#
#     return touch_, distance_cen, distance_size
#
#
# def connect_clump(outcat_record, mult=1):
#
#     re = []
#     for i, outcat_i in enumerate(outcat_record.values):
#         aa = [i]
#         for j in range(i+1, outcat_record.shape[0]):
#             outcat_j = outcat_record.values[j]
#             touch_, distance_cen, distance_size = touch_clump(outcat_i, outcat_j, mult)
#             if touch_:
#                 aa.append(j)
#
#         re.append(np.array(aa, np.int64))
#
#     indx = np.array([item for item in range(outcat_record.shape[0])])
#     result = []
#     for i, item in enumerate(re):
#         if i in indx:
#             result.append(item + 1)  # 云核的编号从1开始
#             indx = np.setdiff1d(indx, item)
#     return result


def connect_clump_new(outcat, mult=1):
    """

    :param outcat: 像素核表
    :param mult: 比例系数
        if distance_cen > distance_size * mult
            touch_ = 0; # 没有重叠
        else
            touch_ = 1; # 存在重叠
        end
    :return:
        首先，计算邻接矩阵(touch_mat-->n*n):第i个核与第j个核相互重叠时touch_mat[i,j]=1
        然后，Analyze the connected components of a sparse graph
        最后，整理得到哪些云核是重叠在一起的：touch_all-->包含云核的ID(从1开始)

        同时，返回一个touch_all_direct：
            第i个云核及与其直接重叠的核的ID
    """

    outcat_num = outcat.shape[0]
    touch_mat = np.zeros([outcat_num, outcat_num], np.int64)

    for i in range(outcat_num):
        outcat_i = outcat.iloc[i]
        for j in range(i+1, outcat_num):
            outcat_j = outcat.iloc[j]
            touch_, distance_cen, distance_size = touch_clump_new(outcat_i, outcat_j, mult)
            if touch_:
                # print(distance_cen, distance_size)
                touch_mat[i, j] = 1
    # 计算邻接矩阵
    # touch_mat = touch_mat + touch_mat.T
    newarr = csr_matrix(touch_mat)

    n_components, labels = connected_components(newarr)  # 树的个数，标签
    touch_all = []
    for i in range(n_components):
        idx = np.where(labels == i)[0]
        touch_all.append(idx + 1)  # 云核的编号从1开始

    touch_all_direct = []
    for i in range(outcat_num):
        touch_i = np.array([], np.int64)
        touch_i = np.append(touch_i, i + 1)
        touch_mat_i = touch_mat[i, :]
        idx = np.where(touch_mat_i == 1)[0]
        if idx.shape[0] > 0:
            touch_i = np.append(touch_i, idx + 1)
        touch_all_direct.append(touch_i)

    return touch_all, touch_all_direct


def touch_clump_new(outcat_i, outcat_j, mult):
    """

    :param outcat_i: 第i个分子云核核表记录 [pd.DataFrame]
    :param outcat_j: 第j个分子云核核表记录
   :param mult: 比例系数
        if distance_cen > distance_size * mult
            touch_ = 0; # 没有重叠
        else
            touch_ = 1; # 存在重叠
        end
    :return:
    """
    axis_size = 2.3548   # 将 FWHM 转换成 sigma
    if 'Cen3' in outcat_i.keys():
        clump_cen_i = outcat_i[['Cen1', 'Cen2', 'Cen3']].values
        clump_cen_j = outcat_j[['Cen1', 'Cen2', 'Cen3']].values

        clump_size_i = outcat_i[['Size1', 'Size2', 'Size3']].values / axis_size
        clump_size_j = outcat_j[['Size1', 'Size2', 'Size3']].values / axis_size

    else:
        clump_cen_i = outcat_i[['Cen1', 'Cen2']].values
        clump_cen_j = outcat_j[['Cen1', 'Cen2']].values

        clump_size_i = outcat_i[['Size1', 'Size2']].values / axis_size
        clump_size_j = outcat_j[['Size1', 'Size2']].values / axis_size

    distance_cen = ((clump_cen_i - clump_cen_j) ** 2).sum()**0.5    # % 两个云核中心点间的距离
    distance_size = (clump_size_i**2).sum()**0.5 + (clump_size_j**2).sum()**0.5  # % 轴长之和构成的长度
    if distance_cen > (distance_size * mult):
        touch_ = 0      # 代表没有重叠
    else:
        touch_ = 1

    return touch_, distance_cen, distance_size


if __name__ == '__main__':
    # mult = 1.5   # mult  越大表示判断重叠的条件越宽松
    # outcat_record = pd.read_csv(r'F:\LDC_python\detection\test_data_zhou_again\n_clump_100\outcat_record\gaussian_outcat_000.txt', sep='\t')
    # result = connect_clump(outcat_record, mult)
    #
    # aa = outcat_record.values[[1,71,99],4:10]
    # print(((aa[2,:3] - aa[1,:3])**2).sum()**0.5)
    # print(((aa[2, 3:6] + aa[1, 3:6]) ** 2).sum() ** 0.5)

    outcat_name = r'F:\Parameter_reduction\LDC\0170+010_L\LDC_auto_loc_outcat.csv'
    f_outcat = pd.read_csv(outcat_name, sep=',')

    touch_clump_record = connect_clump_new(f_outcat, mult=0.9)
    print(touch_clump_record)
    a = np.array([])

    for item in touch_clump_record:
        a = np.append(a, item)




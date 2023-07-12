import numpy as np
import pandas as pd
import os


def create_folders(path_list):
    for item_path in path_list:
        if not os.path.exists(item_path):
            os.mkdir(item_path)


def match_simu_detect(simulated_outcat_path, detected_outcat_path, match_save_path):
    # simulated_outcat_path, detected_outcat_path, match_save_path = outcat_name, fit_outcat_name, match_save_path
    if not os.path.exists(match_save_path):
        os.mkdir(match_save_path)

    clump_item = simulated_outcat_path.split('_')[-1].split('.')[0]

    Match_table = os.path.join(match_save_path, 'Match_table')
    Miss_table = os.path.join(match_save_path, 'Miss_table')
    False_table = os.path.join(match_save_path, 'False_table')
    create_folders([Match_table, Miss_table, False_table])
    match_cfg = {}
    Match_table_name = os.path.join(Match_table, 'Match_%s.txt' %clump_item)
    Miss_table_name = os.path.join(Miss_table, 'Miss_%s.txt' % clump_item)
    False_table_name = os.path.join(False_table, 'False_%s.txt' % clump_item)
    match_cfg['Match_table_name'] = Match_table_name
    match_cfg['Miss_table_name'] = Miss_table_name
    match_cfg['False_table_name'] = False_table_name
    table_s = pd.read_csv(simulated_outcat_path, sep='\t')
    table_g = pd.read_csv(detected_outcat_path, sep='\t')
    if table_g.values.shape[1] == 1:
        table_g = pd.read_csv(detected_outcat_path, sep=' ')

    # table_simulate1=pd.read_csv(path_outcat,sep=' ')
    # table_g=pd.read_csv(path_outcat_wcs,sep='\t')
    # table_g.columns = new_cols

    Error_xyz = np.array([2, 2, 2])  # 匹配容许的最大误差(单位：像素)

    Cen_simulate = np.vstack([table_s['Cen1'], table_s['Cen2'], table_s['Cen3']]).T
    # Cen_simulate = np.vstack([table_s['Peak1'], table_s['Peak2'], table_s['Peak3']]).T
    Size_simulate = np.vstack([table_s['Size1'], table_s['Size2'], table_s['Size3']]).T
    Cen_gauss = np.vstack([table_g['Cen1'], table_g['Cen2'], table_g['Cen3']]).T

    Cen_gauss = Cen_gauss[~np.isnan(Cen_gauss).any(axis=1), :]
    # calculate distance
    simu_len = Cen_simulate.shape[0]
    gauss_len = Cen_gauss.shape[0]
    distance = np.zeros([simu_len, gauss_len])

    for i, item_simu in enumerate(Cen_simulate):
        for j, item_gauss in enumerate(Cen_gauss):
            cen_simu = item_simu
            cen_gauss = item_gauss
            temp = np.sqrt(((cen_gauss - cen_simu)**2).sum())
            distance[i,j] = temp
    max_d = 1.2 * distance.max()

    match_record_simu_detect = [] #匹配核表
    match_num = 0

    while 1:
        # 找到距离最小的行和列
        d_ij_value = distance.min()
        if d_ij_value == max_d:  # 表示距离矩阵里面所有元素都匹配上了
            break
        [simu_i, gauss_j] = np.where(distance==d_ij_value)
        simu_i, gauss_j = simu_i[0], gauss_j[0]
        cen_simu_i = Cen_simulate[simu_i]
        size_simu_i = Size_simulate[simu_i]
        cen_gauss_j = Cen_gauss[gauss_j]

        # 确定误差项
        temp = np.array([Error_xyz, size_simu_i / 2.3548])
        Error_xyz1 = temp.min(axis=0)

        d_ij = np.abs(cen_simu_i - cen_gauss_j)
        match_num_ = match_num
        if (d_ij<= Error_xyz1).all():
            # print([d_ij, d_ij_value])
            distance[simu_i,:] = np.ones([gauss_len]) * max_d
            distance[:, gauss_j] = np.ones([simu_len]) * max_d
            match_num = match_num + 1
            match_record_simu_detect.append(np.array([d_ij_value, simu_i + 1, gauss_j + 1]))  # 误差 仿真表索引 检测表索引

        if match_num == match_num_:
            break
    match_record_simu_detect = np.array(match_record_simu_detect)

    F1, precision, recall = 0, 0, 0
    if match_num > 0:
        precision = match_num / gauss_len
        recall = match_num / simu_len
        F1 = 2 * precision * recall / (precision + recall)
        # print("simulated num = %d\t detected num %d\t match num %d" % (simu_len, gauss_len, match_num))
    print("F1_precision_recall = %.3f, %.3f, %.3f" % (F1, precision, recall))

    # new_cols = ['PIDENT', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
    #             'Sum', 'Volume']
    if match_record_simu_detect.shape[0] > 0:

        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium] #列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名
        table_title = names + names1

        match_simu_inx = match_record_simu_detect[:, 1].astype(np.int)
        table_s_np = table_s.values[match_simu_inx - 1, :]

        match_gauss = match_record_simu_detect[:, 2].astype(np.int)
        table_g_np = table_g.values[match_gauss - 1, :]

        match_outcat = np.hstack([table_s_np, table_g_np])

        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        # simu_inx = table_s['ID']
        simu_inx = np.array([item + 1 for item in range(table_s['ID'].shape[0])])

        # x = set([0.0])
        miss_idx = np.setdiff1d(simu_inx, match_simu_inx).astype(np.int)  # 未检测到的云核编号

        miss_names = ['s_' + item for item in new_cols_sium] #列名
        if len(miss_idx) == 0:
            miss_outcat = []
        else:
            miss_outcat = table_s.values[miss_idx - 1, :]
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)
        # miss = Table(names=miss_names)
        # for item in miss_idx:  # 未检出表
        #     miss.add_row(list(table_s[int(item) - 1, :]))
        # miss.write(Miss_table_name, overwrite=True, format='ascii')
        try:
            # gauss_inx = table_g['ID']
            gauss_inx = np.array([item + 1 for item in range(table_g['ID'].shape[0])])

        except KeyError:
            gauss_inx = table_g['PIDENT']
        false_idx = np.setdiff1d(gauss_inx, match_gauss).astype(np.int)

        if len(false_idx) == 0:
            false_outcat = []
        else:
            # print(false_idx)
            false_outcat = table_g.values[false_idx - 1, :]

        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(False_table_name, sep='\t', index=False)

    else:
        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium]  # 列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名

        table_title = names + names1
        match_outcat = []
        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        miss_names = ['s_' + item for item in new_cols_sium]  # 列名
        miss_outcat = []
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)

        false_outcat = []
        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        dataframe.to_csv(False_table_name, sep='\t', index=False)

    return match_cfg


def match_simu_detect_new(sop, dop, msp, s_cen=None, s_size=None, g_cen=None):
    """
    指定两个核表及其对应的列名进行匹配
        :param sop：仿真核表路径(也可以是其他作为标准的核表)
        :param dop：检测核表路径
        :param msp：匹配核表保存路径
        :param s_cen：仿真核表的质心列名，默认为：['Cen1', 'Cen2', 'Cen3']
        :param s_cen：仿真核表的轴长列名，默认为：['Size1', 'Size2', 'Size3']
        :param g_cen：检测核表的质心列名，默认为：['Cen1', 'Cen2', 'Cen3']
    return：
        match_cfg：字典类型
        match_cfg['Match_table_name'] = Match_table_name
        match_cfg['Miss_table_name'] = Miss_table_name
        match_cfg['False_table_name'] = False_table_name
        方便后续计算时调用。
    """
    if s_size is None:
        s_size = ['Size1', 'Size2', 'Size3']
    if s_cen is None:
        s_cen = ['Cen1', 'Cen2', 'Cen3']
    if g_cen is None:
        g_cen = ['Cen1', 'Cen2', 'Cen3']
        # MGM拟合核表质心的列名
        # g_cen = ['Galactic_Longitude', 'Galactic_Latitude', 'Velocity']

    os.makedirs(msp, exist_ok=True)
    clump_item = sop.split('_')[-1].split('.')[0]

    Match_table = os.path.join(msp, 'Match_table')
    Miss_table = os.path.join(msp, 'Miss_table')
    False_table = os.path.join(msp, 'False_table')
    create_folders([Match_table, Miss_table, False_table])

    Match_table_name = os.path.join(Match_table, 'Match_%s.txt' %clump_item)
    Miss_table_name = os.path.join(Miss_table, 'Miss_%s.txt' % clump_item)
    False_table_name = os.path.join(False_table, 'False_%s.txt' % clump_item)
    match_cfg = {}
    match_cfg['Match_table_name'] = Match_table_name
    match_cfg['Miss_table_name'] = Miss_table_name
    match_cfg['False_table_name'] = False_table_name
    table_s = pd.read_csv(sop, sep='\t')
    table_g = pd.read_csv(dop, sep='\t')
    if table_g.values.shape[1] == 1:
        table_g = pd.read_csv(dop, sep=' ')

    Error_xyz = np.array([2, 2, 2])  # 匹配容许的最大误差(单位：像素)

    # Cen_simulate = np.vstack([table_s['Cen1'], table_s['Cen2'], table_s['Cen3']]).T
    Cen_simulate = table_s[s_cen].values
    # Size_simulate = np.vstack([table_s['Size1'], table_s['Size2'], table_s['Size3']]).T
    Size_simulate = table_s[s_size].values
    # Cen_gauss = np.vstack([table_g['Cen1'], table_g['Cen2'], table_g['Cen3']]).T
    Cen_gauss = table_g[g_cen].values
    Cen_gauss = Cen_gauss[~np.isnan(Cen_gauss).any(axis=1), :]
    # calculate distance
    simu_len = Cen_simulate.shape[0]
    gauss_len = Cen_gauss.shape[0]
    distance = np.zeros([simu_len, gauss_len])

    for i, item_simu in enumerate(Cen_simulate):
        for j, item_gauss in enumerate(Cen_gauss):
            cen_simu = item_simu
            cen_gauss = item_gauss
            temp = np.sqrt(((cen_gauss - cen_simu)**2).sum())
            distance[i,j] = temp
    max_d = 1.2 * distance.max()

    match_record_simu_detect = [] #匹配核表
    match_num = 0

    while 1:
        # 找到距离最小的行和列
        d_ij_value = distance.min()
        if d_ij_value == max_d:  # 表示距离矩阵里面所有元素都匹配上了
            break
        [simu_i, gauss_j] = np.where(distance==d_ij_value)
        simu_i, gauss_j = simu_i[0], gauss_j[0]
        cen_simu_i = Cen_simulate[simu_i]
        size_simu_i = Size_simulate[simu_i]
        cen_gauss_j = Cen_gauss[gauss_j]

        # 确定误差项
        temp = np.array([Error_xyz, size_simu_i / 2.3548])
        Error_xyz1 = temp.min(axis=0)

        d_ij = np.abs(cen_simu_i - cen_gauss_j)
        match_num_ = match_num
        if (d_ij<= Error_xyz1).all():
            # print([d_ij, d_ij_value])
            distance[simu_i,:] = np.ones([gauss_len]) * max_d
            distance[:, gauss_j] = np.ones([simu_len]) * max_d
            match_num = match_num + 1
            match_record_simu_detect.append(np.array([d_ij_value, simu_i + 1, gauss_j + 1]))  # 误差 仿真表索引 检测表索引

        if match_num == match_num_:
            break
    match_record_simu_detect = np.array(match_record_simu_detect)

    F1, precision, recall = 0, 0, 0
    if match_num > 0:
        precision = match_num / gauss_len
        recall = match_num / simu_len
        F1 = 2 * precision * recall / (precision + recall)
        # print("simulated num = %d\t detected num %d\t match num %d" % (simu_len, gauss_len, match_num))
    print("F1_precision_recall = %.3f, %.3f, %.3f" % (F1, precision, recall))

    # new_cols = ['PIDENT', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
    #             'Sum', 'Volume']
    if match_record_simu_detect.shape[0] > 0:

        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium] #列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名
        table_title = names + names1

        match_simu_inx = match_record_simu_detect[:, 1].astype(np.int)
        table_s_np = table_s.values[match_simu_inx - 1, :]

        match_gauss = match_record_simu_detect[:, 2].astype(np.int)
        table_g_np = table_g.values[match_gauss - 1, :]

        match_outcat = np.hstack([table_s_np, table_g_np])

        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        # simu_inx = table_s['ID']
        simu_inx = np.array([item + 1 for item in range(table_s['ID'].shape[0])])

        # x = set([0.0])
        miss_idx = np.setdiff1d(simu_inx, match_simu_inx).astype(np.int)  # 未检测到的云核编号

        miss_names = ['s_' + item for item in new_cols_sium] #列名
        if len(miss_idx) == 0:
            miss_outcat = []
        else:
            miss_outcat = table_s.values[miss_idx - 1, :]
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)
        # miss = Table(names=miss_names)
        # for item in miss_idx:  # 未检出表
        #     miss.add_row(list(table_s[int(item) - 1, :]))
        # miss.write(Miss_table_name, overwrite=True, format='ascii')
        try:
            # gauss_inx = table_g['ID']
            gauss_inx = np.array([item + 1 for item in range(table_g['ID'].shape[0])])

        except KeyError:
            gauss_inx = table_g['PIDENT']
        false_idx = np.setdiff1d(gauss_inx, match_gauss).astype(np.int)

        if len(false_idx) == 0:
            false_outcat = []
        else:
            # print(false_idx)
            false_outcat = table_g.values[false_idx - 1, :]

        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(False_table_name, sep='\t', index=False)

    else:
        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium]  # 列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名

        table_title = names + names1
        match_outcat = []
        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        miss_names = ['s_' + item for item in new_cols_sium]  # 列名
        miss_outcat = []
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)

        false_outcat = []
        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        dataframe.to_csv(False_table_name, sep='\t', index=False)

    return match_cfg


def match_simu_detect_2d(simulated_outcat_path, detected_outcat_path, match_save_path):
    # simulated_outcat_path, detected_outcat_path, match_save_path = outcat_name, fit_outcat_name, match_save_path

    if not os.path.exists(match_save_path):
        os.mkdir(match_save_path)

    clump_item = simulated_outcat_path.split('_')[-1].split('.')[0]

    Match_table = os.path.join(match_save_path, 'Match_table')
    Miss_table = os.path.join(match_save_path, 'Miss_table')
    False_table = os.path.join(match_save_path, 'False_table')
    create_folders([Match_table, Miss_table, False_table])

    Match_table_name = os.path.join(Match_table, 'Match_%s.txt' % clump_item)
    Miss_table_name = os.path.join(Miss_table, 'Miss_%s.txt' % clump_item)
    False_table_name = os.path.join(False_table, 'False_%s.txt' % clump_item)

    # table_simulate1 = np.loadtxt(simulated_outcat_path, skiprows=1)
    # table_g = np.loadtxt(detected_outcat_path, skiprows=1)

    table_s = pd.read_csv(simulated_outcat_path, sep='\t')
    table_g = pd.read_csv(detected_outcat_path, sep='\t')
    if table_g.values.shape[1] == 1:
        table_g = pd.read_csv(detected_outcat_path, sep=' ')

    # table_simulate1=pd.read_csv(path_outcat,sep=' ')
    # table_g=pd.read_csv(path_outcat_wcs,sep='\t')
    # table_g.columns = new_cols

    Error_xyz = np.array([2, 2])  # 匹配容许的最大误差(单位：像素)

    Cen_simulate = np.vstack([table_s['Cen1'], table_s['Cen2']]).T
    Size_simulate = np.vstack([table_s['Size1'], table_s['Size2']]).T
    try:
        Cen_gauss = np.vstack([table_g['Cen1'], table_g['Cen2']]).T
    except KeyError:
        Cen_gauss = np.vstack([table_g['cen1'], table_g['cen2']]).T
    Cen_gauss = Cen_gauss[~np.isnan(Cen_gauss).any(axis=1), :]
    # calculate distance
    simu_len = Cen_simulate.shape[0]
    gauss_len = Cen_gauss.shape[0]
    distance = np.zeros([simu_len, gauss_len])

    for i, item_simu in enumerate(Cen_simulate):
        for j, item_gauss in enumerate(Cen_gauss):
            cen_simu = item_simu
            cen_gauss = item_gauss
            temp = np.sqrt(((cen_gauss - cen_simu) ** 2).sum())
            distance[i, j] = temp
    max_d = 1.2 * distance.max()

    match_record_simu_detect = []  # 匹配核表
    match_num = 0

    while 1:
        # 找到距离最小的行和列
        d_ij_value = distance.min()
        if d_ij_value == max_d:  # 表示距离矩阵里面所有元素都匹配上了
            break
        [simu_i, gauss_j] = np.where(distance == d_ij_value)
        simu_i, gauss_j = simu_i[0], gauss_j[0]
        cen_simu_i = Cen_simulate[simu_i]
        size_simu_i = Size_simulate[simu_i]
        cen_gauss_j = Cen_gauss[gauss_j]

        # 确定误差项
        temp = np.array([Error_xyz, size_simu_i / 2.3548])
        Error_xyz1 = temp.min(axis=0)

        d_ij = np.abs(cen_simu_i - cen_gauss_j)
        match_num_ = match_num
        if (d_ij <= Error_xyz1).all():
            # print([d_ij, d_ij_value])
            distance[simu_i, :] = np.ones([gauss_len]) * max_d
            distance[:, gauss_j] = np.ones([simu_len]) * max_d
            match_num = match_num + 1
            match_record_simu_detect.append(np.array([d_ij_value, simu_i + 1, gauss_j + 1]))  # 误差 仿真表索引 检测表索引

        if match_num == match_num_:
            break
    match_record_simu_detect = np.array(match_record_simu_detect)

    F1, precision, recall = 0, 0, 0
    if match_num > 0:
        precision = match_num / gauss_len
        recall = match_num / simu_len
        F1 = 2 * precision * recall / (precision + recall)
        # print("simulated num = %d\t detected num %d\t match num %d" % (simu_len, gauss_len, match_num))
    print("F1_precision_recall = %.3f, %.3f, %.3f" % (F1, precision, recall))

    # new_cols = ['PIDENT', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
    #             'Sum', 'Volume']
    if match_record_simu_detect.shape[0] > 0:

        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium]  # 列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名
        table_title = names + names1

        match_simu_inx = match_record_simu_detect[:, 1].astype(np.int)
        table_s_np = table_s.values[match_simu_inx - 1, :]

        match_gauss = match_record_simu_detect[:, 2].astype(np.int)
        table_g_np = table_g.values[match_gauss - 1, :]

        match_outcat = np.hstack([table_s_np, table_g_np])

        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        simu_inx = table_s['ID']

        # x = set([0.0])
        miss_idx = np.setdiff1d(simu_inx, match_simu_inx).astype(np.int)  # 未检测到的云核编号

        miss_names = ['s_' + item for item in new_cols_sium]  # 列名
        if len(miss_idx) == 0:
            miss_outcat = []
        else:
            miss_outcat = table_s.values[miss_idx - 1, :]
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)
        # miss = Table(names=miss_names)
        # for item in miss_idx:  # 未检出表
        #     miss.add_row(list(table_s[int(item) - 1, :]))
        # miss.write(Miss_table_name, overwrite=True, format='ascii')
        try:
            gauss_inx = table_g['ID']

        except KeyError:
            gauss_inx = table_g['PIDENT']
        false_idx = np.setdiff1d(gauss_inx, match_gauss).astype(np.int)

        if len(false_idx) == 0:
            false_outcat = []
        else:
            # print(false_idx)
            false_outcat = table_g.values[false_idx - 1, :]

        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(False_table_name, sep='\t', index=False)

    else:
        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium]  # 列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名

        table_title = names + names1
        match_outcat = []
        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        miss_names = ['s_' + item for item in new_cols_sium]  # 列名
        miss_outcat = []
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)

        false_outcat = []
        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        dataframe.to_csv(False_table_name, sep='\t', index=False)
    # print("end")
# 


if __name__ == '__main__':
    simulated_outcat_path = r'data/simulated_outcat_000.txt'
    detected_outcat_path = r'data/detected_outcat.txt'
    match_result = r'match_result'
    match_simu_detect(simulated_outcat_path, detected_outcat_path, match_result)
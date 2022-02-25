import numpy as np

# 判断两个核是否可分
def is_separable(local_outcat):
    # row1 = random.sample(range(0, 1000), 10)
    # test_data = data.iloc[row1]
    # test_data = np.array(test_data)
    local_outcat[:, 7: 10] = local_outcat[:, 7: 10] / 2.3548  # 检测核表的轴长(标准差)
    distance_xy = []  # 表达式二的xy距离矩阵
    distance_xy_sigma = []  # 表达式二的sigma距离矩阵
    distance_v = []  # 表达式一的v距离矩阵
    distance_v_sigma = []  # 表达式一的sigma距离矩阵
    for i in range(local_outcat.shape[0]):
        distance_xy.append(np.sqrt((local_outcat[:, 1] - local_outcat[i, 1]) ** 2 + (local_outcat[:, 2] - local_outcat[i, 2]) ** 2))
        distance_xy_sigma.append(2.3548 * np.sqrt(local_outcat[:, 8] ** 2 + local_outcat[:, 7] ** 2 + local_outcat[i, 8] ** 2 +
                                                  local_outcat[i, 7] ** 2))

        distance_v.append(np.abs(local_outcat[:, 3] - local_outcat[i, 3]))
        distance_v_sigma.append(2.3548 * np.sqrt(local_outcat[:, 9] ** 2 + local_outcat[i, 9] ** 2))
    distance_xy = np.array(distance_xy)
    distance_xy_sigma = np.array(distance_xy_sigma)
    distance_v = np.array(distance_v)
    distance_v_sigma = np.array(distance_v_sigma)

    func1_res = distance_v - distance_v_sigma
    func1_res[func1_res >= 0] = 0  # 可分
    func1_res[func1_res < 0] = 1  # 不可分

    func2_res = distance_xy - distance_xy_sigma
    func2_res[func2_res >= 0] = 0  # 可分
    func2_res[func2_res < 0] = 1  # 不可分

    result = np.zeros_like(func2_res)
    for i in range(func2_res.shape[0]):
        for j in range(func2_res.shape[1]):
            result[i, j] = func2_res[i, j] and func1_res[i, j]
    row, col = np.diag_indices_from(result)
    result[row, col] = np.array(0)  # 将对角线元素的值置为0
    a = np.triu(result)
    b = np.where(a == 1)
    b1 = b[0].reshape((b[0].shape[0], 1))
    b2 = b[1].reshape((b[1].shape[0], 1))
    res = np.hstack((b1, b2))
    return res

def is_separable1(test_data):
    # row1 = random.sample(range(0, 1000), 10)
    # test_data = data.iloc[row1]
    # test_data = np.array(test_data)
    multi = 1
    test_data[:, 7: 10] = test_data[:, 7: 10] / 2.3548
    xy_distance = []  # 表达式二的xy距离矩阵
    sigma_xy_distance = []  # 表达式二的sigma距离矩阵
    v_distance = []  # 表达式一的v距离矩阵
    sigma_v_distance = []  # 表达式一的sigma距离矩阵
    for i in range(test_data.shape[0]):
        xy_distance.append(np.sqrt((test_data[:, 4] - test_data[i, 4]) ** 2 + (test_data[:, 5] - test_data[i, 5]) ** 2))
        sigma_xy_distance.append(multi * np.sqrt(test_data[:, 8] ** 2 + test_data[:, 7] ** 2 + test_data[i, 8] ** 2 +
                                                  test_data[i, 7] ** 2))
        v_distance.append(np.abs(test_data[:, 6] - test_data[i, 6]))
        sigma_v_distance.append(multi * np.sqrt(test_data[:, 9] ** 2 + test_data[i, 9] ** 2))
    xy_distance = np.array(xy_distance)
    sigma_xy_distance = np.array(sigma_xy_distance)
    v_distance = np.array(v_distance)
    sigma_v_distance = np.array(sigma_v_distance)

    func1_res = v_distance - sigma_v_distance
    func1_res[func1_res >= 0] = 0  # 可分
    func1_res[func1_res < 0] = 1  # 不可分

    func2_res = xy_distance - sigma_xy_distance
    func2_res[func2_res >= 0] = 0  # 可分
    func2_res[func2_res < 0] = 1  # 不可分

    result = np.zeros_like(func2_res)
    for i in range(func2_res.shape[0]):
        for j in range(func2_res.shape[1]):
            result[i, j] = func2_res[i, j] and func1_res[i, j]
    row, col = np.diag_indices_from(result)
    result[row, col] = np.array(0)  # 将对角线元素的值置为0
    a = np.triu(result)
    b = np.where(a == 1)
    b1 = b[0].reshape((b[0].shape[0], 1))
    b2 = b[1].reshape((b[1].shape[0], 1))
    res = np.hstack((b1, b2))
    return res
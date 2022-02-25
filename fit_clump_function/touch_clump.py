import numpy as np
import pandas as pd

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

distance_cen = sqrt(sum((clump_1(1:3)-clump_2(1:3)).^2)); % 两个云核中心点间的距离
distance_size = sqrt(sum((clump_1(4:6)+clump_2(4:6)).^2)); % 轴长之和构成的长度

if distance_cen > distance_size * mult
    touch_ = 0;
else
    touch_ = 1;
end
"""


def touch_clump(outcat_i, outcat_j, mult):
    if len(outcat_i) > 10:
        start_ind = 4
        clump_i = outcat_i[start_ind:start_ind + 6]
        clump_j = outcat_j[start_ind:start_ind + 6]

        distance_cen = np.sqrt(((clump_i[0:3] - clump_j[0:3]) ** 2).sum()) # % 两个云核中心点间的距离
        distance_size = np.sqrt(((clump_i[3:6] + clump_j[3:6]) ** 2).sum()) # % 轴长之和构成的长度
    else:
        start_ind = 3
        clump_i = outcat_i[start_ind:start_ind + 4]
        clump_j = outcat_j[start_ind:start_ind + 4]

        distance_cen = np.sqrt(((clump_i[0:2] - clump_j[0:2]) ** 2).sum())  # % 两个云核中心点间的距离
        distance_size = np.sqrt(((clump_i[2:4] + clump_j[2:4]) ** 2).sum())  # % 轴长之和构成的长度
    if distance_cen > (distance_size * mult):
        touch_ = 0 # 代表没有重叠
    else:
        touch_ = 1

    return touch_, distance_cen, distance_size


def connect_clump(outcat, mult=1):

    re = []
    for i, outcat_i in enumerate(outcat.values):
        aa = [i]
        for j in range(i+1, outcat.shape[0]):
            outcat_j = outcat.values[j]
            touch_, distance_cen, distance_size = touch_clump(outcat_i, outcat_j, mult)
            if touch_:
                aa.append(j)

        re.append(np.array(aa, np.int))

    indx = np.array([item for item in range(outcat.shape[0])])
    result = []
    for i, item in enumerate(re):
        if i in indx:
            result.append(item + 1)  # 云核的编号从1开始
            indx = np.setdiff1d(indx, item)
    return result


if __name__ == '__main__':
    mult = 1.5   # mult  越大表示判断重叠的条件越宽松
    outcat = pd.read_csv(r'F:\LDC_python\detection\test_data_zhou_again\n_clump_100\outcat\gaussian_outcat_000.txt', sep='\t')
    result = connect_clump(outcat,mult)

    aa = outcat.values[[1,71,99],4:10]
    print(((aa[2,:3] - aa[1,:3])**2).sum()**0.5)
    print(((aa[2, 3:6] + aa[1, 3:6]) ** 2).sum() ** 0.5)

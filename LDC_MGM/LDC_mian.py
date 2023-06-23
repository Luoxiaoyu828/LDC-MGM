import os
from DensityClust.LocalDensityClustering_main import localDenCluster


def LDC_main(data_name, para, save_folder=None):
    """
        LDC algorithm
        :param data_name: 待检测数据的路径(str)，fits文件
        :param para: 算法参数, dict
            para.rho_min: Minimum density [5*rms]
            para.delta_min: Minimum delta [4]
            para.v_min: Minimum volume [25, 5]
            para.noise: The noise level of the data, used for data truncation calculation [2*rms]
            para.dc: auto
        :param save_folder: 检测结果保存路径
        :param save_loc: 是否保存检测的局部核表，默认为False(不保存)

        :return:
            None

        Usage:
        data_name = r'*******.fits'
        para = Param(delta_min=4, gradmin=0.01, v_min=[25, 5], noise_times=2, rms_times=5, rms_key='RMS')
        save_folder = r'#####'
        LDC_main(data_name, para, save_folder)

    """

    if save_folder is None:
        save_folder = data_name.replace('.fits', '')
    os.makedirs(save_folder, exist_ok=True)

    localDenCluster(data_name, para, save_folder=save_folder)


if __name__ == '__main__':
    pass
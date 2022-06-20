import os
import astropy.io.fits as fits
from tools.show_clumps import deal_data
# from DensityClust.locatDenClust3 import Data, Param, LocalDensityCluster
from DensityClust.localDenClust2 import Data, Param, LocalDensityCluster


def ldc_base(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name, fig_name):
    ldc = LocalDensityCluster(data=data, para=para)
    ldc.detect()
    ldc.save_detect_log(detect_log)

    ldc.result.save_outcat(outcat_name, loc=0)
    ldc.result.save_outcat_wcs(outcat_wcs_name, loc=0)

    ldc.result.save_outcat(loc_outcat_name, loc=1)
    ldc.result.save_outcat_wcs(loc_outcat_wcs_name, loc=1)

    ldc.result.save_mask(mask_name)
    ldc.result.make_plot_wcs_1(fig_name)
    print(ldc.data.data_path + ' has finished!')


def localDenCluster(data_name, mask_name=None, outcat_name=None, outcat_wcs_name=None, loc_outcat_name=None,
                    loc_outcat_wcs_name=None, detect_log=None, fig_name='', paras_set=None):
    """
    LDC algorithm
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数，dict
        para.rho_min: Minimum density [5*rms]
        para.delta_min: Minimum delta [4]
        para.v_min: Minimum volume [27]
        para.noise: The noise level of the data, used for data truncation calculation [2*rms]
        para.dc: auto
    :param mask_name: 掩模数据的保存路径(str) [*.fits]
    :param outcat_name: 基于像素单位的核表保存路径(str) [*.csv]
    :param outcat_wcs_name: 基于wcs的核表保存路径(str) [*.csv]
    :param loc_outcat_name: 基于像素单位的局部区域核表保存路径(str) [*.csv]
    :param loc_outcat_wcs_name: 基于wcs的局部区域核表保存路径(str) [*.csv]
    :param detect_log: 检测中的信息保存文件(str) [*.txt]
    :param flags: 代码调用还是软件界面调用，默认为True(代码调用)
    :return:
    """
    data = Data(data_name)

    para = Param(delta_min=4, gradmin=0.01, v_min=27, noise_times=2, rms_times=3)
    para.set_rms_by_data(data)

    if paras_set is not None:
        para.set_para(paras_set)

    ldc_base(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name,
             fig_name)


def localDenCluster_split_mode(data_name, mask_name=None, outcat_name=None, outcat_wcs_name=None, loc_outcat_name=None,
                               loc_outcat_wcs_name=None, detect_log=None, fig_name='', paras_set=None):
    """
    LDC algorithm
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数，dict
        para.rho_min: Minimum density [5*rms]
        para.delta_min: Minimum delta [4]
        para.v_min: Minimum volume [27]
        para.noise: The noise level of the data, used for data truncation calculation [2*rms]
        para.dc: auto
    :param mask_name: 掩模数据的保存路径(str) [*.fits]
    :param outcat_name: 基于像素单位的核表保存路径(str) [*.csv]
    :param outcat_wcs_name: 基于wcs的核表保存路径(str) [*.csv]
    :param loc_outcat_name: 基于像素单位的局部区域核表保存路径(str) [*.csv]
    :param loc_outcat_wcs_name: 基于wcs的局部区域核表保存路径(str) [*.csv]
    :param detect_log: 检测中的信息保存文件(str) [*.txt]
    :param flags: 代码调用还是软件界面调用，默认为True(代码调用)
    :return:
    """
    data = Data(data_name)
    data_cube = fits.getdata(data_name)

    data.set_data_cube(data_cube[:, :50, :50])
    # data.set_wcs(data.wcs[:100, :100, :100])

    para = Param(delta_min=4, gradmin=0.01, v_min=27, noise_times=2, rms_times=5)
    para.set_rms_by_data(data)

    if paras_set is not None:
        para.set_para(paras_set)

    ldc_base(data, para, detect_log, outcat_name, outcat_wcs_name, loc_outcat_name, loc_outcat_wcs_name, mask_name,
             fig_name)


def LDC_fast(data_name, save_folder, split=False):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    mask_name = os.path.join(save_folder, 'LDC_auto_mask.fits')
    outcat_name = os.path.join(save_folder, 'LDC_auto_outcat.csv')
    outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_outcat_wcs.csv')

    loc_outcat_name = os.path.join(save_folder, 'LDC_auto_loc_outcat.csv')
    loc_outcat_wcs_name = os.path.join(save_folder, 'LDC_auto_loc_outcat_wcs.csv')

    detect_log = os.path.join(save_folder, 'LDC_auto_detect_log.txt')
    fig_name = os.path.join(save_folder, 'LDC_auto_detect_result.png')

    if split:
        localDenCluster_split_mode(data_name, mask_name=mask_name, outcat_name=outcat_name,
                                   outcat_wcs_name=outcat_wcs_name,
                                   loc_outcat_name=loc_outcat_name, loc_outcat_wcs_name=loc_outcat_wcs_name,
                                   detect_log=detect_log,
                                   fig_name=fig_name)
    else:
        localDenCluster(data_name, mask_name=mask_name, outcat_name=outcat_name, outcat_wcs_name=outcat_wcs_name,
                        loc_outcat_name=loc_outcat_name, loc_outcat_wcs_name=loc_outcat_wcs_name, detect_log=detect_log,
                        fig_name=fig_name)


if __name__ == '__main__':
    data_name = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000.fits'
    save_folder = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000_split'
    LDC_fast(data_name, save_folder)

    deal_data(save_folder, os.path.join(save_folder, 'ei'))
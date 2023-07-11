from distrib.fit_clump_function import main_fit_gauss_3d
from distrib.DensityClust import LocalDensityClustering_main
import yaml
import os
from distrib.DensityClust.localDenClust2 import Param

this_cwd = os.getcwd()


def read_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(this_cwd, 'config.yaml')
    if not os.path.exists(config_path):
        raise OSError('file dose not exist!')
    file = open(config_path, 'r', encoding='utf-8')
    para = yaml.load(file, Loader=yaml.FullLoader)
    return para


def main_fun():
    para = read_config()
    delta_min = para['detect']['para']['delta_min']
    gradmin = para['detect']['para']['gradmin']
    v_min = para['detect']['para']['v_min']
    noise_times = para['detect']['para']['noise_times']
    rms_times = para['detect']['para']['rms_times']
    data_file = para['detect']['data_name']
    save_folder = para['detect']['save_folder']
    split = para['detect']['split']
    save_loc = para['detect']['save_loc']
    outcat_name_pix = para['mgm']['outcat_name_loc']
    origin_name = para['mgm']['origin_name']
    mask_name = para['mgm']['mask_name']
    save_path = para['mgm']['save_path']
    thresh_num = para['mgm']['thresh_num']
    save_png = para['mgm']['save_png']
    para_fun = Param(delta_min=delta_min, gradmin=gradmin, v_min=v_min, noise_times=noise_times, rms_times=rms_times)
    LocalDensityClustering_main.LDC_main(data_file, para_fun, save_folder, split=split, save_loc=save_loc)
    if para['detect']['use_mgm'] is True:
        main_fit_gauss_3d.MGM_main(outcat_name_pix, origin_name, mask_name, save_path, thresh_num, save_png)


if __name__ == '__main__':
    main_fun()

import os.path

import pandas as pd
from spectral_cube import SpectralCube
from tools.show_clumps import make_plot_wcs_1
from DensityClust.localDenClust2 import LocalDensityCluster as LDC
from DensityClust.localDenClust2 import DetectResult, Data

split_list = [[0, 150, 0, 150], [0, 150, 90, 271], [0, 150, 210, 361],
              [90, 241, 0, 150], [90, 241, 90, 271], [90, 241, 210, 361]]

def split_cube_lxy(data_path, split_list=split_list):
    if split_list is None:
        split_list = split_list
    data_cube = SpectralCube.read(data_path)
    sub_cube_path_list = []
    for i, item in enumerate(split_list):
        sub_cude = data_cube[:, item[0]: item[1], item[2]: item[3]]
        sub_cube_path = data_path.replace('.fits', '_%02d.fits' % i)
        if not os.path.exists(sub_cube_path):
            sub_cude.write(sub_cube_path)
        sub_cube_path_list.append(sub_cube_path)
    return sub_cube_path_list


def get_outcat_i(outcat, i):
    outcat_split = [[0, 120, 0, 120], [0, 120, 30, 150], [0, 120, 30, 150],
                    [30, 150, 0, 120], [30, 150, 30, 150], [30, 150, 30, 150]]
    [ cen2_min, cen2_max, cen1_min, cen1_max] = outcat_split[i]
    aa = outcat.loc[outcat['Cen1'] > cen1_min]
    aa = aa.loc[outcat['Cen1'] <= cen1_max]

    aa = aa.loc[outcat['Cen2'] > cen2_min]
    loc_outcat = aa.loc[outcat['Cen2'] <= cen2_max]
    return loc_outcat


def get_outcat_wcs_all():
    outcat_wcs_all = pd.DataFrame([])
    for i in range(6):
        # i = 1
        outcat_i_path = r'test_data/synthetic/synthetic_model_0000_%02d/LDC_auto_outcat.csv' % i
        outcat_i = pd.read_csv(outcat_i_path, sep='\t')
        loc_outcat_i = get_outcat_i(outcat_i, i)

        origin_data_name = r'test_data/synthetic/synthetic_model_0000_%02d.fits' % i
        data = Data(origin_data_name)
        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(loc_outcat_i)
        outcat_wcs_all = pd.concat([outcat_wcs_all, outcat_wcs], axis=0)
        data_wcs = ldc.data.wcs
        data_cube = ldc.data.data_cube
        make_plot_wcs_1(outcat_wcs, data_wcs, data_cube)

if __name__ == '__main__':
    data_path = r'test_data/synthetic/synthetic_model_0000.fits'
    # sub_cube_path_list = split_cube_lxy(data_path, split_list=split_list)
    outcat_wcs_all = pd.DataFrame([])
    for i in range(6):
        # i = 1
        outcat_i = pd.read_csv(r'test_data/synthetic/synthetic_model_0000_%02d/LDC_auto_outcat.csv' % i, sep='\t')
        loc_outcat_i = get_outcat_i(outcat_i, i)

        origin_data_name = r'test_data/synthetic/synthetic_model_0000_%02d.fits' % i
        data = Data(origin_data_name)
        ldc = LDC(data=data, para=None)
        outcat_wcs = ldc.change_pix2world(loc_outcat_i)
        outcat_wcs_all = pd.concat([outcat_wcs_all, outcat_wcs], axis=0)
        data_wcs = ldc.data.wcs
        data_cube = ldc.data.data_cube
        make_plot_wcs_1(outcat_wcs, data_wcs, data_cube)

    data = Data(r'test_data/synthetic/synthetic_model_0000.fits')
    ldc = LDC(data=data, para=None)
    data_wcs = ldc.data.wcs
    data_cube = ldc.data.data_cube
    make_plot_wcs_1(outcat_wcs_all, data_wcs, data_cube)
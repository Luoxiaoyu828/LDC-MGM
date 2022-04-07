import matplotlib.pyplot as plt
import astropy.io.fits as fits
import bm4d
from DensityClust.locatDenClust2 import Data, Param, LocalDensityCluster
from t_match import match_6_ as match


def test(data_name):
    data = Data(data_path=data_name)
    data_cube = data.data_cube
    import bm4d
    denoised_image = bm4d.bm4d(data_cube, sigma_psd=0.23)
    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    ax1.imshow(denoised_image.sum(0))

    ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    ax2.imshow(data_cube.sum(0))
    plt.show()

    dd_img = data_cube - denoised_image
    print(dd_img.std())
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im0 = ax.imshow(dd_img.sum(0))
    pos = ax.get_position()
    pad = 0.01
    width = 0.02
    axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

    ax.set_title('rms = %.3f' % dd_img.std())
    cbar = fig.colorbar(im0, cax=axes1)
    plt.show()

    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    ax1.imshow(denoised_image[33, ...])

    ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    ax2.imshow(data_cube[33, ...])
    plt.show()

    dd_img = data_cube[33, ...] - denoised_image[33, ...]
    print(dd_img.std())
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im0 = ax.imshow(dd_img)
    pos = ax.get_position()
    pad = 0.01
    width = 0.02
    axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

    ax.set_title('rms = %.3f' % dd_img.std())
    cbar = fig.colorbar(im0, cax=axes1)
    plt.show()

    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(denoised_image[:, 82, 142], label='denoise_image')
    ax1.plot(data_cube[:, 82, 142], label='M16_image')
    plt.legend()
    plt.show()
    # ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    # ax2.plot(data_cube[:,120,180])
    plt.show()


def main():
    # data_name = r'F:\LDC_python\detection\R2_data\data_9\data_9\0175+000\0175+000_L.fits'
    # data_name = r'F:\LDC_python\detection\R2_data\data_9\0185-005\0185-005_L.fits'
    # data_name = r'F:\LDC_python\detection\test_data\2d_simulated_clump\gaussian_out_360.fits'
    # data_name = r'F:\LDC_python\detection\test_data\high_density_clump_data\s_out_000.fits'
    # data_name = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_025\out\gaussian_out_000.fits'
    data_name = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_010\out\gaussian_out_003.fits'
    # data_name = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_010\model\gaussian_model_001.fits'
    # data_name = r'F:\LDC_python\detection\test_data\M16 data\hdu0_mosaic_L_3D.fits'
    data = Data(data_path=data_name)
    para = Param(rms=0.23, dc=0.6)

    data.summary()
    para.summary()
    ldc = LocalDensityCluster(data=data, para=para)
    ldc.result.log()
    ldc.result.make_plot_wcs_1()
    ldc.result.save_result('mask.fits', 'out.fits')
    ldc.result.save_outcat_wcs('outcat_wcs.txt')
    ldc.result.save_outcat('outcat_record.txt')

    outcat_s = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_010\outcat\gaussian_outcat_003.txt'
    from t_match import match_6_ as match

    match.match_simu_detect(simulated_outcat_path=outcat_s, detected_outcat_path='outcat_record.txt',
                            match_save_path='data/match_result')

    import DensityClust.make_plot as mpl

    mpl.make_plot(outcat_s, ldc.data.data_cube)
    mpl.make_plot('outcat_record.txt', ldc.data.data_cube)
    data_denoised = bm4d.bm4d(data.data_cube, sigma_psd=0.23)
    mpl.make_plot('outcat_record.txt', data_denoised)

    model = fits.getdata(r'F:\LDC_python\detection\test_data_zhou_again\n_clump_010\model\gaussian_model_002.fits')
    dd = ldc.data.data_cube - model
    print(dd.std())

    dd = ldc.data.data_cube - data_denoised
    print(dd.std())

    ldc.data.data_cube = model
    ldc.detect()


def test1():
    for item in range(2,10,1):
        data_name = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_100\out\gaussian_out_%03d.fits' % item
        outcat_s = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_100\outcat\gaussian_outcat_%03d.txt'% item
        data = Data(data_path=data_name)
        para = Param(rms=0.23, dc=0.6)
        ldc = LocalDensityCluster(data=data, para=para)
        # ldc.result.log()
        ldc.result.save_result('mask.fits', 'out.fits')
        ldc.result.save_outcat_wcs('outcat_wcs.txt')
        ldc.result.save_outcat('outcat_record.txt')
        match.match_simu_detect(simulated_outcat_path=outcat_s, detected_outcat_path='outcat_record.txt',
                                match_save_path='data/match_result')

    for item in range(2,10,1):
        data_name = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_100\out\gaussian_out_%03d.fits' % item
        outcat_s = r'F:\LDC_python\detection\test_data_zhou_again\n_clump_100\outcat\gaussian_outcat_%03d.txt'% item
        data = Data(data_path=data_name)
        para = Param(rms_times=3)
        ldc = LocalDensityCluster(data=data, para=para)
        # ldc.result.log()
        ldc.result.save_result('mask.fits', 'out.fits')
        ldc.result.save_outcat_wcs('outcat_wcs.txt')
        ldc.result.save_outcat('outcat_record.txt')
        match.match_simu_detect(simulated_outcat_path=outcat_s, detected_outcat_path='outcat_record.txt',
                                match_save_path='data/match_result')

        outcat_name_old = r'F:\LDC_python\detection\test_data\M16 data\hdu0_mosaic_L_3D_1.15_4_0.01_27_0.6\LDC_outcat.txt'
        outcat_name = r'F:\DensityClust_distribution_class\DensityClust\m16_denoised\LDC_outcat.txt'
        outcat_name = 'outcat_record.txt'

        match.match_simu_detect(simulated_outcat_path=outcat_name_old, detected_outcat_path=outcat_name,
                                match_save_path='data/match_result111')


if __name__ == '__main__':
    # data_name = r'F:\DensityClust_distribution_class\data\0175-010_L.fits'
    data_name = r'F:\LDC_python\detection\R2_data\data_9\data_9\0175+000\0175+000_L.fits'
    # data_name = r'F:\LDC_python\detection\test_data\M16 data\hdu0_mosaic_L_3D.fits'
    data = Data(data_name)
    para = Param(rms_times=5)
    # para.set_para_dc(dc=[1.2, 0.6, 0.6])
    ldc = LocalDensityCluster(data=data, para=para)
    ldc.detect()
    ldc.save_detect_log('ehriohg.txt')
    print('*' * 30)
    # ldc.result.log()
    # outcat_edge = ldc.touch_edge(ldc.result.outcat_record)
    # ldc.result.outcat_record = outcat_edge
    # outcat_detect = r'1.15_4_0.46_auto_3_12_27_0.01_touch_LDC_outcat.txt'
    # ldc.result.save_outcat(outcat_detect)
    # #
    # from DensityClust import make_plot
    # make_plot.make_plot(outcat_detect, ldc.data.data_cube)
    # match.match_simu_detect(r'/data/result/1.15_4_0.46_auto_27_0.01_touch_LDC_outcat.txt',
    #                         r'/data/result/1.15_4_0.46_auto_3_12_27_0.01_touch_LDC_outcat.txt',
    #                         'fse')


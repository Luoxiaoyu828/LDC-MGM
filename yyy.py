from DensityClust.LocalDensityClustering_main import LDC_main
from t_match.match_6_ import match_simu_detect as msd
from fit_clump_function.main_fit_gauss_3d import MGM_main


if __name__ == '__main__':

    data_name = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000.fits'
    save_folder = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000_5sigma'
    LDC_main(data_name)

    # sop = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_outcat_0000.csv'
    # dop = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000_5sigma\LDC_auto_outcat.csv'
    # msp = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000_5sigma\Match'
    # msd(simulated_outcat_path=sop, detected_outcat_path=dop, match_save_path=msp)
    #
    # outcat_name = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000_5sigma\LDC_auto_outcat.csv'
    # origin_name = data_name
    # mask_name = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000_5sigma\LDC_auto_mask.fits'
    # save_path = r'F:\Parameter_reduction\LDC\0170+010_L\test_localDen\R2_synthetic\synthetic_model_0000_5sigma\MGM'
    #
    # MGM_main(outcat_name, origin_name, mask_name, save_path)


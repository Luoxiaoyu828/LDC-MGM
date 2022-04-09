from scipy.ndimage import gaussian_filter
import astropy.io.fits as fits
from tools.show_clumps import display_data


if __name__ == '__main__':
    data_path = r'F:\Parameter_reduction\LDC\0170+010_L\simulate_data\gaussian_model_002.fits'
    # data_path = r'F:\Parameter_reduction\LDC\MWISP013.896+03.254+15.834.fits'
    data = fits.getdata(data_path)
    display_data(data)
    sigma_ = 0.4
    fs = 1.05
    for _ in range(15):

        sigma_1 = sigma_ * fs
        data_1 = gaussian_filter(data, sigma=sigma_) - gaussian_filter(data, sigma=sigma_1)
        print(data_1.max())
        display_data(data_1)
        sigma_ = sigma_1

import os
import astropy.io.fits as fits
import tqdm
from astropy import wcs
from astropy.coordinates import SkyCoord
from skimage import filters
import numpy as np
from skimage import measure
from scipy import ndimage
import pandas as pd
import time
from scipy.spatial import KDTree as kdt
import matplotlib.pyplot as plt
from DensityClust.clustring_subfunc import \
    get_xyz, setdiff_nd, my_print
from DensityClust.clustring_subfunc import get_feature_loc_threshing, get_feature_loc, get_feature, \
    get_feature_threshing, assignation, divide_boundary_by_grad
from Generate.fits_header import Header


class Data:
    def __init__(self, data_path=''):
        self.data_path = data_path
        self.wcs = None
        self.rms = -1
        self.data_cube = None
        self.shape = None
        self.exist_file = False
        self.state = False
        self.n_dim = None
        self.file_type = None
        self.data_header = None
        self.data_inf = None
        self.read_file()
        self.calc_background_rms()
        self.get_wcs()

    def read_file(self):
        if os.path.exists(self.data_path):
            self.exist_file = True
            self.file_type = self.data_path.split('.')[-1]
            if self.file_type == 'fits':
                data_cube = fits.getdata(self.data_path)
                data_cube[np.isnan(data_cube)] = 0  # 去掉NaN
                self.data_cube = data_cube
                self.shape = self.data_cube.shape
                self.n_dim = self.data_cube.ndim
                self.data_header = fits.getheader(self.data_path)
                self.state = True
            else:
                print('data read error!')
        else:
            print('the file not exists!')

    def calc_background_rms(self):
        """
         This functions finds an estimate of the RMS noise in the supplied data data_cube.
        :return: bkgrms_value
        """
        # sigma_clip = SigmaClip(sigma=3.0)
        # bkgrms = StdBackgroundRMS(sigma_clip)
        # data = self.data_cube
        # bkgrms_value = bkgrms.calc_background_rms(data)
        # self.rms_ = bkgrms_value
        if self.exist_file:
            data_header = self.data_header
            keys = data_header.keys()
            key = [k for k in keys]
            if 'RMS' in key:
                self.rms = data_header['RMS']
                print('the rms of cell is %.4f\n' % data_header['RMS'])
            else:
                data_rms_path = self.data_path.replace('.fits', '_rms.fits')
                if os.path.exists(data_rms_path):
                    data_rms = fits.getdata(data_rms_path)
                    data_rms[np.isnan(data_rms)] = 0  # 去掉NaN
                    self.rms = np.median(data_rms)
                    print('The data header not have rms, and the rms is the median of the file:%s.' % data_rms_path)
                    print('The rms of cell is %.4f\n' % self.rms)
                else:
                    self.rms = 0.23256
                    print('the data header not have rms, and the rms of data is set 0.23.\n')

    def get_wcs(self):
        """
        得到wcs信息
        :return:
        data_wcs
        """
        if self.exist_file:
            data_header = self.data_header
            keys = data_header.keys()
            try:
                key = [k for k in keys if k.endswith('4')]
                [data_header.remove(k) for k in key]
                data_header.remove('VELREF')
            except KeyError:
                pass

            data_wcs = wcs.WCS(data_header)

            if data_wcs.axis_type_names[0] in ['GLON', 'GLAT'] and data_wcs.axis_type_names[0] in ['GLON', 'GLAT']:
                self.wcs = data_wcs
            else:
                # sigma_clip = SigmaClip(sigma=3.0)
                # bkgrms = StdBackgroundRMS(sigma_clip)
                # bkgrms_value = bkgrms.calc_background_rms(self.data_cube)
                header = Header(self.data_cube.ndim, self.data_cube.shape, self.rms)
                self.data_header = header.write_header()
                data_wcs = wcs.WCS(self.data_header)
                self.wcs = data_wcs

    def summary(self):
        print('=' * 30)
        data_file = 'data file: \n%s' % self.data_path
        data_rms = 'the rms of data: %.5f' % self.rms
        print(data_file)
        print(data_rms)
        if self.n_dim == 3:
            data_shape = 'data shape: [%d %d %d]' % self.data_cube.shape
            print(data_shape)
        elif self.n_dim == 2:
            data_shape = 'data shape: [%d %d]' % self.data_cube.shap
            print(data_shape)
        else:
            data_shape = ''
            print(data_shape)
        print('=' * 30)


class Param:
    """
        para.rho_min: Minimum density
        para.delta_min: Minimum delta
        para.v_min: Minimum volume
        para.noise: The noise level of the data, used for data truncation calculation
        para.dc: Standard deviation of Gaussian filtering
    """

    def __init__(self, delta_min=4, gradmin=0.01, v_min=27, noise_times=2, rms_times=3, thread_num=8, dc=None):
        self.noise = None
        self.rho_min = None
        self.v_min = v_min
        self.gradmin = gradmin
        self.delta_min = delta_min
        self.touch = True
        self.para_inf = None
        self.thread_num = thread_num

        self.noise_times = noise_times
        self.rms_times = rms_times
        self.dc = dc

    def set_rms_by_data(self, data):
        if data.state and data.rms is not None:
            self.rho_min = data.rms * self.rms_times
            self.noise = data.rms * self.noise_times
        else:
            raise ValueError('rms is not exists!')

    def set_para(self, paras_set):
        self.rho_min = paras_set['rho_min']
        self.v_min = paras_set['v_min']
        self.gradmin = paras_set['gradmin']
        self.delta_min = paras_set['delta_min']
        self.noise = paras_set['noise']

    def summary(self):
        table_title = ['rho_min[%.1f*rms]' % self.rms_times, 'delta_min[4]', 'v_min[27]', 'gradmin[0.01]',
                       'noise[%.1f*rms]' % self.noise_times, 'dc']
        para = np.array([[self.rho_min, self.delta_min, self.v_min, self.gradmin, self.noise, self.dc]])
        para_pd = pd.DataFrame(para, columns=table_title)
        print('=' * 30)
        print(para_pd)
        print('=' * 30)


class DetectResult:
    def __init__(self):
        self.out = None
        self.mask = None
        self.outcat = None
        self.outcat_wcs = None
        self.loc_outcat = None
        self.loc_outcat_wcs = None
        self.data = None
        self.para = None
        self.detect_num = [0, 0, 0]
        self.calculate_time = [0, 0]

    def save_outcat(self, outcat_name, loc):
        """
        # 保存LDC检测的直接结果，即单位为像素
        :param loc: 是否保存局部核表
            1：保存局部核表
            0：保存去除接触边界的核表
        :param outcat_name: 核表的路径
        :return:
        """
        if loc == 1:
            outcat = self.loc_outcat
        else:
            outcat = self.outcat
        outcat_colums = outcat.shape[1]
        if outcat_colums == 10:
            # 2d result
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
            dataframe = pd.DataFrame(outcat, columns=table_title)
            dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 13:
            # 3d result
            dataframe = outcat.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                      'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 11:
            # fitting 2d data result
            fit_outcat = outcat
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                           'Sum', 'Volume']
            dataframe = pd.DataFrame(fit_outcat, columns=table_title)
            dataframe = dataframe.round(
                {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
                 'Sum': 3, 'Volume': 3})
        else:
            print('outcat_record columns is %d' % outcat_colums)
            return

        dataframe.to_csv(outcat_name, sep='\t', index=False)

    def save_outcat_wcs(self, outcat_wcs_name, loc):
        """
        # 保存LDC检测的直接结果，即单位为像素
        :param outcat_wcs_name: 核表名字
        :param loc: 是否保存局部核表
            1：保存局部核表
            0：保存去除接触边界的核表
        :return:
        """
        if loc == 1:
            outcat_wcs = self.loc_outcat_wcs
        else:
            outcat_wcs = self.outcat_wcs

        outcat_colums = outcat_wcs.shape[1]
        if outcat_colums == 10:
            # 2d result
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
            dataframe = pd.DataFrame(outcat_wcs, columns=table_title)
            dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 13:
            # 3d result
            dataframe = outcat_wcs.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                          'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 11:
            # fitting 2d data result
            fit_outcat = outcat_wcs
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                           'Sum', 'Volume']
            dataframe = pd.DataFrame(fit_outcat, columns=table_title)
            dataframe = dataframe.round(
                {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
                 'Sum': 3, 'Volume': 3})
        else:
            print('outcat_record columns is %d' % outcat_colums)
            return

        dataframe.to_csv(outcat_wcs_name, sep='\t', index=False)

    def save_mask(self, mask_name):
        mask = self.mask
        if mask is not None:
            if os.path.isfile(mask_name):
                os.remove(mask_name)
                fits.writeto(mask_name, mask)
            else:
                fits.writeto(mask_name, mask)
        else:
            print('mask is None!')

    def save_out(self, out_name):
        out = self.out
        if out is not None:
            if os.path.isfile(out_name):
                os.remove(out_name)
                fits.writeto(out_name, self.out)
            else:
                fits.writeto(out_name, self.out)
        else:
            print('out is None!')

    def make_plot_wcs_1(self, fig_name=''):
        """
        在积分图上绘制检测结果
        当没有检测到云核时，只画积分图
        """
        markersize = 2
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['xtick.color'] = 'red'
        plt.rcParams['ytick.color'] = 'red'
        # data_name = r'R2_data\data_9\0180-005\0180-005_L.fits'
        # fits_path = data_name.replace('.fits', '')
        # title = fits_path.split('\\')[-1]
        # fig_name = os.path.join(fits_path, title + '.png')

        outcat_wcs = self.outcat_wcs
        data_wcs = self.data.wcs
        data_cube = self.data.data_cube

        fig = plt.figure(figsize=(10, 8.5), dpi=100)

        axes0 = fig.add_axes([0.15, 0.1, 0.7, 0.82], projection=data_wcs.celestial)
        axes0.set_xticks([])
        axes0.set_yticks([])
        if self.data.n_dim == 3:
            im0 = axes0.imshow(data_cube.sum(axis=0))
        else:
            im0 = axes0.imshow(data_cube)
        if outcat_wcs.values.shape[0] > 0:
            outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Cen1'].values, b=outcat_wcs['Cen2'].values,
                                    unit="deg")
            axes0.plot_coord(outcat_wcs_c, 'r*', markersize=markersize)

        axes0.set_xlabel("Galactic Longitude", fontsize=12)
        axes0.set_ylabel("Galactic Latitude", fontsize=12)
        # axes0.set_title(title, fontsize=12)
        pos = axes0.get_position()
        pad = 0.01
        width = 0.02
        axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

        cbar = fig.colorbar(im0, cax=axes1)
        cbar.set_label('K m s${}^{-1}$')
        if fig_name == '':
            plt.show()
        else:
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close(fig=fig)

    def log(self):
        outcat = self.outcat
        self.para.summary()
        self.data.summary()

        print('%10s' % 'Result' + '=' * 30)
        print('The number of clumps: %d' % outcat.shape[0])
        print('=' * 30)


class LocalDensityCluster:

    def __init__(self, data, para):
        # 参数初始化

        self.rho = None
        self.INN = None
        self.grad = None
        self.kdt_xx = None
        self.ND_num = None
        self.kd_tree_loc = None
        self.data = data
        self.para = para
        # self.para.set_rms_by_data(data)

        self.result = DetectResult()
        self.xx = None
        self.delta = None
        self.IndNearNeigh = None
        self.Gradient = None
        self.vosbe = False

        maxed = 0
        ND = 1
        for item in self.data.shape:
            maxed += item ** 2
            ND *= item

        self.maxed = maxed ** 0.5
        self.ND = ND

    def kc_coord(self, point_ii_xy, r):
        """
        :param point_ii_xy: 当前点坐标(x,y,z)
        :param r: 2 * r + 1
        :return:
        返回delta_ii_xy点r邻域的点坐标

        : xm: size_x
        : ym: size_y
        : zm: size_z
        """

        n_dim = self.data.n_dim
        if n_dim == 3:
            # xm, ym, zm = self.size_z, self.size_y, self.size_x
            zm, ym, xm = self.data.shape
            it = point_ii_xy[0]
            jt = point_ii_xy[1]
            kt = point_ii_xy[2]

            xyz_min = np.array([[1, it - r], [1, jt - r], [1, kt - r]])
            xyz_min = xyz_min.max(axis=1)

            xyz_max = np.array([[xm, it + r], [ym, jt + r], [zm, kt + r]])
            xyz_max = xyz_max.min(axis=1)

            x_arange = np.arange(xyz_min[0], xyz_max[0] + 1)
            y_arange = np.arange(xyz_min[1], xyz_max[1] + 1)
            v_arange = np.arange(xyz_min[2], xyz_max[2] + 1)

            [p_k, p_i, p_j] = np.meshgrid(x_arange, y_arange, v_arange, indexing='ij')
            Index_value = np.column_stack([p_k.flatten(), p_i.flatten(), p_j.flatten()])
            Index_value = setdiff_nd(Index_value, np.array([point_ii_xy]))

            ordrho_jj = np.matmul(Index_value - 1, np.array([[1], [xm], [ym * xm]]))
            ordrho_jj.reshape([1, ordrho_jj.shape[0]])

        else:
            """
            kc_coord_2d(point_ii_xy, xm, ym, r):
            bt = kc_coord_2d(point_ii_xy, size_y, size_x, k)
            size_x, size_y = data.shape
            :param point_ii_xy: 当前点坐标(x,y)
            :param xm: size_x
            :param ym: size_y
            :param r: 2 * r + 1
            :return:
            返回point_ii_xy点r邻域的点坐标

            """
            ym, xm = self.data.shape
            it = point_ii_xy[0]
            jt = point_ii_xy[1]

            xyz_min = np.array([[1, it - r], [1, jt - r]])
            xyz_min = xyz_min.max(axis=1)

            xyz_max = np.array([[xm, it + r], [ym, jt + r]])
            xyz_max = xyz_max.min(axis=1)

            x_arrange = np.arange(xyz_min[0], xyz_max[0] + 1)
            y_arrange = np.arange(xyz_min[1], xyz_max[1] + 1)

            [p_k, p_i] = np.meshgrid(x_arrange, y_arrange, indexing='ij')
            Index_value = np.column_stack([p_k.flatten(), p_i.flatten()])
            Index_value = setdiff_nd(Index_value, np.array([point_ii_xy]))

            ordrho_jj = np.matmul(Index_value - 1, np.array([[1], [xm]]))
            ordrho_jj.reshape([1, ordrho_jj.shape[0]])

        return ordrho_jj[:, 0], Index_value

    def kc_coord_new(self, ordrho_ii, r):
        """
        :param point_ii_xy: 当前点坐标(x,y,z)
        :param r: 2 * r + 1
        :return:
        返回delta_ii_xy点r邻域的点坐标

        : xm: size_x
        : ym: size_y
        : zm: size_z
        """
        point_ii_xy = self.kdt_xx.data[ordrho_ii]
        n_dim = self.data.n_dim
        idex1 = self.kdt_xx.query_ball_point(point_ii_xy, r=r * np.sqrt(n_dim) + 0.0001, workers=4)
        idex1.remove(ordrho_ii)  # 删除本身
        idex1 = np.array(idex1, np.int32)
        bt1 = self.kdt_xx.data[idex1]

        bb = np.abs(bt1 - point_ii_xy).max(1)
        d_idx = np.where(bb <= r)[0]

        bt1 = bt1[d_idx, :]
        idex1 = idex1[d_idx]
        if n_dim == 3:
            aa = np.lexsort((bt1[:, 2], bt1[:, 1], bt1[:, 0]))
            # 按照x>y>z的优先级排序
        elif n_dim == 2:
            aa = np.lexsort((bt1[:, 1], bt1[:, 0]))
            # 按照x>y>z的优先级排序
        bt1 = bt1[aa, :]
        idex1 = idex1[aa]
        return point_ii_xy, idex1, bt1

    def esti_rho(self, save_esti=True):
        esti_file = self.data.data_path.replace('.fits', '_esti.fits')
        if os.path.exists(esti_file):
            print('find the rho file.')
            data_esmit_rr = fits.getdata(esti_file)
        else:
            data = self.data.data_cube
            n_dim = self.data.n_dim
            dc_ = np.arange(0.3, 0.9, 0.01)
            dc_len = dc_.shape[0]
            data_esmit = np.zeros((data.shape + dc_.shape), np.float32)
            for i, dc in enumerate(dc_):
                data_esmit[..., i] = filters.gaussian(data, dc)
            data_esmit_mean = data_esmit.mean(axis=n_dim)
            data_esmit_mean_repeat = np.expand_dims(data_esmit_mean, n_dim).repeat(dc_len, axis=n_dim)
            d_data_esmi = np.abs(data_esmit - data_esmit_mean_repeat)

            min_indx = d_data_esmi.argmin(axis=n_dim)
            min_indx_ = min_indx.flatten()
            one_hot = np.eye(dc_len, dtype=np.float32)[min_indx_]
            data_esti = np.reshape(one_hot, data_esmit.shape)
            data_esmit_rr = (data_esti * data_esmit).sum(axis=n_dim)
            if save_esti:
                fits.writeto(esti_file, data_esmit_rr)
        return data_esmit_rr

    def build_kd_tree(self):
        aa = self.xx[self.rho > self.para.noise]
        self.ND_num = aa.shape[0]
        self.kd_tree_loc = kdt(aa)
        self.kdt_xx = kdt(self.xx)

    def detect(self):
        thread_num = self.para.thread_num
        t0_ = time.time()
        delta_min = self.para.delta_min
        data = self.data.data_cube
        self.xx = get_xyz(data)  # xx: 3D data coordinates  坐标原点是 1

        # 密度估计
        if self.para.dc is None:
            data_estim = self.esti_rho()
        else:
            data_estim = filters.gaussian(data, self.para.dc)

        self.rho = data_estim.flatten()
        self.build_kd_tree()
        rho_Ind = np.argsort(-self.rho)
        rho_up = self.rho[self.rho > self.para.noise]
        rho_Ind_loc = np.argsort(-rho_up)
        idx_rho_loc_glob = np.where(self.rho > self.para.noise)[0]
        # delta 记录距离，
        # INN 记录：两个密度点的联系 % index of nearest neighbor with higher density
        # INN_, delta_, grad_ = get_feature(self.rho, rho_Ind, self.kdt_xx, self.ND_num, self.para.delta_min)
        # INN, delta, grad = get_feature_threshing(self.rho, rho_Ind, self.kdt_xx, self.ND_num,
        #                                                      self.para.delta_min, thresh_num=40)
        if thread_num == 1:
            self.INN, self.delta, self.grad = get_feature_loc(rho_up, rho_Ind_loc, idx_rho_loc_glob, self.kd_tree_loc,
                                                              self.para.delta_min, self.ND_num, self.ND)
        else:
            print(time.ctime())
            # self.INN, self.delta, self.grad = get_feature_loc_threshing(rho_up, rho_Ind_loc, idx_rho_loc_glob,
            #                                                             self.kd_tree_loc, self.para.delta_min,
            #                                                             self.ND_num, self.ND, thresh_num=thread_num)
            self.INN, self.delta, self.grad = get_feature_threshing(self.rho, rho_Ind, self.kdt_xx, self.ND_num,
                                                                    self.para.delta_min, thresh_num=thread_num)
            print(time.ctime())
        t1_ = time.time()
        d_t = t1_ - t0_
        my_print(' ' * 10 + 'delata, rho and Gradient are calculated, using %.2f seconds.' % d_t, self.vosbe)
        self.result.calculate_time[0] = d_t

        t0_ = time.time()
        clusterInd = assignation(self.rho, self.delta, delta_min, self.para.rho_min, self.para.v_min, rho_Ind, self.INN)
        clumpInd = divide_boundary_by_grad(clusterInd, self.rho, self.grad, self.para.gradmin, self.para.v_min)
        mask = clumpInd.reshape(data.shape)
        loc_LDC_outcat, LDC_outcat = self.extra_record(mask)

        t1_ = time.time()
        d_t = t1_ - t0_
        self.result.calculate_time[1] = d_t
        my_print(' ' * 10 + 'Outcats are calculated, using %.2f seconds.' % d_t, self.vosbe)

        self.result.outcat = LDC_outcat
        self.result.outcat_wcs = self.change_pix2world(self.result.outcat)

        self.result.loc_outcat = loc_LDC_outcat
        self.result.loc_outcat_wcs = self.change_pix2world(self.result.loc_outcat)

        self.result.mask = mask
        self.result.data = self.data
        self.result.para = self.para

    def extra_record(self, label_data):
        print('props')
        dim = self.data.data_cube.ndim
        props = measure.regionprops_table(label_image=label_data, intensity_image=self.data.data_cube,
                                          properties=['weighted_centroid', 'area', 'mean_intensity',
                                                      'weighted_moments_central', 'max_intensity',
                                                      'image_intensity', 'bbox'])
        image_intensity = props['image_intensity']
        max_intensity = props['max_intensity']
        bbox = np.array([props['bbox-%d' % item] for item in range(dim)])
        Peak123 = np.zeros([dim, image_intensity.shape[0]])
        for ps_i, item in enumerate(max_intensity):
            max_idx = np.argwhere(image_intensity[ps_i] == item)[0]
            peak123 = max_idx + bbox[:, ps_i]
            Peak123[:, ps_i] = peak123.T

        clump_Cen = np.array([props['weighted_centroid-2'], props['weighted_centroid-1'], props['weighted_centroid-0']])
        size_3 = (props['weighted_moments_central-0-0-2'] / props['weighted_moments_central-0-0-0']) ** 0.5
        size_2 = (props['weighted_moments_central-0-2-0'] / props['weighted_moments_central-0-0-0']) ** 0.5
        size_1 = (props['weighted_moments_central-2-0-0'] / props['weighted_moments_central-0-0-0']) ** 0.5

        clump_Volume = props['area']
        clump_Peak = props['max_intensity']
        clump_Sum = clump_Volume * props['mean_intensity']
        clump_Size = 2.3548 * np.array([size_3, size_2, size_1])
        clump_Peak123 = Peak123 + 1
        clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
        id_clumps = np.array([item + 1 for item in range(label_data.max())], np.int32).T
        id_clumps = id_clumps.reshape([id_clumps.shape[0], 1])
        LDC_outcat = np.column_stack(
            (id_clumps, clump_Peak123.T, clump_Cen.T, clump_Size.T, clump_Peak.T, clump_Sum.T, clump_Volume.T))
        if dim == 3:
            table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak',
                           'Sum', 'Volume']
        else:
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']

        LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)
        self.result.detect_num[0] = LDC_outcat.shape[0]
        if self.para.touch:
            LDC_outcat = self.touch_edge(LDC_outcat)
            self.result.detect_num[1] = LDC_outcat.shape[0]

        loc_LDC_outcat = self.get_outcat_local(LDC_outcat)
        self.result.detect_num[2] = loc_LDC_outcat.shape[0]
        return loc_LDC_outcat, LDC_outcat

    def extra_outcat(self, rho_Ind):
        rho = self.rho
        deltamin = self.para.delta_min
        data = self.data.data_cube
        dim = self.data.n_dim

        # Initialize the return result: mask and out
        mask = np.zeros_like(data, dtype=np.int64)
        my_print('Second step: calculating Outcats.' + '-' * 30, self.vosbe)
        # 根据密度和距离来确定类中心
        clusterInd = -1 * np.ones(self.ND + 1)
        clust_index = np.intersect1d(np.where(rho > self.para.rho_min), np.where(self.delta > deltamin))

        clust_num = len(clust_index)
        # icl是用来记录第i个类中心在xx中的索引值
        icl = clust_index
        n_clump = 0
        for ii in clust_index:
            n_clump += 1
            clusterInd[ii] = n_clump
        # assignation 将其他非类中心分配到离它最近的类中心中去
        # clusterInd = -1 表示该点不是类的中心点，属于其他点，等待被分配到某个类中去
        # 类的中心点的梯度Gradient被指定为 - 1

        for i in range(self.ND):
            ordrho_i = rho_Ind[i]
            if clusterInd[ordrho_i] == -1:  # not centroid
                clusterInd[ordrho_i] = clusterInd[self.IndNearNeigh[ordrho_i]]
            else:
                self.Gradient[ordrho_i] = -1  # 将类中心点的梯度设置为-1

        clump_volume = np.zeros(n_clump)
        for i in range(n_clump):
            clump_volume[i] = np.where(clusterInd == (i + 1))[0].shape[0]
        # centInd [类中心点在xx坐标下的索引值，类中心在centInd的索引值: 代表类别编号]
        centInd = []
        for i, item in enumerate(clump_volume):
            if item >= self.para.v_min:
                centInd.append([icl[i], i])
        centInd = np.array(centInd, np.int64)

        # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
        n_clump = centInd.shape[0]
        clump_sum, clump_volume, clump_peak = np.zeros([n_clump, 1]), np.zeros([n_clump, 1]), np.zeros([n_clump, 1])
        clump_Cen, clump_size = np.zeros([n_clump, dim]), np.zeros([n_clump, dim])
        clump_Peak = np.zeros([n_clump, dim], np.int64)
        clump_ii = 0
        if dim == 3:
            for item_cent in tqdm.tqdm(centInd):
                rho_cluster_i = np.zeros(self.ND, np.float32)
                index_cluster_i = np.where(clusterInd == (item_cent[1] + 1))[0]  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
                clump_rho = rho[index_cluster_i]
                rho_max_min = clump_rho.max() - clump_rho.min()
                Gradient_ = self.Gradient.copy()
                grad_clump_i = Gradient_ / rho_max_min
                mask_grad = np.where(grad_clump_i > self.para.gradmin)[0]
                index_cc = np.intersect1d(mask_grad, index_cluster_i)
                rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
                rho_cc_mean = rho[index_cc].mean()
                index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
                index_cluster_rho = np.union1d(index_cc, index_cc_rho)

                # cl_i_index_xx = self.xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
                # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
                cl_i_index = np.zeros_like(self.rho, np.int32)
                cl_i_index[index_cluster_rho] = 1
                cl_i = cl_i_index.reshape(self.data.shape)
                # cl_i = np.zeros(data.shape, np.int64)
                # for j, item in enumerate(cl_i_index_xx):
                #     cl_i[item[2], item[1], item[0]] = 1

                # 形态学处理
                L = ndimage.binary_fill_holes(cl_i).astype(np.int32)
                L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
                # STATS = measure.regionprops(L)

                props = measure.regionprops_table(label_image=L, intensity_image=data,
                                                  properties=['weighted_centroid', 'area', 'mean_intensity',
                                                              'weighted_moments_central'])

                Ar = props['mean_intensity'] * props['area']
                ind = np.where(Ar == Ar.max())[0]
                L[L != (ind[0] + 1)] = 0
                cl_i = L / (ind[0] + 1)

                clustNum = props['area'][ind[0]]
                weighted_centroid = np.array(
                    [props['weighted_centroid-2'], props['weighted_centroid-1'], props['weighted_centroid-0']])
                if clustNum > self.para.v_min:

                    clump_Cen[clump_ii, :] = weighted_centroid[:, ind[0]]
                    clump_volume[clump_ii, 0] = clustNum
                    clump_sum[clump_ii, 0] = Ar[ind[0]]

                    size_3 = (props['weighted_moments_central-0-0-2'] / props['weighted_moments_central-0-0-0']) ** 0.5
                    size_2 = (props['weighted_moments_central-0-2-0'] / props['weighted_moments_central-0-0-0']) ** 0.5
                    size_1 = (props['weighted_moments_central-2-0-0'] / props['weighted_moments_central-0-0-0']) ** 0.5
                    # x_i = coords - clump_Cen[clump_ii, :]
                    # clump_size[clump_ii, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                    #                                            - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
                    clump_size[clump_ii, :] = 2.3548 * np.array([size_3[ind[0]], size_2[ind[0]], size_1[ind[0]]])

                    clump_i = data * cl_i

                    mask = mask + cl_i * (clump_ii + 1)
                    clump_peak[clump_ii, 0] = clump_i.max()
                    clump_Peak[clump_ii, [2, 1, 0]] = np.argwhere(clump_i == clump_i.max())[0]
                    clump_ii += 1
                else:
                    pass
        else:
            for i, item_cent in enumerate(centInd):  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
                rho_cluster_i = np.zeros(self.ND)
                index_cluster_i = np.where(clusterInd == (item_cent[1] + 1))[0]  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
                clump_rho = rho[index_cluster_i]
                rho_max_min = clump_rho.max() - clump_rho.min()
                Gradient_ = self.Gradient.copy()
                grad_clump_i = Gradient_ / rho_max_min
                mask_grad = np.where(grad_clump_i > self.para.gradmin)[0]
                index_cc = np.intersect1d(mask_grad, index_cluster_i)
                rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
                rho_cc_mean = rho[index_cc].mean()
                index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
                index_cluster_rho = np.union1d(index_cc, index_cc_rho)

                cl_i_index_xx = self.xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
                # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
                # clustNum = cl_i_index_xx.shape[0]

                cl_i = np.zeros(data.shape, np.int64)
                index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
                index_clust_rho = np.union1d(index_cc, index_cc_rho)

                cl_i_index_xx = self.xx[index_clust_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
                # clustInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
                for j, item in enumerate(cl_i_index_xx):
                    cl_i[item[1], item[0]] = 1
                # 形态学处理
                L = ndimage.binary_fill_holes(cl_i).astype(np.int64)
                L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
                STATS = measure.regionprops(L)

                Ar_sum = []
                for region in STATS:
                    coords = region.coords  # 经过验证，坐标原点为0
                    coords = coords[:, [1, 0]]
                    temp = 0
                    for j, item in enumerate(coords):
                        temp += data[item[1], item[0]]
                    Ar_sum.append(temp)
                Ar = np.array(Ar_sum)
                ind = np.where(Ar == Ar.max())[0]
                L[L != ind[0] + 1] = 0
                cl_i = L / (ind[0] + 1)
                coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标

                clustNum = coords.shape[0]

                if clustNum > self.para.v_min:
                    coords = coords[:, [1, 0]]
                    clump_i_ = np.zeros(coords.shape[0])
                    for j, item in enumerate(coords):
                        clump_i_[j] = data[item[1], item[0]]

                    clustsum = sum(clump_i_) + 0.0001  # 加一个0.0001 防止分母为0
                    clump_Cen[clump_ii, :] = np.matmul(clump_i_, coords) / clustsum
                    clump_volume[clump_ii, 0] = clustNum
                    clump_sum[clump_ii, 0] = clustsum

                    x_i = coords - clump_Cen[clump_ii, :]
                    clump_size[clump_ii, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                                               - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
                    clump_i = data * cl_i
                    # out = out + clump_i
                    mask = mask + cl_i * (clump_ii + 1)
                    clump_peak[clump_ii, 0] = clump_i.max()
                    clump_Peak[clump_ii, [1, 0]] = np.argwhere(clump_i == clump_i.max())[0]
                    clump_ii += 1
                else:
                    pass

        clump_Peak = clump_Peak + 1
        clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
        id_clumps = np.array([item + 1 for item in range(n_clump)], np.int64).T
        id_clumps = id_clumps.reshape([n_clump, 1])

        LDC_outcat = np.column_stack(
            (id_clumps, clump_Peak, clump_Cen, clump_size, clump_peak, clump_sum, clump_volume))
        LDC_outcat = LDC_outcat[:clump_ii, :]
        if dim == 3:
            table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak',
                           'Sum', 'Volume']
        else:
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']

        LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)
        self.result.detect_num[0] = LDC_outcat.shape[0]
        if self.para.touch:
            LDC_outcat = self.touch_edge(LDC_outcat)
            self.result.detect_num[1] = LDC_outcat.shape[0]

        loc_LDC_outcat = self.get_outcat_local(LDC_outcat)
        self.result.detect_num[2] = loc_LDC_outcat.shape[0]
        return loc_LDC_outcat, LDC_outcat, mask

    def change_pix2world(self, outcat):
        """
        将算法检测的结果(像素单位)转换到天空坐标系上去
        :return:
        outcat_wcs
        ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume']
        -->3d

         ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
         -->2d
        """
        # outcat_record = self.result.outcat_record
        table_title = outcat.keys()
        if outcat is None:
            return None
        else:
            data_wcs = self.data.wcs
            if 'Cen3' not in table_title:
                # 2d result
                peak1, peak2 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], 1)
                cen1, cen2 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1)
                size1, size2 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30])

                clump_Peak = np.column_stack([peak1, peak2])
                clump_Cen = np.column_stack([cen1, cen2])
                clustSize = np.column_stack([size1, size2])
                clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

                id_clumps = []  # MWSIP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
                for item_l, item_b in zip(cen1, cen2):
                    str_l = 'MWSIP' + ('%.03f' % item_l).rjust(7, '0')
                    if item_b < 0:
                        str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    else:
                        str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    id_clumps.append(str_l + str_b)
                id_clumps = np.array(id_clumps)

            elif 'Cen3' in table_title:
                # 3d result
                peak1, peak2, peak3 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], outcat['Peak3'], 1)
                clump_Peak = np.column_stack([peak1, peak2, peak3 / 1000])
                cen1, cen2, cen3 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1)
                size1, size2, size3 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30, outcat['Size3'] * 0.166])

                clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])
                clump_Cen = np.column_stack([cen1, cen2, cen3 / 1000])
                clustSize = np.column_stack([size1, size2, size3])
                id_clumps = []  # MWISP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
                for item_l, item_b, item_v in zip(cen1, cen2, cen3 / 1000):
                    str_l = 'MWISP' + ('%.03f' % item_l).rjust(7, '0')
                    if item_b < 0:
                        str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    else:
                        str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    if item_v < 0:
                        str_v = '-' + ('%.03f' % abs(item_v)).rjust(6, '0')
                    else:
                        str_v = '+' + ('%.03f' % abs(item_v)).rjust(6, '0')
                    id_clumps.append(str_l + str_b + str_v)
                id_clumps = np.array(id_clumps)
            else:
                print('outcat_record columns name are: ' % table_title)
                return None

            outcat_wcs = np.column_stack(
                (id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))
            outcat_wcs = pd.DataFrame(outcat_wcs, columns=table_title)
            return outcat_wcs

    def touch_edge(self, outcat):
        """
        判断云核是否到达边界
        :param outcat:
        :return:
        """
        # data_name = r'F:\DensityClust_distribution_class\DensityClust\m16_denoised.fits'
        # outcat_name = r'F:\DensityClust_distribution_class\DensityClust\m16_denoised\LDC_outcat.txt'
        # data = fits.getdata(data_name)
        # outcat_record = pd.read_csv(outcat_name, sep='\t')
        if self.data.n_dim == 3:
            [size_x, size_y, size_v] = self.data.data_cube.shape
            indx = []

            # condition 1: 峰值到达边界-->接触边界
            for item_peak, item_size in zip(['Peak1', 'Peak2', 'Peak3'], [size_v, size_y, size_x]):
                indx.append(np.where(outcat[item_peak] == item_size)[0])
                indx.append(np.where(outcat[item_peak] == 1)[0])

            # condition 2: 中心位置加减2倍sigma超出数据块边界-->接触边界
            for item_cen, item_size in zip([['Cen1', 'Size1'], ['Cen2', 'Size2'], ['Cen3', 'Size3']],
                                           [size_v, size_y, size_x]):
                indx.append(np.where((outcat[item_cen[0]] + 2 / 2.3548 * outcat[item_cen[1]]) > item_size)[0])
                indx.append(np.where((outcat[item_cen[0]] - 2 / 2.3548 * outcat[item_cen[1]]) < 1)[0])

            inde_all = []
            for item_ in indx:
                [inde_all.append(item) for item in item_]

            inde_all = np.array(list(set(inde_all)))
        else:
            [size_x, size_y] = self.data.data_cube.shape
            indx = []

            for item_peak, item_size in zip(['Peak1', 'Peak2'], [size_y, size_x]):
                indx.append(np.where(outcat[item_peak] == item_size)[0])
                indx.append(np.where(outcat[item_peak] == 1)[0])

            for item_cen, item_size in zip([['Cen1', 'Size1'], ['Cen2', 'Size2']],
                                           [size_y, size_x]):
                indx.append(np.where((outcat[item_cen[0]] + 2 / 2.3548 * outcat[item_cen[1]]) > item_size)[0])
                indx.append(np.where((outcat[item_cen[0]] - 2 / 2.3548 * outcat[item_cen[1]]) < 1)[0])

            inde_all = []
            for item_ in indx:
                [inde_all.append(item) for item in item_]

            inde_all = np.array(list(set(inde_all)))
        if inde_all.shape[0] > 0:
            outcat = outcat.drop(outcat.index[inde_all])
        else:
            outcat = outcat
        return outcat

    def get_outcat_local(self, outcat):
        """
        返回局部区域的检测结果：
        原始图为120*120  局部区域为30-->90, 30-->90 左开右闭
        :param outcat: LDC outcat_record
        :return:
        """
        size_x, size_y, size_v = self.data.data_cube.shape
        # outcat_record = pd.read_csv(txt_name, sep='\t')
        cen1_min = 30
        cen1_max = size_v - 30 - 1
        cen2_min = 30
        cen2_max = size_y - 30 - 1
        cen3_min = 60
        cen3_max = size_x - 60 - 1
        aa = outcat.loc[outcat['Cen1'] > cen1_min]
        aa = aa.loc[outcat['Cen1'] <= cen1_max]

        aa = aa.loc[outcat['Cen3'] > cen3_min]
        aa = aa.loc[outcat['Cen3'] <= cen3_max]

        aa = aa.loc[outcat['Cen2'] > cen2_min]
        loc_outcat = aa.loc[outcat['Cen2'] <= cen2_max]
        return loc_outcat

    def get_para_inf(self):
        table_title = ['rho_min[%.1f*rms]' % self.para.rms_times, 'delta_min[4]', 'v_min[27]', 'gradmin[0.01]',
                       'noise[%.1f*rms]' % self.para.noise_times, 'dc']
        para = [self.para.rho_min, self.para.delta_min, self.para.v_min, self.para.gradmin, self.para.noise,
                self.para.dc]
        para_inf = []
        for item_title, item_value in zip(table_title, para):
            if item_value is None:
                para_inf.append(item_title + ' = %s\n' % 'None')
            else:
                para_inf.append(item_title + ' = %.3f\n' % item_value)

        return para_inf

    def get_data_inf(self):
        print('=' * 30)
        data_file = 'data file: %s\n' % self.data.data_path
        data_rms = 'the rms of data: %.5f\n' % self.data.rms
        if self.data.n_dim == 3:
            data_shape = 'data shape: [%d %d %d]\n' % self.data.data_cube.shape
        elif self.data.n_dim == 2:
            data_shape = 'data shape: [%d %d]\n' % self.data.data_cube.shape
        else:
            data_shape = ''
        data_inf = [data_file, data_rms, data_shape]
        return data_inf

    def get_detect_inf(self):
        first_num = self.result.detect_num[0]
        second_num = self.result.detect_num[1]
        loc_num = self.result.detect_num[2]
        detect_inf = []
        d_num = first_num - second_num
        if d_num == 0:
            detect_inf.append('%d clumps are rejected.\n' % d_num)
        else:
            detect_inf.append('%d clumps are rejected because they touched the border.\n' % d_num)

        detect_inf.append('The number of clumps: %d\n' % second_num)
        detect_inf.append('The number of local region clumps: %d\n' % loc_num)
        detect_inf.append(
            'delata, rho and Gradient are calculated, using %.2f seconds.\n' % self.result.calculate_time[0])
        detect_inf.append('Outcats are calculated, using %.2f seconds.\n' % self.result.calculate_time[1])
        return detect_inf

    def save_detect_log(self, detect_log):

        data_inf = self.get_data_inf()
        para_inf = self.get_para_inf()
        detect_inf = self.get_detect_inf()
        f = open(detect_log, 'w')

        f.writelines('Data information\n')
        [f.writelines(item) for item in data_inf]
        f.writelines('=' * 20 + '\n\n')

        f.writelines('Algorithm parameter information\n')
        [f.writelines(item) for item in para_inf]
        f.writelines('=' * 20 + '\n\n')

        f.writelines('Detect result\n')
        [f.writelines(item) for item in detect_inf]

        f.close()


if __name__ == '__main__':
    pass

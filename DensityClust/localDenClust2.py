import os
import astropy.io.fits as fits
import astropy.units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
from scipy.spatial import KDTree as kdt
from skimage import filters, measure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tqdm

from DensityClust.clustring_subfunc import get_xyz, my_print, get_area_v_len, get_clump_angle
from DensityClust.clustring_subfunc import assignation, divide_boundary_by_grad
from Generate.fits_header import Header


"""
1.4.0 2022/11/04
"""


class Data:
    def __init__(self, data_path='', l_st_end=None, b_st_end=None, v_st_end=None,
                 save_folder=None):
        """
        处理数据的Data类，
        """
        self.data_path = data_path
        self.save_folder = save_folder
        self.wcs = None
        self.rms = None
        self.res = np.array([30, 30, 0.166])
        self.data_cube = None
        self.shape = None
        self.exist_file = False
        self.state = False
        self.n_dim = None
        self.file_type = None
        self.data_header = None
        self.data_inf = None
        self.read_file(v_st_end, l_st_end, b_st_end)
        self.get_wcs()

    def read_file(self, v_st_end, l_st_end, b_st_end):
        """
        根据给定的范围对原始数据进行切分，并把切出来的数据子块保存到self.save_folder路径下
        如果不指定范围，则不做任何处理
        其中：
            v_st_end表示速度的范围，单位必须是 km/s
            l_st_end表示经度的范围，单位必须是 degree
            b_st_end表示纬度的范围，单位必须是 degree
        """
        if not os.path.exists(self.data_path):
            raise FileExistsError('The file %s not exists!' % self.data_path)
        else:
            self.exist_file = True
            self.file_type = self.data_path.split('.')[-1]
            if self.file_type != 'fits':
                raise TypeError('Only support *.fits file!')
            else:
                self.state = True
                xlo = 'min'
                xhi = 'max'
                ylo = 'min'
                yhi = 'max'
                zlo = 'min'
                zhi = 'max'
                try:
                    # the case of 3d data (contain velocity axis)
                    data_cube = SpectralCube.read(self.data_path)
                    data_cube = data_cube.with_spectral_unit(u.km / u.s)
                except:
                    # the case of 2d data
                    data_array = fits.getdata(self.data_path)
                    data_cube = data_array.squeeze()  # 将空的维度去掉
                    data_cube[np.isnan(data_cube)] = 0  # 将NaN的标记为-1000
                    self.data_cube = data_cube

                    self.shape = self.data_cube.shape
                    self.n_dim = self.data_cube.ndim

                    data_header = fits.getheader(self.data_path)

                    keys = data_header.keys()
                    try:
                        key = [k for k in keys if k.endswith('3')]
                        [data_header.remove(k) for k in key]
                        data_header.remove('VELREF')
                    except KeyError:
                        pass

                    self.data_header = data_header
                    self.res = self.res[:self.n_dim]
                    return

                l_b_v_name = []
                b_unit = u.deg
                l_unit = u.deg
                v_unit = u.km / u.s
                if l_st_end is not None:

                    xlo = l_st_end[0] * l_unit
                    xhi = l_st_end[1] * l_unit

                    longitude_extrema = data_cube.longitude_extrema.to(l_unit).value

                    if l_st_end[0] < longitude_extrema[0]:
                        print('WARNING: Galactic longitude Lower Boundary Exceeded! Using min instead')
                        xlo = longitude_extrema[0] * l_unit
                    if l_st_end[1] > longitude_extrema[1]:
                        print('WARNING: Galactic longitude upper bound exceeded! Using max instead')
                        xhi = longitude_extrema[1] * l_unit
                    l_name = 'l_%.2f--%.2f' % (xlo.value, xhi.value)
                    l_b_v_name.append(l_name)

                if b_st_end is not None:

                    ylo = b_st_end[0] * b_unit
                    yhi = b_st_end[1] * b_unit

                    # g_info = [item for item in data_cube.header['HISTORY'] if item.startswith('B')]
                    # num_list = np.array([item for item in g_info[0].split(' ') if not item.isalpha()], np.float32)
                    latitude_extrema = data_cube.latitude_extrema.to(b_unit).value
                    if b_st_end[0] < latitude_extrema[0]:
                        print('WARNING: Galactic latitude Lower Boundary Exceeded! Using min instead')
                        ylo = latitude_extrema[0] * b_unit
                    if b_st_end[1] > latitude_extrema[1]:
                        print('WARNING: Galactic latitude upper bound exceeded! Using max instead')
                        yhi = latitude_extrema[1] * b_unit
                    b_name = 'b_%.2f--%.2f' % (ylo.value, yhi.value)
                    l_b_v_name.append(b_name)

                if v_st_end is not None:
                    zlo = v_st_end[0] * v_unit
                    zhi = v_st_end[1] * v_unit

                    spectral_extrema = data_cube.spectral_extrema.to(v_unit).value
                    if v_st_end[0] < spectral_extrema[0]:
                        print('WARNING: Velocity Lower Boundary Exceeded! Using min instead')
                        zlo = spectral_extrema[0] * v_unit
                    if v_st_end[1] > spectral_extrema[1]:
                        print('WARNING: Velocity upper bound exceeded! Using max instead')
                        zhi = spectral_extrema[1] * v_unit
                    v_name = 'v_%.1f--%.1f' % (zlo.value, zhi.value)
                    l_b_v_name.append(v_name)

                data_cube = data_cube.subcube(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi, rest_value=None)

                if l_st_end is None and b_st_end is None and v_st_end is None:
                    pass
                else:
                    l_b_v_name = '_' + '_'.join(l_b_v_name)
                    sub_cube_name = os.path.basename(self.data_path).replace('.fits', l_b_v_name[:-1] + '_cube.fits')
                    sub_cube_path = os.path.join(self.save_folder, sub_cube_name)
                    if os.path.exists(sub_cube_path):
                        os.remove(sub_cube_path)
                    data_cube.write(sub_cube_path)
                    self.data_path = sub_cube_path

                self.data_header = data_cube.header
                data_cube = data_cube._data

                data_cube = data_cube.squeeze()  # 将空的维度去掉
                data_cube[np.isnan(data_cube)] = 0  # 将NaN的标记为-1000
                self.data_cube = data_cube

                self.shape = self.data_cube.shape
                self.n_dim = self.data_cube.ndim
                self.res = self.res[:self.n_dim]

    def calc_background_rms(self, rms_key='RMS', data_rms_path='', rms=0.23):
        """
         This functions finds an estimate of the RMS noise in the supplied data data_cube.
        :return: bkgrms_value
        """
        data_header = self.data_header
        keys = data_header.keys()
        key = [k for k in keys]
        if rms_key in key:
            # the case of header.
            self.rms = data_header[rms_key]
            print('the rms of cell is %.4f\n' % self.rms)
        elif os.path.exists(data_rms_path):
            # the case of rms file
            data_rms = fits.getdata(data_rms_path)
            data_rms[np.isnan(data_rms)] = 0  # 去掉NaN
            self.rms = np.median(data_rms)
            print('The data header not have rms, and the rms is the median of the file:%s.' % data_rms_path)
            print('The rms of cell is %.4f\n' % self.rms)
        else:
            # the case of rms value.
            self.rms = rms
            print('the data header not have rms, and the rms of data is set %.4f\n' % rms)

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
            data_header['NAXIS'] = self.n_dim
            self.data_header = data_header
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

    def get_res(self):
        res = np.zeros(self.n_dim, np.float32)
        if self.n_dim == 3:
            CUNIT1 = self.data_header['CUNIT1']
            CUNIT2 = self.data_header['CUNIT2']
            CUNIT3 = self.data_header['CUNIT3']

            CDELT1 = self.data_header['CDELT1']
            CDELT2 = self.data_header['CDELT2']
            CDELT3 = self.data_header['CDELT3']
            if CUNIT1 == 'deg':
                res[0] = (CDELT1 * u.deg).to(u.arcsec).value
            if CUNIT2 == 'deg':
                res[1] = (CDELT2 * u.deg).to(u.arcsec).value
            if CUNIT3 == 'm/s':
                res[2] = (CDELT3 * u.m / u.s).to(u.km / u.s).value
            else:
                res[2] = (CDELT1 * u.m / u.s).value
        elif self.n_dim == 2:

            CUNIT1 = self.data_header['CUNIT1']
            CUNIT2 = self.data_header['CUNIT2']

            CDELT1 = self.data_header['CDELT1']
            CDELT2 = self.data_header['CDELT2']

            if CUNIT1 == 'deg':
                res[0] = (CDELT1 * u.deg).to(u.arcsec).value
            if CUNIT2 == 'deg':
                res[1] = (CDELT2 * u.deg).to(u.arcsec).value
        else:
            raise ValueError('The data n_dim must be 2 or 3.')

        self.res = np.abs(res)


class Param:
    def __init__(self, delta_min=4, gradmin=0.01, v_min=None, noise_times=2, rms_times=5, dc=None, v_st_end=None,
                 l_st_end=None, b_st_end=None, res=None, save_loc=False, rms_key='', data_rms_path='', rms=None):
        """
        para.rho_min:
            rho_min represents the minimum peak intensity value of a candidate clump. The value can be supplied as
            a mutliple of the RMS noise using the syntax "rms_times * RMS"
            rho is the local density of a point_i.
            if rho of point_i >= rho_min,
                the point_i may be one center of cluster (also considering the delta of points)
            Note: The higher the value, the more likely it is to merge the two candidate clumps,
            and the smaller the value, the more likely it is to separate the clump.

        para.delta_min:
            delta_min represents the minimum distance between the centers of the two points, where delta is measured
            by computing the minimum distance between point_i and any other point with higher density.
            if delta of point_i >= delta_min,
                the point_i may be one center of cluster (also considering the rho of points)

        para.grad_min:
            grad_min is used to determine the region of a clump.

        para.thresh:
            The smallest significant peak height. Peaks which have a maximum data value less than
            this value are ignored. The value can be supplied as a mutliple of the RMS noise using
            the syntax "noise_times*RMS"

        para.v_min: [A_, B_]
            The lowest number of pixel which a clump can contain.
            If a candidate clump has fewer than this number of pixels, it will be ignored.
            The default value is [9, 3] pixels. A_ represents the minimum integral area in the space direction,
            and b represents the minimum span in the velocity direction.
            If it is 2-dimensional data, only A_ will be used. If it is 3-dimensional data, both A_ and B_ will be used.

        para.noise: The noise level of the data, used for data truncation calculation
        para.dc: Standard deviation of Gaussian filtering
        para.rms_key : The rms key of header.

        """
        if res is None:
            res = [30, 30, 0.166]
        if v_min is None:
            v_min = [25, 5]

        if rms_key.__len__() and data_rms_path.__len__() and rms is None:
            raise ValueError('the rms setting must be the one of: 1. rms key, 2. rms file, 3. rms value.')
        self.rms_key = rms_key
        self.data_rms_path = data_rms_path
        self.rms = rms

        self.noise = None
        self.rho_min = None
        self.res = res
        self.v_min = v_min
        self.gradmin = gradmin
        self.delta_min = delta_min
        self.distance_w = np.array([1, 1, 1], np.float32)
        self.rm_touch_edge = True
        self.save_loc = save_loc
        self.para_inf = None

        self.noise_times = noise_times
        self.rms_times = rms_times
        self.dc = dc
        self.v_st_end = v_st_end
        self.l_st_end = l_st_end
        self.b_st_end = b_st_end

    def calculate_w(self, data_ndim):
        """
        根据给定的空间、速度分辨率计算算法中距离参数的权重，以适应不同观测仪器得到的数据
        以 MWISP项目的数据分辨率[30, 30, 0.166]为基准
        """
        if isinstance(self.res, np.ndarray):
            self.res = np.array(self.res)
        self.res = self.res[: data_ndim]
        if data_ndim == 3:
            distance_w = self.res / np.array([30, 30, 0.166])
            # self.v_min[0] = self.v_min[0] / (distance_w[0] * distance_w[1])
            # self.v_min[-1] = self.v_min[-1] / distance_w[-1]

        elif data_ndim == 2:
            distance_w = self.res / np.array([30, 30])
            # self.v_min[0] = self.v_min[0] / distance_w[0] ** 2
            self.v_min[-1] = 1
        else:
            raise ValueError('The value of resolution must be spatial or velocity and the length of array at least 2.')
        # distance_w[distance_w <= 1] = 1
        self.distance_w = distance_w

    def set_params_by_data(self, data):
        if data.state and data.rms is not None:
            self.rho_min = data.rms * self.rms_times
            self.noise = data.rms * self.noise_times
        else:
            raise ValueError('rms is not exists!')

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
        outcat_colums = outcat.keys()
        if 'Cen3' not in outcat_colums:
            # 2d result

            dataframe = outcat.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3, 'Angle': 1})

        else:
            # 3d result
            dataframe = outcat.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                      'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3, 'Angle': 1})

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

        outcat_colums = outcat_wcs.keys()
        if 'Cen3' not in outcat_colums:
            # 2d result
            dataframe = outcat_wcs.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3, 'Angle': 1})

        else:
            # 3d result
            dataframe = outcat_wcs.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                          'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3, 'Angle': 1})

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

        self.kd_tree = None
        self.ND_num = None
        self.data = data
        self.para = para

        self.result = DetectResult()
        self.xx = None
        self.delta = None
        self.INN = None
        self.grad = None
        self.vosbe = False

        para.calculate_w(self.data.n_dim)

        maxed = 0
        ND = 1
        for item in self.data.shape:
            maxed += item ** 2
            ND *= item

        self.maxed = maxed ** 0.5
        self.ND = ND

    def kc_coord_new_(self, ordrho_ii, r):
        """
        :param point_ii_xy: 当前点坐标(x,y,z)
        :param r: 2 * r + 1
        :return:
        返回delta_ii_xy点r邻域(立方体)的点坐标
        """
        r_temp = np.zeros([1, 3])
        r_temp[0, :2] = r[0]
        r_temp[0, 2] = r[-1]
        point_ii_xy = self.kd_tree.data[ordrho_ii]
        rr = np.sqrt((r_temp ** 2).sum())
        n_dim = self.data.n_dim
        idex1 = self.kd_tree.query_ball_point(point_ii_xy, r=rr, workers=4)
        idex1.remove(ordrho_ii)  # 删除本身
        idex1 = np.array(idex1, np.int32)
        point_j_xy = self.kd_tree.data[idex1]

        # 将点限制在长方体里面
        bb = np.abs(point_j_xy - point_ii_xy)
        d_bb = (bb - r_temp) <= 0

        d_idx = np.where(d_bb.astype(np.int32).sum(1) == 3)[0]
        point_j_xy = point_j_xy[d_idx, :]

        idex1 = idex1[d_idx]
        if n_dim == 3:
            aa = np.lexsort((point_j_xy[:, 2], point_j_xy[:, 1], point_j_xy[:, 0]))
            # 按照x>y>z的优先级排序
        elif n_dim == 2:
            aa = np.lexsort((point_j_xy[:, 1], point_j_xy[:, 0]))
            # 按照x>y>z的优先级排序
        point_j_xy = point_j_xy[aa, :]
        idex1 = idex1[aa]
        return point_ii_xy, idex1, point_j_xy

    def kc_coord_new(self, ordrho_ii, r):
        """
        :param point_ii_xy: 当前点坐标(x,y,z)
        :param r: 2 * r + 1
        :return:
        返回delta_ii_xy点r邻域(立方体)的点坐标
        """
        point_ii_xy = self.kd_tree.data[ordrho_ii]
        n_dim = self.data.n_dim
        idex1 = self.kd_tree.query_ball_point(point_ii_xy, r=r * np.sqrt(n_dim) + 0.0001, workers=4)
        idex1.remove(ordrho_ii)  # 删除本身
        idex1 = np.array(idex1, np.int32)
        point_j_xy = self.kd_tree.data[idex1]

        # 将点限制在立方体里面
        bb = np.abs(point_j_xy - point_ii_xy).max(1)
        d_idx = np.where(bb <= r)[0]
        point_j_xy = point_j_xy[d_idx, :]

        idex1 = idex1[d_idx]
        if n_dim == 3:
            aa = np.lexsort((point_j_xy[:, 2], point_j_xy[:, 1], point_j_xy[:, 0]))
            # 按照x>y>z的优先级排序
        elif n_dim == 2:
            aa = np.lexsort((point_j_xy[:, 1], point_j_xy[:, 0]))
            # 按照x>y>z的优先级排序
        point_j_xy = point_j_xy[aa, :]
        idex1 = idex1[aa]

        d_points = (point_ii_xy - point_j_xy) * self.para.distance_w
        dist_ii_jj = np.sqrt((d_points ** 2).sum(1))  # 计算两点间的加权距离
        return dist_ii_jj, idex1

    def esti_rho(self, save_esti=False):
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

    def build_kd_tree(self, rho, xx):
        aa = xx[rho > self.para.noise]
        self.ND_num = aa.shape[0]
        self.kd_tree = kdt(xx)

    def detect(self):
        t0_ = time.time()
        delta_min = self.para.delta_min
        data = self.data.data_cube

        # delta 记录距离，
        # IndNearNeigh 记录：两个密度点的联系 % index of nearest neighbor with higher density
        self.delta = np.zeros(self.ND, np.float32)  # np.iinfo(np.int32).max-->2147483647-->1290**3
        self.INN = np.zeros(self.ND, np.int64)
        self.grad = np.zeros(self.ND, np.float32)

        k1 = 1  # 第1次计算点的邻域大小
        k2 = delta_min * self.para.distance_w.min()  # 第2次计算点的邻域大小
        xx = get_xyz(data)  # xx: 3D data coordinates  坐标原点是 1
        # 密度估计
        if self.para.dc is None:
            data_estim = self.esti_rho()
        else:
            data_estim = filters.gaussian(data, self.para.dc)

        rho = data_estim.flatten()
        self.build_kd_tree(rho, xx)
        rho_Ind = np.argsort(-rho)
        rho_sorted = rho[rho_Ind[: self.ND_num]]

        for ii in tqdm.tqdm(range(1, self.ND_num)):
            # 密度降序排序后，即密度第ii大的索引(在rho中)
            ordrho_ii = rho_Ind[ii]
            rho_ii = rho_sorted[ii]  # 第ii大的密度值

            delta_ordrho_ii = self.maxed
            get_value = True  # 判断是否需要在大循环中继续执行，默认需要，一旦在小循环中赋值成功，就不在大循环中运行

            dist_ii_jj, idex1 = self.kc_coord_new(ordrho_ii, k1)
            for ordrho_jj, dist_i_j in zip(idex1, dist_ii_jj):
                rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                gradient = (rho_jj - rho_ii) / dist_i_j
                if dist_i_j <= delta_ordrho_ii and gradient >= 0:
                    delta_ordrho_ii = dist_i_j
                    Gradient_ordrho_ii = gradient
                    IndNearNeigh_ordrho_ii = ordrho_jj
                    get_value = False

            if get_value:
                # 表明，在(2 * k1 + 1) * (2 * k1 + 1) * (2 * k1 + 1)的邻域中没有找到比该点高，距离最近的点，则在更大的邻域中搜索
                dist_ii_jj, idex1 = self.kc_coord_new(ordrho_ii, k2)
                for ordrho_jj, dist_i_j in zip(idex1, dist_ii_jj):
                    rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                    gradient = (rho_jj - rho_ii) / dist_i_j
                    if dist_i_j <= delta_ordrho_ii and gradient >= 0:
                        delta_ordrho_ii = dist_i_j
                        Gradient_ordrho_ii = gradient
                        IndNearNeigh_ordrho_ii = ordrho_jj
                        get_value = False

            if get_value:
                delta_ordrho_ii = k2 + 0.0001
                Gradient_ordrho_ii = -1
                IndNearNeigh_ordrho_ii = 0

            self.delta[ordrho_ii] = delta_ordrho_ii
            self.grad[ordrho_ii] = Gradient_ordrho_ii
            self.INN[ordrho_ii] = IndNearNeigh_ordrho_ii

        self.delta[rho_Ind[0]] = max(self.delta.max(), self.para.delta_min + 0.001)
        self.INN[rho_Ind[0]] = rho_Ind[0]
        my_print('First step: calculating rho, delta and Gradient.' + '-' * 20, vosbe_=self.vosbe)

        t1_ = time.time()
        d_t = t1_ - t0_
        my_print(' ' * 10 + 'delta, rho and Gradient are calculated, using %.2f seconds.' % d_t, self.vosbe)
        self.result.calculate_time[0] = d_t

        t0_ = time.time()
        clusterInd = assignation(rho, self.delta, delta_min, self.para.rho_min, rho_Ind, self.INN)
        mask = divide_boundary_by_grad(clusterInd, rho, self.grad, self.para.gradmin, self.data.shape)

        LDC_outcat, mask = self.extra_record(mask)

        t1_ = time.time()
        d_t = t1_ - t0_
        self.result.calculate_time[1] = d_t
        my_print(' ' * 10 + 'Outcats are calculated, using %.2f seconds.' % d_t, self.vosbe)

        self.result.outcat = LDC_outcat
        self.result.outcat_wcs = self.change_pix2world(self.result.outcat)

        self.result.mask = mask
        self.result.data = self.data
        self.result.para = self.para

    def extra_record(self, label_data):
        print('props')
        dim = self.data.data_cube.ndim
        if dim == 3:
            table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak',
                           'Sum', 'Volume', 'Angle']
        else:
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume', 'Angle']
        label_data_all = np.zeros_like(label_data, np.int32)
        if label_data.max() == 0:
            LDC_outcat = pd.DataFrame([], columns=table_title)
        else:
            props = measure.regionprops_table(label_image=label_data, intensity_image=self.data.data_cube,
                                              properties=['weighted_centroid', 'area', 'mean_intensity',
                                                          'weighted_moments_central', 'max_intensity',
                                                          'image_intensity', 'bbox'],
                                              extra_properties=(get_area_v_len, get_clump_angle))
            id_temp = 1
            area_v_idx = []
            props_bbox = np.array([props['bbox-%d' % bbox_i] for bbox_i in range(dim * 2)], np.int32)
            for props_i, props_item in enumerate(props['get_area_v_len']):
                g_l_area = props_item[0]
                v_len = props_item[1]
                label_image = props_item[2]
                props_bbox_i = props_bbox[:, props_i]
                if g_l_area >= self.para.v_min[0] and v_len >= self.para.v_min[1]:
                    label_temp = np.zeros_like(label_data_all, np.int32)
                    if dim == 3:
                        label_temp[props_bbox_i[0]: props_bbox_i[3], props_bbox_i[1]: props_bbox_i[4], props_bbox_i[2]: props_bbox_i[5]] = id_temp * label_image
                    else:
                        label_temp[props_bbox_i[0]: props_bbox_i[2], props_bbox_i[1]: props_bbox_i[3]] = id_temp * label_image

                    label_data_all += label_temp
                    id_temp += 1
                    area_v_idx.append(props_i)
            area_v_idx = np.array(area_v_idx, np.int32)

            clump_Angle = props['get_clump_angle-0'][area_v_idx]
            image_intensity = props['image_intensity'][area_v_idx]
            max_intensity = props['max_intensity'][area_v_idx]
            bbox = np.array([props['bbox-%d' % item][area_v_idx] for item in range(dim)])
            Peak123 = np.zeros([dim, area_v_idx.shape[0]])
            for ps_i, item in enumerate(max_intensity):
                max_idx = np.argwhere(image_intensity[ps_i] == item)[0]
                peak123 = max_idx + bbox[:, ps_i]
                Peak123[:, ps_i] = peak123.T
            if dim == 3:
                clump_Cen = np.array(
                    [props['weighted_centroid-2'][area_v_idx], props['weighted_centroid-1'][area_v_idx],
                     props['weighted_centroid-0'][area_v_idx]])
                size_3 = (props['weighted_moments_central-0-0-2'][area_v_idx] / props['weighted_moments_central-0-0-0'][area_v_idx]) ** 0.5
                size_2 = (props['weighted_moments_central-0-2-0'][area_v_idx] / props['weighted_moments_central-0-0-0'][area_v_idx]) ** 0.5
                size_1 = (props['weighted_moments_central-2-0-0'][area_v_idx] / props['weighted_moments_central-0-0-0'][area_v_idx]) ** 0.5
                clump_Size = 2.3548 * np.array([size_3, size_2, size_1])
            elif dim == 2:
                clump_Cen = np.array(
                    [props['weighted_centroid-1'][area_v_idx], props['weighted_centroid-0'][area_v_idx]])
                size_2 = (props['weighted_moments_central-0-2'][area_v_idx] / props['weighted_moments_central-0-0'][area_v_idx]) ** 0.5
                size_1 = (props['weighted_moments_central-2-0'][area_v_idx] / props['weighted_moments_central-0-0'][area_v_idx]) ** 0.5
                clump_Size = 2.3548 * np.array([size_2, size_1])
            else:
                clump_Cen = None
                clump_Size = None
                print('Only 2D and 3D are supported!')

            clump_Volume = props['area'][area_v_idx]
            clump_Peak = props['max_intensity'][area_v_idx]
            clump_Sum = clump_Volume * props['mean_intensity'][area_v_idx]
            clump_Peak123 = Peak123 + 1
            clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
            id_clumps = np.array([item + 1 for item in range(len(area_v_idx))], np.int32)
            id_clumps = id_clumps.reshape([id_clumps.shape[0], 1])

            LDC_outcat = np.column_stack((id_clumps, clump_Peak123.T[:, ::-1], clump_Cen.T, clump_Size.T, clump_Peak.T,
                                          clump_Sum.T, clump_Volume.T, clump_Angle.T))

            LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)
            self.result.detect_num[0] = LDC_outcat.shape[0]
            if self.para.rm_touch_edge:
                LDC_outcat, label_data_all = self.touch_edge(LDC_outcat, label_data_all)
                self.result.detect_num[1] = LDC_outcat.shape[0]

            if self.para.save_loc:
                loc_LDC_outcat = self.get_outcat_local(LDC_outcat)
                self.result.detect_num[2] = loc_LDC_outcat.shape[0]
                self.result.loc_outcat = loc_LDC_outcat
                self.result.loc_outcat_wcs = self.change_pix2world(loc_LDC_outcat)

        return LDC_outcat, label_data_all

    def change_pix2world(self, outcat, id_prefix='MWISP'):
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
                size1, size2 = np.array([outcat['Size1'], outcat['Size2']])

                clump_Peak = np.column_stack([peak1, peak2])
                clump_Cen = np.column_stack([cen1, cen2])
                clustSize = np.column_stack([size1, size2]) * self.para.res
                clustPeak, clustSum, clustVolume, clustAngle = np.array(
                    [outcat['Peak'], outcat['Sum'], outcat['Volume'], outcat['Angle']])

                id_clumps = []  # MWSIP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
                for item_l, item_b in zip(cen1, cen2):
                    str_l = id_prefix + ('%.03f' % item_l).rjust(7, '0')
                    if item_b < 0:
                        str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    else:
                        str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    id_clumps.append(str_l + str_b)
                id_clumps = np.array(id_clumps)

            elif 'Cen3' in table_title:
                # 3d result
                if abs(self.data.data_header['CDELT3']) > 10:
                    # case m/s  CUNIT3 = m/s
                    cdelt3 = 1000
                else:
                    # case km/s
                    cdelt3 = 1
                peak1, peak2, peak3 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], outcat['Peak3'], 1)
                clump_Peak = np.column_stack([peak1, peak2, peak3 / cdelt3])
                cen1, cen2, cen3 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1)
                size1, size2, size3 = np.array([outcat['Size1'], outcat['Size2'], outcat['Size3']])

                clustPeak, clustSum, clustVolume, clustAngle = np.array(
                    [outcat['Peak'], outcat['Sum'], outcat['Volume'], outcat['Angle']])
                clump_Cen = np.column_stack([cen1, cen2, cen3 / cdelt3])
                clustSize = np.column_stack([size1, size2, size3]) * self.para.res
                id_clumps = []  # MWISP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
                for item_l, item_b, item_v in zip(cen1, cen2, cen3 / cdelt3):
                    str_l = id_prefix + ('%.03f' % item_l).rjust(7, '0')
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
                (id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume, clustAngle))
            outcat_wcs = pd.DataFrame(outcat_wcs, columns=table_title)
            return outcat_wcs

    def touch_edge(self, outcat, label_data_all):
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
            label_data_all_ = label_data_all.copy()
        if inde_all.shape[0] > 0:
            outcat = outcat.drop(outcat.index[inde_all])
            for item_i in outcat.index[inde_all]:
                label_data_all_[label_data_all == item_i] = 0

        else:
            outcat = outcat
        return outcat, label_data_all_

    def get_outcat_local(self, outcat, cen1_edge=None, cen2_edge=None, cen3_edge=None):
        """
        返回局部区域的检测结果：
        原始图为120*120  局部区域为30-->90, 30-->90 左开右闭

        :param outcat: LDC outcat_record

        :return:
        """
        if cen3_edge is None:
            cen3_edge = [60, 60]
        if cen2_edge is None:
            cen2_edge = [30, 30]
        if cen1_edge is None:
            cen1_edge = [30, 30]
        if self.data.n_dim == 3:
            size_x, size_y, size_v = self.data.data_cube.shape
            # outcat_record = pd.read_csv(txt_name, sep='\t')
            cen1_min = cen1_edge[0]
            cen1_max = size_v - cen1_edge[1] - 1
            cen2_min = cen2_edge[0]
            cen2_max = size_y - cen2_edge[1] - 1
            cen3_min = cen3_edge[0]
            cen3_max = size_x - cen3_edge[1] - 1

            aa = outcat.loc[outcat['Cen1'] > cen1_min]
            aa = aa.loc[outcat['Cen1'] <= cen1_max]
            aa = aa.loc[outcat['Cen3'] > cen3_min]
            aa = aa.loc[outcat['Cen3'] <= cen3_max]

            aa = aa.loc[outcat['Cen2'] > cen2_min]
            loc_outcat = aa.loc[outcat['Cen2'] <= cen2_max]
        else:
            size_x, size_y = self.data.data_cube.shape
            cen1_min = cen1_edge[0]
            cen1_max = size_x - cen1_edge[1] - 1
            cen2_min = cen2_edge[0]
            cen2_max = size_y - cen2_edge[1] - 1

            aa = outcat.loc[outcat['Cen1'] > cen1_min]
            aa = aa.loc[outcat['Cen1'] <= cen1_max]
            aa = aa.loc[outcat['Cen2'] > cen2_min]
            loc_outcat = aa.loc[outcat['Cen2'] <= cen2_max]
        return loc_outcat

    def get_para_inf(self):
        table_title = ['rho_min[%.1f*rms]' % self.para.rms_times, 'delta_min=%d' % self.para.delta_min,
                       'pix_min[25, 5]=[%d, %d]' % (self.para.v_min[0], self.para.v_min[1]), 'gradmin[0.01]',
                       'noise[%.1f*rms]' % self.para.noise_times, 'dc']
        para = [self.para.rho_min, -1, -1, self.para.gradmin, self.para.noise,
                self.para.dc]
        para_inf = []
        for item_title, item_value in zip(table_title, para):
            if item_value is None:
                para_inf.append(item_title + ' = %s\n' % 'None')
            elif item_value == -1:
                para_inf.append(item_title + '\n')
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
        """
        整理检测过程中的参数日志
        """
        first_num = self.result.detect_num[0]   # 保留边界核的情况下，云核总数
        second_num = self.result.detect_num[1]   # 不保留边界核的情况下，云核总数
        detect_inf = []

        detect_inf.append('The number of clumps: %d\n' % second_num)
        if self.para.rm_touch_edge:
            d_num = first_num - second_num
            detect_inf.append('%d clumps are rejected because they touched the border.\n' % d_num)

        if self.para.save_loc:
            loc_num = self.result.detect_num[2]  # 局部核表下的云核总数
            detect_inf.append('The number of local region clumps: %d\n' % loc_num)
        detect_inf.append(
            'delta, rho and Gradient are calculated, using %.2f seconds.\n' % self.result.calculate_time[0])
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

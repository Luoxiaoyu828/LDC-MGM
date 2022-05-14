from astropy.io import fits
import datetime


class Header:
    def __init__(self, dim, size, rms, history_info=None, information=None):
        self.history_info = history_info
        self.dim = dim
        self.info = information
        self.keys_3d = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'DATAMIN', 'DATAMAX', 'BUNIT',
                        'CTYPE1', 'CRVAL1', 'CDELT1', 'CRPIX1', 'CROTA1', 'CTYPE2', 'CRVAL2', 'CDELT2', 'CRPIX2',
                        'CROTA2', 'CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CROTA3', 'EQUINOX', 'LINE', 'ALTRVAL',
                        'ALTRPIX', 'RESTFREQ', 'BMAJ', 'BMIN', 'BPA', 'ORIGIN', 'DATE', 'RMS', 'CUNIT3']
        self.keys_2d = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'DATAMIN', 'DATAMAX', 'BUNIT',
                        'CTYPE1', 'CRVAL1', 'CDELT1', 'CRPIX1', 'CROTA1', 'CTYPE2', 'CRVAL2', 'CDELT2', 'CRPIX2',
                        'CROTA2', 'EQUINOX', 'LINE', 'ALTRVAL',
                        'ALTRPIX', 'RESTFREQ', 'BMAJ', 'BMIN', 'BPA', 'ORIGIN', 'DATE', 'RMS', 'CUNIT3']
        if dim == 3:
            self.values = [(True,), (-64,), (3,), (size[0],), (size[1],), (size[2],), (-1003.975830078,), (338.0236816406,),
                              ('K (T_MB)',), ('GLON-CAR',), (16.75,), (-0.00833333333333,), (181,), (0.0,), ('GLAT-CAR',),
                              (0.75,), (0.00833333333333,), (91.0,), (0.0,), ('VELOCITY',), (15939.88032,), (166.04042,),
                              (1,), (0.0,), (0.0,), ('13CO(1-0)',), (0.0,), (10826.0,), (110201353000.0,),
                              (0.0152257818068,), (0.0152257818068,), (0.0,), ('GILDAS Consortium',),
                              (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),), (rms,), ('m/s',)]
        elif dim == 2:
            self.values = [(True,), (-64,), (3,), (size[0],), (size[1],), (-1003.975830078,), (338.0236816406,),
                              ('K (T_MB)',), ('GLON-CAR',), (16.75,), (-0.00833333333333,), (181,), (0.0,), ('GLAT-CAR',),
                              (0.75,), (0.00833333333333,), (91.0,), (0.0,), (0.0,), ('13CO(1-0)',), (0.0,), (10826.0,),
                              (110201353000.0,), (0.0152257818068,), (0.0152257818068,), (0.0,), ('GILDAS Consortium',),
                              (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),), (rms,), ('m/s',)]
        else:
            raise ValueError('only support 2-dimension and 3-dimension!')

    def write_header(self):
        header = fits.Header()
        if self.dim == 3:
            if self.info is not None:
                for key in self.info.keys():
                    idx = self.keys_3d.index(key)
                    self.values[idx] = (self.info[key],)
            for item1, item2 in zip(self.keys_3d, self.values):
                header[item1] = item2
        elif self.dim == 2:
            if self.info is not None:
                for key in self.info.keys():
                    idx = self.keys_2d.index(key)
                    self.values[idx] = (self.info[key],)
            for item1, item2 in zip(self.keys_2d, self.values):
                header[item1] = item2
        if self.history_info is not None:
            for key in self.history_info.keys():
                header.add_history(self.history_info[key])
        return header

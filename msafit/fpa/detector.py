import numpy as np
import os
from astropy.io import fits
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from copy import copy
from ..utils.array_utils import find_pixel, oversample_array

# Class to describe the NIRSpec detectors based on latest version of geometric model

class Detector:

    def __init__(self,oversampling=1,refdir=None):

        self.pix_size = 1.8e-5 # meter
        self.pix_per_shutter = 5    # approximate, as each pixel is ~0.1 arcsec and the pitch is on average ~0.5 arcsec
        if refdir is not None: detector_cood = fits.open(refdir+ f'/coordinates.fits')
        else: detector_cood = fits.open(os.path.expandvars('${msa_refdata}') + f'/detector/coordinates.fits')
        self.sca491 = detector_cood[1].data
        self.sca492 = detector_cood[2].data
        detector_cood.close()
        self.det_oversample = oversampling
        if self.det_oversample>1:
            self._oversample_detector()
        elif self.det_oversample<1:
            raise ValueError("Oversampling factor must be equal to or greater than 1")

    @classmethod
    def _read_trace_lib(cls,fwa,gwa,refdir):

        if refdir is not None:
           detector_trace = fits.open(refdir + f'/trace_lib_{fwa.upper()}_{gwa.upper()}.fits')
        else: detector_trace = fits.open(os.path.expandvars('${msa_refdata}') + f'/detector/trace_lib_{fwa.upper()}_{gwa.upper()}.fits')

        return detector_trace


    def _oversample_detector(self):

        self.sca491 = zoom(self.sca491,[1,self.det_oversample,self.det_oversample],order=1,mode='nearest')
        self.sca492 = zoom(self.sca492,[1,self.det_oversample,self.det_oversample],order=1,mode='nearest')




class DetectorCutout(Detector):


    def __init__(self,fwa,gwa,quadrant,shutter_i,shutter_j,N_shutter=3,source_shutter=0,oversampling=1):

        super().__init__(oversampling=1,refdir=None)    # we only oversample the cutout region if oversampling>1, not the full detector
        self.fwa = fwa
        self.gwa = gwa
        self.qd = quadrant
        self.N_shutter = N_shutter
        self.source_shutter = source_shutter
        self.cutout_oversample = oversampling
        self.s_i = shutter_i
        self.s_j = shutter_j - source_shutter        
        self._get_trace_func(refdir)



    def _get_trace_func(self,refdir=None):

        trace_lib = Detector._read_trace_lib(self.fwa,self.gwa,refdir=refdir)

        wl_in = trace_lib[5].data * 1e4
        trace_cube = trace_lib[int(self.qd)].data
        trace_x,trace_y,trace_theta = trace_cube[self.s_i-1,self.s_j-1,:,:]
        self.get_trace_x = interp1d(wl_in,trace_x,bounds_error=False,fill_value="extrapolate")
        self.get_trace_y = interp1d(wl_in,trace_y,bounds_error=False,fill_value="extrapolate")
        self.get_trace_theta = interp1d(wl_in,trace_theta,bounds_error=False,fill_value="extrapolate")
        trace_lib.close()

    def _find_aperture(self,xpix,ypix,pad_x,pad_y):

        Npix = int(self.det_oversample * self.sca491.shape[1])

        ylow_ext = (self.N_shutter/2 ) * self.pix_per_shutter # + self.source_shutter
        ypix_low = np.maximum(0, (ypix - int(np.floor(ylow_ext)) - pad_y))

        yup_ext = (self.N_shutter/2 ) * self.pix_per_shutter # - self.source_shutter
        ypix_up = np.minimum(self.sca491.shape[1], ( ypix + int(np.ceil(yup_ext)) + pad_y))

        xpix_low = np.maximum(0, xpix - pad_x)
        xpix_up = np.minimum(self.sca491.shape[1], xpix + pad_x)

        return xpix_low,xpix_up,ypix_low,ypix_up


    def _oversample_cutout(self,cutout_x,cutout_y):

        xgrid = oversample_array(cutout_x,self.cutout_oversample)
        ygrid = oversample_array(cutout_y,self.cutout_oversample)

        return xgrid,ygrid


    def make_cutout_wl(self,wl,pad_x=15,pad_y=10):


        # find the true (x,y) position on the detector [unit: meter]
        self.x_fpa, self.y_fpa = self.get_trace_x(wl), self.get_trace_y(wl)
        self.wl_fpa = wl
        # translate this to a pixel index and make a cutout
        if (self.x_fpa<np.min(self.sca491[0].T[-1])) and (self.x_fpa>np.max(self.sca491[0].T[0])):
            xpix,ypix = find_pixel(self.x_fpa, self.y_fpa,self.sca491[0],self.sca491[1])
            xlow,xup,ylow,yup = self._find_aperture(xpix,ypix,pad_x,pad_y)
            cutout_x = copy(self.sca491[0])[ylow-1:yup+1,xlow-1:xup+1]
            cutout_y = copy(self.sca491[1])[ylow-1:yup+1,xlow-1:xup+1]
            self.xgrid_491, self.ygrid_491 = self._oversample_cutout(cutout_x,cutout_y)
            self.xgrid_492, self.ygrid_492 = None, None

        elif (self.x_fpa>np.max(self.sca492[0].T[-1])) and  (self.x_fpa<np.min(self.sca492[0].T[0])):
            xpix,ypix = find_pixel(self.x_fpa, self.y_fpa,self.sca492[0],self.sca492[1])
            xlow,xup,ylow,yup = self._find_aperture(xpix,ypix,pad_x,pad_y)
            cutout_x = copy(self.sca492[0])[ylow-1:yup+1,xlow-1:xup+1]
            cutout_y = copy(self.sca492[1])[ylow-1:yup+1,xlow-1:xup+1]
            self.xgrid_492, self.ygrid_492 = self._oversample_cutout( np.rot90(cutout_x,2), np.rot90(cutout_y,2))
            self.xgrid_491, self.ygrid_491 = None, None

        # raise errors if not on the detector
        elif (self.x_fpa>np.min(self.sca491[0].T[-1])) and (self.x_fpa<np.max(self.sca492[0].T[-1])):
            raise ValueError("Congratulations! Your line falls inside the chip gap")
        else: raise ValueError("Wavelength falls outside of the detector area")




    def make_cutout_range(self,wl_range,pad_x=20,pad_y=5,pix_limits=None,**kwargs):

        self.xgrid_491, self.ygrid_491 = None, None
        self.xgrid_492, self.ygrid_492 = None, None

        self.x_fpa, self.y_fpa = self.get_trace_x(wl_range), self.get_trace_y(wl_range)

        # now check which of these fall on detector 1 and which on detector 2
        ind_491 = np.where((self.x_fpa<np.min(self.sca491[0].T[-1])) & (self.x_fpa>np.max(self.sca491[0].T[0])) )[0]
        ind_492 = np.where((self.x_fpa>np.max(self.sca492[0].T[-1])) & (self.x_fpa<np.min(self.sca492[0].T[0])) )[0]


        
        if len(ind_491)==0 and len(ind_492)==0:
            raise ValueError("Wavelength range outside of detector")

        if len(ind_491)>0:

            if pix_limits is None:
                x_491_min, x_491_max = self.x_fpa[ind_491][0], self.x_fpa[ind_491][-1]            
                y_491_min, y_491_max = self.y_fpa[ind_491][0], self.y_fpa[ind_491][-1]            
                xpix_491_min,ypix_491_min = find_pixel(x_491_min, y_491_min,self.sca491[0],self.sca491[1])
                xpix_491_max,ypix_491_max = find_pixel(x_491_max, y_491_max,self.sca491[0],self.sca491[1])

                xlow_491_min,xup_491_min,ylow_491_min,yup_491_min = self._find_aperture(xpix_491_min,ypix_491_min,pad_x,pad_y)
                xlow_491_max,xup_491_max,ylow_491_max,yup_491_max = self._find_aperture(xpix_491_max,ypix_491_max,pad_x,pad_y)

                xlow_491 = np.minimum(xlow_491_min,xlow_491_max)
                xup_491 = np.maximum(xup_491_min,xup_491_max)
                ylow_491 = np.minimum(ylow_491_min,ylow_491_max)
                yup_491 = np.maximum(yup_491_min,yup_491_max)
            else:
                ylow_491,yup_491,xlow_491,xup_491 = pix_limits[:4]

            ylow_491 = np.maximum(1,ylow_491)
            xlow_491 = np.maximum(1,xlow_491)                   
            cutout_x_491 = copy(self.sca491[0])[ylow_491-1:yup_491+1,xlow_491-1:xup_491+1]
            cutout_y_491 = copy(self.sca491[1])[ylow_491-1:yup_491+1,xlow_491-1:xup_491+1]                
            self.xgrid_491,self.ygrid_491 = self._oversample_cutout(cutout_x_491,cutout_y_491)

        if len(ind_492)>0:

            if pix_limits is None:            
                x_492_min, x_492_max = self.x_fpa[ind_492][0], self.x_fpa[ind_492][-1]            
                y_492_min, y_492_max = self.y_fpa[ind_492][0], self.y_fpa[ind_492][-1]            
                xpix_492_min,ypix_492_min = find_pixel(x_492_min, y_492_min,self.sca492[0],self.sca492[1])
                xpix_492_max,ypix_492_max = find_pixel(x_492_max, y_492_max,self.sca492[0],self.sca492[1])

                xlow_492_min,xup_492_min,ylow_492_min,yup_492_min = self._find_aperture(xpix_492_min,ypix_492_min,pad_x,pad_y)
                xlow_492_max,xup_492_max,ylow_492_max,yup_492_max = self._find_aperture(xpix_492_max,ypix_492_max,pad_x,pad_y)

                xlow_492 = np.minimum(xlow_492_min,xlow_492_max)
                xup_492 = np.maximum(xup_492_min,xup_492_max)
                ylow_492 = np.minimum(ylow_492_min,ylow_492_max)
                yup_492 = np.maximum(yup_492_min,yup_492_max)
            elif len(pix_limits)==4:
                ylow_492,yup_492,xlow_492,xup_492 = pix_limits[:4]
            else:                
                ylow_492,yup_492,xlow_492,xup_492 = pix_limits[4:]
            
            ylow_492 = np.maximum(1,ylow_492)
            xlow_492 = np.maximum(1,xlow_492)            
            cutout_x_492 = copy(self.sca492[0])[ylow_492-1:yup_492+1,xlow_492-1:xup_492+1]
            cutout_y_492 = copy(self.sca492[1])[ylow_492-1:yup_492+1,xlow_492-1:xup_492+1]                
            self.xgrid_492,self.ygrid_492 = self._oversample_cutout(np.rot90(cutout_x_492,2),np.rot90(cutout_y_492,2))

        ind_all = np.append(ind_491,ind_492)
        self.x_fpa = self.x_fpa[ind_all]
        self.y_fpa = self.y_fpa[ind_all]
        self.wl_fpa = wl_range[ind_all]













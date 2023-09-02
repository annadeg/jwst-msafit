import numpy as np
import os
from msafit.utils.array_utils import downsample_array,find_pixel
from msafit.utils.convolution import convolve_cube_psf
from msafit.fpa.detector import DetectorCutout
from msafit.fpa.psf import PSFLib
from scipy import ndimage
from astropy.convolution import convolve as kernel_convolve

__all__ = ["Spec2D"]

class Spec2D(DetectorCutout):

    def __init__(self,parameter_dict,return_full_detector=False,refdir=None,**kwargs):

        super().__init__(fwa=parameter_dict["instrument"]["filter"],
            gwa=parameter_dict["instrument"]["disperser"],
            quadrant=parameter_dict["geometry"]["quadrant"],
            shutter_i=parameter_dict["geometry"]["shutter_i"],
            shutter_j=parameter_dict["geometry"]["shutter_j"],
            N_shutter=int(parameter_dict["geometry"]["shutter_array"][-1]),
            source_shutter=parameter_dict["geometry"]["source_shutter"],
            oversampling=parameter_dict["instrument"]["psf_oversampling"],refdir=refdir)

        self.params = parameter_dict
        self._make_full_detector = return_full_detector

        if return_full_detector:
            self.make_cutout_range(parameter_dict["grid"]["wave_grid"],pad_x=2048,pad_y=2048,**kwargs)
        else:
            self.make_cutout_range(parameter_dict["grid"]["wave_grid"],**kwargs)

    def _get_psfgrid(self,filename=None):

        psfs = PSFLib(filename=fname)

        return psfs


    def _convolve_crosstalk(self,array,sca,refdir=None,**kwargs):

        if refdir is not None:
            kernel = np.loadtxt(refdir + f'kernel_sca{sca}.txt')
        else: kernel = np.loadtxt(os.path.expandvars('${msa_refdata}') + f'/detector/kernel_sca{sca}.txt')

        if array is not None: return kernel_convolve(array,kernel,boundary='fill',fill_value=0.)
        else: return None


    def make_spec2d(self,ModelCube,PSF,wave_thresh=2e2,downsampled=True,return_fluxes=False,
                    rotate_slit=True,add_crosstalk=True,verbose=False,**kwargs):

        if type(PSF)==str:
            PSF = self._get_psfgrid(filename=PSF)

        conv_cube = convolve_cube_psf(ModelCube,PSF,**kwargs)

        dim_wave = conv_cube.shape[0]

        try:
            self.spec_491 = np.zeros(self.xgrid_491.shape)
        except AttributeError: 
            self.spec_491 = None
            if verbose: print("No source on SCA491")
        try:
            self.spec_492 = np.zeros(self.xgrid_492.shape)
        except AttributeError: 
            self.spec_492 = None
            if verbose: print("No source on SCA492")

        for i in range(dim_wave):

            wl = ModelCube._wave[i]

            if np.min(np.fabs(self.wl_fpa-wl))>wave_thresh:
                if verbose: print(f"Skipping wavelength {wl:.0f}. \
                    Increase the wave_thresh if you want this included.\n"
                    + "PSF may be inaccurate or wavelength may fall off detector.")
                continue 


            im = conv_cube[i,:,:]
            dim_x = conv_cube.shape[2]
            dim_y = conv_cube.shape[1]
            x_fpa,y_fpa,theta_fpa = self.get_trace_x(wl), self.get_trace_y(wl), self.get_trace_theta(wl)


            if rotate_slit: rim = ndimage.rotate(im,theta_fpa+180.,order=1,reshape=False)
            else: rim = np.rot90(im,2)

            if x_fpa<0:
                x_pix,y_pix = find_pixel(x_fpa,y_fpa,self.xgrid_491,self.ygrid_491)
                #print(x_pix,self.xgrid_491.shape)
                y_min = np.maximum(0,y_pix-(dim_y//2))
                y_max = np.minimum(self.ygrid_491.shape[0],y_pix-(dim_y//2)+dim_y)
                x_min = np.maximum(0,x_pix-(dim_x//2))
                x_max = np.minimum(self.xgrid_491.shape[1],x_pix-(dim_x//2)+dim_x)    

                if y_min==0:
                    dy_low = rim.shape[0] - (y_max-y_min)
                else: dy_low = 0
                if y_max==self.ygrid_491.shape[0]:
                    dy_up = (y_max-y_min) - rim.shape[0]
                    if dy_up == 0: dy_up = None
                else: dy_up = None
                if x_min==0:
                    dx_low = rim.shape[1] - (x_max-x_min)
                else: dx_low = 0
                if x_max==self.xgrid_491.shape[1]:
                    dx_up = (x_max-x_min) - rim.shape[1]
                    if dx_up == 0: dx_up = None
                else: dx_up = None       

                self.spec_491[y_min:y_max, x_min:x_max] += rim[dy_low:dy_up,dx_low:dx_up]         
                   
            else:
            
                x_pix,y_pix = find_pixel(x_fpa,y_fpa,self.xgrid_492,self.ygrid_492)
                y_min = np.maximum(0,y_pix-(dim_y//2))
                y_max = np.minimum(self.ygrid_492.shape[0],y_pix-(dim_y//2)+dim_y)
                x_min = np.maximum(0,x_pix-(dim_x//2))
                x_max = np.minimum(self.xgrid_492.shape[1],x_pix-(dim_x//2)+dim_x)         
                if y_min==0:
                    dy_low = rim.shape[0] - (y_max-y_min)
                else: dy_low = 0
                if y_max==self.ygrid_492.shape[0]:
                    dy_up = (y_max-y_min) - rim.shape[0]
                    if dy_up == 0: dy_up = None
                else: dy_up = None
                if x_min==0:
                    dx_low = rim.shape[1] - (x_max-x_min)
                else: dx_low = 0
                if x_max==self.xgrid_492.shape[1]:
                    dx_up = (x_max-x_min)-rim.shape[1]
                    if dx_up == 0: dx_up = None
                else: dx_up = None       

                self.spec_492[y_min:y_max, x_min:x_max] += rim[dy_low:dy_up,dx_low:dx_up]                 


        if downsampled:

            if add_crosstalk:
                self.spec_491 = self._convolve_crosstalk(self.spec_491,491,**kwargs)             
                self.spec_492 = self._convolve_crosstalk(self.spec_492,492,**kwargs)             

            if self.spec_491 is not None: 
                self.spec_491 = downsample_array(self.spec_491,self.cutout_oversample)
                if self._make_full_detector: self.spec_491 = np.pad(self.spec_491,1,mode='constant',constant_values=0)
            elif self.spec_491 is None and self._make_full_detector:
                self.spec_491 = np.zeros((2048,2048))

            if self.spec_492 is not None: 
                self.spec_492 = downsample_array(self.spec_492,self.cutout_oversample)
                if self._make_full_detector: self.spec_492 = np.pad(self.spec_492,1,mode='constant',constant_values=0)
            elif self.spec_492 is None and self._make_full_detector:
                self.spec_492 = np.zeros((2048,2048))

        if return_fluxes:
            flux_on_detector = np.sum(np.sum(conv_cube,axis=2),axis=1)
            return flux_on_detector



    def to_1d(self,downsampled=True):

        if self.spec_491 is None and self.spec_492 is None:
            raise TypeError("Spectra are None type - make sure to first run make_spec2d !")

        x_1d = np.array([])
        spec_1d = np.array([])

        if self.spec_491 is not None:
            if downsampled: x491,s491_1d = get_spectrum_1d(downsample_array(self.xgrid_491,self.cutout_oversample,func=np.mean),self.spec_491)
            else: x491,s491_1d = get_spectrum_1d(self.xgrid_491,self.spec_491)
            x_1d = np.append(x_1d,x491)
            spec_1d = np.append(spec_1d,s491_1d)

        if self.spec_492 is not None:

            if downsampled: x492,s492_1d = get_spectrum_1d(downsample_array(self.xgrid_492,self.cutout_oversample,func=np.mean),self.spec_492)
            else: x492,s492_1d = get_spectrum_1d(self.xgrid_492,self.spec_492)

            if len(x_1d)>0:
                x_max_491 = x_1d[-1]
                x_min_492 = x492[0]
                dx = np.diff(x_1d)[0]
                x_fill = np.arange(x_max_491+dx,x_min_492,dx)
                y_fill = np.zeros(x_fill.shape)

                x_1d = np.append(x_1d,x_fill)
                spec_1d = np.append(spec_1d,y_fill)            

            x_1d = np.append(x_1d,x492)
            spec_1d = np.append(spec_1d,s492_1d)           
                       

        return x_1d, spec_1d


def get_spectrum_1d(xgrid,spectrum_2d):

    # we don't care about subtleties of the trace for our purposes
    # we only need to worry about delta_x for the LSF and this is constant regardless of detector rotation

    spectrum_1d = np.sum(spectrum_2d,axis=0)
    xcood = xgrid[xgrid.shape[0]//2]

    return xcood, spectrum_1d


import numpy as np
import json
import os
import glob
from astropy.io import fits
from msafit.utils.array_utils import downsample_array
from msafit.utils.convolution import extend_hypercube
from scipy.interpolate import interpn

__all__ = ["PSFLib"]

class PSFLib:

    def __init__(self,fname=None,fwa='',gwa='',
        quadrant=3,shutter_i=183,shutter_j=85,N_shutter=3,source_shutter=0,
        oversampling=1,refdir=None,**kwargs):

        self.fwa = fwa.upper()
        self.gwa = gwa.upper()
        self.qd = int(quadrant)
        self.s_i = int(shutter_i)
        self.s_j = int(shutter_j)
        self.N_shutter = N_shutter
        self.source_shutter = source_shutter        
        self.psf_oversample = oversampling
        
        try:
            self.__load_from_file(fname,refdir=refdir,**kwargs)
        except FileNotFoundError: print("No PSF library found. Generate new libary or supply filename")

    @property    
    def oversampling(self):
        return self.psf_oversample

    @oversampling.setter
    def oversampling(self,oversampling):
        self.psf_oversample = oversampling


    def _get_refwl(self):
        if self.fwa=="CLEAR": ref_wl = 2.5e-6
        elif self.fwa=="F070LP": ref_wl = 1.0e-6
        elif self.fwa=="F100LP": ref_wl = 1.4e-6
        elif self.fwa=="F170LP": ref_wl = 2.3e-6
        elif self.fwa=="F290LP": ref_wl = 4.0e-6
        return ref_wl


    def __load_from_file(self,fname=None,refdir=None,verbose=False):
        if fname is None:
            fname = os.path.expandvars("${msa_refdata}")+f"/psf/1x{self.N_shutter}_{self.fwa}_{self.gwa}_{self.qd}_{self.s_i}_{self.s_j}_PSFLib.fits"

        if refdir is None: refdir = os.path.expandvars("${msa_refdata}") + '/psf/'

        with fits.open(refdir+fname) as hdu:

            self.psf_cube = hdu[4].data
            self.psf_wave = hdu[1].data*1e10
            self.psf_x = hdu[2].data
            self.psf_y = hdu[3].data
            self.fwa = hdu[4].header['FWA']
            self.gwa = hdu[4].header['GWA']

            psf_lib_oversample = hdu[4].header['OVERSAMP']
            if self.psf_oversample != psf_lib_oversample: 
                if verbose: print(f"setting the oversampling from {self.psf_oversample} to {psf_lib_oversample} to match library" )       
                self.psf_oversample = hdu[4].header['OVERSAMP']




    def slice_cube(self,xsel=(None,None),ysel=(None,None),wsel=(None,None)):
        """Slice the psf cube by x, y and/or wavelength
        
        Parameters
        ----------
        xsel : tuple, optional
            min and max indices selected from the x_grid
        ysel : tuple, optional
            min and max indices selected from the y_grid
        wsel : tuple, optional
            min and max indices selected from the wave_grid
        """

        assert isinstance(self.psf_cube,np.ndarray)
        
        if isinstance(xsel,list):
            slcx = xsel
        elif isinstance(xsel,tuple):
            slcx = slice(xsel[0],xsel[1])
        else: raise TypeError("input needs to be tuple or list")

        if isinstance(ysel,list):
            slcy = ysel
        elif isinstance(ysel,tuple):
            slcy = slice(ysel[0],ysel[1])
        else: raise TypeError("input needs to be tuple or list")

        if isinstance(wsel,list):
            slcw = wsel
        elif isinstance(wsel,tuple):
            slcw = slice(wsel[0],wsel[1])
        else: raise TypeError("input needs to be tuple or list")

        self.psf_x = self.psf_x[slcy,slcx]
        self.psf_y = self.psf_y[slcy,slcx]
        self.psf_wave = self.psf_wave[slcw]
        self.psf_cube = self.psf_cube[slcw,slcy,slcx,:,:]


    def slice_psfs(self,npix_x=None,npix_y=None):

        if npix_y is not None:
            self.psf_cube = self.psf_cube[:,:,:,npix_y*self.psf_oversample:-npix_y*self.psf_oversample,:]
        if npix_x is not None:
            self.psf_cube = self.psf_cube[:,:,:,:,npix_x*self.psf_oversample:-npix_x*self.psf_oversample]        


    def extend_cube(self,N):
        """Extend the psf cube by multiples of itself.
        This is a useful feature when fitting velocity fields for emission lines
        
        Parameters
        ----------
        N : int
            number of multiplications

        returns: None
        edits the existing self.psf_cube and self.psf_wave to reflect changes
        """

        self.psf_wave = np.repeat(self.psf_wave,N)
        self.psf_cube = extend_hypercube(self.psf_cube,N)


    def interp_new_wave(self,new_wl):
    
        idx_wl = np.argsort(np.abs(self.psf_wave-new_wl))[:2]
        idx_low = np.min(idx_wl)
        idx_up = np.max(idx_wl)

        weight_low = new_wl - self.psf_wave[idx_low]
        weight_up = self.psf_wave[idx_up] - new_wl       
        norm = self.psf_wave[idx_up] - self.psf_wave[idx_low]

        slice_low = self.psf_cube[idx_low]
        slice_up = self.psf_cube[idx_up]
        new_slice = (weight_low*slice_low + weight_up*slice_up) / norm
        print('check sum:', np.sum(slice_low), np.sum(slice_up), np.sum(new_slice))

        self.psf_cube = np.insert(self.psf_cube,idx_up,new_slice,axis=0)
        self.psf_wave = np.insert(self.psf_wave,idx_up,new_wl)






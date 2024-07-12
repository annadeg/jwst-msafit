import numpy as np
import os
import warnings
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.units import UnitsWarning

# some helper functions to compute Nyquist sampling
__all__ = ["get_fwhm_spatial","nyquist_spatial","get_fwhm_lambda","nyquist_wave"]

_pix_size = 0.1 # arcsec
_pix_fwhm = 2.2 # pix/fwhm
_c = 3e5        # km/s

def get_fwhm_spatial(wl):

    fwhm = wl*1e-10/6.6
    return fwhm*180*3600/np.pi


def nyquist_spatial(wl,oversampling_factor=True):

    fwhm = get_fwhm_spatial(wl)
    if oversampling_factor: return _pix_size/(fwhm/2)
    else: return fwhm/2

def get_fwhm_lambda(wl,disperser,refdir=None):

    warnings.filterwarnings("ignore",category=UnitsWarning, module='astropy')

    if refdir is not None:
        disp_curve = Table.read(refdir+f'/detector/jwst_nirspec_{disperser.lower()}_disp.fits')
    else: disp_curve = Table.read(os.path.expandvars('${msa_refdata}')+f'/detector/jwst_nirspec_{disperser.lower()}_disp.fits')
    fdisp = interp1d(disp_curve['WAVELENGTH']*1e4,disp_curve['R'],kind='cubic') 

    fwhm = wl/fdisp(wl)     # lsf fwhm in angstrom
    return fwhm

def nyquist_wave(wl,disperser,return_deltav=True):

    fwhm = get_fwhm_lambda(wl,disperser)
    dlambda = fwhm/_pix_fwhm    # A/pix

    if return_deltav: 
        dv = _c * dlambda/wl
        return dlambda/2, dv/2
    else: return dlambda/2
















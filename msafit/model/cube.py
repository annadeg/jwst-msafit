import numpy as np
from .velocity_field import arctan1D, const_vdisp, VelField2D 
from .light_profile import sersic2D, gauss2D, LightDist2D
from astropy.io import fits


__all__ = ["Cube"]

_gauss_prefact = np.sqrt(2.*np.pi)
_c = 2.998e5  # km/s

def convolve_gauss1D(x, flux, x0, sigma):
    """Function to convolve a 2D velocity field with a 2D light profile, assuming the dispersion is Gaussian

    
    Parameters
    ----------
    x : 1D array
        wavelengths
    flux : 2D array
        light profile
    x0 : 2D array
        central wavelength
    sigma : 2D array
        dispersion (not fwhm!). Set to small value if sigma==0 (sigma~1 km/s)
    
    Returns
    -------
    3D array
        flux cube L(lambda,y,x)
    """
    dx = np.diff(x)[0]
    check_sigma = np.where(sigma<(0.01*dx))
    if len(check_sigma[0])>0: sigma += 0.01*dx

    cube = flux[np.newaxis,:, :]*np.exp(-0.5*((x[:,np.newaxis, np. newaxis]-x0[np.newaxis,:, :])**2/sigma[np.newaxis,:, :]**2))/(_gauss_prefact*sigma[np.newaxis,:, :])

    return cube


class Cube:

    """
    Class to construct a noisefree, unconvolved cube
    
    Parameters
    ----------
    x_grid
        1D array, dispersion direction
    y_grid
        1D array, cross-dispersion direction
    wave_grid
        1D array, wavelengths
    wave_sampling
        str, specify whether wave_grid is sampled in linear or log space for the convolution
    
    Attributes
    ----------
    data : ndarray
        model flux cube 
    
    """
    def __init__(self, x_grid, y_grid, wave_grid,wave_sampling='linear'):

        self._x = x_grid
        self._y = y_grid
        self._wave = wave_grid
        self._wave_sampling = wave_sampling
        self._dim = (wave_grid.shape[0], x_grid.shape[0],x_grid.shape[1])
        self.data = None

        
    def get_component2D(self,modelLight2D,modelVel2D,parameter_dict,i_comp=0,obs_wave=None,rest_wave=None,z_spec=None,**kwargs):
 
        """
        create single cube by convolving one 2D light profile model with one 2D velocity field model
        
        Parameters
        ----------
        modelLight2D
            class instance
        modelVel2D
            class instance
        parameter_dict
            dict, has to contain all parameters relevant for the light and velocity models
        i_comp : int, optional
            Description
        obs_wave
            float or None, *observed* wavelength at which to place the line
        rest_wave
            float or None, *rest-frame* wavelength at which to place the line. Important: must be accompanied by a valid z_spec
        z_spec
            float or None, redshift. Only needs to be defined if using rest_wave
        **kwargs
        
        Returns
        -------
        ndarray
            flux cube for single component
        
        Raises
        ------
        ValueError
            if user input is insufficient (no wavelength or redshift, invalid wavelength sampling)
        
        """
        if (obs_wave is None) and (rest_wave is None): 
            raise ValueError("No wavelength defined for line")
        elif (rest_wave is not None) and (z_spec is None): 
            raise ValueError("Need to specify redshift when using a rest-frame wavelength")
        elif (rest_wave is not None) and (z_spec is not None): 
            obs_wave = rest_wave*(z_spec+1)

        cube_out = np.zeros(self._dim)
        velocity_map, dispersion_map = modelVel2D.evaluate(self._x, self._y, parameter_dict["vfield"][i_comp])
        light_map = modelLight2D.evaluate(self._x, self._y,parameter_dict["morph"][i_comp],**kwargs)

        if self._wave_sampling=='linear':
            wave_cent = ((velocity_map/_c)+1)*obs_wave
            wave_sigma = (dispersion_map/_c)*obs_wave      
            cube_out = convolve_gauss1D(self._wave, light_map, wave_cent, wave_sigma) 

        elif self._wave_sampling=='log':
            wave_cent = np.log( ((velocity_map/_c)+1)*obs_wave )
            wave_sigma = np.log(dispersion_map/_c)
            cube_out = convolve_gauss1D(np.log(self._wave), light_map, wave_cent, wave_sigma) 

        else: raise ValueError('Choose between linear or log for the Gaussian convolution')

        return cube_out

    def compute_line_from2d(self, modelLight2D, modelVel2D, parameter_dict,obs_wave=None,rest_wave=None,z_spec=None,**kwargs):
        """
        create cube by co-adding multiple 2D light profile models convolved with multiple 2D velocity field models
        
        Parameters
        ----------
        modelLight2D
            class instance or list
        modelVel2D
            class instance or list
        parameter_dict
            dict or list of dicts, has to contain all parameters relevant for the light and velocity models
        obs_wave
            list or None, *observed* wavelength at which to place the line
        rest_wave
            list or None, *rest-frame* wavelength at which to place the line. Important: must be accompanied by a valid z_spec
        z_spec
            float or None, redshift. Only needs to be defined if using rest_wave
        **kwargs
        
        Raises
        ------
        ValueError
            if user input is insufficient (no wavelength or redshift, invalid wavelength sampling)
        
        """

        if (obs_wave is None) and (rest_wave is None): 
            raise ValueError("No wavelength defined for line")
        elif (rest_wave is not None) and (z_spec is None): 
            raise ValueError("Need to specify redshift when using a rest-frame wavelength")            
        elif obs_wave is None:
            obs_wave = np.asarray(rest_wave)*z_spec

        obs_wave = np.atleast_1d(obs_wave)

        if type(modelLight2D) != list:
            modelLight2D = [modelLight2D]
        if type(modelVel2D) != list:
            modelVel2D = [modelVel2D]                       

        if (len(modelLight2D) == len(modelVel2D)) and (len(modelLight2D) == len(parameter_dict["morph"])):
            cube_out = np.zeros(self._dim)
            for i in range(len(modelLight2D)):
                cube_out += self.get_component2D(modelLight2D[i], modelVel2D[i], parameter_dict,i,obs_wave[i],**kwargs) 
            self.data = cube_out

        else: raise ValueError("number of light models is incompatible with number of velocity models or parameters")


    def compute_line_zerovel(self, modelLight2D=None, parameter_dict=None,obs_wave=None,rest_wave=None,z_spec=None,**kwargs):

        if (obs_wave is None) and (rest_wave is None): 
            raise ValueError("No wavelength defined for line")
        elif (rest_wave is not None) and (z_spec is None): 
            raise ValueError("Need to specify redshift when using a rest-frame wavelength")            
        elif obs_wave is None:
            obs_wave = (np.asarray(rest_wave)*z_spec).reshape((1,))

        obs_wave = np.atleast_1d(obs_wave)
        cube_out = np.zeros(self._dim)

        # loop over wavelengths to add to cube
        for i in range(len(obs_wave)):
            index = np.argmin(np.abs(self._wave - obs_wave[i]))

            # add all morphological components
            for j in range(len(self.param_dict["morph"])):
                cube_out[index,:,:] += self._model_light2D.evaluate(self._x, self._y, self.param_dict["morph"][j],**kwargs)

        self.data = cube_out



    def compute_from2d(self, modelLight2D=None, modelVel2D=None, parameter_dict=None,**kwargs):


        if modelLight2D is None or parameter_dict is None:
            raise ValueError("no morphological model or parameters given")
            
        elif modelVel2D is not None:
            self.compute_line_from2d(modelLight2D, modelVel2D, parameter_dict,**kwargs)

        else:
            self.compute_line_zerovel(modelLight2D,parameter_dict,**kwargs)
            




    def write_data(self, fileout):
        """
        write cube to .fits file
        Parameters:
        fileout: str, path+name of output file
        """
        hdu = fits.PrimaryHDU(self.data)
        hdu.header['CRVAL3'] = self._wave[0]
        hdu.header['CDELT3'] = self._wave[1]-self._wave[0]
        hdu.header['CRPIX3'] = 1
        hdu.writeto(fileout, overwrite=True)
        


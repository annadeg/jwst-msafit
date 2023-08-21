import numpy as np
from astropy.modeling import models
from astropy.io import fits
from msafit.utils.array_utils import find_pixel, downsample_array, oversample_array
from scipy.special import gamma, gammaincinv, gammainc


__all__ = ["LightDist2D","sersic2D","gauss2D","point2D","uniform2D"]

_det_pixsize = 0.1 # arcsec (approximate)

class LightDist2D:
    
    def __init__(self, func_light2D):
        self.__function2D = func_light2D


    def integrate_profile(self,xgrid,ygrid,parameter_dict):
            surface_brightness = self.__function2D(xgrid, ygrid, parameter_dict)
            dx = np.fabs(xgrid[0,1]-xgrid[0,0]) 
            dy = np.fabs(ygrid[1,0]-ygrid[0,0])
            pixel_area = dy*dx
            return surface_brightness*pixel_area


    def integrate_oversampled_profile(self,xgrid,ygrid,parameter_dict,N_re,tot_zoom,min_pix):

        dx = np.fabs(xgrid[0,1]-xgrid[0,0]) 
        dy = np.fabs(ygrid[1,0]-ygrid[0,0])

        zoom_factor = int(np.round( np.maximum(dx,dy) / (_det_pixsize/tot_zoom) ))
        if zoom_factor<= 1.: return self.integrate_profile(xgrid,ygrid,parameter_dict), [[None,None],[None,None]]
        else:
            N_re_pix = int(np.ceil(N_re*parameter_dict["r_e"] / np.minimum(dx,dy) ))
            ind_x,ind_y = find_pixel(parameter_dict["x0_sky"],parameter_dict["y0_sky"],xgrid,ygrid)

            N_pix_x = np.maximum(min_pix//2,N_re_pix) + 1
            N_pix_y = np.maximum(min_pix//2,N_re_pix) + 1

            ylim_up = xgrid.shape[0]-1
            xlim_up = xgrid.shape[1]-1

            ind_y_min = np.maximum(0,ind_y-N_pix_y)
            ind_y_max = np.minimum(ylim_up,ind_y+N_pix_y)
            ind_x_min = np.maximum(0,ind_x-N_pix_x)
            ind_x_max = np.minimum(xlim_up,ind_x+N_pix_x)

            #print([[ind_y_min,ind_y_max+1],[ind_x_min,ind_x_max+1]])
            x_zoom = oversample_array(xgrid[ind_y_min:ind_y_max+1,ind_x_min:ind_x_max+1],zoom_factor)
            y_zoom = oversample_array(ygrid[ind_y_min:ind_y_max+1,ind_x_min:ind_x_max+1],zoom_factor)

            light_inner = self.integrate_profile(x_zoom,y_zoom,parameter_dict)

            # return indices that account for the fact that we chop off one pixel during downsampling
            light_dist2D = downsample_array(light_inner,zoom_factor)
            return light_dist2D, [[ind_y_min+1,ind_y_max],[ind_x_min+1,ind_x_max]]


    def evaluate(self, xgrid, ygrid, parameter_dict,zoom_centre=True,r_e_1=0.5,r_e_2=0.2,**kwargs):
        
        if parameter_dict["profile"] == "point" or  parameter_dict["profile"] == "uniform":
            return self.__function2D(xgrid, ygrid, parameter_dict)  

        elif zoom_centre==False:
            light_dist2D = self.integrate_profile(xgrid,ygrid,parameter_dict)
            return light_dist2D

        else:
            ### integrate the profile to go from surface brightness to flux

            # we want to progressively oversample the inner regions because the Sersic profile can be very steep in the centre 
            # (following Haeussler et al. 2007 and discussion https://github.com/astropy/astropy/issues/11179)
            # the curvature of the profile means a default Sersic2D profile is not good for computing the integral on a coarse grid
            # -- within 0.2 Re oversample by factor 1000 wrt the NIRSpec pixels
            # -- within 0.2-0.5 Re oversample by factor 100
            # -- for >.5 Re oversample by factor 10
            

            light_dist2D = self.integrate_profile(xgrid,ygrid,parameter_dict)
            light_outer, ind_outer = self.integrate_oversampled_profile(xgrid,ygrid,parameter_dict,100,10,2*xgrid.shape[0])

            light_dist2D[ind_outer[0][0]:ind_outer[0][1],ind_outer[1][0]:ind_outer[1][1]] = 0.
            light_dist2D[ind_outer[0][0]:ind_outer[0][1],ind_outer[1][0]:ind_outer[1][1]] += light_outer

            light_med, ind_med = self.integrate_oversampled_profile(xgrid,ygrid,parameter_dict,r_e_1,100,3)
            light_dist2D[ind_med[0][0]:ind_med[0][1],ind_med[1][0]:ind_med[1][1]] = 0.
            light_dist2D[ind_med[0][0]:ind_med[0][1],ind_med[1][0]:ind_med[1][1]] += light_med

            light_inner, ind_inner = self.integrate_oversampled_profile(xgrid,ygrid,parameter_dict,r_e_2,500,1)
            light_dist2D[ind_inner[0][0]:ind_inner[0][1],ind_inner[1][0]:ind_inner[1][1]] = 0.
            light_dist2D[ind_inner[0][0]:ind_inner[0][1],ind_inner[1][0]:ind_inner[1][1]] += light_inner

            return light_dist2D




def sersic_flux2Ie(flux,n,r_e_maj,q):
    # calculate surface brightness at Re [in arcsec] from total flux [arbitrary flux units]
    bn = gammaincinv(2*n, 0.5)
    G2n = gamma(2*n)
    Ie = flux / (r_e_maj**2 * q * 2*np.pi * n * np.exp(bn) * bn**(-2*n) * G2n )
    return Ie
    
def sersic2D(xgrid, ygrid, parameter_dict):
    amplitude = sersic_flux2Ie(flux=parameter_dict["flux"],n=parameter_dict["n"],r_e_maj=parameter_dict["r_e"],q=parameter_dict["q"])
    model = models.Sersic2D(amplitude=amplitude,r_eff=parameter_dict["r_e"], n=parameter_dict["n"],
        x_0=parameter_dict["x0_sky"],y_0=parameter_dict["y0_sky"], 
        ellip=1-parameter_dict["q"], theta=(parameter_dict["PA"]/180.0)*np.pi)
    light_dist2D = model(xgrid, ygrid)
    return light_dist2D
        

def gauss_peak2tot(flux,sigma_x,sigma_y):

    return flux/(2*np.pi*sigma_x*sigma_y)

def gauss2D(xgrid,ygrid,parameter_dict):
    amplitude = gauss_peak2tot(flux=parameter_dict["flux"],sigma_x=parameter_dict["sigma_x"],sigma_y=parameter_dict["sigma_y"])
    model = models.Gaussian2D(amplitude=amplitude,x_mean=parameter_dict["x0_sky"],y_mean=parameter_dict["y0_sky"],
        x_stddev=parameter_dict["sigma_x"],y_stddev=parameter_dict["sigma_y"],
        theta= (parameter_dict["PA"]/180.0)*np.pi )
    light_dist2D = model(xgrid, ygrid)
    return light_dist2D
                

def point2D(xgrid,ygrid,parameter_dict):

    light_dist2D = np.zeros(xgrid.shape)
    x_pix,y_pix = find_pixel(parameter_dict["x0_sky"],parameter_dict["y0_sky"],xgrid,ygrid)
    light_dist2D[y_pix,x_pix] = parameter_dict["flux"]
    return light_dist2D

def uniform2D(xgrid,ygrid,parameter_dict):

    light_dist2D = np.ones(xgrid.shape)

    return light_dist2D*parameter_dict["flux"]/np.sum(light_dist2D)




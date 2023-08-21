import numpy as np
from astropy.io import fits 

__all__ = ["VelField2D","arctan1D","const_vdisp"]


class VelField2D:
    
    """
    Class to construct a 2D velocity field for an infinitely thin disk from arbitrary 1D velocity profiles

    """

    def __init__(self, func_velcurve1D,func_veldisp1D):
        self.__velcurve1D = func_velcurve1D
        self.__veldisp1D = func_veldisp1D

    def evaluate(self, x, y, parameter_dict):
        
        #Taking necessary parameters from parameter dictonary with 
        PA_rad = parameter_dict["PA_vel"]/180.0*np.pi 
        inclination = parameter_dict["inclination"]/180.0*np.pi
        x_cent = parameter_dict["x0_vel"]
        y_cent = parameter_dict["y0_vel"]
        # Applying projection regarding inclination and projection angle 
        sini = np.sin(inclination)
        cosi = np.cos(inclination)
        cos2i = 1.0-sini**2
        x_rot = (x-x_cent)*np.cos(PA_rad)+(y-y_cent)*np.sin(PA_rad)
        y_rot = -(x-x_cent)*np.sin(PA_rad)+(y-y_cent)*np.cos(PA_rad)

        # Mapping 1D velocity curve onto the grid + some workaround for the case where the infinitely thin disk is seen edge-on
        if parameter_dict["inclination"] == 90:
            velocity = self.__velcurve1D(x_rot, parameter_dict)
            veldisp = self.__veldisp1D(x_rot, parameter_dict)                        
            pscale = np.diff(y,axis=0)[0][0]
            select = np.abs(y_rot) > (pscale/2)
            velocity[select] = 0.
            veldisp[select] = np.max(veldisp)*1e-5            

        else:
            r = np.sqrt(x_rot**2+(y_rot**2/cos2i))        
            velocity = self.__velcurve1D(r, parameter_dict)*(x_rot/r)*sini
            select = r==0
            velocity[select] = 0.0 
            veldisp = self.__veldisp1D(r, parameter_dict)
        
        return velocity, veldisp



def arctan1D(r, parameter_dict):
    """
    arctangent rotation curve

    parameters:
    v_a: float, asymptotic velocity (km/s)
    r_t: float, turnover radius (arcsec)
    """
    v_out = (2.0 / np.pi) * parameter_dict["v_a"] * np.arctan(r / parameter_dict["r_t"])
    return v_out


def const_vdisp(r,parameter_dict):
    """
    constant velocity dispersion profile

    parameters:
    sigma_0: float, velocity dispersion (km/s)
    """    
    return np.ones(r.shape)*parameter_dict["sigma_0"]




# to be removed later on probably
class VelCurve1D:
    def __init__(self, function):
        self.__function1D = function
        
    def evaluate(self, r, parameter_dict):
        velocity = self.__function1D(r, parameter_dict)
        return velocity
    
class VelDisp1D:
    def __init__(self, function):
        self.__function1D = function
        
    def evaluate(self, r, parameter_dict):
        veldisp = self.__function1D(r, parameter_dict)
        return veldisp




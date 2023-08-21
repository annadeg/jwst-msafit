import numpy as np
from msafit.utils.trace import get_dlambda_dx
from lmfit import Parameters, minimize
from scipy.interpolate import interp1d
from .fit_func import *


class LSF:

    def __init__(self,spec2d,wl_lines,x_lines,downsampled=True):

        if type(spec2d) == list: self.spec = spec2d
        else: self.spec = [spec2d]
        self.x_fpa = x_lines
        self.wl_fpa = wl_lines
        self.downsampled = downsampled


    def _fit_lines(self,xdata,ydata):

        norm = np.max(ydata)
        x = np.atleast_2d(xdata)
        delta_x = np.diff(x)[0][0]
        data = np.atleast_2d(ydata)
        #print(x.shape,data.shape)
        nlines = len(self.wl_fpa)
        fit_params = Parameters()
        for iy, y in enumerate(data):
            for jx, xfpa in enumerate(self.x_fpa):
                fit_params.add(f'amp_{iy}_{jx}', value=norm/2, min=0.0, max=2*norm)
                fit_params.add(f'cen_{iy}_{jx}', value=xfpa, min=xfpa-5*delta_x, max=xfpa+5*delta_x)
                if iy==0: fit_params.add(f'sig_{jx}', value=delta_x, min=0.0, max=5*delta_x)

        output = minimize(objective, fit_params, args=(nlines,x, data))
        return output


    def _dx_to_dlambda(self,fit_params,dlambda=100.):

        nlines = len(self.wl_fpa)
        sigma_x = [fit_params[f'sig_{jx}'] for jx in range(nlines)]
        slopes = get_dlambda_dx(self.spec[0].get_trace_x,self.spec[0].get_trace_y,self.wl_fpa,dlambda)

        return sigma_x*slopes

    def _check_edges(self,Nmin=5):
        '''
        Parameters:
        -----------
        Nmin: int, default=5. Number of pixels we want to be away from the edge of the detector for fitting a robust dispersion.
        
        '''

        dx = 1.8e-5
        ind_491 = np.where((self.x_fpa>np.max(self.spec[0].sca491[0].T[0])+Nmin*dx) & (self.x_fpa<np.min(self.spec[0].sca491[0].T[-1])-Nmin*dx) )[0]
        ind_492 = np.where((self.x_fpa>np.max(self.spec[0].sca492[0].T[-1])+Nmin*dx) & (self.x_fpa<np.min(self.spec[0].sca492[0].T[0])-Nmin*dx) )[0]    

        ind_ok = np.append(ind_491,ind_492)
        return ind_ok


    def _interp_lsf(self,lsf_property,Nmin=5,**kwargs):

        ind_ok = self._check_edges(Nmin)
        lsf = interp1d(self.wl_fpa[ind_ok],lsf_property[ind_ok],bounds_error=False,fill_value=np.nan,**kwargs)
        return lsf


    def compute_lsf(self,dlambda=100.,Nmin=5,**kwargs):


        for i in range(len(self.spec)):

            x1d,spec1d = self.spec[i].to_1d(downsampled=self.downsampled)
            if i==0:
                xdata = np.atleast_2d(x1d)
                ydata = np.atleast_2d(spec1d)
            elif xdata.shape[1]==x1d.shape[0]:
                xdata = np.vstack((xdata,x1d))
                ydata = np.vstack((ydata,spec1d))
            elif xdata.shape[1]<x1d.shape[0]:                
                diff = np.abs(xdata.shape[1] - x1d.shape[0])
                xdata = np.vstack((xdata,x1d[diff:]))
                ydata = np.vstack((ydata,spec1d[diff:]))            
            elif xdata.shape[1]>x1d.shape[0]:                
                diff = np.abs(xdata.shape[1] - x1d.shape[0])
                xdata = np.vstack((xdata[:,diff:],x1d))
                ydata = np.vstack((ydata[:,diff:],spec1d))  


        fit_output = self._fit_lines(xdata,ydata)
        sigma_lambda = self._dx_to_dlambda(fit_output.params,dlambda)
        
        if len(self.wl_fpa)> 1:
            self.dispersion = self._interp_lsf(sigma_lambda,Nmin,**kwargs)
            self.dispersion_kms = self._interp_lsf(3e5*sigma_lambda/self.wl_fpa,Nmin,**kwargs)
            self.resolution = self._interp_lsf(self.wl_fpa/(2*np.sqrt(2*np.log(2))* sigma_lambda), Nmin ,**kwargs)

        return fit_output, sigma_lambda






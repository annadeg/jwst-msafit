import numpy as np


__all__ = ["gauss_dataset","objective"]

def gauss(x, amp, cen, sigma):
    return amp * np.exp(-(x-cen)**2 / (2.*sigma**2))


def multi_gauss(x,amp_list,cen_list,sigma_list):
    nlines = len(amp_list)
    y = np.zeros(x.shape)

    for i in range(nlines):
        y += gauss(x,amp_list[i],cen_list[i],sigma_list[i])
    return y

def gauss_dataset(params,idata,nlines, x):
    """Calculate Gaussian lineshape from parameters for data set."""
    amp_list = [params[f'amp_{idata}_{i}'] for i in range(nlines) ]
    cen_list = [params[f'cen_{idata}_{i}'] for i in range(nlines) ]
    sig_list = [params[f'sig_{i}'] for i in range(nlines) ]
    return multi_gauss(x, amp_list, cen_list, sig_list)


def objective(params,nlines, x, data):
    """Calculate total residual for fits of Gaussians to several data sets."""
    ndata, _ = data.shape
    resid = 0.0*data[:]

    # make residual per data set
    for idata in range(ndata):
        resid[idata, :] = data[idata, :] - gauss_dataset(params, idata,nlines, x[idata,:])

    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


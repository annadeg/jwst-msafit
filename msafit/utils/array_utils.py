import numpy as np
from scipy.ndimage import zoom

__all__ = ["downsample_array","oversample_array","find_pixel"]


def downsample_array(arr,factor,func=np.sum):

    # reshape and then sum every N pixels along an axis
    reshaped_arr = arr.reshape((arr.shape[0],arr.shape[1]//factor,factor))
    comb_arr = func(reshaped_arr,-1)

    # now repeat for the other axis
    reshaped_arr = comb_arr.reshape(arr.shape[0]//factor,factor,arr.shape[1]//factor)
    downsampled_arr = func(reshaped_arr,1)
    return downsampled_arr


def oversample_array(arr,factor):

    if factor>1:
        # interpolate onto finer grid and trim the edges to avoid extrapolation issues
        new_arr = zoom(arr,factor,order=1,mode='nearest',grid_mode=True)[factor:-factor,factor:-factor] 

    else: 
        new_arr = arr[1:-1,1:-1]

    return new_arr


def find_pixel(x,y,xarr,yarr):

    ypix = np.unravel_index(np.argmin(np.fabs(yarr-y)),yarr.shape)[0]
    xpix = np.argmin(np.fabs(xarr[ypix,:]-x))
    return xpix,ypix







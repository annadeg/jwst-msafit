import numpy as np

__all__ = ["extend_hypercube","convolve_cube_psf"]


def multiply_arrays(cube_arr,psf_arr,out_arr=None):

    if len(cube_arr.shape)==3:
        return np.multiply(cube_arr[:,:,:,np.newaxis,np.newaxis], psf_arr,out=out_arr)
    elif len(cube_arr.shape)==2:
        return np.multiply(cube_arr[np.newaxis,:,:,np.newaxis,np.newaxis], psf_arr,out=out_arr)


def collapse_xy(hypercube):

    sum_y = np.sum(hypercube,axis=2)
    sum_x = np.sum(sum_y,axis=1)

    return sum_x


def extend_hypercube(hypercube,N):
    # method to repeat a hypercube N times for faster multiplication
    dim_in = hypercube.shape
    dim_out = [N] + list(dim_in[1:])
    new_psf_cube = np.zeros(dim_out)

    for i in range(N):
        new_psf_cube[i] = hypercube[0]

    return new_psf_cube


def convolve_cube_psf(modelCube,modelPSF,extend_cube=False,out_arr=None,**kwargs):

    dim_cube = modelCube.data.shape
    try:
        dim_psf = modelPSF.psf_cube.shape
        psf_hc = modelPSF.psf_cube
    except AttributeError:
        try:
            dim_psf = modelPSF.shape
            psf_hc = modelPSF    
        except AttributeError:
            print("invalid PSF library")


    conv_cube = None

    if dim_cube == dim_psf[:3]:
        hc = multiply_arrays(modelCube.data,psf_hc,out_arr)
        conv_cube = collapse_xy(hc)
        
    elif dim_cube[1]!=dim_psf[1] or dim_cube[2]!=dim_psf[2]:
        raise ValueError("dimensions are not compatible")
        
    elif dim_cube[0]>dim_psf[0] and dim_psf[0]==1:
    
        if extend_cube:
            print("extending the cube")
            extend_lib = extend_hypercube(psf_hc,N=dim_cube[0])
            hc = multiply_arrays(modelCube.data,extend_lib,out_arr)
            conv_cube = collapse_xy(hc)
        else:
            conv_cube = np.zeros((len(modelCube._wave),dim_psf[3],dim_psf[4]))
            for i in range(len(modelCube._wave)):
                hc = multiply_arrays(modelCube.data[i],psf_hc,out_arr) 
                conv_cube[i] = collapse_xy(hc)[0]       
        
    else: raise RuntimeError("something went wrong in the convolution")

    #print(conv_cube.shape)
    return conv_cube

import numpy as np
import glob
from msafit.utils.array_utils import find_pixel
from astropy.io import fits
from msafit.fpa.detector import DetectorCutout
from astropy.nddata import bitmask


__all__ = ["select_obs"]

def retrieve_files(obs_dir,obj_id,filt='',disperser='',gto_convention=False,fits_ext='FFLT_I2D',**kwargs):
    """find data files, either within datalabs durectory structure or custom folder
    
    Parameters
    ----------
    obs_dir : str
        path to observation directory
    obj_id : int
        galaxy ID number
    filt : str, optional
        filter used, only needed if using datalabs structure
    disperser : str, optional
        disperser used, only needed if using datalabs structure
    gto_convention : bool, optional
        flag to indicate whether or not the data are stored following
        the directory structure on datalabs. Default is True
    
    Returns
    -------
    list
        contains filenames of all I2D files found for given object ID
    """

    if gto_convention:
        flist = []
        dlist = glob.glob(obs_dir+f"/*_COMM_{filt.upper()}_{disperser.upper()}_NOLAMP")
        for datadir in dlist:
            obj_dir = glob.glob(datadir+f"/*_{str(obj_id).zfill(6)}_*")[0]
            fname = glob.glob(obj_dir+f"/*_{fits_ext}.fits")
            flist += fname

    else:
        flist = glob.glob(obs_dir+f"/*{obj_id}*_{fits_ext}.fits")

    return flist


def extract_keyword_info(hdr):
    """Extract important keyword:value pairs from the fits file header
    
    Parameters
    ----------
    hdr : astropy fits header object
            
    Returns
    -------
    dict
        dictionary that stores the crucial outputs
        this follows the same structure needed for the FitModel class
    """
    hdr_dict = {"instrument":{},"geometry":{}}
    hdr_dict["OBS_ID"] = hdr["OBSID"]
    hdr_dict["unit"] = hdr["UNIT"]
    hdr_dict["instrument"]["roll_angle"] = hdr["ROLL_REF"]
    hdr_dict["instrument"]["filter"] = hdr["FWA_POS"]
    hdr_dict["instrument"]["disperser"] = hdr["GWA_POS"]
    hdr_dict["geometry"]["shutter_array"] = hdr["APERTURE"].lower()
    hdr_dict["geometry"]["psky_x"] = hdr["PWID_SKY"]
    hdr_dict["geometry"]["psky_y"] = hdr["PLEN_SKY"]

    slit = hdr["SLIT"].replace('-','/').split('/')
    hdr_dict["geometry"]["quadrant"] = int(slit[0])
    hdr_dict["geometry"]["shutter_i"] = int(slit[1])
    hdr_dict["geometry"]["shutter_j"] = int(slit[2])
    central_shutter = int(np.mean(np.array([int(slit[3]),int(slit[4])])))
    hdr_dict["geometry"]["source_shutter"] = int(slit[2])-central_shutter

    return hdr_dict


def read_obs(fname):
    """Read in .fits file and extract key parameters and data
    
    Parameters
    ----------
    fname : str
        filename path
    
    Returns
    -------
    tuple
        contains dictionary with header info and spectral data
        also includes coordinates needed for making cutouts, as well as quality flags
    """
    data = []
    err = []
    icood = []
    jcood = []
    wl = []
    quality = [] 
    
    with fits.open(fname) as hdu_list:

        obs_info = extract_keyword_info(hdu_list[0].header)
        s_exp = fname.split('_')[-3]
        if len(s_exp)==6:
            obs_info['NEXP'] = int(s_exp)
        else: obs_info['NEXP'] = 1

        for i in range(len(hdu_list)):
            if hdu_list[i].header["EXTNAME"]=="DATA1" or hdu_list[i].header["EXTNAME"]=="DATA2":
                data.append(hdu_list[i].data)

            elif hdu_list[i].header["EXTNAME"]=="ERR1" or hdu_list[i].header["EXTNAME"]=="ERR2":
                err.append(hdu_list[i].data)

            elif hdu_list[i].header["EXTNAME"]=="QUALITY1" or hdu_list[i].header["EXTNAME"]=="QUALITY2":
                quality.append(hdu_list[i].data)

            elif hdu_list[i].header["EXTNAME"]=="WAVE1" or hdu_list[i].header["EXTNAME"]=="WAVE2":
                wl.append(hdu_list[i].data)

            elif hdu_list[i].header["EXTNAME"]=="I1" or hdu_list[i].header["EXTNAME"]=="I2":
                icood.append(hdu_list[i].data)

            elif hdu_list[i].header["EXTNAME"]=="J1" or hdu_list[i].header["EXTNAME"]=="J2":
                jcood.append(hdu_list[i].data)

    return (obs_info, data, err, icood, jcood, wl, quality)


def _cutout_line(line_list,obs_info,data, err, icood, jcood, wl, quality,pad_x,pad_y,qflag_bad,norm_const):
    """Summary
    
    Parameters
    ----------
    line_list : list
         wavelengths in Angstrom
    obs_info : dict
        observation dictionary with slit position and filter/grating combination
    data : ndarray
        observed flux
    err : ndarray
        uncertainties on the flux
    icood : ndarray
        detector i coordinates
    jcood : ndarray
        detector j coordinates
    wl : ndarray
        detector wavelength for this slit
    quality : ndarray
        pixel quality (see nips documentation: https://jwst-tools.cosmos.esa.int/usage.html#qflags)
    pad_x : int
        number of padded pixels in the x-direction. Cutout size will be minimum of 2*pad_x+1
    pad_y : int
        number of padded pixels in the y-diretion. Cutout size will be 5*N_shutters+2*pad_y
    qflag_bad : list
        qflag values to use when creating a mask (otherwise all pixels are selected)
    norm_const : float
        normalisation constant
    
    Returns
    -------
    tuple
        contains the cutout data, uncertainties, quality mask and corner coordinates
    
    Raises
    ------
    RunTimeError
        raises error when there is an unexpected mix up between the two detectors
    """
    lmin = np.min(line_list)
    lmax = np.max(line_list)

    Nshutter = int(obs_info["geometry"]["shutter_array"][-1])
    Npix = Nshutter*5
    pad_y = Npix//2 + 2*pad_y

    detector = DetectorCutout(obs_info["instrument"]["filter"],
               obs_info["instrument"]["disperser"],
               obs_info["geometry"]["quadrant"],
               obs_info["geometry"]["shutter_i"],
               obs_info["geometry"]["shutter_j"],
               Nshutter,obs_info["geometry"]["source_shutter"])

    x_lmin = detector.get_trace_x(lmin)
    x_lmax = detector.get_trace_x(lmax)
    y_lmin = detector.get_trace_y(lmin)
    y_lmax = detector.get_trace_y(lmax)

    is_sca492 = False

    if x_lmin<0 and x_lmax<0: 
        ind_xlmin, ind_ylmin = find_pixel(x_lmin,y_lmin,detector.sca491[0],detector.sca491[1]) 
        ind_xlmax, ind_ylmax = find_pixel(x_lmax,y_lmax,detector.sca491[0],detector.sca491[1]) 

    elif x_lmin>0 and x_lmax>0: 
        is_sca492 = True
        ind_xlmin, ind_ylmin = find_pixel(x_lmin,y_lmin,detector.sca492[0],detector.sca492[1]) 
        ind_xlmax, ind_ylmax = find_pixel(x_lmax,y_lmax,detector.sca492[0],detector.sca492[1]) 
    else:
        raise RunTimeError("Something is wrong with the data -\
                            unexpected mix between SCA491 and SCA492")

    selection = np.where((jcood>=ind_ylmin-pad_y)&(jcood>=ind_ylmax-pad_y) & \
                (jcood<=ind_ylmin+pad_y)&(jcood<=ind_ylmax+pad_y) & \
                (icood>=ind_xlmin-pad_x)&(icood>=ind_xlmax-pad_x) & \
                (icood<=ind_xlmin+pad_x)&(icood<=ind_xlmax+pad_x) )

    ydim = len(np.unique(selection[0]))
    xdim = len(np.unique(selection[1]))

    data = data[selection].reshape(ydim,xdim) * norm_const
    err = err[selection].reshape(ydim,xdim) * norm_const
    quality = quality[selection].reshape(ydim,xdim)
    mask = bitmask.bitfield_to_boolean_mask(quality, ignore_flags=2**(np.asarray(qflag_bad,dtype=np.uint64)-1),flip_bits=True,good_mask_value=True)
    cood = [np.min(jcood[selection])-1,np.max(jcood[selection]),np.min(icood[selection])-1,np.max(icood[selection])]

    #if is_sca492:
    #    return (np.flipud(data), np.flipud(err), np.flipud(mask), np.flipud(quality), cood)
    #else: 
    return (data, err, mask, quality, cood)


def make_cutout(line_list,fname,pad_x=7,pad_y=3,qflag_bad=[2,3,4,5,10,15,17,36,38,40,64],renorm=False,norm_flux=1e-12,**kwargs):
    """cut out relevant part of the spectrum for a single I2D file. Accounts for the fact that lines 
    might fall on one or both detectors
    
    
    Parameters
    ----------
    line_list : list of floats
        wavelengths of interest in Angstrom
    fname : str
        filename of the I2D spectrum
    pad_x : int, optional
        number of padded pixels in the x-direction. Cutout size will be minimum of 2*pad_x+1
    pad_y : int, optional
        number of padded pixels in the y-diretion. Cutout size will be 5*N_shutters+2*pad_y
    qflag_bad : list, optional
        qflag values to use when creating a mask (otherwise all pixels are selected)
        see https://jwst-tools.cosmos.esa.int/usage.html#qflags
    renorm : bool, optional
        renormalise spectra using flux measurement supplied as norm_flux, useful to avoid rounding errors due to small numbers
        default is False.
    norm_flux : float, optional
        estimated line flux
    **kwargs
    
    Returns
    -------
    dict
        contains header info, spectral data, uncertainties and quality flags
    
    Raises
    ------
    ValueError
        raised when the supplied line list is invalid (none or wrong unit)
    
    """
    obs_info, data, err, icood, jcood, wl, quality = read_obs(fname)

    if renorm: norm_const = 1./norm_flux
    else: norm_const = 1.

    line_wl = [[],[]]

    for line in line_list: 
        for i in range(len(wl)):    
            if line*1e-10> np.min(wl[i]) and line*1e-10<np.max(wl[i]):
                line_wl[i].append(line)

    #print(line_wl)
    #print(wl)

    if len(wl)==1 and len(line_wl[0])>=1:
        data, err, mask, quality, cood = _cutout_line(line_wl[0],obs_info,data[0], err[0], icood[0], jcood[0], wl[0], quality[0],pad_x,pad_y,qflag_bad,norm_const)

    elif len(wl)==2 and len(line_wl[0])>=1 and len(line_wl[1])==0:
        data, err, mask, quality, cood = _cutout_line(line_wl[0],obs_info,data[0], err[0], icood[0], jcood[0], wl[0], quality[0],pad_x,pad_y,qflag_bad,norm_const)

    elif len(wl)==2 and len(line_wl[0])==0 and len(line_wl[1])>=1:
        data, err, mask, quality, cood = _cutout_line(line_wl[1],obs_info,data[1], err[1], icood[1], jcood[1], wl[1], quality[1],pad_x,pad_y,qflag_bad,norm_const)

    elif len(wl)==2 and len(line_wl[0])>=1 and len(line_wl[1])>=1:
        data1, err1, mask1, quality1, cood1 = _cutout_line(line_wl[0],obs_info,data[0], err[0], icood[0], jcood[0], wl[0], quality[0],pad_x,pad_y,qflag_bad,norm_const)
        data2, err2, mask2, quality2, cood2 = _cutout_line(line_wl[1],obs_info,data[1], err[1], icood[1], jcood[1], wl[1], quality[1],pad_x,pad_y,qflag_bad,norm_const)
        data = [data1,data2]
        err = [err1,err2]
        quality = [quality1,quality2]
        cood = cood1 + cood2

    else:
        raise ValueError("invalid line_list entered")



    obs_info["data"] = data
    obs_info["unc"] = err
    obs_info["mask"] = mask
    obs_info["qflag"] = quality
    obs_info["pix_cood"] = cood

    return obs_info


def select_obs(obs_dir,obj_id,line_list,skip_obs=[],**kwargs):
    """select
    
    Parameters
    ----------
    obs_dir : str
        path to observation directory
    obj_id : int
        galaxy ID number
    line_list : list of floats
        wavelengths of interest in Angstrom
    skip_obsid : list, optional
        observation ID numbers to be skipped (convenient when data quality is bad)
    **kwargs
        optional arguments for the pixel padding, creation of the pixel mask
        and picking the file structure of the data
    
    Returns
    -------
    list
        Contains a dictionary for each observation ID, containing crucial keywords,
        data, uncertainties and a pixel mask. Can directly be used for the fitting.
    """
    try:
        skip_obs_ids = [t[0] for t in skip_obs]
        d_skip = dict(skip_obs)
    except TypeError:
        skip_obs_ids = skip_obs
        d_skip = dict(zip(skip_obs,[1]*len(skip_obs)))

    flist = retrieve_files(obj_id=obj_id,obs_dir=obs_dir,**kwargs)
    obs_list = []
    for fname in flist:

        obs_dict = make_cutout(line_list,fname,**kwargs)
        if int(obs_dict["OBS_ID"]) not in skip_obs_ids:
            obs_list.append(obs_dict)
        else:
            if obs_dict['NEXP'] != d_skip[int(obs_dict["OBS_ID"])]:
                obs_list.append(obs_dict)


    return sorted(obs_list, key=lambda d: d["OBS_ID"]) 









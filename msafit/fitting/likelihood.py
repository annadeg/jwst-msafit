import numpy as np
from .fit_model import FitModel
from functools import partial


__all__ = ["lnprob","make_lnprob","chi2","make_chi2","compute_lnlike","compute_constant"]


def compute_lnlike(model_spec,obs_spec,obs_unc,mask_spec=slice(None),const=None,**kwargs):
    """compute the log likelihood given the model and data. Accounts for masking
    
    Parameters
    ----------
    model_spec : 1D array or list
        model spectrum (flattened)
    obs_spec : 1D array or list
        observed spectrum (flattened)
    obs_unc : 1D array or list
        observed uncertainties (flattened)
    mask_spec : 1D array or list, optional
        spectrum mask for bad pixels and artefacts (flattened)
    const : None, optional
        pre-computed constant for the log likelihood
    **kwargs
        accepts extra kwargs (not used)
    
    Returns
    -------
    lnp: float
        log likelihood
    """
    df = (obs_spec-model_spec)[mask_spec]
    var = obs_unc[mask_spec]**2

    chi2 = np.nansum(df**2/var)
    if const is None: const = -0.5*np.sum(np.log(2*np.pi*var))
    return (-0.5* chi2) + const 


def compute_constant(obs):
    """pre-compute constant for the log likelihood function from the observational uncertainties
    
    Parameters
    ----------
    obs : list
        contains the flattened spectrum, uncertainties and mask
    
    Returns
    -------
    const: float
        log likelihood constant
    """
    obs_spec,obs_unc,mask_spec = obs

    var = obs_unc[mask_spec]**2
    const = np.log(2*np.pi*var)
    return -0.5*np.sum(const)


def lnprob(theta,fit_model,obs,sampling,**kwargs):
    """computes the log likelihood or probability given the parameter vector theta, model and observational data
    
    Parameters
    ----------
    theta : 1D array
        contains parameter values for the N free parameters (shape (N,))
    fit_model : FitModel 
        object storing the model and fit configuration parameters
    obs : list
        contains the flattened spectrum, uncertainties and maskon
    sampling : str
        either set to 'mcmc' or 'nested' to indicate the type of sampler used
    **kwargs
    
    Returns
    -------
    float
        log likelihood or probability given the parameter vector theta, model and observational data
    """
    lnp_prior = fit_model.ln_prior_prob(theta)

    if np.isfinite(lnp_prior)==False:
        return -np.inf

    else:
        model_list = fit_model.theta2model(theta,**kwargs)
        Nmod = len(model_list)

        model_spec = np.empty(0) # np.zeros(Nmod*rdim)
        for i in range(Nmod):
            model_spec = np.append(model_spec,np.ravel(model_list[i]))

        obs_spec,obs_unc,mask_spec = obs
        lnlike = compute_lnlike(model_spec,obs_spec,obs_unc,mask_spec,**kwargs)

        if sampling=='mcmc': return lnlike + lnp_prior
        elif sampling=='nested': return lnlike


def make_lnprob(fit_model,obs,sampling,**kwargs):
    """wraps the probability function with fixed parameters, which is needed for sampling algorithms
    
    Parameters
    ----------
    fit_model : FitModel 
        object storing the model and fit configuration parameters
    obs : list
        contains the flattened spectrum, uncertainties and maskon
    sampling : str
        either set to 'mcmc' or 'nested' to indicate the type of sampler used
    **kwargs
    
    Returns
    -------
    function
        log probability function, takes one parameter (theta)
    """
    const = compute_constant(obs)
    lnprob_func = partial(lnprob,fit_model=fit_model,obs=obs,sampling=sampling,const=const,**kwargs)
    return lnprob_func


def chi2(theta,fit_model,obs,**kwargs):
    """computes the chi squared value given the parameter vector theta, model and observational data
    
    Parameters
    ----------
    theta : 1D array
        contains parameter values for the N free parameters (shape (N,))
    fit_model : FitModel 
        object storing the model and fit configuration parameters
    obs : list
        contains the flattened spectrum, uncertainties and maskon
    **kwargs
    
    Returns
    -------
    float
        chi squared given the parameter vector theta, model and observational data
    """
    model_list = fit_model.theta2model(theta,**kwargs)
    Nmod = len(model_list)

    obs_spec,obs_unc,mask_spec = obs
    model_spec = np.empty(0) # np.zeros(Nmod*rdim)
    for i in range(Nmod):
        model_spec = np.append(model_spec,np.ravel(model_list[i]))

    df = (obs_spec-model_spec)[mask_spec]
    var = obs_unc[mask_spec]**2
    chi2 = np.nansum(df**2/var)
    return chi2


def make_chi2(fit_model,obs,**kwargs):
    """wraps the probability function with fixed parameters, which is useful for the optimizer algorithms
    
    Parameters
    ----------
    fit_model : FitModel 
        object storing the model and fit configuration parameters
    obs : list
        contains the flattened spectrum, uncertainties and maskon
    **kwargs
    
    Returns
    -------
    function
        chi squared function, takes one parameter (theta)
    """
    chi2_func = partial(chi2,fit_model=fit_model,obs=obs,**kwargs)
    return chi2_func










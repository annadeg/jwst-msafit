import numpy as np 
import matplotlib.pyplot as plt
import emcee


__all__ = ["make_init_ball","get_scale_estimate","plot_autocorr"]


def get_scale_estimate(prior_func):
    """find a reasonable typical value for the prior distribution.
    For uniform priors this is somewhere around the middle, for other priors
    it is the sigma or scale.
    
    Parameters
    ----------
    prior_func : Prior
        prior class instance from msafit.fitting.priors
    
    Returns
    -------
    float
        value from the prior distribution
    """
    pf_name = str(type(prior_func))
    params = prior_func.params

    if 'Uniform' in pf_name:
        diff = params['maxi']-params['mini']
        if 'Log' in pf_name:
            return np.exp(diff/2.)
        else: return diff/10.
    elif 'Normal' in pf_name:
        if 'Log' in pf_name:
            return np.exp(params['mode']+params['sigma'])-np.exp(params['mode'])
        else:
            return params['sigma']
    elif 'Beta' in pf_name:
        return prior_func.loc + np.sqrt( (params['alpha']*params['beta']) / ((params['alpha']+params['beta'])**2 * (params['alpha']+params['beta']+1)) )

    elif 'StudentT' in pf_name:
        return params['scale']

    else: return 1.



def make_init_ball(init_point,fit_model,nwalkers,e=0.01):
    """creates a small Gaussian ball around the initial parameter vector
    
    Parameters
    ----------
    init_point : 1D array
        parameter values of the initial point
    fit_model : FitModel instance
        holds fit details, the physical model and psf library
    nwalkers : int
        number of walkers for emcee
    e : float, optional
        value used to scale the dispersion of the Gaussian
    
    Returns
    -------
    ndarray
        array of size (nwalkers,nparams) 
    """
    ball = np.zeros((len(init_point),nwalkers))

    for i, ival in enumerate(init_point):
        pn,pn_mod = fit_model.free_params[i]
        prior_func = fit_model.fit_config["fit_params"][pn]["prior"]
        pval = get_scale_estimate(prior_func)
        ball[i] = np.random.normal(loc=ival,scale=np.abs(e*pval),size=nwalkers)

    return ball.T


def plot_autocorr(tau_mean,niter_check,fdir):
    """make basic plot of the autocorrelation timescale vs the iteration number

    Parameters
    ----------
    tau_mean : float
      autocorr 
    niter_check : int
      interval to plot
    fname : str
      output file
    """

    plt.figure()
    plt.plot(np.arange(len(tau_mean))[niter_check::niter_check],tau_mean[niter_check::niter_check],ls='solid')
    plt.xlabel('iteration number',fontsize=14)
    plt.ylabel(r'$\tau$',fontsize=14)
    plt.tick_params(direction='in',top=True,right=True)
    plt.tight_layout()
    plt.savefig(fdir+'autocorr.pdf')
    plt.close()
















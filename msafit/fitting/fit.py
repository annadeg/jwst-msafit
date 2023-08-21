import numpy as np
import pickle
import sys, os
import time
import h5py
import glob
from .likelihood import make_lnprob, chi2, make_chi2, lnprob
from scipy import optimize
import multiprocessing 

try: 
    import ultranest
except(ImportError):
    pass

try: 
    import emcee
    from .mcmc import make_init_ball, plot_autocorr
except(ImportError):
    pass



__all__ = ["prep_obs","run_fit_ultra","run_optimizer","run_fit_emcee"]


def prep_obs(obs_list):
    """flattens the observed data to be used in the likelihood function
    
    Parameters
    ----------
    obs_list : list
        list of observation dicts. Each dict must contain "data" 
        and "unc" keywords, and optionally also a "mask"
    
    Returns
    -------
    list
        holds flattened arrays for the data, errors and mask
    """
    Nmod = len(obs_list)

    obs_spec = np.empty(0) # np.zeros(Nmod*rdim)    
    obs_unc = np.empty(0) #np.zeros(Nmod*rdim)        
    mask_spec = np.zeros(0,dtype=bool)

    for i in range(Nmod):
        dim = obs_list[i]["data"].shape
        rdim = dim[0]*dim[1]

        obs_spec = np.append(obs_spec,np.ravel(obs_list[i].get("data")))
        obs_unc = np.append(obs_unc,np.ravel(obs_list[i].get("unc")))
        mask_spec = np.append(mask_spec,np.zeros(rdim,dtype=bool))
        mask2d = obs_list[i].get("mask",None)   
        if mask2d is None: mask = slice(None)
        else: mask = np.ravel(mask2d)
        mask_spec[-rdim:][mask] = True  

    return [obs_spec, obs_unc, mask_spec]


def run_fit_ultra(lnp_func,fit_model,obs_list,log_dir=None,use_mpi=False,resume='subfolder',run_num=None,**kwargs):
    """run ultranest
    
    Parameters
    ----------
    fit_model : FitModel instance
        holds fit details, the physical model and psf library
    obs_list : list
        list of observation dicts. Each dict must contain "data" 
        and "unc" keywords, and optionally also a "mask"
    log_dir : str, optional
        directory where fit results are stored
    use_mpi : bool, optional
        flag to set multiprocessing
    **kwargs
        additional keywords for ultranest 
    
    Returns
    -------
    dict
        results of the fit from ultranest
    """

    obs = prep_obs(obs_list)

    free_param_names = [t[0] for t in fit_model.free_params]
    sampler = ultranest.ReactiveNestedSampler(free_param_names,lnp_func,fit_model.prior_transform,log_dir=log_dir,resume=resume,run_num=run_num)
    result = sampler.run(**kwargs)

    return result



def run_fit_optimizer(fit_model,obs_list,opt_method,param_bounds=None,disp=True,maxiter=1000,keep_feasible=True,x0=None,workers=1,fname_out=None,**kwargs):
    """run scipy.optimize
    
    Parameters
    ----------
    fit_model : FitModel instance
        holds fit details, the physical model and psf library
    obs_list : list
        list of observation dicts. Each dict must contain "data" 
        and "unc" keywords, and optionally also a "mask"
    opt_method : str
        specify which scipy.optimization method to use: set to "min" or "diff" for minimize or differential_evolution, respectively
    **kwargs
        additional keywords  
    
    Returns
    -------
    obj
       scipy.optimize.OptimizeResult object storing the results 
    """
    obs = prep_obs(obs_list)
    chi2_func = make_chi2(fit_model,obs)
    N_params = len(fit_model.free_params)
    
    if param_bounds is not None:
        bounds = optimize.Bounds(lb=param_bounds[0],ub=param_bounds[1],keep_feasible=keep_feasible)
    else:
        bounds = optimize.Bounds(lb=[-np.inf]*N_params,ub=[np.inf]*N_params,keep_feasible=keep_feasible)

    if opt_method=="min":
    
        if x0 is None: 
            raise ValueError("Need to specify an initial parameter vector")
        if 'options' in kwargs.keys():
            options = {'disp':disp,'maxiter':maxiter}
            for key in kwargs['options']:
                options[key] = kwargs['options'][key]
            del kwargs['options']
        else:
            options = {'disp':disp,'maxiter':maxiter}
        output = optimize.minimize(chi2_func,x0=x0,options=options,
                                   bounds=bounds,**kwargs)

    elif opt_method=="diff":
        if isinstance(workers,multiprocessing.pool.Pool):
            print('user supplied pool')
            updating = 'deferred'        
        elif workers>1:
            updating = 'deferred'
        else: updating = 'immediate'
        output = optimize.differential_evolution(chi2_func,bounds=bounds,disp=disp,
                                                 maxiter=maxiter,workers=workers,
                                                 updating=updating,**kwargs) # chi2,args=(fit_model,obs,kwargs)

    if fname_out is not None:
        import pickle
        from astropy.table import Table
        with open(fname_out+'.pickle','wb') as fout:
            pickle.dump(output,fout)
        tab = Table(output['x'],names=[t[0] for t in fit_model.free_params])
        tab.write(fname_out+'.csv',format='csv',overwrite=True)

    return output





def run_fit_emcee(lnp_func,fit_model,obs_list,init_point,nwalkers,niter=100000,niter_check=100,nstop=10,fdir=None,verbose=True,check_convergence=False,e_scale=0.1,skip_initial_state_check=False,**kwargs):


    obs = prep_obs(obs_list)
    ndim = len(fit_model.free_params)

    if fdir is not None:
        store_chain = True
        ldir = glob.glob(fdir+'/*')
        if len(ldir)==0:
            outdir = fdir+'/run1/'
        else:
            nums = []
            for dname in ldir:
                nums.append(int(dname.split('/')[-1][3:]))
            last_num = sorted(nums)[-1]
            outdir = fdir + f'/run{last_num+1}/'
        os.makedirs(outdir,exist_ok=True)
        hdf = h5py.File(outdir+'chains.h5','a')
        chain = hdf.create_dataset("chain",(niter_check,nwalkers,ndim),maxshape=(None,nwalkers,ndim))
        lnp = hdf.create_dataset("lnp",(niter_check,nwalkers),maxshape=(None,nwalkers))

    else: store_chain = False

    init_ball = make_init_ball(init_point,fit_model,nwalkers,e=e_scale)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp_func,**kwargs)  # lnprob,args=(fit_model,obs,'mcmc')

    autocorr_results = np.ones(niter)

    for idx,sample in enumerate(sampler.sample(init_ball, iterations=niter,store=True, progress=verbose,skip_initial_state_check=skip_initial_state_check)):
        if store_chain:
            chain[idx,:,:] = sample[0]
            lnp[idx,:] = sample[1]

        if sampler.iteration % niter_check:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        autocorr_results[(idx-niter_check)+1:idx+1] = np.mean(tau)
        hdf.flush()
        np.savetxt(outdir+'autocorr.txt',autocorr_results)

        if check_convergence:
            converged = np.all(tau * nstop < sampler.iteration)
            converged &= np.all(np.abs(autocorr_results[idx-niter_check] - tau) / tau < 0.01)
        else: converged = False

        if converged:
            break
        elif idx<(niter-niter_check):
            chain.resize(chain.shape[0]+niter_check,axis=0)
            lnp.resize(lnp.shape[0]+niter_check,axis=0)

    if store_chain: 
        plot_autocorr(autocorr_results,niter_check,outdir)
        hdf.close()

    return autocorr_results, sampler


















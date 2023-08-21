import numpy as np
from . import priors
from msafit.utils.sampling import nyquist_wave
from msafit.fpa import Spec2D
from copy import deepcopy

# set up parameters for the model that needs to be fit
# parameters will need a prior attached to them and/or a fixed value

'''
fit parameters need to be of following form
specify model type (i.e. light profile "morph" or velocity field "vfield")
except if the param name is "line_wave" (any "model" value then is ignored)

"fixed_param" : {
    "model":"morph"
    "fixed":True,
    "value":1.0
}

"fit_param" : {
    "model":"vfield"
    "fixed":False
    "prior":priors.SomePrior(**kwargs)
}

'''

class FitModel:

    """Summary
    
    Attributes
    ----------
    depend_params : dict
        parameters with dependencies
    fit_config : dict
        dictionary holding basic parameteres needed to set up the models
    fixed_params : list
        names of parameters held fixed
    free_params : list
        names of free parameters
    group_models : list or None
        list containing info on which models are grouped in the computation
    model_dicts : list
        list of dicts holding the model parameters
    param_names : list
        all parameters
    psf_lib : PSFLib instance
        psf library used
    """
    
    def __init__(self,fit_config,model_class,psf_lib,group_models=None,**kwargs): 
        """
        Parameters
        ----------
        fit_config : dict
            dictionary with critical fit parameters
        model_class : cls
            the physical model to be used
        psflib : PSFLib instance
            holds 
        **kwargs
            Description
        
        """

        self.fit_config = fit_config
        self.psf_lib = psf_lib
        self._check_config(**kwargs)
        self._parse_params()
        self._make_model_dicts()
        self.group_models = group_models

        kwargs["pad_x"] = kwargs.get("pad_x",(self.psf_lib.psf_cube.shape[-1]//self.psf_lib.psf_oversample)//2 +1)
        kwargs["pad_y"] = kwargs.get("pad_y",((self.psf_lib.psf_cube.shape[-2]//self.psf_lib.psf_oversample)-15)//2+1)

        self._init_model(model_class) 
        self._init_spec(**kwargs)

    @property
    def group_models(self):
        return self._group_models
    
    @group_models.setter 
    def group_models(self,init_list):
        if isinstance(init_list,list):
            self._group_models = init_list
        else:
            self._group_models = [1]*self._N_dict


    def _check_config(self,**kwargs):
        """Perform basic checks on the input and sets the x and y grids automatically from the input PSFLib
        
        Parameters
        ----------
        **kwargs
            any extra parameters for the wavelength grid making
        """
        assert "instrument" in self.fit_config.keys(), "need to specify filter + disperser"
        assert "geometry" in self.fit_config.keys(), "need to specify observing setup"
        assert "fit_params" in self.fit_config.keys(), "need to specify parameters for model"

        if "grid" not in self.fit_config.keys():
            self.fit_config["grid"] = {"x_grid":self.psf_lib.psf_x, "y_grid":self.psf_lib.psf_y}
        
        if "x_grid" not in self.fit_config["grid"].keys():
            self.fit_config["grid"]["x_grid"] = self.psf_lib.psf_x
        if "y_grid" not in self.fit_config["grid"].keys():
            self.fit_config["grid"]["y_grid"] = self.psf_lib.psf_y          

        if "wave_grid" not in self.fit_config["grid"].keys():
            self.fit_config["grid"]["wave_grid"] = self._make_wave_grid(**kwargs)

        # make sure oversampling factor is consistent
        self.fit_config["instrument"]["psf_oversampling"] = self.psf_lib.psf_oversample


    def _make_wave_grid(self,fid_line_wave=None,velrange=1000.,**kwargs):
        """create wavelength grid for model evaluation
        
        Parameters
        ----------
        fid_line_wave : None, optional
            fiducial wavelength of the line centroid, needed to make the grid
        velrange : float, optional
            velocity range considered in the forward modelling - the larger the range, the slower the fit
        **kwargs
            Description
        
        Returns
        -------
        1D numpy.array
            wavelength grid
        """
        l0 = fid_line_wave
        assert isinstance(l0,float), "need to enter (fiducial) line wavelength"
        dl, dv = nyquist_wave(l0,self.fit_config["instrument"]["disperser"], True)
        N = int(np.rint(velrange/dv))
        if N%2 != 0:
            wave_range = np.linspace(l0-dl*(N//2),l0+dl*(N//2),N)
        else: wave_range = np.linspace(l0-dl*(N//2),l0+dl*(N//2),N+1)
        
        return wave_range


    def _parse_params(self):
        """Translates the supplied configuration file with fitting parameters to lists separating
        free parameters, fixed parameters, and parameters that depend on other (free) parameters.
        Subsequently configuration files for individual models (one per exposure) are initialised.
        """

        self.param_names = list(self.fit_config["fit_params"].keys())

        self.fixed_params = []
        self.free_params = []
        self._wave_ind = None
        self.depend_params = {}

        for i,pn in enumerate(self.param_names):

            pn_props = self.fit_config["fit_params"][pn]
            if pn_props["fixed"]:
                self.fixed_params.append((pn,pn_props.get("model",None)))
                if pn=="line_wave": self.fit_config["line_wave"] = pn_props["value"]
                elif pn_props["model"] not in self.fit_config.keys():
                    if callable(pn_props["value"]): 
                        self.fit_config[pn_props["model"]] = [{pn:None}]
                        self.depend_params[pn] = pn_props
                    else: self.fit_config[pn_props["model"]] = [{pn:pn_props["value"]}]
                else:
                    if callable(pn_props["value"]):
                        self.fit_config[pn_props["model"]][0][pn] = None
                        self.depend_params[pn] = pn_props
                    else: self.fit_config[pn_props["model"]][0][pn] = pn_props["value"]

            else:
                self.free_params.append((pn,pn_props.get("model",None)))
                if pn=="line_wave": 
                    self._wave_ind = i
                    self.fit_config["line_wave"] = None
                elif pn_props["model"] not in self.fit_config.keys():
                    self.fit_config[pn_props["model"]] = [{pn:None}]
                else: 
                    self.fit_config[pn_props["model"]][0][pn] = None

        if "n" in self.param_names:
            self.fit_config["morph"][0]["profile"] = "sersic"
        elif "sigma_x" in self.param_names:
            self.fit_config["morph"][0]["profile"] = "gauss"
        elif "x0" in self.param_names or "x0_0" in self.param_names:
            self.fit_config["morph"][0]["profile"] = "point"
        else:
            self.fit_config["morph"][0]["profile"] = "uniform"



    def _make_model_dicts(self):
        """create dictionaries needed for each of the exposure models.
        To do: incorporate multi-component models
        """
        self.model_dicts = []

        if isinstance(self.fit_config["geometry"]["shutter_i"],list):
            N = len(self.fit_config["geometry"]["shutter_i"])

            for i in range(N):
                new_dict = deepcopy(self.fit_config)
                new_dict.pop("fit_params")  # don't need this

                for k in list(self.fit_config["morph"][0].keys()):
                    if "x0_" in k: new_dict["morph"][0].pop(k)
                    elif "y0_" in k: new_dict["morph"][0].pop(k)
                new_dict["morph"][0]["x0"] = self.fit_config["morph"][0].get(f"x0_{i}",0)
                new_dict["morph"][0]["y0"] = self.fit_config["morph"][0].get(f"y0_{i}",0)

                new_dict["geometry"]["shutter_i"] = self.fit_config["geometry"]["shutter_i"][i]
                new_dict["geometry"]["shutter_j"] = self.fit_config["geometry"]["shutter_j"][i]
                new_dict["geometry"]["quadrant"] = self.fit_config["geometry"]["quadrant"][i]
                new_dict["geometry"]["shutter_array"] = self.fit_config["geometry"]["shutter_array"][i]
                new_dict["geometry"]["source_shutter"] = self.fit_config["geometry"]["source_shutter"][i]
                new_dict["geometry"]["psky_x"] = self.fit_config["geometry"]["psky_x"][i]
                new_dict["geometry"]["psky_y"] = self.fit_config["geometry"]["psky_y"][i]

                self.model_dicts.append(new_dict)

        else: 
            new_dict = deepcopy(self.fit_config)
            new_dict.pop("fit_params")  # don't need this
            self.model_dicts.append(new_dict)

        # cleaning up
        self.fit_config.pop("morph",None) 
        self.fit_config.pop("vfield",None)

        self._N_dict = len(self.model_dicts)


    def _update_model_dicts(self,theta):
        '''update model dictionaries based on parameter sampling from prior
        
        Parameters
        ----------
        theta : array
            parameter vector
        '''

        for i in range(self._N_dict):

            for j in range(len(theta)):

                pn,pn_mod = self.free_params[j]
                #print(pn, theta[j])
                if pn=="line_wave": self.model_dicts[i][pn] = theta[j]
                elif pn==f"x0_{i}": self.model_dicts[i][pn_mod][0]["x0"] = theta[j]
                elif pn==f"y0_{i}": self.model_dicts[i][pn_mod][0]["y0"] = theta[j]
                elif "x0_" in pn or "y0_" in pn: continue
                else: self.model_dicts[i][pn_mod][0][pn] = theta[j]

        self._update_dependent_params()


    def _update_dependent_params(self):
        """update parameters that explicitly depend on other free parameters
        """
        if len(self.depend_params)>0:
            for i in range(self._N_dict):
                for pn in list(self.depend_params.keys()):
                    pn_dict = self.depend_params[pn]                    
                    if pn==f"x0_{i}": pn="x0"
                    elif pn==f"y0_{i}": pn="y0"
                    self.model_dicts[i][pn_dict["model"]][0][pn] = pn_dict["value"](self.model_dicts[i])


    def _init_spec(self,**kwargs):
        """initialise the Spec2D models for each exposure
        
        Parameters
        ----------
        **kwargs
            optional arguments for the Spec2D class 
        
        Raises
        ------
        ValueError
            raises error for edge cases -- this is probably fixed now
        """
        self._specs = []
        self._is_sca491 = np.ones(self._N_dict,dtype=bool)


        for i in range(self._N_dict):
            
            if "pix_cood" in kwargs.keys():
                kwargs["pix_limits"] = kwargs["pix_cood"][i]

            self._specs.append(Spec2D(self.model_dicts[i],**kwargs))
            if self._specs[i].xgrid_491 is None: self._is_sca491[i] = False

            if i==0: 
                if self._is_sca491[i]: self._dim_spec = self._specs[i].xgrid_491.shape
                else: self._dim_spec = self._specs[i].xgrid_492.shape
            else:
                if self._is_sca491[i]: dim_spec = self._specs[i].xgrid_491.shape
                else: dim_spec = self._specs[i].xgrid_492.shape 

                if dim_spec != self._dim_spec: 
                    raise ValueError("Line is near the edge, leading to a mismatch between \
                    detector cutout shapes. Workaround is not yet implemented.")


    def _init_model(self,model_class):
        """create a (PSF-free) model for each exposure
        
        Parameters
        ----------
        model_class : Cube 
            callable model cube from msafit.model.models
        """
        self._models = []

        for i in range(self._N_dict):

            self._models.append(model_class(self.model_dicts[i]))



    def prior_transform(self,cube):
        """transforms the unit cube to physical parameter values
        
        Parameters
        ----------
        cube : nd array
            unit cube of size (N,) where N is the number of free parameters
        
        Returns
        -------
        1D array
            parameter vector theta
        """
        theta = cube.copy()
        for i in range(len(theta)):
            pn,pn_mod = self.free_params[i]
            prior_func = self.fit_config["fit_params"][pn]["prior"].unit_transform
            theta[i] = prior_func(cube[i])

        return theta


    def ln_prior_prob(self,theta):
        """computes the log prior probability (used for MCMC)
        
        Parameters
        ----------
        theta : 1D array
            parameter vector
        
        Returns
        -------
        float
            log prior probability
        """
        lnp = 0
        for i in range(len(theta)):
            pn,pn_mod = self.free_params[i]
            prior_func = self.fit_config["fit_params"][pn]["prior"]
            lnp += prior_func(theta[i])

        return lnp


    def theta2model(self,theta,**kwargs):
        """generate a model from a parameter vector
        
        Parameters
        ----------
        theta : 1D array
            parameter vector
        **kwargs
            additional kwargs for the Cube and Spec2D classes
        
        Returns
        -------
        list
            list consists of N spectra (2D arrays), where N is the number of exposures
        """
        self._update_model_dicts(theta)


        models = [] #np.zeros((self._N_dict,*self._dim_spec))

        imod = 0

        for igroup,nmods in enumerate(self._group_models):
            #print(imod)
            self._models[imod](obs_wave=self.model_dicts[imod]["line_wave"],new_param_dict=self.model_dicts[imod],**kwargs)

            for nmod in range(nmods):
                jmod = imod + nmod
                #print(jmod)
                new_spec = deepcopy(self._specs[jmod])
                new_spec.make_spec2d(self._models[imod],self.psf_lib,downsampled=True,rotate_slit=True,return_fluxes=False,**kwargs)

                if self._is_sca491[jmod]:
                    models.append(new_spec.spec_491)
                else:
                    models.append(new_spec.spec_492)

            imod += nmods

        return models




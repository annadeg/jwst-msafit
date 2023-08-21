import numpy as np
from .velocity_field import *
from .light_profile import *
from .cube import Cube
from .config import check_input_config

__all__ = ["ThinDisk","MorphOnly","Sersic","Point","Gauss","UniformIllum"]


class ThinDisk(Cube):

    """
    Class for easy instantiating of a simple thin rotating disk model
    """
    def __init__(self,param_dict):
        self.param_dict = param_dict
        super().__init__(self._param_dict["grid"]["x_grid_sky"], self._param_dict["grid"]["y_grid_sky"], 
            self._param_dict["grid"]["wave_grid"], self._param_dict["grid"].get("wave_sampling","linear"))
        self.model_vel2D = (arctan1D,const_vdisp)
        self.model_light2D = sersic2D

    @property
    def param_dict(self):
        return self._param_dict

    @param_dict.setter
    def param_dict(self,param_dict):
        self._param_dict = check_input_config(param_dict)

    @property
    def model_light2D(self):
        return self._model_light2D

    @model_light2D.setter
    def model_light2D(self,func_light2D):
        self._model_light2D = LightDist2D(func_light2D)

    @property
    def model_vel2D(self):
        return self._model_vel2D

    @model_vel2D.setter
    def model_vel2D(self,vel_funcs):
        func_vel2D, func_sigma2D = vel_funcs
        self._model_vel2D = VelField2D(func_vel2D,func_sigma2D)


    def __call__(self,obs_wave=None,rest_wave=None,z_spec=None,new_param_dict={},**kwargs):
    
        if len(new_param_dict)>0:
            self.param_dict = new_param_dict
            
        self.compute_from2d(modelLight2D=self._model_light2D,modelVel2D=self._model_vel2D,parameter_dict=self.param_dict,
                            obs_wave=obs_wave,rest_wave=rest_wave,z_spec=z_spec,**kwargs)
        



class MorphOnly(Cube):

    """
    Class for a simple model without any velocity info (for path losses & LSFs)

    """

    def __init__(self,param_dict,func_light2D):

        self.param_dict = param_dict
        super().__init__(self._param_dict["grid"]["x_grid_sky"], self._param_dict["grid"]["y_grid_sky"], 
                         self._param_dict["grid"]["wave_grid"], self._param_dict["grid"].get("wave_sampling","linear"))
        self.model_light2D = func_light2D

    @property
    def model_light2D(self):
        return self._model_light2D

    @model_light2D.setter
    def model_light2D(self,func_light2D):
        self._model_light2D = LightDist2D(func_light2D)

    @property
    def param_dict(self):
        return self._param_dict

    @param_dict.setter
    def param_dict(self,param_dict):
        self._param_dict = check_input_config(param_dict)

    def compute(self,obs_wave=None,rest_wave=None,z_spec=None,**kwargs):
        self.compute_from2d(modelLight2D=self._model_light2D,modelVel2D=None,parameter_dict=self.param_dict,
                            obs_wave=obs_wave,rest_wave=rest_wave,z_spec=z_spec,**kwargs)


class Sersic(MorphOnly):

    def __init__(self,param_dict):

        super().__init__(param_dict,sersic2D)

    def __call__(self,obs_wave=None,rest_wave=None,z_spec=None,new_param_dict={},**kwargs):
    
        if len(new_param_dict)>0:
            self.param_dict = new_param_dict
            
        self.compute(obs_wave,rest_wave,z_spec,**kwargs)

        
        
class UniformIllum(MorphOnly):

    def __init__(self,param_dict):
        super().__init__(param_dict,uniform2D)         
        
        
    def __call__(self,obs_wave=None,rest_wave=None,z_spec=None,new_param_dict={},**kwargs):
    
        if len(new_param_dict)>0:
            self.param_dict = new_param_dict
            
        self.compute(obs_wave,rest_wave,z_spec,**kwargs)        
   


class Point(MorphOnly):

    def __init__(self,param_dict):
        super().__init__(param_dict,point2D)    


    def __call__(self,obs_wave=None,rest_wave=None,z_spec=None,new_param_dict={},**kwargs):
    
        if len(new_param_dict)>0:
            self.param_dict = new_param_dict
            
        self.compute(obs_wave,rest_wave,z_spec,**kwargs)



class Gauss(MorphOnly):

    def __init__(self,param_dict):
        super().__init__(param_dict,gauss2D) 
        
        
    def __call__(self,obs_wave=None,rest_wave=None,z_spec=None,new_param_dict={},**kwargs):
    
        if len(new_param_dict)>0:
            self.param_dict = new_param_dict
            
        self.compute(obs_wave,rest_wave,z_spec,**kwargs)        
        
        
        
        
        
        
        
        
        
        
        


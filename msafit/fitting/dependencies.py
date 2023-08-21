import numpy as np 


__all__ = ["thindisk_inclination","fix_turnover","rturnover_from_reff"]


def thindisk_inclination(param_dict,comp_idx=0):

    return np.arccos(param_dict["morph"][comp_idx]["q"])*180./np.pi


def fix_turnover(param_dict,comp_idx=0,fraction=0.25):

    return fraction*param_dict["morph"][comp_idx]["r_e"]


def rturnover_from_reff(param_dict,comp_idx=0):

    return param_dict["vfield"][comp_idx]["frac_rt_re"]*param_dict["morph"][comp_idx]["r_e"]









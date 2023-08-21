import numpy as np



def get_default_morph():

    p_light = {"profile":"sersic","flux":1.0,"x0":0.,"y0":0.,"x0_sky":None,"y0_sky":None,"r_e":0.3, "n":1., "q":0.6, "PA":90.}
    return p_light


def get_default_vfield():

    p_vel = {"r_t":0.2,"v_a":200.,"sigma_0":50.,"PA_vel":90.,"inclination":60.,"x0_vel":0.,"y0_vel":0.}
    return p_vel


def get_default_geometry():
    # important: the pitch here is the average of all shutters and *not* the pitch corresponding to (q=3,i=183,j=85)
    p_geometry = {"quadrant":3,"shutter_i":183,"shutter_j":85,"shutter_array":"1x3","source_shutter":0,"psky_x":0.268,"psky_y":0.530}
    return p_geometry


def get_default_instrument():

    p_inst = {"filter":"F170LP","disperser":"G235M","psf_oversampling":5} 
    return p_inst

def get_default_grid():

    x_range = np.linspace(-0.65,0.65,15)
    y_range = np.linspace(-1.65,1.65,71)
    xx,yy = np.meshgrid(x_range,y_range)
    p_grid = {"x_grid":xx,"y_grid":yy,
                "wave_grid":np.arange(0.6e4,5.3e4,1e3),"wave_sampling":"linear"}
    return p_grid


def get_default_config():

    p_light = get_default_morph()
    p_vel = get_default_vfield()
    p_geometry = get_default_geometry()
    p_inst = get_default_instrument()
    p_grid = get_default_grid()

    parameter_dict = {"geometry":p_geometry, "instrument":p_inst,"morph":[p_light], "vfield":[p_vel],"grid":p_grid}

    return parameter_dict


def check_input_config(parameter_dict):

	# make sure all crucial keys are there
    if "instrument" not in parameter_dict.keys():
        raise KeyError("No instrument setup specified")
    elif "geometry" not in parameter_dict.keys():
        print("Warning: no geometry specified. Assuming 1x3 shutter at q3_i183_j85")
        parameter_dict["geometry"] = get_default_geometry()
    elif "grid" not in parameter_dict.keys():
        print("Warning: no grids specified. Assuming default of factor ~4 oversampling and 1x3 shutter")
        parameter_dict["grid"] = get_default_grid()


    # now compute other critical inputs from the parameter_dict if not already there
    # this converts from pitch units to physical units

    if "x_grid_sky" not in parameter_dict["grid"].keys() or "y_grid_sky" not in parameter_dict["grid"].keys():
        parameter_dict["grid"]["x_grid_sky"] = parameter_dict["grid"]["x_grid"]*parameter_dict["geometry"]["psky_x"]
        parameter_dict["grid"]["y_grid_sky"] = parameter_dict["grid"]["y_grid"]*parameter_dict["geometry"]["psky_y"]


    if "morph" in parameter_dict.keys():

        if isinstance(parameter_dict["morph"],list) == False:
            parameter_dict["morph"] = [parameter_dict["morph"]]

        for i in range(len(parameter_dict["morph"])):
            #if "x0_sky" not in parameter_dict["morph"][i].keys() or "y0_sky" not in parameter_dict["morph"][i].keys():
                
            try:
                # y0 is the *in-shutter* offset, so for a source in the top/bottom shutter we need to subtract one full pitch length
                # lower j is in positive y direction on the detector, because it's flipped 180 degrees
                sloc = parameter_dict["geometry"]["source_shutter"]
                
                if parameter_dict["morph"][i]["x0"] is not None and parameter_dict["morph"][i]["y0"] is not None:
                    parameter_dict["morph"][i]["x0_sky"] = parameter_dict["morph"][i]["x0"]*parameter_dict["geometry"]["psky_x"]
                    parameter_dict["morph"][i]["y0_sky"] = (parameter_dict["morph"][i]["y0"]+sloc)*parameter_dict["geometry"]["psky_y"]
                else:
                    parameter_dict["morph"][i]["x0_sky"] = parameter_dict["morph"][i]["x0_sky"]
                    parameter_dict["morph"][i]["y0_sky"] = parameter_dict["morph"][i]["y0_sky"]+(sloc*parameter_dict["geometry"]["psky_y"])            
            except KeyError:
                print("WARNING: no shutter offsets specified, assuming (x0=0,y0=0)")
                parameter_dict["morph"][i]["x0_sky"] = 0.
                parameter_dict["morph"][i]["y0_sky"] = 0.




    if "vfield" in parameter_dict.keys(): 

        if isinstance(parameter_dict["vfield"],list) == False:
            parameter_dict["vfield"] = list(parameter_dict["vfield"])

        for j in range(len(parameter_dict["vfield"])):
            # we switch off the option to offset the velocity field from the light profile for now...
            # need to implement prior dependencies for this to work properly
            #if "x0_vel" not in parameter_dict["vfield"][j].keys() or "y0_vel" not in parameter_dict["vfield"][j].keys() :
            parameter_dict["vfield"][j]["x0_vel"] = parameter_dict["morph"][j]["x0_sky"]
            parameter_dict["vfield"][j]["y0_vel"] = parameter_dict["morph"][j]["y0_sky"]
                
            #if "PA_vel" not in parameter_dict["vfield"][j].keys():
            parameter_dict["vfield"][j]["PA_vel"] = parameter_dict["morph"][j]["PA"]        

    return parameter_dict




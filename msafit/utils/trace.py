import numpy as np


def get_dlambda_dx(f_trace_x,f_trace_y,wl_range,dlambda=100.):

    x_fpa, y_fpa = f_trace_x(wl_range), f_trace_y(wl_range)
    x_fpa_up, y_fpa_up = f_trace_x(wl_range+dlambda), f_trace_y(wl_range+dlambda)
    x_fpa_low, y_fpa_low = f_trace_x(wl_range-dlambda), f_trace_y(wl_range-dlambda)

    d1 = dlambda / np.fabs(x_fpa_up-x_fpa)
    d2 = dlambda / np.fabs(x_fpa_low-x_fpa)

    return np.mean(np.array([d1,d2]),axis=0)

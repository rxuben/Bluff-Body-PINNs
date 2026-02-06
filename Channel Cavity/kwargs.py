import inspect
import numpy as np

###### Safe kwargs helpers ######
def _kwargs_for_init(cls, **kwargs):
    try:
        params = inspect.signature(cls.__init__).parameters
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return {}

def _kwargs_for_call(func, **kwargs):
    try:
        params = inspect.signature(func).parameters
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return {}

###### SDF helper ######
# not really supposed to be in this kwargs file but no obvious other place to put it
def compute_sdf_for_geom(geom, x_arr, y_arr):
    invar = {"x": x_arr.astype(np.float32), "y": y_arr.astype(np.float32)}
    sdf = geom.sdf(invar, params={})["sdf"].astype(np.float32)
    if sdf.ndim == 1:
        sdf = sdf.reshape(-1, 1)
    return sdf
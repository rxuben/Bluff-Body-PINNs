import numpy as np
from physicsnemo.sym.domain.constraint import PointwiseConstraint
from kwargs import _kwargs_for_call

# load csv of sampled CFD data and return it as a dictionary
def load_line_csv_xyuv(path: str):

    # csv MUST have header with columns: x,y,u,v
    arr = np.genfromtxt(path, delimiter=",", names=True)
    cols = {c.lower(): c for c in arr.dtype.names}
    need = ["x", "y", "u", "v"]
    miss = [k for k in need if k not in cols]
    if miss:
        raise ValueError(f"CSV {path} missing columns {miss}. Found: {arr.dtype.names}")

    out = {}
    for k in need:
        a = arr[cols[k]].astype(np.float32)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        out[k] = a
    return out

# reduce the number of points along a line to keep training manageable
# if line is already small: return unchanged
def downsample_line(line, max_points=4000, method="uniform"):

    x, y, u, v = line["x"], line["y"], line["u"], line["v"]
    n = x.shape[0]
    if (max_points is None) or (max_points <= 0) or (n <= max_points):
        return line

    if method == "random":
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        idx.sort()
    else:
        idx = np.linspace(0, n - 1, max_points).astype(int)

    return {
        "x": x[idx],
        "y": y[idx],
        "u": u[idx],
        "v": v[idx],
    }

# remove points that lie outside of the cavity
def filter_to_cavity(line, y_max=0.0):
    x, y, u, v = line["x"], line["y"], line["u"], line["v"]
    mask = (y[:, 0] <= float(y_max))
    return {"x": x[mask], "y": y[mask], "u": u[mask], "v": v[mask]}

# turn line data into PINN constraint
def add_line_data_constraint(domain, nodes, name, line, batch_size, weight, shuffle=False):

    n = line["x"].shape[0]
    if n == 0:
        raise ValueError(f"Line dataset '{name}' is empty after filtering/downsampling.")

    bs = int(min(batch_size, n)) if batch_size is not None else int(n)

    lam_u = np.full_like(line["u"], float(weight), dtype=np.float32)
    lam_v = np.full_like(line["v"], float(weight), dtype=np.float32)

    extra = _kwargs_for_call(
        PointwiseConstraint.from_numpy,
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
    )

    c = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"x": line["x"], "y": line["y"]},
        outvar={"u": line["u"], "v": line["v"]},
        batch_size=bs,
        lambda_weighting={"u": lam_u, "v": lam_v},
        **extra,
    )
    domain.add_constraint(c, name)
import os
import numpy as np
import sympy as sp
from sympy import Symbol, Eq, Abs, And, Le, Ge
import torch

#### NVIDIA imports ####
import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

# file imports (functions i've made)
from plotting import CanyonFlowPlotter
from kwargs import compute_sdf_for_geom, _kwargs_for_init
from supervised_functions import (
    load_line_csv_xyuv,
    downsample_line,
    filter_to_cavity,
    add_line_data_constraint,
)

# copy either of the below lines into the terminal to open tensorboard
# tensorboard --logdir="./Channel Cavity/outputs" --port=7007
# tensorboard --logdir="./Results" --port=7007

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # general dimensions and parameters
    channel_height = float(0.75)
    channel_length = float(6.0)
    cavity_height = float(3.0)
    cavity_width = float(3.0)
    ledge = float( (channel_length - cavity_width) / 2 )
    U_in = float(0.3)
    V_in = float(0.0)

    # define sympy varaibles to parametize domain curves
    x, y = Symbol("x"), Symbol("y")

    # Geometry definitions
    channel_bottomleft = (-ledge, 0)
    channel_topright = (cavity_width + ledge, channel_height)
    channel = Rectangle(channel_bottomleft, channel_topright)

    cavity_bottomleft = (0, -cavity_height)
    cavity_topright = (cavity_width, 0)
    cavity = Rectangle(cavity_bottomleft, cavity_topright)

    geom = channel + cavity

    # Initialise NavierStokes and ZeroEquation
    ze = ZeroEquation(nu=0.0001, rho=1.0, dim=2, max_distance=(channel_height / 2))
    ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=2, time=False)
    ndv = NormalDotVec(["u", "v"])

    # Create neural network
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    # create a list of all the nodes
    nodes = ns.make_nodes() + ze.make_nodes() + ndv.make_nodes() + [flow_net.make_node(name="flow_network")]

    # the domain object contains the PDE and its boundary conditions
    domain = Domain()

    # important robustness knobs for constraints
    constraint_common = dict(
        shuffle=False,
        drop_last=False,
        num_workers=0,
        batch_per_epoch=1,
    )

    ## Boundary conditions
    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": U_in, "v": V_in},
        batch_size=cfg.batch_size.Inlet,
        criteria=And(Eq(x, -ledge), Ge(y, 0), Le(y, channel_height)),
        **_kwargs_for_init(PointwiseBoundaryConstraint, **constraint_common),
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"p": 0, "u__x": 0, "v__x": 0},
        batch_size=cfg.batch_size.Outlet,
        criteria=And(Eq(x, cavity_width + ledge), Ge(y, 0), Le(y, channel_height)),
        **_kwargs_for_init(PointwiseBoundaryConstraint, **constraint_common),
    )
    domain.add_constraint(outlet, "outlet")

    # symmetry on top channel wall (slip)
    symmetry_channel_topwall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"v": 0, "u__y": 0},
        batch_size=cfg.batch_size.Symmetry,
        criteria=Eq(y, channel_height),
        **_kwargs_for_init(PointwiseBoundaryConstraint, **constraint_common),
    )
    domain.add_constraint(symmetry_channel_topwall, "symmetry_channel_topwall")

    # no slip on channel ledge
    no_slip_ledge = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlipLedge,
        criteria = Eq(y, 0) & (((x >= -1.5) & (x <= 0)) | ((x >= 3) & (x <= 4.5))),
        **_kwargs_for_init(PointwiseBoundaryConstraint, **constraint_common),
    )
    domain.add_constraint(no_slip_ledge, "no_slip_ledge")

    # no slip on cavity walls and floor
    no_slip_cavity = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cavity,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlipCavity,
        criteria=y < 0,
        **_kwargs_for_init(PointwiseBoundaryConstraint, **constraint_common),
    )
    domain.add_constraint(no_slip_cavity, "no_slip_cavity")

    # weighting near boundaries/opening
    w0 = float(getattr(cfg.custom, "w0", 0.05))
    w = sp.Float(w0) + sp.Symbol("sdf")

    # interior
    # this is a PDE constraint rather than a boundary condition
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        compute_sdf_derivatives=True,
        lambda_weighting={"continuity": w, "momentum_x": w, "momentum_y": w},
        **_kwargs_for_init(PointwiseInteriorConstraint, **constraint_common),
    )
    domain.add_constraint(interior, "interior")

    # integral continuity
    Q_in = float(U_in * channel_height)

    def integral_criteria_union(invar, params):
        sdf_c = cavity.sdf(invar, params)["sdf"]
        sdf_h = channel.sdf(invar, params)["sdf"]
        return np.logical_or(sdf_c > 0.0, sdf_h > 0.0)

    # Create 16 fixed vertical lines spanning from x = -1.5 to x = 4.5
    num_lines = cfg.batch_size.num_integral_continuity
    x_positions = np.linspace(-ledge, cavity_width + ledge, num_lines)

    for i, x_pos in enumerate(x_positions):
        # Determine vertical extent based on x position
        if 0 <= x_pos <= cavity_width:
            # Inside cavity region: line spans full height (cavity + channel)
            y_bottom = -cavity_height
        else:
            # Outside cavity: line spans only channel height
            y_bottom = 0.0

        integral_line = Line(
            (x_pos, y_bottom),
            (x_pos, channel_height),
            1,
        )

        integral_cont = IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=integral_line,
            outvar={"normal_dot_vel": Q_in},
            batch_size=1,
            integral_batch_size=cfg.batch_size.integral_continuity,
            lambda_weighting={"normal_dot_vel": 0.1},
            **_kwargs_for_init(IntegralBoundaryConstraint, **constraint_common),
        )
        domain.add_constraint(integral_cont, f"integral_continuity_{i}")

    # supervised learning
    # enforce conditions from valid cfd data at certain horizontal and vertical lines
    data_cfg = getattr(cfg.custom, "data", None)
    if data_cfg and bool(getattr(data_cfg, "use_line_data", True)):

        v1 = to_absolute_path(data_cfg.Vline1_csv)
        v2 = to_absolute_path(data_cfg.Vline2_csv)
        v3 = to_absolute_path(data_cfg.Vline3_csv)
        h1 = to_absolute_path(data_cfg.Hline1_csv)
        h2 = to_absolute_path(data_cfg.Hline2_csv)
        h3 = to_absolute_path(data_cfg.Hline3_csv)

        for p in [v1, v2, v3, h1, h2, h3]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing line CSV: {p}")

        # downsample controls
        max_pts = int(getattr(data_cfg, "max_points_per_line", 4000))
        method = str(getattr(data_cfg, "downsample_method", "uniform"))
        restrict = bool(getattr(data_cfg, "restrict_to_canyon", True))

        V1 = downsample_line(load_line_csv_xyuv(v1), max_pts, method)
        V2 = downsample_line(load_line_csv_xyuv(v2), max_pts, method)
        V3 = downsample_line(load_line_csv_xyuv(v3), max_pts, method)
        H1 = downsample_line(load_line_csv_xyuv(h1), max_pts, method)
        H2 = downsample_line(load_line_csv_xyuv(h2), max_pts, method)
        H6 = downsample_line(load_line_csv_xyuv(h3), max_pts, method)

        if restrict:
            V1 = filter_to_cavity(V1, 0.0)
            V2 = filter_to_cavity(V2, 0.0)
            V3 = filter_to_cavity(V3, 0.0)
            H1 = filter_to_cavity(H1, 0.0)
            H2 = filter_to_cavity(H2, 0.0)
            H6 = filter_to_cavity(H6, 0.0)

        # weights
        w_H1 = float(getattr(cfg.custom, "w_Hline1", 3.0))
        w_mid = float(getattr(cfg.custom, "w_mid", 1.5))
        w_low = float(getattr(cfg.custom, "w_low", 1.0))

        bs_line = int(cfg.batch_size.LineData)

        add_line_data_constraint(domain, nodes, "data_Vline1", V1, bs_line, w_low, shuffle=False)
        add_line_data_constraint(domain, nodes, "data_Vline2", V2, bs_line, w_mid, shuffle=False)
        add_line_data_constraint(domain, nodes, "data_Vline3", V3, bs_line, w_low, shuffle=False)
        add_line_data_constraint(domain, nodes, "data_Hline1", H1, bs_line, w_H1, shuffle=False)
        add_line_data_constraint(domain, nodes, "data_Hline2", H2, bs_line, w_mid, shuffle=False)
        add_line_data_constraint(domain, nodes, "data_Hline6", H6, bs_line, w_low, shuffle=False)

        print(f"[INFO] Line data loaded (downsample max={max_pts}, method={method}, canyon_only={restrict}).", flush=True)

    # measure and report how well the physics is being satisfied across the domain
    global_monitor = PointwiseMonitor(
        geom.sample_interior(2000),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(
                var["area"] * torch.abs(var["continuity"])
            ),
            "momentum_imbalance": lambda var: torch.sum(
                var["area"]* (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
            ),
        },
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(global_monitor)

    # plotting settings
    use_inferencer = bool(getattr(cfg.custom, "use_inferencer", True))
    grid_n = int(getattr(cfg.custom, "plot_grid_n", 140))
    quiver_stride = int(getattr(cfg.custom, "quiver_stride", 16))
    stream_density = float(getattr(cfg.custom, "stream_density", 1.2))

    if use_inferencer:
        infer_bs = int(cfg.batch_size.Inferencer)
        plotter = CanyonFlowPlotter(grid_n=grid_n, quiver_stride=quiver_stride, stream_density=stream_density)

        Nx = int(getattr(cfg.custom, "Nx_canyon", 129))
        Ny = int(getattr(cfg.custom, "Ny_canyon", 129))
        xs = np.linspace(0.0, cavity_width, Nx, dtype=np.float32)
        ys = np.linspace(- cavity_height, 0.0, Ny, dtype=np.float32)
        Xc, Yc = np.meshgrid(xs, ys, indexing="ij")
        Xc1 = Xc.reshape(-1, 1)
        Yc1 = Yc.reshape(-1, 1)
        sdf_c = compute_sdf_for_geom(geom, Xc1, Yc1)
        invar_canyon = {"x": Xc1, "y": Yc1, "sdf": sdf_c}

        domain.add_inferencer(
            PointwiseInferencer(
                nodes=nodes,
                invar=invar_canyon,
                output_names=["u", "v", "p", "nu"],
                batch_size=infer_bs,
                plotter=plotter,
                requires_grad=True,
            ),
            "infer_canyon",
        )

        Nx2 = int(getattr(cfg.custom, "Nx_channel", 129))
        Ny2 = int(getattr(cfg.custom, "Ny_channel", 65))
        xs2 = np.linspace(-ledge, cavity_width + ledge, Nx2, dtype=np.float32)
        ys2 = np.linspace(0.0, channel_height, Ny2, dtype=np.float32)
        Xh, Yh = np.meshgrid(xs2, ys2, indexing="ij")
        Xh1 = Xh.reshape(-1, 1)
        Yh1 = Yh.reshape(-1, 1)
        sdf_h = compute_sdf_for_geom(geom, Xh1, Yh1)
        invar_channel = {"x": Xh1, "y": Yh1, "sdf": sdf_h}

        domain.add_inferencer(
            PointwiseInferencer(
                nodes=nodes,
                invar=invar_channel,
                output_names=["u", "v", "p", "nu"],
                batch_size=infer_bs,
                plotter=plotter,
                requires_grad=True,
            ),
            "infer_channel",
        )

        print("[INFO] Inferencers registered (will write PNGs at rec_inference_freq).", flush=True)

    # solver
    print("[DEBUG] Domain built. Creating solver...", flush=True)
    slv = Solver(cfg, domain)
    print("[DEBUG] Solver created. Starting solve()...", flush=True)
    slv.solve()

################## call the function ##################
if __name__ == "__main__":
    run()
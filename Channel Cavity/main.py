import os
import warnings

from sympy import Symbol, Eq, Abs
import matplotlib.pyplot as plt
import torch

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key

from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,)

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # general dimensions
    channel_height = 0.1
    channel_length = 1.2
    cavity_height = 1.0
    cavity_width = 1.0

    # define sympy varaibles to parametize domain curves
    x, y = Symbol("x"), Symbol("y")

    # Geometry definitions
    channel_bottomleft = (- channel_length / 2, - channel_height / 2)
    channel_topright = (channel_length / 2, channel_height / 2)
    channel = Rectangle(channel_bottomleft, channel_topright)

    cavity_bottomleft = (- cavity_width / 2, - (channel_height / 2) - cavity_height)
    cavity_topright = (cavity_width / 2, - channel_height / 2)
    cavity = Rectangle(cavity_bottomleft, cavity_topright)

    geom = channel + cavity

    # Initialise NavierStokes equations
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)

    # Create neural network
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    # create a list of all the nodes
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # the domain object contains the PDE and its boundary conditions
    domain = Domain()

    ## Boundary conditions
    # no slip on cavity wall
    no_slip_cavity = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cavity,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlipCavity,
        criteria=y < -(channel_height / 2),
    )
    domain.add_constraint(no_slip_cavity, "no_slip_cavity")

    # no slip on channel ledge
    no_slip_ledge = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlipLedge,
        criteria = Eq(y, -(channel_height / 2)) & (Abs(x) > 0.5),
    )
    domain.add_constraint(no_slip_ledge, "no_slip_ledge")

    # symmetry on top channel wall
    symmetry_channel_topwall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"v": 0, "u__y": 0, "p__y": 0},
        batch_size=cfg.batch_size.Symmetry,
        criteria=Eq(y, channel_height / 2),
    )
    domain.add_constraint(symmetry_channel_topwall, "symmetry_channel_topwall")

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": 0.3, "v": 0},
        batch_size=cfg.batch_size.Inlet,
        criteria=Eq(x, -(channel_length / 2)),
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"p": 0, "u__x": 0, "v__x": 0},
        batch_size=cfg.batch_size.Outlet,
        criteria=Eq(x, (channel_length / 2)),
    )
    domain.add_constraint(outlet, "outlet")

    # interior
    # this is a PDE constraint rather than a boundary condition
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        # compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")

    global_monitor = PointwiseMonitor(
        geom.sample_interior(100),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(
                var["area"] * torch.abs(var["continuity"])
            ),
            "momentum_imbalance": lambda var: torch.sum(
                var["area"]
                * (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
            ),
        },
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(global_monitor)

    # solver
    slv = Solver(cfg, domain)
    slv.solve()

################## call the function ##################
if __name__ == "__main__":
    run()
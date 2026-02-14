import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from physicsnemo.sym.utils.io import ValidatorPlotter
from physicsnemo.sym.utils.io.plotter import InferencerPlotter

# Plotter: contour + quiver + streamlines
class CanyonFlowPlotter(InferencerPlotter):
    def __init__(self, grid_n=140, quiver_stride=16, stream_density=1.2):
        super().__init__()
        self.grid_n = int(grid_n)
        self.quiver_stride = int(quiver_stride)
        self.stream_density = float(stream_density)

    def __call__(self, invar, outvar):
        if "x" not in invar or "y" not in invar:
            return []
        if "u" not in outvar or "v" not in outvar:
            return []

        extent, out_i = self._interpolate_2D(
            self.grid_n,
            {"x": invar["x"], "y": invar["y"]},
            outvar,
        )
        x1 = np.linspace(extent[0], extent[1], self.grid_n)
        y1 = np.linspace(extent[2], extent[3], self.grid_n)

        # (nx, ny) -> transpose to (ny, nx) for plotting
        U = np.nan_to_num(out_i["u"].T)
        V = np.nan_to_num(out_i["v"].T)
        P = out_i.get("p", None)
        if P is not None:
            P = np.nan_to_num(P.T)

        speed = np.sqrt(U**2 + V**2)
        X, Y = np.meshgrid(x1, y1)

        figs = []

        # speed contour
        f1 = plt.figure(figsize=(6, 4), dpi=150)
        plt.imshow(speed, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="|V|")
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Speed |V|")
        plt.tight_layout()
        figs.append((f1, "speed_contour"))

        # quiver
        s = max(1, self.quiver_stride)
        f2 = plt.figure(figsize=(6, 4), dpi=150)
        plt.imshow(speed, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="|V|")
        plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s],
                   angles="xy", scale_units="xy")
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Quiver on |V|")
        plt.tight_layout()
        figs.append((f2, "quiver"))

        # streamlines
        f3 = plt.figure(figsize=(6, 4), dpi=150)
        plt.imshow(speed, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="|V|")
        plt.streamplot(x1, y1, U, V, density=self.stream_density,
                       linewidth=1.0, arrowsize=1.0)
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Streamlines on |V|")
        plt.tight_layout()
        figs.append((f3, "streamlines"))

        # pressure contour
        if P is not None:
            f4 = plt.figure(figsize=(6, 4), dpi=150)
            plt.imshow(P, origin="lower", extent=extent, aspect="auto")
            plt.colorbar(label="p")
            plt.xlabel("x"); plt.ylabel("y"); plt.title("Pressure p")
            plt.tight_layout()
            figs.append((f4, "pressure_contour"))

        return figs

class MaskedValidatorPlotter(ValidatorPlotter):
    def __init__(self, max_triangle_size=0.05):
        self.max_triangle_size = max_triangle_size

    def __call__(self, invar, true_outvar, pred_outvar):
        x = invar["x"].flatten()
        y = invar["y"].flatten()

        # build triangulation with mask
        triang = tri.Triangulation(x, y)
        triangles = triang.triangles
        xT = x[triangles]
        yT = y[triangles]
        too_large = (
            (np.max(xT, axis=1) - np.min(xT, axis=1) > self.max_triangle_size) |
            (np.max(yT, axis=1) - np.min(yT, axis=1) > self.max_triangle_size)
        )
        triang.set_mask(too_large)

        figs = []
        for var in true_outvar:
            true_vals = true_outvar[var].flatten()
            pred_vals = pred_outvar[var].flatten()
            diff_vals = pred_vals - true_vals

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, vals, title in zip(axes,
                                       [true_vals, pred_vals, diff_vals],
                                       [f"{var}_true", f"{var}_pred", f"{var}_diff"]):
                tcf = ax.tricontourf(triang, vals, levels=50, cmap="jet")
                plt.colorbar(tcf, ax=ax)
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal")

            plt.tight_layout()
            figs.append((fig, var))

        return figs
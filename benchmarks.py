"""
Particle Swarm Optimization applied to 3D test functions for unconstrained non-convex optimization algorithms.
"""
import numpy as np
from matplotlib import pyplot as plt

from swarm import Swarm
from utils import TEXT, ANIMATION, get_out_mode, gen_data, animate, save


def rastrigin(x, y):
    """
    Rastrigin function R^2 -> R.

    Global minimum = 0 = f(0, 0).

    Single global minimum, lots of local minima around it.
    """
    return x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y) + 20


def rosenbrock(x, y):
    """
    Rosenbrock function R^2 -> R.

    Global minimum = 0 = f(1, 1).

    The global minimum is inside a long, narrow, parabolic-shaped flat valley.
    Finding the valley is trivial; converging to the global minimum, however, is not.
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


INFO = {  # function parameters for the animation
    rastrigin: {
        "min_point": (0, 0),
        "plot_3d": "contour",
        "levels": 5,
        "x_min": -5,
        "x_max": 5,
        "y_min": -5,
        "y_max": 5,
        "elevation": 30,
        "azimuth": -60,
        "fps": 20
    },
    rosenbrock: {
        "min_point": (1, 1),
        "plot_3d": "surface",
        "levels": 50,
        "x_min": -1.5,
        "x_max": 2,
        "y_min": -0.5,
        "y_max": 3,
        "elevation": 20,
        "azimuth": -140,
        "fps": 20
    }
}


def plot(f):
    """Plot surface and contour of function ``f``: R^2 -> R."""
    fig = plt.figure(f.__name__.capitalize() + " function", figsize=(12, 7))

    x = np.arange(INFO[f]["x_min"], INFO[f]["x_max"], .01)
    y = np.arange(INFO[f]["y_min"], INFO[f]["y_max"], .01)
    x, y = np.meshgrid(x, y)
    z = f(x, y)

    ax1 = plt.subplot(121, projection="3d")
    ax1.set_title("Surface")
    if INFO[f]["plot_3d"] == "contour":
        ax1.contour3D(x, y, z, 100, cmap="viridis")
    else:
        ax1.plot_surface(x, y, z, cmap="viridis")
    ax1.view_init(INFO[f]["elevation"], INFO[f]["azimuth"])

    ax2 = plt.subplot(122)
    ax2.set_title("Contour")
    ax2.set_aspect("equal")
    ax2.contour(x, y, z, INFO[f]["levels"])

    ax2.plot(INFO[f]["min_point"][0], INFO[f]["min_point"][1], "rx")

    return fig, ax1, ax2


def display(f):
    """Display surface and contour of ``f`` without animation."""
    plot(f)
    plt.show()


def init(f):
    """Initialize animated objects."""
    fig, ax1, ax2 = plot(f)

    title = ax2.text(INFO[f]["x_min"], INFO[f]["y_max"], "", bbox={"facecolor": "w", "pad": 5}, fontsize=12)
    particles, = ax2.plot([], [], "r.")
    particles_3d, = ax1.plot([], [], "r.")

    return fig, title, particles, particles_3d


def update(data):
    """Perform animation step."""
    x, y, z, count = data

    particles.set_data(x, y)
    particles_3d.set_data(x, y)
    particles_3d.set_3d_properties(z)
    title.set_text("Epochs: {}".format(count))

    return particles, particles_3d, title


if __name__ == "__main__":
    out_mode = get_out_mode()

    for f in INFO.keys():
        print("\n" + f.__name__.capitalize() + " function")

        swarm = Swarm([(INFO[f]["x_min"], INFO[f]["x_max"]), (INFO[f]["y_min"], INFO[f]["y_max"])], f, 1e-4)

        if out_mode == ANIMATION:
            display(f)
            fig, title, particles, particles_3d = init(f)
            anim = animate(fig, update, lambda: gen_data(swarm), INFO[f]["fps"])
#            save(anim, f.__name__, INFO[f]["fps"], ".mp4")
        elif out_mode == TEXT:
            swarm.minimize()

    print()

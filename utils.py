"""
Functions shared by both ``benchmarks.py`` and ``detection.py``.
"""
from matplotlib import pyplot as plt, animation as anim

TEXT = "t"
ANIMATION = "a"


def get_out_mode():
    """Let the user decide how to display the output."""
    mode = input("\nHow do you want the output to be displayed? Type \"t\" for text-only, \"a\" for animation: ")
    while mode != TEXT and mode != ANIMATION:
        mode = input("Please type either \"t\" or \"a\": ")
    return mode


def gen_data(swarm):
    """Generate data for the animation step."""
    while swarm.has_not_converged():
        swarm.update_swarm()

        x = swarm.positions[:, 0]
        y = swarm.positions[:, 1]
        z = swarm.get_f_values()
        count = swarm.epoch_count

        yield x, y, z, count


def animate(fig, update, gen_data, fps):
    """
    Create and show an animation.

    :param fig: matplotlib.pyplot.Figure object on which to perform the animation.
    :param update: function performing the animation step.
    :param gen_data: function generating the data for the animation step.
    :param fps: frames per second.
    :return: matplotlib.animation.FuncAnimation object.
    """
    animation = anim.FuncAnimation(fig, update, gen_data, interval=1e3/fps, blit=True, repeat=False, save_count=1500)
    plt.show()
    return animation


def save(animation, name, fps, ext=".gif"):
    """
    Save animation as either a .gif or .mp4 file.

    :param animation: matplotlib.animation.FuncAnimation object.
    :param name: name of the saved file.
    :param fps: frames per second.
    :param ext: extension of the saved file, either .gif or .mp4.
    """
    writer = anim.PillowWriter(fps) if ext == ".gif" else anim.FFMpegWriter(fps)
    animation.save("images/" + name + ext, writer)

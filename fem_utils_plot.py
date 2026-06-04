import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
import imageio
from IPython.display import Image, display


def xtplot(grid, u, t, plot_type="fade", color="b", linestyle="-", name_gif="solution_animation"):
    """
    Plot FEM solution in different formats.

    Args:
        grid (Grid): Finite element grid.
        u (np.ndarray): Solution matrix (space x time).
        t (np.ndarray): Time vector.
        plot_type (str): Plot type: "animation", "surface", "fade".
        color (str): Plot color.
        linestyle (str): Line style.
        name_gif (str): GIF filename base.

    Returns:
        None
    """
    match plot_type:
        case "animation":
            animate_fem(grid, u, t, color, name_gif)
        case "surface":
            surface_fem(grid, u, t)
        case "fade":
            fade_fem(grid, u, t, color, linestyle)
        case _:
            raise RuntimeError("Error: il plot_type in input è sbagliato")


def animate_fem(grid, u, t, color, name_gif):
    """
    Create and display GIF animation.

    Args:
        grid (Grid): Finite element grid.
        u (np.ndarray): Solution matrix.
        t (np.ndarray): Time vector.
        color (str): Plot color.
        name_gif (str): Output GIF name.

    Returns:
        None
    """
    def draw_frame(i):
        return plot_frame(grid, u, t, i, color)

    save_gif(draw_frame, len(t), name_gif)
    name_gif = name_gif + '.gif'
    display(Image(name_gif))


def plot_frame(grid, u, t, i, color='blue'):
    """
    Plot a single time frame.

    Args:
        grid (Grid): Finite element grid.
        u (np.ndarray): Solution matrix.
        t (np.ndarray): Time vector.
        i (int): Time index.
        color (str): Plot color.

    Returns:
        matplotlib.figure.Figure: Figure object.

    Raises:
        ValueError: If time index is out of range.
    """
    if i >= len(t):
        raise ValueError("Errore: indice frame fuori intervallo")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_fem(grid, u[:, i], color)
    ax.set(xlabel='x', ylabel='u(x,t)',
           title=f'Current time: t = {t[i]:.2f}',
           xlim=(grid.a - grid.h, grid.b + grid.h),
           ylim=(np.min(u) - 0.3 * (np.max(u) - np.min(u)),
                 np.max(u) + 0.3 * (np.max(u) - np.min(u))))
    return fig


def plot_fem(grid, u, color="b", linestyle="-", linewidth=2, alpha=1.0, label=None):
    """
    Plot FEM solution.

    Args:
        grid (Grid): Finite element grid.
        u (np.ndarray or callable): Solution values or function.
        color (str): Line color.
        linestyle (str): Line style.
        linewidth (float): Line width.
        alpha (float): Transparency.
        label (str): Legend label.

    Returns:
        None

    Raises:
        RuntimeError: If u is neither an ndarray nor a callable.
    """
    if isinstance(u, np.ndarray):
        plt.plot(grid.nodes, u, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
    elif callable(u):
        xplot = np.linspace(grid.a, grid.b, 1000)
        plt.plot(xplot, u(xplot), color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
    else:
        raise RuntimeError("u deve essere ndarray oppure callable")


def save_gif(draw_frame, frames, filename, dt=1.0 / 24.0):
    """
    Save animation as GIF.

    Args:
        draw_frame (callable): Frame drawing function.
        frames (int): Number of frames.
        filename (str): Output filename.
        dt (float): Frame duration in seconds.

    Returns:
        None
    """
    images = []
    for i in range(frames):
        draw_frame(i)
        fig = plt.gcf()
        fig.canvas.draw()
        images.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)
    imageio.mimsave(filename.replace(".gif", "") + ".gif", images, duration=dt)


def surface_fem(grid, u, t):
    """
    Plot heatmap and 3D surface of the solution.

    Args:
        grid (Grid): Finite element grid.
        u (np.ndarray): Solution matrix (space × time).
        t (np.ndarray): Time vector.

    Returns:
        None
    """
    X, T = np.meshgrid(grid.nodes, t, indexing="ij")

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    pcm = ax1.pcolormesh(X, T, u, shading="auto", cmap="plasma")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title("Vista dall'alto")
    fig1.colorbar(pcm, ax=ax1)

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf = ax2.plot_surface(X, T, u, cmap="plasma", linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u(x,t)")
    ax2.set_title("Surface FEM")
    fig2.colorbar(surf, ax=ax2, shrink=0.6)
    ax2.view_init(elev=30, azim=-90)
    plt.show()


def fade_fem(grid, u, t, color, linestyle):
    """
    Plot time evolution as overlapping faded curves.

    Args:
        grid (Grid): Finite element grid.
        u (np.ndarray): Solution matrix (space × time).
        t (np.ndarray): Time vector.
        color (str): Line color.
        linestyle (str): Line style.

    Returns:
        None
    """
    nt = t.size
    if nt > 10:
        dt = ceil(nt / 10) if (nt % 2 == 0) else ceil((nt - 1) / 10)
    else:
        dt = 1

    idx = range(0, nt, dt)
    plt.figure(figsize=(6, 4))

    for k, j in enumerate(idx):
        alpha_value = 0.4 + 0.6 * (j / (nt - 1))
        plot_fem(grid, u[:, j], color=color, linestyle=linestyle, alpha=alpha_value, label=f"t= {t[j]:.2f}")

    plt.title("Evoluzione temporale")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.show()
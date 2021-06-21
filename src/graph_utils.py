import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_sample(sample, fig_name):
    """
    Plots a sample of shape (num_items, 3) in 3D.

    :param sample: Sample from dynamical system (num_samples, n_x).
    :type sample: numpy.ndarray

    :param fig_name: The name of the file to save the plot to.
    :type fig_name: string
    """
    fig, axs = plt.subplots(3, figsize=(5, 15))
    axs[0].plot(sample[:, 0], sample[:, 1], "k.")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].plot(sample[:, 0], sample[:, 2], "k.")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[2].plot(sample[:, 1], sample[:, 2], "k.")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("z")
    plt.savefig(fig_name)


def plot_contour_2D(
    xv1,
    yv1,
    z_cont,
    z_level,
    sample,
    fig_name,
    xv2=None,
    zv1=None,
    y_cont=None,
    y_level=None,
    yv2=None,
    zv2=None,
    x_cont=None,
    x_level=None,
):
    """
    Plots the contours provided (output of utils.compute_contour) in 2D.

    :param xv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type xv1: numpy.ndarray

    :param yv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type yv1: numpy.ndarray

    :param z_cont: A (grid_n, grid_n) matrix, each point a function of xv1 and yv1.
    :type z_cont: numpy.ndarray

    :param z_level: The z-value at which the level set is to be drawn.
    :type z_level: float

    :param sample: Sample from dynamical system (num_samples, n_x).
    :type sample: numpy.ndarray

    :param fig_name: The name of the file to save the plot to.
    :type fig_name: string

    :param xv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type xv2: numpy.ndarray

    :param zv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type zv1: numpy.ndarray

    :param y_cont: A (grid_n, grid_n) matrix, each point a function of xv2 and zv1.
    :type y_cont: numpy.ndarray

    :param y_level: The y-value at which the level set is to be drawn.
    :type y_level: float

    :param yv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type yv2: numpy.ndarray

    :param zv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type zv2: numpy.ndarray

    :param x_cont: A (grid_n, grid_n) matrix, each point a function of yv2 and zv2.
    :type x_cont: numpy.ndarray

    :param x_level: The x-value at which the level set is to be drawn.
    :type x_level: float
    """
    if y_cont is not None and x_cont is not None:
        fig, axs = plt.subplots(3, figsize=(5, 15))

    elif y_cont is not None:
        fig, axs = plt.subplots(2, figsize=(5, 10))

    else:
        fig, axs = plt.subplots(1, figsize=(5, 5))
        axs.plot(sample[:, 0], sample[:, 1], "k.")
        axs.contour(xv1, yv1, z_cont, levels=[z_level], colors="b")
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        axs.set_title("contour in x-y plane")
        plt.show()
        return

    axs[0].plot(sample[:, 0], sample[:, 1], "k.")
    axs[0].contour(xv1, yv1, z_cont, levels=[z_level], colors="b")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("contour in x-y plane")

    if y_cont is not None:
        axs[1].plot(sample[:, 0], sample[:, 2], "k.")
        axs[1].contour(xv2, zv1, y_cont, levels=[y_level], colors="b")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("z")
        axs[1].set_title("contour in x-z plane")

        if x_cont is not None:
            axs[2].plot(sample[:, 1], sample[:, 2], "k.")
            axs[2].contour(yv2, zv2, x_cont, levels=[x_level], colors="b")
            axs[2].set_xlabel("y")
            axs[2].set_ylabel("z")
            axs[2].set_title("contour in y-z plane")

    plt.savefig(fig_name)


def plot_contour_3D(
    xv,
    yv,
    z_cont,
    z_min,
    z_max,
    sample,
    fig_name,
    gif_name=None,
    xv2=None,
    yv2=None,
    z_cont2=None,
):
    """
    Plots the contours in 3D with the option for saving an animated gif of the rotating graph.

    :param xv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type xv1: numpy.ndarray

    :param yv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type yv1: numpy.ndarray

    :param z_cont: A (grid_n, grid_n) matrix, each point a function of xv1 and yv1 (corresponding to
    the maximum optimal solutions that satisfy the p-norm).
    :type z_cont: numpy.ndarray

    :param z_level: The z-value at which the level set is to be drawn.
    :type z_level: float

    :param z_min: The minimum z-value of the samples.
    :type z_min: float

    :param z_max: The maximum z-value of the samples.
    :type z_max: float

    :param sample: Sample from dynamical system (num_samples, n_x).
    :type sample: numpy.ndarray

    :param fig_name: The name of the file to save the plot to.
    :type fig_name: string

    :param fig_name: The name of the file to save the gif to.
    :type fig_name: string

    :param xv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type xv2: numpy.ndarray

    :param yv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector
    repeated along the first dimension.
    :type yv2: numpy.ndarray

    :param z_cont2: A (grid_n, grid_n) matrix, each point a function of xv2 and yv2 (corresponding to
    the minimum optimal solutions that satisfy the p-norm).
    :type z_cont2: numpy.ndarray
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    v_min = z_min - 0.25 * (z_max - z_min)
    v_max = z_max + 0.25 * (z_max - z_min)
    data_2 = [xv2, yv2, z_cont2]
    ax.scatter3D(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c=sample[:, 2],
        cmap="binary",
        vmin=v_min,
        vmax=z_max,
    )

    if all(d is None for d in data_2):
        ax.contour3D(
            xv, yv, z_cont, 50, cmap="Blues", vmin=v_min, vmax=v_max, alpha=0.8
        )
    else:
        abs_min = np.min(z_cont)
        abs_max = np.max(z_cont2)
        abs_avg = (abs_min + abs_max) / 2
        levels_1 = list(np.linspace(abs_min, abs_avg, 25))
        levels_2 = list(np.linspace(abs_avg, abs_max, 25))
        ax.contour3D(
            xv,
            yv,
            z_cont,
            50,
            cmap="Blues",
            vmin=v_min,
            vmax=v_max,
            levels=levels_1,
        )
        ax.contour3D(
            xv2,
            yv2,
            z_cont2,
            50,
            cmap="Blues",
            vmin=v_min,
            vmax=v_max,
            levels=levels_2,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D contour")

    if gif_name is not None:

        def rotate(angle):
            ax.view_init(azim=angle)

        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 362, 2), interval=100
        )
        rot_animation.save(gif_name, dpi=100)

    plt.savefig(fig_name)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_sample(sample, fig_name):
    """Plots samples in 2D

    :param sample: Sample from dynamical system (num_samples, n_x)
    :type sample: numpy.ndarray
    :param fig_name: The name of the file to save the plot to
    :type fig_name: string
    """
    if sample.shape[1] == 2:
        fig, axs = plt.subplots(1, figsize=(5, 5))
        axs.plot(sample[:, 0], sample[:, 1], "k.")
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        axs.set_title("contour in x-y plane")
    else:
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
        plt.savefig(fig_name, bbox_inches="tight", facecolor="white")


def plot_sample_time(
    time_x, samples, fig_name, figsize=(10, 10), color="default", **kwargs
):
    """Plots trajectory of samples over time.

    :param time_x: The time corresponding to the independent variable
    :type time_x: numpy.array
    :param samples: The samples of shape (num_samples, parts) or (num_samples, parts, 2)
    :type samples: numpy.array
    :param fig_name: The name of the file to save the plot to
    :type fig_name: str
    :param figsize: the size of the figure to be saved, defaults to (10, 10)
    :type figsize: tuple, optional
    :param color: The color to draw the lines in, defaults to "default"
    :type color: str, optional
    :raises ValueError: If samples is not an array of shape (num_samples, parts)
    """
    if len(samples.shape) == 3 and samples.shape[2] != 2:
        raise ValueError(
            "The shape of samples must be either (num_samples, parts) or (num_samples, parts, 2)"
        )

    fig, axs = plt.subplots(1, figsize=figsize)
    if len(samples.shape) == 3:
        for sample in samples:
            if color == "default":
                axs.plot(sample[:, 0], sample[:, 1])
            else:
                axs.plot(sample[:, 0], sample[:, 1], color=color)
    else:
        for sample in samples:
            if color == "default":
                axs.plot(time_x, sample)
            else:
                axs.plot(time_x, sample, color=color)

    xmin, xmax, ymin, ymax = plt.axis()

    for key in kwargs:
        if key == "x":
            for val in kwargs["x"]:
                axs.plot([val, val], [ymin, ymax], "k", alpha=0.5)
        elif key == "y":
            for val in kwargs["y"]:
                axs.plot([xmin, xmax], [val, val], "k", alpha=0.5)
        elif key == "title":
            axs.set_title(kwargs["title"])
        elif key == "ylabel":
            axs.set_ylabel(kwargs["ylabel"])

    axs.set_xlabel("time")
    plt.savefig(fig_name, bbox_inches="tight", facecolor="white")


def plot_reach_time(
    time_x,
    samples,
    min_vals,
    max_vals,
    fig_name,
    num_samples_show=0,
    figsize=(10, 10),
    c1="palevioletred",
    c2="lavenderblush",
    **kwargs,
):
    """Plots the reachable set estimate (as well as possibly some samples) over time.

    :param time_x: The time corresponding to the independent variable
    :type time_x: numpy.array
    :param samples: The samples of shape (num_samples, parts)
    :type samples: numpy.array
    :param min_vals: The lower bound of the reachable set estimate
    :type min_vals: list
    :param max_vals: The upper bound of the reachable set estimate
    :type max_vals: list
    :param fig_name: The name of the file to save the plot to
    :type fig_name: str
    :param num_samples_show: The number of sample trajectories to show in addition to the reachable set, defaults to 0
    :type num_samples_show: int, optional
    :param figsize: the size of the figure to be saved, defaults to (10, 10)
    :type figsize: tuple, optional
    :param c1: Color 1, used for the reachable set estimate, defaults to "palevioletred"
    :type c1: str, optional
    :param c2: Color 2, used for the samples, defaults to "lavendarblush"
    :type c2: str, optional
    """
    fig, axs = plt.subplots(1, figsize=figsize)

    axs.plot(time_x, min_vals, c=c1)
    axs.plot(time_x, max_vals, c=c1)
    axs.fill_between(time_x, min_vals, max_vals, color=c1)

    for i in range(num_samples_show):
        axs.plot(time_x, samples[i], color=c2, alpha=0.2)

    xmin, xmax, ymin, ymax = plt.axis()

    # assume kwargs are of the form {"x"=[1], "y"=[0.98, 1.4]}
    for key in kwargs:
        if key == "x":
            for val in kwargs["x"]:
                axs.plot([val, val], [ymin, ymax], "k", alpha=0.5)
        elif key == "y":
            for val in kwargs["y"]:
                axs.plot([xmin, xmax], [val, val], "k", alpha=0.5)
        elif key == "title":
            axs.set_title(kwargs["title"])
        elif key == "ylabel":
            axs.set_ylabel(kwargs["ylabel"])

    axs.set_xlabel("time")
    plt.savefig(fig_name, bbox_inches="tight", facecolor="white")


def plot_reach_time_2D(samples, all_p_norm, fig_name, **kwargs):
    """Plots the reachable set estimate for two state variables across all timesteps, as well as possibly some samples.

    :param samples: The samples to be plotted along with the reachable set estimate. An array of shape (num_samples, timesteps, 2)
    :type samples: numpy.ndarray
    :param all_p_norm: A list of tuples, each one containing two lists of x and y values corresponding to the coordinates included in the reachable set estimate at a given timestep
    :type all_p_norm: list
    :param fig_name: The name of the figure to be saved
    :type fig_name: str
    """
    fig = plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    c1 = kwargs.get("c1", "palevioletred")
    c2 = kwargs.get("c2", "lavenderblush")

    min_vals_x, max_vals_x, min_vals_y, max_vals_y = [], [], [], []
    for i in range(len(all_p_norm)):
        xs, ys = all_p_norm[i][0], all_p_norm[i][1]
        min_y = min(ys)
        max_y = max(ys)
        min_index = ys.index(min_y)
        max_index = ys.index(max_y)
        min_x = xs[min_index]
        max_x = xs[max_index]
        if i == 0:
            for x, y in zip(xs, ys):
                if x <= min_x:
                    plt.plot(x, y, color=c1, marker=".", markersize=4)
        elif i == len(all_p_norm) - 1:
            for x, y in zip(xs, ys):
                if x >= min_x:
                    plt.plot(x, y, color=c1, marker=".", markersize=4)
        min_vals_x.append(min_x)
        min_vals_y.append(min_y)
        max_vals_x.append(max_x)
        max_vals_y.append(max_y)

    plt.plot(min_vals_x, min_vals_y, color=c1)
    plt.plot(max_vals_x, max_vals_y, color=c1)
    plt.fill_between(max_vals_x, min_vals_y, max_vals_y, color=c1)
    plt.fill_between(min_vals_x, min_vals_y, max_vals_y, color=c1)

    for sample in samples:
        plt.plot(sample[:, 0], sample[:, 1], color=c2, alpha=0.5)

    xmin, xmax, ymin, ymax = plt.axis()
    # assume kwargs are of the form {"x"=[1], "y"=[0.98, 1.4]}
    for key in kwargs:
        if key == "x":
            for val in kwargs["x"]:
                plt.plot([val, val], [ymin, ymax], "k", alpha=0.5)
        elif key == "y":
            for val in kwargs["y"]:
                plt.plot([xmin, xmax], [val, val], "k", alpha=0.5)

    plt.xlabel(kwargs.get("xlabel", "x"))
    plt.ylabel(kwargs.get("ylabel", "y"))
    plt.title(kwargs.get("title", "Reachable set estimate over time"))
    plt.savefig(fig_name, bbox_inches="tight", facecolor="white")


def plot_reach_time_3D(samples, tuple_list_3D, fig_name, gif_name=None, **kwargs):
    """Plots the reachable set estimate for three state variables across all timesteps, as well as possibly some samples.

    :param samples: The samples to be plotted along with the reachable set estimate. An array of shape (num_samples, timesteps, 3)
    :type samples: numpy.ndarray
    :param tuple_list_3D: A list of tuples, each one containing three lists of x, y, and z values corresponding to the coordinates included in the reachable set estimate at a given timestep
    :type tuple_list_3D: list
    :param fig_name: The name of the figure to be saved
    :type fig_name: str
    :param gif_name: The name of the gif to be saved, defaults to None
    :type gif_name: str, optional
    """
    fig = plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    ax = plt.axes(projection="3d")

    c1 = kwargs.get("c1", "cornflowerblue")
    surface = kwargs.get("surface", None)

    if surface is not None:
        min_z = np.min(surface[2])
        max_z = np.max(surface[2])

        ax.plot_surface(
            surface[0],
            surface[1],
            surface[2],
            cmap="binary",
            vmin=min_z - 0.15 * (max_z - min_z),
            vmax=max_z,
            alpha=0.8,
        )

    line = kwargs.get("line", None)
    if line is not None:
        ax.plot(line[0], line[1], line[2], color="black", alpha=1.0)

    for x_list, y_list, z_list in tuple_list_3D:
        ax.plot(x_list, y_list, z_list, alpha=0.8, color=c1, linewidth=10)

    for sample in samples:
        ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], alpha=0.9)
        ax.plot(sample[0, 0], sample[0, 1], sample[0, 2], marker="o", color="green")
        ax.plot(sample[-1, 0], sample[-1, 1], sample[-1, 2], marker="o", color="red")

    ax.set_xlabel(kwargs.get("xlabel", "x"))
    ax.set_ylabel(kwargs.get("ylabel", "y"))
    ax.set_zlabel(kwargs.get("zlabel", "z"))
    ax.set_title(kwargs.get("title", "reachable set estimate over time"))
    ax.view_init(elev=kwargs.get("elev", 30), azim=kwargs.get("azim", -30))

    if gif_name is not None:

        def rotate(angle):
            ax.view_init(elev=kwargs.get("elev", 30), azim=angle)

        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 362, 2), interval=100
        )
        rot_animation.save(gif_name, dpi=100)

    plt.savefig(fig_name, bbox_inches="tight", facecolor="white")


def clover_leaf(t, V=13, r=4):
    if t <= 0:
        return [-t, 0, 0]
    elif t >= 2:
        return [0, -t, 0]
    else:
        x = 0.5 * r * np.sin(V / r * t) + r * np.sin(0.5 * V / r * t)
        y = 0.5 * r * np.sin(V / r * t) - r * np.sin(0.5 * V / r * t)
        z = -r + r * np.cos(V / r * t)
        return [-x, -y, -z]


def grow_plot_3d(samples, fig_name, gif_name, figsize, **kwargs):
    """Creates a gif of a three dimensional trajectory that is animated to grow over time

    :param samples: Array of shape (num_samples, time, 3)
    :type samples: numpy.ndarray
    :param fig_name: The name of the static figure to be saved
    :type fig_name: str
    :param gif_name: The name of the animated gif to be saved, defaults to None
    :type gif_name: str, optional
    :raises ValueError: If the samples are not of shape (num_samples, time, 3)
    """
    if len(samples.shape) != 3 or samples.shape[2] != 3:
        raise ValueError("The shape of samples must be (num_samples, time, 3)")

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")

    surface = kwargs.get("surface", None)
    if surface is not None:
        min_z = np.min(surface[2])
        max_z = np.max(surface[2])

        ax.plot_surface(
            surface[0],
            surface[1],
            surface[2],
            cmap="binary",
            vmin=min_z - 0.15 * (max_z - min_z),
            vmax=max_z,
            alpha=0.8,
        )

    line = kwargs.get("line", None)
    if line is not None:
        ax.plot(line[0], line[1], line[2], color="black", alpha=1.0)

    lines = []
    for sample in samples:
        (line,) = ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], alpha=0.9)
        lines.append(line)
        ax.plot(sample[0, 0], sample[0, 1], sample[0, 2], marker="o", color="green")
        ax.plot(sample[-1, 0], sample[-1, 1], sample[-1, 2], marker="o", color="red")

    ax.view_init(elev=kwargs.get("elev", 30), azim=kwargs.get("azim", 45))

    if gif_name is not None:
        ax.set_xlabel(kwargs.get("xlabel", "x"))
        ax.set_ylabel(kwargs.get("ylabel", "y"))
        ax.set_zlabel(kwargs.get("zlabel", "z"))
        ax.set_title(kwargs.get("title", "trajectory over time"))

        def update(num, samples, lines):
            """Inner function to update line plot in 3D over time

            :param num: The number of frames, and the number of points to plot
            :type num: int
            :param samples: Array of shape (num_samples, time, 3)
            :type samples: numpy.ndarray
            :param lines: A list of Line3D objects
            :type lines: list
            :return: An updated list of Line3D objects
            :rtype: list
            """
            for line, sample in zip(lines, samples):
                line.set_xdata(sample[:num, 0])
                line.set_ydata(sample[:num, 1])
                line.set_3d_properties(sample[:num, 2])
            ax.view_init(elev=kwargs.get("elev", 30), azim=num)
            return lines

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=np.arange(0, 362, 2),
            fargs=[samples, lines],
            interval=100,
        )

        ani.save(gif_name)

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
    """Plots the contours provided (output of utils.compute_contour) in 2D

    :param xv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension
    :type xv1: numpy.ndarray
    :param yv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension
    :type yv1: numpy.ndarray
    :param z_cont: A (grid_n, grid_n) matrix, each point a function of xv1 and yv1
    :type z_cont: numpy.ndarray
    :param z_level: The z-value at which the level set is to be drawn
    :type z_level: float
    :param sample: Sample from dynamical system (num_samples, n_x)
    :type sample: numpy.ndarray
    :param fig_name: The name of the file to save the plot to
    :type fig_name: string
    :param xv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension, defaults to None
    :type xv2: numpy.ndarray, optional
    :param zv1: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension, defaults to None
    :type zv1: numpy.ndarray, optional
    :param y_cont: A (grid_n, grid_n) matrix, each point a function of xv2 and zv1, defaults to None
    :type y_cont: numpy.ndarray, optional
    :param y_level: The y-value at which the level set is to be drawn, defaults to None
    :type y_level: float, optional
    :param yv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension, defaults to None
    :type yv2: numpy.ndarray, optional
    :param zv2: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension, defaults to None
    :type zv2: numpy.ndarray, optional
    :param x_cont: A (grid_n, grid_n) matrix, each point a function of yv2 and zv2, defaults to None
    :type x_cont: numpy.ndarray, optional
    :param x_level: The x-value at which the level set is to be drawn, defaults to None
    :type x_level: float, optional
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
        axs.set_title("samples in x-y plane")
        plt.show()
        plt.savefig(fig_name, bbox_inches="tight", facecolor="white")
        return

    axs[0].plot(sample[:, 0], sample[:, 1], "k.")
    axs[0].contour(xv1, yv1, z_cont, levels=[z_level], colors="b")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("samples in x-y plane")

    if y_cont is not None:
        axs[1].plot(sample[:, 0], sample[:, 2], "k.")
        axs[1].contour(xv2, zv1, y_cont, levels=[y_level], colors="b")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("z")
        axs[1].set_title("samples in x-z plane")

        if x_cont is not None:
            axs[2].plot(sample[:, 1], sample[:, 2], "k.")
            axs[2].contour(yv2, zv2, x_cont, levels=[x_level], colors="b")
            axs[2].set_xlabel("y")
            axs[2].set_ylabel("z")
            axs[2].set_title("samples in y-z plane")

    plt.savefig(fig_name, bbox_inches="tight", facecolor="white")


def plot_contour_3D(
    xv,
    yv,
    z_cont,
    z_min,
    z_max,
    sample,
    fig_name,
    gif_name=None,
    z_cont2=None,
):
    """Plots the contours in 3D with the option for saving an animated gif of the rotating graph

    :param xv: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension
    :type xv: numpy.ndarray
    :param yv: A (grid_n, grid_n) matrix with the elements of a (grid_n, 1) vector repeated along the first dimension
    :type yv: numpy.ndarray
    :param z_cont: A (grid_n, grid_n) matrix, each point a function of xv1 and yv1 (corresponding to the maximum optimal solutions that satisfy the p-norm)
    :type z_cont: numpy.ndarray
    :param z_min: The minimum z-value
    :type z_min: float
    :param z_max: The maximum z-value
    :type z_max: float
    :param sample: Sample from dynamical system (num_samples, n_x)
    :type sample: numpy.ndarray
    :param fig_name: The name of the file to save the plot to
    :type fig_name: string
    :param gif_name: The name of the file to save the gif to, defaults to None
    :type gif_name: string, optional
    :param z_cont2: A (grid_n, grid_n) matrix, each point a function of xv and yv (corresponding to the minimum optimal solutions that satisfy the p-norm), defaults to None
    :type z_cont2: numpy.ndarray, optional
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    v_min = z_min - 0.25 * (z_max - z_min)
    v_max = z_max + 0.25 * (z_max - z_min)
    ax.scatter3D(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c=sample[:, 2],
        cmap="binary",
        vmin=v_min,
        vmax=z_max,
    )
    if z_cont2 is None:
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
            xv,
            yv,
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

    plt.savefig(fig_name, bbox_inches="tight", facecolor="white")


def plot_contour_3D_time(
    samples, dict_list, fig_name, gif_name=None, figsize=(10, 10), **kwargs
):
    """Plots the contours of the p-norm reachable set estimate along with the samples over time

    :param samples: Array of shape (num_samples, time, 3)
    :type samples: numpy.ndarray
    :param dict_list: A list of dictionaries, each containing the contour information for a p-norm ball reachable set estimate at a given time
    :type dict_list: list
    :param fig_name: The name of the file to save the plot to
    :type fig_name: string
    :param gif_name: The name of the file to save the gif to, defaults to None
    :type gif_name: string, optional
    :param figsize: The size of the figure to be saved, defaults to (10, 10)
    :type figsize: tuple, optional
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")

    ax.set_xlabel(kwargs.get("xlabel", "x"))
    ax.set_ylabel(kwargs.get("ylabel", "y"))
    ax.set_zlabel(kwargs.get("zlabel", "z"))
    ax.set_title(kwargs.get("title", "reachable set over time"))

    for sample in samples:
        ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], alpha=0.8)

    for curr_dict in dict_list:
        xv = curr_dict["xv"]
        yv = curr_dict["yv"]
        z_cont = curr_dict["z_cont"]
        z_min = curr_dict["z_min"]
        z_max = curr_dict["z_max"]
        z_cont2 = curr_dict["z_cont2"]

        v_min = z_min - 0.25 * (z_max - z_min)
        v_max = z_max + 0.4 * (z_max - z_min)

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
            alpha=0.1,
        )

        ax.contour3D(
            xv,
            yv,
            z_cont2,
            50,
            cmap="Blues",
            vmin=v_min,
            vmax=v_max,
            levels=levels_2,
            alpha=0.1,
        )

    if gif_name is not None:

        def rotate(num):
            ax.view_init(elev=20, azim=num)

        ani = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 362, 2), interval=50
        )

        ani.save(gif_name)

    plt.savefig(fig_name, bbox_inches="tight", facecolor="white")

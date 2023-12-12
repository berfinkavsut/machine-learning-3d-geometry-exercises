"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An SDF grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution)
             with positive values outside the shape and negative values inside.
    """

    #####################################################################################
    x_ = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True)
    y_ = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True)
    z_ = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True)
    xx, yy, zz = np.meshgrid(x_, y_, z_)
    x, y, z = xx.flatten(), yy.flatten(), zz.flatten()

    sdf_values = sdf_function(x, y, z)
    sdf_grid = np.reshape(sdf_values, newshape=(resolution, resolution, resolution))
    return sdf_grid
    #####################################################################################

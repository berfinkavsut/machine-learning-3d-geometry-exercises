"""SDF to Occupancy Grid"""
import numpy as np

def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution)
             with value 0 outside the shape and 1 inside.
    """
    # ###############
    x_ = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True, dtype=np.float64)
    y_ = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True, dtype=np.float64)
    z_ = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x_, y_, z_, indexing='ij')
    x, y, z = xx.flatten(), yy.flatten(), zz.flatten()

    sdf_values = sdf_function(x, y, z)
    sdf_grid = np.reshape(sdf_values, newshape=(resolution, resolution, resolution), order='F')

    occupancy_grid = np.zeros_like(sdf_grid, dtype=np.int32)
    occupancy_grid[sdf_grid < 0] = 1  # negative values are inside 

    return occupancy_grid
    # ###############
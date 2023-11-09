"""Creating an SDF grid"""
import numpy as np
# import matplotlib.pyplot as plt


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
    # ###############
    x = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True, dtype=np.float64)
    y = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True, dtype=np.float64)
    z = np.linspace(start=-0.5, stop=0.5, num=resolution, endpoint=True, dtype=np.float64)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='xy')
    sdf_grid = sdf_function(xx, yy, zz)

    return sdf_grid
    # ###############

"""
if __name__ == '__main__':
    xx, yy, zz = sdf_grid(10)

    print(type(xx), type(yy), type(zz))
    print(xx.shape, yy.shape, zz.shape)  # Check shapes

    # Plotting code
    plt.figure()
    plt.subplot(131)
    plt.imshow(xx[:, :, 0])  # Fix indexing to display a 2D slice
    plt.title("xx")

    plt.subplot(132)
    plt.imshow(yy[:, :, 0])  # Fix indexing to display a 2D slice
    plt.title("yy")

    plt.subplot(133)
    plt.imshow(zz[:, 0, :])  # Fix indexing to display a 2D slice
    plt.title("zz")

    plt.show()
"""
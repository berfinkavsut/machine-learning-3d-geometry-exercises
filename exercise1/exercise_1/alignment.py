""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    ##########################################################################################
    # Source: https://neurodatascience.github.io/fmralign-tutorials/1-2-procrustes.html#id1

    # 1. get centered pc_x and centered pc_y
    x_center = np.mean(pc_x, axis=0)
    y_center = np.mean(pc_y, axis=0)
    X = pc_x - x_center
    Y = pc_y - y_center

    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    X = X.T
    Y = Y.T

    # 3. estimate rotation
    M = np.matmul(Y, X.T)
    U, _, V = np.linalg.svd(M, compute_uv=True)

    S = np.zeros(shape=(3, 3))
    if (np.linalg.det(U) * np.linalg.det(V.T) - 1) <= 1e-9:
        S = np.eye(3)
    else:
        S = np.eye(3)
        S[-1, -1] = -1

    R = np.matmul(U, np.matmul(S, V))

    # 4. estimate translation
    t = (y_center - np.matmul(R, x_center)).reshape(3)
    ##########################################################################################

    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)
    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)

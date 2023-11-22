"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    total_area = 0
    for idx, face in enumerate(faces):
        A, B, C = vertices[face]
        edge_vector_1 = B - A
        edge_vector_2 = C - A
        triangle_area = 0.5 * np.linalg.norm(np.cross(edge_vector_1, edge_vector_2))
        total_area += triangle_area

    points = np.zeros(shape=(n_points, 3))
    total_n = 0
    for idx, face in enumerate(faces):
        A, B, C = vertices[face]

        edge_vector_1 = B - A
        edge_vector_2 = C - A
        triangle_area = 0.5 * np.linalg.norm(np.cross(edge_vector_1, edge_vector_2))

        # TODO: wrong
        prob = (triangle_area / total_area)
        n = int(prob * n_points)
        total_n = total_n + n

        r1 = np.random.rand(n)
        r2 = np.random.rand(n)

        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1) * (1-r2)
        w = np.sqrt(r1) * r2

        P = u * A[:, np.newaxis] + v * B[:, np.newaxis] + w * C[:, np.newaxis]
        points[idx] = np.array(P.reshape((3, 1)))

    return points
    # ###############

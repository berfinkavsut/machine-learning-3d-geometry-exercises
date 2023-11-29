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
    areas = []
    face_num = len(faces)
    for idx, face in enumerate(faces):
        A, B, C = vertices[face]
        edge_vector_1 = B - A
        edge_vector_2 = C - A
        triangle_area = 0.5 * np.linalg.norm(np.cross(edge_vector_1, edge_vector_2))
        areas.append(triangle_area)

    probs = [area / np.sum(areas) for area in areas]

    selected_indices = np.random.choice(len(faces), size=n_points, p=probs)

    points = np.zeros(shape=(n_points, 3))

    for i, idx in enumerate(selected_indices):
        v_ind = faces[idx]
        A, B, C = vertices[v_ind[0]], vertices[v_ind[1]], vertices[v_ind[2]]

        # Generate random indices based on the probabilities
        r1 = np.random.rand(1)
        r2 = np.random.rand(1)

        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1) * (1-r2)
        w = np.sqrt(r1) * r2

        P = u[:, np.newaxis] * A + v[:, np.newaxis] * B + w[:, np.newaxis] * C
        # P = u * A[:, np.newaxis] + v * B[:, np.newaxis] + w * C[:, np.newaxis]
        points[i] = np.array(P)

    return points
    # ###############

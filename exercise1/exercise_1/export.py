"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # faces = faces + 1
    with open(path, 'w') as file:
        for vertex in vertices:
            v_str = 'v' + ' ' + str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + ' \n'
            file.write(v_str)
        for face in faces:
            f_str = 'f' + ' ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + ' \n'
            file.write(f_str)

    """file_str = ''
    for vertex in vertices:
        file_str += f'v {vertex[0]} {vertex[1]} {vertex[2]}\n'
    for face in faces:
        file_str += f'f {face[0]} {face[1]} {face[2]}\n'

    with open(path, 'w') as f:
        f.write(file_str)"""
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    with open(path, 'w') as file:
        for point in pointcloud:
            p_str = 'v' + ' ' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' \n'
            file.write(p_str)
    # ###############

"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """
    ####################################################################
    faces = faces + 1
    with open(path, 'w') as file:
        # write vertices starting with "v "
        for vertex in vertices:
            v_str = f'v {vertex[0]} {vertex[1]} {vertex[2]}\n'
            file.write(v_str)

        # write faces starting with "f "
        for face in faces:
            f_str = f'f {face[0]} {face[1]} {face[2]}\n'
            file.write(f_str)
    ####################################################################


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """
    ####################################################################
    with open(path, 'w') as file:
        for point in pointcloud:
            p_str = f'v {point[0]} {point[1]} {point[2]}\n'
            file.write(p_str)
    ####################################################################

from ...logger import logger
from ..block_mesh import BlockMeshVertex, BlockMeshEdge, unit_vector


def vertex_gen_fcn(start_point, face_normal, perp_vec, tube_diameter, outer_pipe=True):
    dist = tube_diameter / 4

    p0 = BlockMeshVertex(
        position=unit_vector(-face_normal + perp_vec) * dist + start_point)
    p1 = BlockMeshVertex(
        position=unit_vector(-face_normal - perp_vec) * dist + start_point)
    p2 = BlockMeshVertex(
        position=unit_vector(face_normal - perp_vec) * dist + start_point)
    p3 = BlockMeshVertex(
        position=unit_vector(face_normal + perp_vec) * dist + start_point)

    # pipe block
    p4 = p0 + unit_vector(-face_normal + perp_vec) * dist
    p5 = p1 + unit_vector(-face_normal - perp_vec) * dist
    p6 = p2 + unit_vector(face_normal - perp_vec) * dist
    p7 = p3 + unit_vector(face_normal + perp_vec) * dist

    cp0 = start_point - (face_normal * tube_diameter / 2)
    cp1 = start_point - (perp_vec * tube_diameter / 2)
    cp2 = start_point + (face_normal * tube_diameter / 2)
    cp3 = start_point + (perp_vec * tube_diameter / 2)

    if not outer_pipe:
        return [p0, p1, p2, p3, p4, p5, p6, p7], [cp0, cp1, cp2, cp3]
    else:
        p8 = p4 + (-face_normal + perp_vec) * dist
        p9 = p4 + -face_normal * dist

        p10 = p5 + -face_normal * dist
        p11 = p5 + (-face_normal - perp_vec) * dist
        p12 = p5 - perp_vec * dist

        p13 = p6 - perp_vec * dist
        p14 = p6 + (face_normal - perp_vec) * dist
        p15 = p6 + face_normal * dist

        p16 = p7 + face_normal * dist
        p17 = p7 + (face_normal + perp_vec) * dist
        p18 = p7 + perp_vec * dist

        p19 = p4 + perp_vec * dist

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19], [cp0, cp1,
                                                                                                            cp2, cp3]


layer_edge_const_def = [([0, 1], 'line'),
                        ([1, 2], 'line'),
                        ([2, 3], 'line'),
                        ([3, 0], 'line'),
                        ([0, 4], 'line'),
                        ([1, 5], 'line'),
                        ([2, 6], 'line'),
                        ([3, 7], 'line'),
                        ([4, 5], 'arc', [0]),
                        ([5, 6], 'arc', [1]),
                        ([6, 7], 'arc', [2]),
                        ([7, 4], 'arc', [3])
                        ]

block_vertex_indices = [[0, 1, 2, 3],
                        [0, 4, 5, 1],
                        [1, 5, 6, 2],
                        [2, 6, 7, 3],
                        [3, 7, 4, 0],
                        ]

# id of edges in the section
block_edge_indices = [[0, 1, 2, 3],
                      [4, 8, 5, 0],
                      [5, 9, 6, 1],
                      [6, 10, 7, 2],
                      [7, 11, 4, 3],
                      ]

cell_zones = ['pipe', 'pipe', 'pipe', 'pipe', 'pipe']


pipe_wall_def = [([1], [5]),
                 ([2], [3]),
                 ([3], [4]),
                 ([4], [2])]


# outer blocks:

# id of nodes in the section which create a block:
outer_block_vertex_indices = [[4, 19, 8, 9],
                              [4, 9, 10, 5],
                              [5, 10, 11, 12],
                              [5, 12, 13, 6],
                              [6, 13, 14, 15],
                              [6, 15, 16, 7],
                              [7, 16, 17, 18],
                              [7, 18, 19, 4]]

layer_outer_pipe_edge_const_def = [([4, 19], 'line'),
                                   ([4, 9], 'line'),
                                   ([5, 10], 'line'),
                                   ([5, 12], 'line'),
                                   ([6, 13], 'line'),
                                   ([6, 15], 'line'),
                                   ([7, 16], 'line'),
                                   ([7, 18], 'line'),
                                   ([8, 9], 'line'),
                                   ([9, 10], 'line'),
                                   ([10, 11], 'line'),
                                   ([11, 12], 'line'),
                                   ([12, 13], 'line'),
                                   ([13, 14], 'line'),
                                   ([14, 15], 'line'),
                                   ([15, 16], 'line'),
                                   ([16, 17], 'line'),
                                   ([17, 18], 'line'),
                                   ([18, 19], 'line'),
                                   ([19, 8], 'line')]

# ids of
outer_block_edge_indices = [[12, 31, 20, 13],
                            [13, 21, 14, 8],
                            [14, 22, 23, 15],
                            [15, 24, 16, 9],
                            [16, 25, 26, 17],
                            [17, 27, 18, 10],
                            [18, 28, 29, 19],
                            [19, 30, 12, 11]]


outer_cell_zones = [None, None, None, None, None, None, None, None]

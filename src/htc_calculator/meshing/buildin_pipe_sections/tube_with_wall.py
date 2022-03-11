from ...logger import logger
from ..block_mesh import BlockMeshVertex, BlockMeshEdge, unit_vector
from ..pipe_sections import PipeSection
import numpy as np

import FreeCAD
from FreeCAD import Base


def vertex_gen_fcn(start_point, face_normal, perp_vec, tube_inner_diameter, tube_diameter, outer_pipe=True):
    dist1 = tube_inner_diameter / 4
    dist = tube_diameter / 4

    pipe_wall_thickness = (tube_diameter - tube_inner_diameter)/2

    if isinstance(start_point, np.ndarray):
        start_point = Base.Vector(start_point)

    if isinstance(face_normal, np.ndarray):
        face_normal = Base.Vector(face_normal)

    if isinstance(perp_vec, np.ndarray):
        perp_vec = Base.Vector(perp_vec)

    p0 = BlockMeshVertex(
        position=np.array((-face_normal + perp_vec).normalize() * dist1 + start_point))
    p1 = BlockMeshVertex(
        position=np.array((-face_normal - perp_vec).normalize() * dist1 + start_point))
    p2 = BlockMeshVertex(
        position=np.array((face_normal - perp_vec).normalize() * dist1 + start_point))
    p3 = BlockMeshVertex(
        position=np.array((face_normal + perp_vec).normalize() * dist1 + start_point))

    # pipe block
    p4 = p0 + (-face_normal + perp_vec).normalize() * dist1
    p5 = p1 + (-face_normal - perp_vec).normalize() * dist1
    p6 = p2 + (face_normal - perp_vec).normalize() * dist1
    p7 = p3 + (face_normal + perp_vec).normalize() * dist1

    # inner tube wall arc construction points
    cp0 = np.array(start_point - (face_normal * tube_inner_diameter / 2))
    cp1 = np.array(start_point - (perp_vec * tube_inner_diameter / 2))
    cp2 = np.array(start_point + (face_normal * tube_inner_diameter / 2))
    cp3 = np.array(start_point + (perp_vec * tube_inner_diameter / 2))

    if outer_pipe:

        # pipe wall
        p8 = p4 + (-face_normal + perp_vec).normalize() * pipe_wall_thickness
        p9 = p5 + (-face_normal - perp_vec).normalize() * pipe_wall_thickness
        p10 = p6 + (face_normal - perp_vec).normalize() * pipe_wall_thickness
        p11 = p7 + (face_normal + perp_vec).normalize() * pipe_wall_thickness

        # outer blocks:
        p12 = p8 + perp_vec * dist
        p13 = p8 + (-face_normal + perp_vec) * dist
        p14 = p8 + -face_normal * dist
        p15 = p9 + -face_normal * dist
        p16 = p9 + (-face_normal - perp_vec) * dist
        p17 = p9 + -perp_vec * dist
        p18 = p10 + -perp_vec * dist
        p19 = p10 + (face_normal - perp_vec) * dist
        p20 = p10 + face_normal * dist
        p21 = p11 + face_normal * dist
        p22 = p11 + (face_normal + perp_vec) * dist
        p23 = p11 + perp_vec * dist

        # outer tube wall arc construction points
        cp4 = np.array(start_point - (face_normal * tube_diameter / 2))
        cp5 = np.array(start_point - (perp_vec * tube_diameter / 2))
        cp6 = np.array(start_point + (face_normal * tube_diameter / 2))
        cp7 = np.array(start_point + (perp_vec * tube_diameter / 2))

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
                p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23], \
               [cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7]

    else:
        return [p0, p1, p2, p3, p4, p5, p6, p7], [cp0, cp1, cp2, cp3]

edge_def = [([0, 1], 'line'),          # inner pipe edges
            ([1, 2], 'line'),
            ([2, 3], 'line'),
            ([3, 0], 'line'),
            ([0, 4], 'line'),
            ([1, 5], 'line'),
            ([2, 6], 'line'),
            ([3, 7], 'line'),
            ([4, 5], 'arc', [0]),       # last value are interpolation / construction points
            ([5, 6], 'arc', [1]),
            ([6, 7], 'arc', [2]),
            ([7, 4], 'arc', [3])
            ]

outer_edge_def = [([4, 8], 'line'),           # tube wall edges
                  ([5, 9], 'line'),
                  ([6, 10], 'line'),
                  ([7, 11], 'line'),
                  ([8, 9], 'arc', [4]),
                  ([9, 10], 'arc', [5]),
                  ([10, 11], 'arc', [6]),
                  ([11, 8], 'arc', [7]),
                  ([8, 12], 'line'),
                  ([8, 14], 'line'),
                  ([9, 15], 'line'),
                  ([9, 17], 'line'),
                  ([10, 18], 'line'),
                  ([10, 20], 'line'),
                  ([11, 21], 'line'),
                  ([11, 23], 'line'),
                  ([12, 13], 'line'),
                  ([13, 14], 'line'),
                  ([14, 15], 'line'),
                  ([15, 16], 'line'),
                  ([16, 17], 'line'),
                  ([17, 18], 'line'),
                  ([18, 19], 'line'),
                  ([19, 20], 'line'),
                  ([20, 21], 'line'),
                  ([21, 22], 'line'),
                  ([22, 23], 'line'),
                  ([23, 12], 'line')]

vertex_indices = [[0, 1, 2, 3],
                  [0, 4, 5, 1],
                  [1, 5, 6, 2],
                  [2, 6, 7, 3],
                  [3, 7, 4, 0]]

outer_vertex_indices = [[4, 8, 9, 5],
                        [5, 9, 10, 6],
                        [6, 10, 11, 7],
                        [7, 11, 8, 4],
                        [8, 12, 13, 14],
                        [8, 14, 15, 9],
                        [9, 15, 16, 17],
                        [9, 17, 18, 10],
                        [10, 18, 19, 20],
                        [10, 20, 21, 11],
                        [11, 21, 22, 23],
                        [11, 23, 12, 8]]

# TODO: add outer edge indices
edge_indices = [[0, 1, 2, 3],       # Block 0
                [4, 8, 5, 0],       # Block 1
                [5, 9, 6, 1],       # Block 2
                [6, 10, 7, 2],      # Block 3
                [7, 11, 4, 3],      # Block 4
                ]

outer_edge_indices = [[8, 12, 16, 13],       # Block 5
                      [9, 13, 17, 14],       # Block 6
                      [10, 14, 18, 15],      # Block 7
                      [11, 15, 19, 12],      # Block 8
                      [20, 39, 28, 21],      # Block 9
                      [16, 21, 29, 22],      # Block 10
                      [22, 30, 31, 23],      # Block 11
                      [17, 23, 32, 24],      # Block 12
                      [24, 33, 34, 25],      # Block 13
                      [18, 25, 35, 26],      # Block 14
                      [26, 36, 37, 27],      # Block 15
                      [19, 27, 38, 20],      # Block 16
                      ]

# number of cells; if None number of cells is calculated by block length and cell_size
n_cell = [10, 10, None]

# size of cells in mm; if None n_cell must be defined
cell_size = [None, None, 50]

cell_zones = ['pipe', 'pipe', 'pipe', 'pipe', 'pipe']
outer_cell_zones = ['pipe_wall', 'pipe_wall', 'pipe_wall', 'pipe_wall', None, None, None, None, None, None, None, None]

# id of block, id of faces which are pipe wall
pipe_wall_def = [(1, [5]),
                 (2, [3]),
                 (3, [4]),
                 (4, [2])]

block_inlet_faces = [(0, [0]),
                     (1, [0]),
                     (2, [0]),
                     (3, [0]),
                     (4, [0]),
                     ]

block_outlet_faces = [(0, [1]),
                      (1, [1]),
                      (2, [1]),
                      (3, [1]),
                      (4, [1]),
                      ]

pipe_section = PipeSection(name='Plain Tube',
                           layer_vertex_gen_function=vertex_gen_fcn,
                           edge_def=[edge_def, outer_edge_def],
                           vertex_indices=[vertex_indices, outer_vertex_indices],
                           edge_indices=[edge_indices, outer_edge_indices],
                           cell_zones=[cell_zones, outer_cell_zones],
                           n_cell=n_cell,
                           cell_size=cell_size,
                           pipe_wall_def=pipe_wall_def,
                           block_inlet_faces=block_inlet_faces,
                           block_outlet_faces=block_outlet_faces)

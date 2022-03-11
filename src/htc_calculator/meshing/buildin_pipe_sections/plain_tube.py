from ...logger import logger
from ..block_mesh import BlockMeshVertex, BlockMeshEdge, unit_vector
from ..pipe_sections import PipeSection
import numpy as np

import FreeCAD
from FreeCAD import Base


def vertex_gen_fcn(start_point, face_normal, perp_vec, tube_diameter, outer_pipe=True, pipe_wall_thickness=2):
    dist = tube_diameter / 4

    if isinstance(start_point, np.ndarray):
        start_point = Base.Vector(start_point)

    if isinstance(face_normal, np.ndarray):
        face_normal = Base.Vector(face_normal)

    if isinstance(perp_vec, np.ndarray):
        perp_vec = Base.Vector(perp_vec)

    p0 = BlockMeshVertex(
        position=np.array((-face_normal + perp_vec).normalize() * dist + start_point))
    p1 = BlockMeshVertex(
        position=np.array((-face_normal - perp_vec).normalize() * dist + start_point))
    p2 = BlockMeshVertex(
        position=np.array((face_normal - perp_vec).normalize() * dist + start_point))
    p3 = BlockMeshVertex(
        position=np.array((face_normal + perp_vec).normalize() * dist + start_point))

    # pipe block
    p4 = p0 + (-face_normal + perp_vec).normalize() * dist
    p5 = p1 + (-face_normal - perp_vec).normalize() * dist
    p6 = p2 + (face_normal - perp_vec).normalize() * dist
    p7 = p3 + (face_normal + perp_vec).normalize() * dist

    cp0 = np.array(start_point - (face_normal * tube_diameter / 2))
    cp1 = np.array(start_point - (perp_vec * tube_diameter / 2))
    cp2 = np.array(start_point + (face_normal * tube_diameter / 2))
    cp3 = np.array(start_point + (perp_vec * tube_diameter / 2))

    return [p0, p1, p2, p3, p4, p5, p6, p7], [cp0, cp1, cp2, cp3]


edge_def = [([0, 1], 'line'),
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

outer_edge_def = []


vertex_indices = [[0, 1, 2, 3],
                  [0, 4, 5, 1],
                  [1, 5, 6, 2],
                  [2, 6, 7, 3],
                  [3, 7, 4, 0],
                  ]

outer_vertex_indices = []


edge_indices = [[0, 1, 2, 3],
                [4, 8, 5, 0],
                [5, 9, 6, 1],
                [6, 10, 7, 2],
                [7, 11, 4, 3],
                ]

outer_edge_indices = []

# number of cells; if None number of cells is calculated by block length and cell_size
n_cell = [10, 10, None]

# size of cells in mm; if None n_cell must be defined
cell_size = [None, None, 50]

cell_zones = ['pipe', 'pipe', 'pipe', 'pipe', 'pipe']
outer_cell_zones = []

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

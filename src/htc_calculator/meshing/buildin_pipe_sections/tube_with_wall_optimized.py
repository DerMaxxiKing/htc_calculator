from ...logger import logger
from ..block_mesh import BlockMeshVertex, BlockMeshEdge, unit_vector, CellZone
from ..pipe_sections import PipeSection
from ...buildin_materials import water
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
        p12 = p8 + (-face_normal + perp_vec) * dist
        p13 = p9 + (-face_normal - perp_vec) * dist
        p14 = p10 + (face_normal - perp_vec) * dist
        p15 = p11 + (face_normal + perp_vec) * dist

        # outer tube wall arc construction points
        cp4 = np.array(start_point - (face_normal * tube_diameter / 2))
        cp5 = np.array(start_point - (perp_vec * tube_diameter / 2))
        cp6 = np.array(start_point + (face_normal * tube_diameter / 2))
        cp7 = np.array(start_point + (perp_vec * tube_diameter / 2))

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15], \
               [cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7]

    else:
        return [p0, p1, p2, p3, p4, p5, p6, p7], [cp0, cp1, cp2, cp3]


n_edge_tube = 10
n_inner_pipe = 10
n_tube_thickness = 3

edge_def = [([0, 1], 'line', n_edge_tube),           # Edge 0          # inner pipe edges
            ([1, 2], 'line', n_edge_tube),           # Edge 1
            ([2, 3], 'line', n_edge_tube),           # Edge 2
            ([3, 0], 'line', n_edge_tube),           # Edge 3
            ([0, 4], 'line', n_inner_pipe),           # Edge 4
            ([1, 5], 'line', n_inner_pipe),           # Edge 5
            ([2, 6], 'line', n_inner_pipe),           # Edge 6
            ([3, 7], 'line', n_inner_pipe),           # Edge 7
            ([4, 5], 'arc', [0], n_edge_tube),           # Edge 8       # last value are interpolation / construction points
            ([5, 6], 'arc', [1], n_edge_tube),           # Edge 9
            ([6, 7], 'arc', [2], n_edge_tube),           # Edge 10
            ([7, 4], 'arc', [3], n_edge_tube)            # Edge 11
            ]

outer_edge_def = [([4, 8], 'line', n_tube_thickness),           # Edge 12 tube wall edges
                  ([5, 9], 'line', n_tube_thickness),           # Edge 13
                  ([6, 10], 'line', n_tube_thickness),           # Edge 14
                  ([7, 11], 'line', n_tube_thickness),           # Edge 15
                  ([8, 9], 'arc', [4], n_edge_tube),           # Edge 16
                  ([9, 10], 'arc', [5], n_edge_tube),           # Edge 17
                  ([10, 11], 'arc', [6], n_edge_tube),           # Edge 18
                  ([11, 8], 'arc', [7], n_edge_tube),           # Edge 19
                  ([8, 12], 'line'),           # Edge 20
                  ([9, 13], 'line'),           # Edge 21
                  ([10, 14], 'line'),           # Edge 22
                  ([11, 15], 'line'),           # Edge 23
                  ([12, 13], 'line'),           # Edge 24
                  ([13, 14], 'line'),           # Edge 25
                  ([14, 15], 'line'),           # Edge 26
                  ([15, 12], 'line'),           # Edge 27
                  ]

vertex_indices = [[0, 1, 2, 3],         # Block 0
                  [0, 4, 5, 1],         # Block 1
                  [1, 5, 6, 2],         # Block 2
                  [2, 6, 7, 3],         # Block 3
                  [3, 7, 4, 0]]         # Block 4

outer_vertex_indices = [[4, 8, 9, 5],           # Block 5
                        [5, 9, 10, 6],          # Block 6
                        [6, 10, 11, 7],         # Block 7
                        [7, 11, 8, 4],          # Block 8
                        [8, 12, 13, 9],            # Block 9
                        [9, 13, 14, 10],             # Block 10
                        [10, 14, 15, 11],            # Block 11
                        [11, 15, 12, 8],            # Block 12
                        ]

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
                      [16, 20, 24, 21],      # Block 9
                      [17, 21, 25, 22],      # Block 10
                      [18, 22, 26, 23],      # Block 11
                      [19, 23, 27, 20],      # Block 12
                      ]

# number of cells; if None number of cells is calculated by block length and cell_size
n_cell = [5, 5, None]

# size of cells in mm; if None n_cell must be defined
cell_size = [None, None, 100]

inner_cell_zone = CellZone(new=True)
outer_cell_zone = CellZone(new=True)
undefined_cell_zone = CellZone(new=True)

cell_zones = [inner_cell_zone, outer_cell_zone, undefined_cell_zone]

cell_zone_ids = [0, 0, 0, 0, 0]
outer_cell_zones_ids = [1, 1, 1, 1, None, None, None, None]

block_cell_zones = [inner_cell_zone,
                    inner_cell_zone,
                    inner_cell_zone,
                    inner_cell_zone,
                    inner_cell_zone]

outer_block_cell_zones = [outer_cell_zone,
                          outer_cell_zone,
                          outer_cell_zone,
                          outer_cell_zone,
                          None,
                          None,
                          None,
                          None]

# id of block, id of faces which are pipe wall
pipe_wall_def = [(1, [3]),
                 (2, [3]),
                 (3, [3]),
                 (4, [3])]

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

grading = [[1, 1, 1],   # Block 0
           [0.33, 1, 1],   # Block 1
           [0.33, 1, 1],   # Block 2
           [0.33, 1, 1],   # Block 3
           [0.33, 1, 1],   # Block 4
           [1, 1, 1],   # Block 5
           [1, 1, 1],   # Block 6
           [1, 1, 1],   # Block 7
           [1, 1, 1],   # Block 8
           [1, 1, 1],   # Block 9
           [1, 1, 1],   # Block 10
           [1, 1, 1],   # Block 11
           [1, 1, 1]]

# define which faces are on top side and bottom side:
top_side = {11: [3]}          # block id, face id
bottom_side = {9: [3]}       # block id, face id
interface_side = {10: [3],
                  12: [3]}

merge_patch_pairs = [{}, {}]

pipe_section = PipeSection(name='Plain Tube',
                           layer_vertex_gen_function=vertex_gen_fcn,
                           edge_def=[edge_def, outer_edge_def],
                           vertex_indices=[vertex_indices, outer_vertex_indices],
                           edge_indices=[edge_indices, outer_edge_indices],
                           cell_zones=cell_zones,
                           block_cell_zones=[block_cell_zones, outer_block_cell_zones],
                           cell_zone_ids=[cell_zone_ids, outer_cell_zones_ids],
                           n_cell=n_cell,
                           cell_size=cell_size,
                           grading=grading,
                           merge_patch_pairs=merge_patch_pairs,
                           pipe_wall_def=pipe_wall_def,
                           block_inlet_faces=block_inlet_faces,
                           block_outlet_faces=block_outlet_faces,
                           top_side=top_side,
                           bottom_side=bottom_side,
                           interface_side=interface_side)

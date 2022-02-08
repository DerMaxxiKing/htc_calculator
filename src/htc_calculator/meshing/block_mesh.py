import uuid
import itertools
import numpy as np
from functools import lru_cache, wraps

import FreeCAD
import Part as FCPart
from FreeCAD import Base

App = FreeCAD


def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


class VertexMetaMock(type):

    instances = []

    @staticmethod
    @np_cache
    def get_vertex(position):
        return next((x for x in VertexMetaMock.instances if np.array_equal(x.position, position)), None)

    def __call__(cls, *args, **kwargs):
        obj = cls.get_vertex(kwargs.get('position', np.array([0, 0, 0])))
        if obj is None:

            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            cls.instances.append(obj)
        return obj


class EdgeMetaMock(type):

    instances = []

    @staticmethod
    @np_cache
    def get_edge(vertices):
        return next((x for x in EdgeMetaMock.instances if np.array_equal(x.vertices, vertices)), None)

    def __call__(cls, *args, **kwargs):
        obj = cls.get_edge(kwargs.get('vertices'))
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            cls.instances.append(obj)
        return obj


class BlockMetaMock(type):

    instances = []

    @staticmethod
    @np_cache
    def get_block(vertices):
        return next((x for x in BlockMetaMock.instances if np.array_equal(x.vertices, vertices)), None)

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        return obj


class BlockMeshVertex(object, metaclass=VertexMetaMock):
    id_iter = itertools.count()

    @classmethod
    def block_mesh_entry(cls):
        return ''

    def __init__(self, *args, **kwargs):
        self.id = next(BlockMeshVertex.id_iter)
        self.position = kwargs.get('position', np.array([0, 0, 0]))

    def __add__(self, vec):
        return BlockMeshVertex(position=self.position + vec)

    def __sub__(self, vec):
        return BlockMeshVertex(position=self.position - vec)

    def __repr__(self):
        return f'Vertex {self.id} (position={self.position[0], self.position[1], self.position[2]})'

    def dist_to_point(self, vertex):
        return np.linalg.norm(self.position - vertex.position)


class BlockMeshEdge(object, metaclass=EdgeMetaMock):
    id_iter = itertools.count()

    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :keyword type:  arc	            Circular arc	    Single interpolation point
                        simpleSpline	Spline curve	    List of interpolation points
                        polyLine	    Set of lines	    List of interpolation points
                        polySpline	    Set of splines	    List of interpolation points
                        line	        Straight line	    —
        """
        self.id = next(BlockMeshEdge.id_iter)
        self.vertices = kwargs.get('vertices')

        self.type = kwargs.get('type')
        self.interpolation_points = kwargs.get('interpolation_points', None)

    def __repr__(self):
        return f'Edge {self.id} (type={self.type}, interpolation_points={self.interpolation_points})'


class Block(object, metaclass=BlockMetaMock):

    id_iter = itertools.count()
    doc = App.newDocument()

    @classmethod
    def block_mesh_entry(cls):
        return ''

    @classmethod
    def save_fcstd(cls, filename):
        """
        save as freecad document
        :param filename: full filename; example: '/tmp/test.FCStd'
        :param shape_type: 'solid', 'faces'
        """
        doc = App.newDocument(f"Blocks")
        for block in cls.instances:
            __o__ = doc.addObject("Part::Feature", f'Block {block.name} {block.id}')
            __o__.Shape = block.fc_box
        doc.recompute()
        doc.saveCopy(filename)

    def __init__(self, *args, **kwargs):

        self.name = kwargs.get('name', 'unnamed_block')
        self.id = next(Block.id_iter)
        self.vertices = kwargs.get('vertices', [])
        self.assigned_feature = kwargs.get('assigned_feature', None)
        self.edge = kwargs.get('edge', None)
        self.num_cells = (kwargs.get('num_cells', None))
        self.block_edges = (kwargs.get('block_edges', None))

        self._fc_box = None

    def __repr__(self):
        return f'Block {self.id} ({self.name})'

    @property
    def fc_box(self):
        if self._fc_box is None:
            self._fc_box = create_box_from_points(self.vertices, self.edge)
        return self._fc_box


def get_position(vertex: FCPart.Vertex):
    return np.array([vertex.X, vertex.Y, vertex.Z])


def vector_to_np_array(vector):
    return np.array([vector.x, vector.y, vector.z])


def perpendicular_vector(x, y):
    return np.cross(x, y)


def unit_vector(vec):
    return vec / np.linalg.norm(vec)


def create_o_grid_blocks(edge, reference_face, n_cell=10):

    face_normal = vector_to_np_array(reference_face.normal)
    direction = vector_to_np_array(edge.Curve.Direction)

    if type(edge.Curve) is FCPart.Line:

        # create vertices
        # -------------------------------------------------------------------------------------------------------------
        # center block
        start_point = get_position(edge.Vertexes[0])
        end_point = get_position(edge.Vertexes[1])
        perp_vec = perpendicular_vector(face_normal, direction)
        dist = reference_face.tube_diameter/4

        # p0 = BlockMeshVertex(
        #     position=unit_vector(-face_normal + perp_vec) * dist + start_point)
        # p1 = BlockMeshVertex(
        #     position=unit_vector(-face_normal - perp_vec) * dist + start_point)
        # p2 = BlockMeshVertex(
        #     position=unit_vector(face_normal - perp_vec) * dist + start_point)
        # p3 = BlockMeshVertex(
        #     position=unit_vector(face_normal + perp_vec) * dist + start_point)
        #
        # # pipe block
        # p4 = p0 + unit_vector(-face_normal + perp_vec) * dist
        # p5 = p1 + unit_vector(-face_normal - perp_vec) * dist
        # p6 = p2 + unit_vector(face_normal - perp_vec) * dist
        # p7 = p3 + unit_vector(face_normal + perp_vec) * dist
        #
        # # outer blocks
        # p8 = p4 + (-face_normal + perp_vec) * dist
        # p9 = p4 + -face_normal * dist
        #
        # p10 = p5 + -face_normal * dist
        # p11 = p5 + (-face_normal - perp_vec) * dist
        # p12 = p5 - perp_vec * dist
        #
        # p13 = p6 - perp_vec * dist
        # p14 = p6 + (face_normal - perp_vec) * dist
        # p15 = p6 + face_normal * dist
        #
        # p16 = p7 + face_normal * dist
        # p17 = p7 + (face_normal + perp_vec) * dist
        # p18 = p7 + perp_vec * dist
        #
        # p19 = p4 + perp_vec * dist
        #
        # layer1_vertices = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19]
        layer1_vertices = create_layer_vertices(start_point, face_normal, perp_vec, dist)
        layer2_vertices = [x + (end_point - start_point) for x in layer1_vertices]

        # create edges
        # -------------------------------------------------------------------------------------------------------------
        i = 0
        layer1_edges = create_layer_edges(layer1_vertices, start_point, face_normal, reference_face, perp_vec)
        layer2_edges = create_layer_edges(layer1_vertices, end_point, face_normal, reference_face, perp_vec)

        # create blocks:
        # -------------------------------------------------------------------------------------------------------------
        blocks = create_blocks(layer1_vertices,
                               layer2_vertices,
                               layer1_edges,
                               layer2_edges,
                               [n_cell, n_cell, int(np.ceil(edge.Length / 50))],
                               edge)

    else:
        pass

    Block.save_fcstd('/tmp/blocks.FCStd')

    return blocks


def create_layer_vertices(start_point, face_normal, perp_vec, dist):

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

    # outer blocks
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

    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19]


def create_layer_edges(layer_vertices, center_point, face_normal, reference_face, perp_vec):
    e0 = BlockMeshEdge(vertices=[layer_vertices[0], layer_vertices[1]], type='line')
    e1 = BlockMeshEdge(vertices=[layer_vertices[1], layer_vertices[2]], type='line')
    e2 = BlockMeshEdge(vertices=[layer_vertices[2], layer_vertices[3]], type='line')
    e3 = BlockMeshEdge(vertices=[layer_vertices[3], layer_vertices[0]], type='line')

    # pipe edges
    e4 = BlockMeshEdge(vertices=[layer_vertices[0], layer_vertices[4]], type='line')
    e5 = BlockMeshEdge(vertices=[layer_vertices[1], layer_vertices[5]], type='line')
    e6 = BlockMeshEdge(vertices=[layer_vertices[2], layer_vertices[6]], type='line')
    e7 = BlockMeshEdge(vertices=[layer_vertices[3], layer_vertices[7]], type='line')

    e8 = BlockMeshEdge(vertices=[layer_vertices[4], layer_vertices[5]],
                       type='arc',
                       interpolation_points=[center_point - (face_normal * reference_face.tube_diameter / 2)])
    e9 = BlockMeshEdge(vertices=[layer_vertices[5], layer_vertices[6]],
                       type='arc',
                       interpolation_points=[center_point - (perp_vec * reference_face.tube_diameter / 2)])
    e10 = BlockMeshEdge(vertices=[layer_vertices[6], layer_vertices[7]],
                        type='arc',
                        interpolation_points=[center_point + (face_normal * reference_face.tube_diameter / 2)])
    e11 = BlockMeshEdge(vertices=[layer_vertices[7], layer_vertices[4]],
                        type='arc',
                        interpolation_points=[center_point + (perp_vec * reference_face.tube_diameter / 2)])

    # outer edges
    e12 = BlockMeshEdge(vertices=[layer_vertices[4], layer_vertices[19]], type='line')
    e13 = BlockMeshEdge(vertices=[layer_vertices[4], layer_vertices[9]], type='line')
    e14 = BlockMeshEdge(vertices=[layer_vertices[5], layer_vertices[10]], type='line')
    e15 = BlockMeshEdge(vertices=[layer_vertices[5], layer_vertices[12]], type='line')
    e16 = BlockMeshEdge(vertices=[layer_vertices[6], layer_vertices[13]], type='line')
    e17 = BlockMeshEdge(vertices=[layer_vertices[6], layer_vertices[15]], type='line')
    e18 = BlockMeshEdge(vertices=[layer_vertices[7], layer_vertices[16]], type='line')
    e19 = BlockMeshEdge(vertices=[layer_vertices[7], layer_vertices[18]], type='line')

    e20 = BlockMeshEdge(vertices=[layer_vertices[8], layer_vertices[9]], type='line')
    e21 = BlockMeshEdge(vertices=[layer_vertices[9], layer_vertices[10]], type='line')
    e22 = BlockMeshEdge(vertices=[layer_vertices[10], layer_vertices[11]], type='line')
    e23 = BlockMeshEdge(vertices=[layer_vertices[11], layer_vertices[12]], type='line')
    e24 = BlockMeshEdge(vertices=[layer_vertices[12], layer_vertices[13]], type='line')
    e25 = BlockMeshEdge(vertices=[layer_vertices[13], layer_vertices[14]], type='line')
    e26 = BlockMeshEdge(vertices=[layer_vertices[14], layer_vertices[15]], type='line')
    e27 = BlockMeshEdge(vertices=[layer_vertices[15], layer_vertices[16]], type='line')
    e28 = BlockMeshEdge(vertices=[layer_vertices[16], layer_vertices[17]], type='line')
    e29 = BlockMeshEdge(vertices=[layer_vertices[17], layer_vertices[18]], type='line')
    e30 = BlockMeshEdge(vertices=[layer_vertices[18], layer_vertices[19]], type='line')
    e31 = BlockMeshEdge(vertices=[layer_vertices[19], layer_vertices[8]], type='line')

    return [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23,
            e24, e25, e26, e27, e28, e29, e30, e31]


def create_blocks(layer1_vertices, layer2_vertices, layer1_edges, layer2_edges,num_cells, edge):

    index = [0, 1, 2, 3]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [0, 1, 2, 3]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b0 = Block(name=f'Center Block edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [0, 4, 5, 1]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [0, 4, 8, 5]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b1 = Block(name=f'Pipe Block 1, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [1, 5, 6, 2]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [1, 5, 9, 6]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b2 = Block(name=f'Pipe Block 2, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [2, 6, 7, 3]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [2, 6, 10, 7]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b3 = Block(name=f'Pipe Block 3, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [3, 7, 4, 0]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [3, 7, 11, 4]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b4 = Block(name=f'Pipe Block 4, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [4, 19, 8, 9]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [12, 31, 20, 13]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b5 = Block(name=f'Outer Block 5, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [4, 9, 10, 5]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [13, 21, 14, 8]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b6 = Block(name=f'Outer Block 6, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [5, 10, 11, 12]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [14, 22, 23, 15]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b7 = Block(name=f'Outer Block 7, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [5, 12, 13, 6]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [15, 24, 16, 9]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b8 = Block(name=f'Outer Block 8, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [6, 13, 14, 15]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [16, 25, 26, 17]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b9 = Block(name=f'Outer Block 9, edge {edge}',
               vertices=vertices,
               edge=edge,
               block_edges=block_edges,
               num_cells=num_cells)

    index = [6, 15, 16, 7]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [17, 27, 18, 10]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b10 = Block(name=f'Outer Block 10, edge {edge}',
                vertices=vertices,
                edge=edge,
                block_edges=block_edges,
                num_cells=num_cells)

    index = [7, 16, 17, 18]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [18, 28, 29, 19]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b11 = Block(name=f'Outer Block 11, edge {edge}',
                vertices=vertices,
                edge=edge,
                block_edges=block_edges,
                num_cells=num_cells)

    index = [7, 18, 19, 4]
    vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    edge_index = [19, 30, 12, 11]
    block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    b12 = Block(name=f'Outer Block 12, edge {edge}',
                vertices=vertices,
                edge=edge,
                block_edges=block_edges,
                num_cells=num_cells)

    return [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12]


def create_box_from_points(vertices, edge):

    points = np.array([x.position for x in vertices])
    edge = FCPart.Wire(edge)
    w0 = FCPart.makePolygon([Base.Vector(row) for row in points[[0, 1, 2, 3, 0], :]])
    base_face = FCPart.Face(w0)
    return edge.makePipe(base_face)

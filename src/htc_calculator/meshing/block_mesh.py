import uuid
import itertools
import numpy as np
from functools import lru_cache, wraps

import FreeCAD
import Part as FCPart


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


class MetaMock(type):

    instances = []

    @staticmethod
    @np_cache
    def get_vertex(position):
        return next((x for x in MetaMock.instances if np.array_equal(x.position, position)), None)

    def __call__(cls, *args, **kwargs):
        obj = cls.get_vertex(kwargs.get('position', np.array([0, 0, 0])))
        if obj is None:

            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            cls.instances.append(obj)
        return obj


class BlockMeshVertex(object, metaclass=MetaMock):
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


class Block(object):

    instances = []

    @classmethod
    def block_mesh_entry(cls):
        return ''

    def __call__(self, *args, **kw):
        x = Block(*args, **kw)
        self.instances.append(x)
        return x

    def __init__(self, *args, **kwargs):

        self.name = kwargs.get('name', 'unnamed_block')
        self.id = kwargs.get('id', uuid.uuid4())
        self.vertices = kwargs.get('vertices', {})
        self.assigned_feature = kwargs.get('assigned_feature', None)


def get_position(vertex: FCPart.Vertex):
    return np.array([vertex.X, vertex.Y, vertex.Z])


def vector_to_np_array(vector):
    return np.array([vector.x, vector.y, vector.z])


def perpendicular_vector(x, y):
    return np.cross(x, y)


def unit_vector(vec):
    return vec / np.linalg.norm(vec)


def create_o_grid_block(edge, reference_face):

    face_normal = vector_to_np_array(reference_face.normal)
    direction = vector_to_np_array(edge.Curve.Direction)

    if type(edge.Curve) is FCPart.Line:

        # create vertices

        # center block
        start_point = get_position(edge.Vertexes[0])
        end_point = get_position(edge.Vertexes[1])
        perp_vec = perpendicular_vector(face_normal, direction)
        dist = reference_face.tube_diameter/4

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

        layer1_vertices = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19]
        layer2_vertices = [x + (end_point - start_point) for x in layer1_vertices]

        # create blocks:


        print('done')



        pass
    else:
        pass

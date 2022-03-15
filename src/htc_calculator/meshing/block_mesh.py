import copy
import os.path
import uuid
import itertools
import numpy as np
from functools import lru_cache, wraps
from ..logger import logger
from ..tools import vector_to_np_array, perpendicular_vector, export_objects, angle_between_vectors, array_row_intersection
from ..geo_tools import search_contacts, surfaces_in_contact, get_position
import trimesh

import FreeCAD
import Part as FCPart
import Draft
from FreeCAD import Base
import DraftVecUtils
import PartDesign
import BOPTools.SplitFeatures

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import meshing_resources as msh_resources

App = FreeCAD


default_cell_size = 100
default_arc_cell_size = 25


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

    instances = {}

    @staticmethod
    # @np_cache
    def get_vertex(position):
        # logger.debug('Getting vertex...')
        # return next((x for x in VertexMetaMock.instances if np.allclose(x.position, position, atol=1e-3)), None)
        vert = VertexMetaMock.instances.get(tuple(position), None)
        # if vert is not None:
        #     logger.debug('Vertex already existing')
        return vert

    def __call__(cls, *args, **kwargs):
        position = np.round(kwargs.get('position', np.array([0, 0, 0])), 3)
        obj = cls.get_vertex(position)
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            # cls.instances.append(obj)
            cls.instances[tuple(position)] = obj
        return obj


class EdgeMetaMock(type):

    instances = {}

    def get_edge(self,
                 vertices=None,
                 vertex_ids: tuple[int] = None,
                 create=False):
        # logger.debug('Getting edge...')
        # return next((x for x in EdgeMetaMock.instances if np.array_equal(x.vertices, vertices)), None)
        if vertices is not None:
            edge = EdgeMetaMock.instances.get(tuple(sorted([vertices[0].id, vertices[1].id])), None)
        elif vertex_ids is not None:
            edge = EdgeMetaMock.instances.get(vertex_ids, None)
        # if edge is None:
        #     edge = EdgeMetaMock.instances.get((vertices[1].id, vertices[0].id), None)
        if edge is None:
            if create:
                edge = self(vertices=[list(BlockMeshVertex.instances.values())[vertex_ids[0]],
                                      list(BlockMeshVertex.instances.values())[vertex_ids[1]]])
        # if edge is not None:
        #     logger.debug('Edge already existing')
        return edge

    def __call__(cls, *args, **kwargs):
        vertices = kwargs.get('vertices', np.array([0, 0, 0]))
        obj = cls.get_edge(kwargs.get('vertices'))
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            # cls.instances.append(obj)
            cls.instances[tuple(sorted([vertices[0].id, vertices[1].id]))] = obj
        return obj


class ParallelEdgeSetMetaMock(type):

    instances = []

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        # cls.instances.append(obj)
        cls.instances.append(obj)
        return obj


class FaceMetaMock(type):

    instances = {}

    @staticmethod
    def get_face(vertices):
        # logger.debug('Getting face...')
        # raise NotImplementedError
        # return next((x for x in EdgeMetaMock.instances if np.array_equal(x.vertices, vertices)), None)
        face = FaceMetaMock.instances.get(tuple(sorted(vertices)), None)
        # if face is not None:
        #     logger.debug('Face already existing')
        return face

    def __call__(cls, *args, **kwargs):
        vertices = kwargs.get('vertices')
        obj = cls.get_face(kwargs.get('vertices'))
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            # cls.instances.append(obj)
            cls.instances[tuple(sorted(vertices))] = obj
        return obj


class BoundaryMetaMock(type):

    instances = {}

    def __call__(cls, *args, **kwargs):

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances[kwargs.get('name')] = obj
        return obj


class BlockMetaMock(type):

    instances = []

    @staticmethod
    @np_cache
    def get_block(vertices):
        logger.debug('Getting block...')
        return next((x for x in BlockMetaMock.instances if np.array_equal(x.vertices, vertices)), None)

    @property
    def comp_solid(self):
        if self._comp_solid is None:
            solids = []
            [solids.extend(x.fc_solid.Solids) for x in self.instances]
            self._comp_solid = FCPart.CompSolid(solids)
        return self._comp_solid

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        cls._comp_solid = None
        return obj


class CompBlockMetaMock(type):

    instances = []

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        return obj


class BlockMeshVertex(object, metaclass=VertexMetaMock):
    id_iter = itertools.count()

    @classmethod
    def block_mesh_entry(cls):
        vertex_entries = [None] * cls.instances.values().__len__()
        for i, vertex in enumerate(VertexMetaMock.instances.values()):
            vertex_entries[i] = '\t' + vertex.dict_entry
        return f'vertices\n(\n' + "\n".join(vertex_entries) + '\n);\n'

    def __init__(self, *args, **kwargs):
        self.id: int = next(BlockMeshVertex.id_iter)
        self.position = kwargs.get('position', np.array([0, 0, 0]))
        self._fc_vertex = None

        self.edges = set()
        self.faces = set()
        self.blocks = set()

    def __add__(self, vec):
        return BlockMeshVertex(position=self.position + vec)

    def __sub__(self, vec):
        return BlockMeshVertex(position=self.position - vec)

    def __repr__(self):
        return f'Vertex {self.id} (position={self.position[0], self.position[1], self.position[2]})'

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def fc_vertex(self):
        if self._fc_vertex is None:
            self._fc_vertex = FCPart.Point(Base.Vector(self.position[0], self.position[1], self.position[2]))
        return self._fc_vertex

    @property
    def dict_entry(self):
        return f'name v{self.id} ({self.position[0]:16.6f} {self.position[1]:16.6f} {self.position[2]:16.6f}) ' \
               f'// vertex {self.id}'

    def dist_to_point(self, vertex):
        return np.linalg.norm(self.position - vertex.position)

    def __lt__(self, obj):
        return self.id < obj.id

    def __gt__(self, obj):
        return self.id > obj.id

    def __le__(self, obj):
        return self.id <= obj.id

    def __ge__(self, obj):
        return self.id >= obj.id

    def __eq__(self, obj):
        return self.id == obj.id

    def __hash__(self):
        return id(self)


class BlockMeshEdge(object, metaclass=EdgeMetaMock):
    id_iter = itertools.count()

    @classmethod
    def block_mesh_entry(cls):

        edge_entries = ['\t' + x.dict_entry for x in EdgeMetaMock.instances.values() if x.dict_entry is not None]

        # edge_entries = [None] * cls.instances.values().__len__()
        # for i, edge in enumerate(EdgeMetaMock.instances.values()):
        #     edge_entries[i] = '\t' + edge.dict_entry
        return f'edges\n(\n' + "\n".join(edge_entries) + '\n);\n'

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

        self.type = kwargs.get('type', 'line')      # arc or line
        self.center = kwargs.get('center', None)
        self.interpolation_points = kwargs.get('interpolation_points', None)

        self._num_cells = None
        self._fc_edge = None
        self._collapsed = False
        self._direction = None

        # number cells:
        self.fixed_num_cells = kwargs.get('fixed_num_cells', False)
        self.num_cells = kwargs.get('num_cells', None)
        self.cell_size = kwargs.get('cell_size', default_cell_size)

        self.vertices[0].edges.add(self)
        self.vertices[1].edges.add(self)

        self.faces = set()
        self.blocks = set()

        self._parallel_edge_set = None

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def direction(self):
        if self._direction is None:
            if isinstance(self.fc_edge.Curve, FCPart.Line):
                self._direction = np.array(self.fc_edge.Curve.Direction)
            else:
                self._direction = None
        return self._direction

    @property
    def length(self):
        if self.fc_edge is not None:
            return self.fc_edge.Length
        else:
            return 0

    @property
    def fc_edge(self):
        if self._fc_edge is None:
            self._fc_edge = self.create_fc_edge()
        return self._fc_edge

    @property
    def parallel_edge_set(self):
        if self._parallel_edge_set is None:
            self._parallel_edge_set = self.get_parallel_edge_set()
        return self._parallel_edge_set

    @parallel_edge_set.setter
    def parallel_edge_set(self, value):
        self._parallel_edge_set = value

    @property
    def num_cells(self):
        if self._num_cells is None:
            if self.length is not None:
                if isinstance(self.fc_edge.Curve, FCPart.Line):
                    self._num_cells = int(np.ceil(self.length / self.cell_size))
                if isinstance(self.fc_edge.Curve, FCPart.Arc):
                    self._num_cells = int(np.ceil(self.length / self.cell_size))
        return self._num_cells

    @num_cells.setter
    def num_cells(self, value):
        self._num_cells = value

    def create_fc_edge(self):
        if self.vertices[0].fc_vertex == self.vertices[1].fc_vertex:
            self._collapsed = True
            return None
        else:
            self._collapsed = False
            if self.type == 'arc':
                return FCPart.Edge(FCPart.Arc(Base.Vector(self.vertices[0].position),
                                              Base.Vector(self.interpolation_points[0]),
                                              Base.Vector(self.vertices[1].position))
                                   )
            elif self.type in ['line', None]:
                return FCPart.Edge(FCPart.LineSegment(Base.Vector(self.vertices[0].position),
                                                      Base.Vector(self.vertices[1].position)))

    def __repr__(self):
        return f'Edge {self.id} (length={self.length}, type={self.type}, v1={self.vertices[0].id}, v2={self.vertices[1].id} interpolation_points={self.interpolation_points})'

    @property
    def dict_entry(self):
        if self.vertices[0] is self.vertices[1]:
            return None

        if self.type == 'line':
            return None
            # return f'line v{self.vertices[0].id} v{self.vertices[1].id}'
        elif self.type == 'arc':
            return f'arc {self.vertices[0].id} {self.vertices[1].id} ' \
                   f'({self.interpolation_points[0][0]:16.6f} ' \
                   f'{self.interpolation_points[0][1]:16.6f} ' \
                   f'{self.interpolation_points[0][2]:16.6f})'

    def __hash__(self):
        return id(self)

    def is_partial_same(self, other):
        if None in [self.fc_edge, other.fc_edge]:
            return False

        if type(self.fc_edge.Curve) is not type(other.fc_edge.Curve):
            return False
        if isinstance(self.fc_edge.Curve, FCPart.Line):
            if not (np.allclose(self.direction, other.direction) or np.allclose(self.direction, -other.direction)):
                return False
        if isinstance(self.fc_edge.Curve, FCPart.Arc):
            if self.fc_edge.Curve.Center != other.fc_edge.Curve.Center:
                return False
            # TODO check radius

        if self.fc_edge.distToShape(other.fc_edge)[0] > 0.1:
            return False

        return True

    def get_parallel_edge_set(self):
        return ParallelEdgesSet.get_edges_set(self)

    def translated_copy(self, translation: Base.Vector):

        if self.interpolation_points is not None:
            interpolation_points = [x + np.array(translation) for x in self.interpolation_points]
        else:
            interpolation_points = self.interpolation_points

        return BlockMeshEdge(vertices=[x + translation for x in self.vertices],
                             type=self.type,  # arc or line
                             center=self.center,
                             interpolation_points=interpolation_points
                             )


class ParallelEdgesSet(object, metaclass=ParallelEdgeSetMetaMock):

    id_iter = itertools.count()

    @classmethod
    def get_edges_set(cls, edge):
        for edge_set in ParallelEdgesSet.instances:
            if edge in edge_set.edges:
                return edge_set
        return None

    @classmethod
    def merge_sets(cls):

        do_merge = True

        while do_merge:
            do_merge = False
            for instance in cls.instances:
                for next_instance in cls.instances:
                    if instance == next_instance:
                        continue
                    if (instance.edges & next_instance.edges).__len__() > 0:
                        instance.edges.update(next_instance.edges)
                        _ = [setattr(x, 'parallel_edge_set', instance) for x in next_instance.edges]
                        instance._cell_size = default_cell_size
                        instance._num_cells = None
                        cls.instances.remove(next_instance)
                        do_merge = True
                        continue

    @classmethod
    def add_set(cls, p_edges):

        p_edges_set = set(p_edges)

        for instance in cls.instances:
            if (p_edges_set & instance.edges).__len__() > 0:
                instance.add_edges(p_edges_set)
                return instance

        return cls(edges=p_edges_set)

    def __init__(self, *args, **kwargs):

        self.id = next(ParallelEdgesSet.id_iter)

        self.edges = kwargs.get('edges', set())
        self._cell_size = kwargs.get('cell_size', default_cell_size)
        self._num_cells = kwargs.get('num_cells', None)

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def cell_size(self):
        return self._cell_size

    @cell_size.setter
    def cell_size(self, value):
        if self.cell_size == value:
            return
        self.cell_size = value
        self.num_cells = None

    @property
    def num_cells(self):
        if self._num_cells is None:
            fixed_num_cells = [x.num_cells for x in self.edges if x.fixed_num_cells]
            if fixed_num_cells:
                self._num_cells = min(fixed_num_cells)
            else:
                max_edges_length = max([x.length for x in self.edges])
                num_cells = int(np.ceil(max_edges_length/self.cell_size))
                if num_cells < 3:
                    num_cells = 3
                self._num_cells = num_cells
        return self._num_cells

    @num_cells.setter
    def num_cells(self, value: int):
        if self._num_cells is value:
            return

        if self._num_cells > value:
            logger.warning(f'Trying to set num cells value to a smaller one than existing')
            return
        self._num_cells = value

    def add_edges(self, edges: set[BlockMeshEdge]):
        self.edges.update(edges)
        _ = [setattr(x, 'parallel_edge_set', self) for x in edges]
        self.num_cells = None


class BlockMeshBoundary(object, metaclass=BoundaryMetaMock):

    id_iter = itertools.count()

    @classmethod
    def block_mesh_entry(cls):
        # boundary_entries = [None] * cls.instances.values().__len__()

        boundary_entries = ['\t' + x.dict_entry for x in BoundaryMetaMock.instances.values() if x.dict_entry is not None]
        # for i, boundary in enumerate(BoundaryMetaMock.instances.values()):
        #     entry = boundary.dict_entry
        #     if entry is not None:
        #         boundary_entries[i] = '\t' + boundary.dict_entry
        return f'boundary\n(\n' + "\n".join(boundary_entries) + '\n);\n'

    def __init__(self, *args, **kwargs):

        self.id = next(BlockMeshBoundary.id_iter)
        self.name = kwargs.get('name')
        self.type = kwargs.get('type')
        self._faces = kwargs.get('faces', set())

    def add_face(self, face):
        self._faces.add(face)

    def add_faces(self, faces):
        self._faces.update(faces)

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        self._faces = value
        [x.set_boundary(self) for x in self._faces]

    @property
    def dict_entry(self):

        if self.type == 'interface':
            return None

        faces_entry = "\n".join(['\t\t\t(' + ' '.join([str(y.id) for y in x.vertices]) + ')' for x in self.faces])

        return (f"\t{self.name}\n"
                f"\t{'{'}\n"
                f"\t\ttype {self.type};\n"
                f"\t\tfaces\n"
                f"\t\t(\n"
                f"{faces_entry}\n"
                f"\t\t);\n"
                f"\t{'}'}")

    def __repr__(self):
        return f'Boundary {self.id} (name={self.name}, type={self.type}, faces={self.faces})'


inlet_patch = BlockMeshBoundary(name='inlet', type='patch')
outlet_patch = BlockMeshBoundary(name='outlet', type='patch')
wall_patch = BlockMeshBoundary(name='wall', type='wall')
pipe_wall_patch = BlockMeshBoundary(name='pipe_wall', type='interface')
top_side_patch = BlockMeshBoundary(name='top_side', type='patch')
bottom_side_patch = BlockMeshBoundary(name='bottom_side', type='patch')


class BlockMeshFace(object, metaclass=FaceMetaMock):
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
        self.name = kwargs.get('name', 'unnamed face')
        self.id = next(BlockMeshFace.id_iter)
        self.vertices = kwargs.get('vertices')

        self.merge = kwargs.get('merge', True)
        self.check_merge_patch_pairs = kwargs.get('check_merge_patch_pairs', True)
        self.contacts = kwargs.get('contacts', set())

        self._boundary = kwargs.get('boundary', None)
        self._fc_face = None

        self._edge0 = None
        self._edge1 = None
        self._edge2 = None
        self._edge3 = None

        self.extruded = kwargs.get('extruded', False)   # false or: [base_profile, top_profile, path1, path2]

        _ = [x.faces.add(self) for x in self.vertices]

        self.blocks = set()
        _ = self.edges

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def area(self):
        if self.fc_face is not None:
            return self.fc_face.Area
        else:
            return 0

    @property
    def normal(self):
        if self.fc_face is None:
            return None
        return self.fc_face.Normal

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, value: BlockMeshBoundary):
        if not value == self._boundary:
            value.add_face(self)
        self._boundary = value

    @property
    def edge0(self):
        if self._edge0 is None:
            self._edge0 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[0].id,
                                                                          self.vertices[1].id])),
                                                 create=True)
            self._edge0.faces.add(self)
        return self._edge0

    @property
    def edge1(self):
        if self._edge1 is None:
            self._edge1 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[1].id,
                                                                          self.vertices[2].id])),
                                                 create=True)
            self._edge1.faces.add(self)
        return self._edge1

    @property
    def edge2(self):
        if self._edge2 is None:
            self._edge2 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[2].id,
                                                                          self.vertices[3].id])),
                                                 create=True)
            self._edge2.faces.add(self)
        return self._edge2

    @property
    def edge3(self):
        if self._edge3 is None:
            self._edge3 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[3].id,
                                                                          self.vertices[0].id])),
                                                 create=True)
            self._edge3.faces.add(self)
        return self._edge3

    @property
    def edges(self):
        return [self.edge0, self.edge1, self.edge2, self.edge3]

    @property
    def fc_face(self):
        if self._fc_face is None:
            self._fc_face = self.create_fc_face()
        return self._fc_face

    @fc_face.setter
    def fc_face(self, value):
        self._fc_face = value

    def set_boundary(self, value: BlockMeshBoundary):
        self._boundary = value

    def create_fc_face(self):

        try:
            if any([isinstance(x.fc_edge.Curve, FCPart.Arc) for x in self.edges if x.fc_edge is not None]):
                print('debug')
        except Exception as e:
            raise e

        if isinstance(self.extruded, list):
            # https://forum.freecadweb.org/viewtopic.php?t=21636

            # save_fcstd([x.fc_edge for x in self.extruded], f'/tmp/extrude_edges_face{self.id}')
            # ex_wire = FCPart.Wire([x.fc_edge for x in self.extruded])
            # save_fcstd([ex_wire], f'/tmp/extrude_wire{self.id}')

            doc = App.newDocument()
            sweep = doc.addObject('Part::Sweep', 'Sweep')
            section0 = doc.addObject("Part::Feature", f'section0')
            section0.Shape = FCPart.Wire(self.extruded[0].fc_edge)
            section1 = doc.addObject("Part::Feature", f'section1')
            section1.Shape = FCPart.Wire(self.extruded[1].fc_edge)
            spine = doc.addObject("Part::Feature", f'spine')
            spine.Shape = FCPart.Wire(self.extruded[2].fc_edge)

            sweep.Sections = [section0, section1]
            sweep.Spine = spine
            sweep.Solid = False
            sweep.Frenet = False

            doc.recompute()

            return sweep.Shape

            # body = doc.addObject('PartDesign::Body', 'Body')
            # AdditivePipe = doc.addObject("PartDesign::AdditivePipe", "AdditivePipe")
            #
            #
            #
            #
            #
            # body.addObject(spine)
            #
            # AdditivePipe.Profile = sketch1
            # AdditivePipe.Sections = [sketch1, sketch2]
            #
            # doc.recompute()
            #
            # doc.saveCopy('/tmp/sweep_test2.FCStd')
            #
            # AdditivePipe.Profile = FCPart.Wire(self.extruded[0].fc_edge)
            #
            # sweep = FCPart.BRepSweep.MakePipeShell(FCPart.Wire(self.extruded[2].fc_edge))
            #
            # ps = FCPart.BRepOffsetAPI.MakePipeShell(FCPart.Wire(self.extruded[2].fc_edge))
            # ps.setFrenetMode(True)
            # # ps.setSpineSupport(FCPart.Wire(edges[0]))
            # # ps.setAuxiliarySpine(FCPart.Wire(self.extruded[3].fc_edge), True, False)
            # ps.add(FCPart.Wire(self.extruded[0].fc_edge), True, True)
            # ps.add(FCPart.Wire(self.extruded[1].fc_edge), True, True)
            # if ps.isReady():
            #     ps.build()
            # face = ps.shape()
        else:
            edges = [x.fc_edge for x in self.edges if x.fc_edge is not None]

            if edges.__len__() < 3:
                return None
            else:
                face_wire = FCPart.Wire(edges)

                try:
                    face = FCPart.Face(face_wire)
                except:
                    face = FCPart.makeFilledFace(FCPart.Wire(edges).OrderedEdges)

            # face = make_complex_face_from_edges(face_wire.OrderedEdges)

        if face.Area < 1e-3:
            logger.warning(f'Face {self.id} area very small: {face.Area}')

        return face

    def common_edges(self, other, partial=True):

        if partial:
            edge_combinations = np.stack(np.meshgrid(self.edges, other.edges), -1).reshape(-1, 2)
            try:
                return edge_combinations[[x[0].is_partial_same(x[1]) for x in edge_combinations], 0]
            except Exception as e:
                raise e
        else:
            raise NotImplementedError

    def is_same(self, other):
        return tuple(sorted(self.vertices)) == tuple(sorted(other.vertices))

    def is_part(self, other):
        common_elements = set(self.edges) & set(other.edges)
        return common_elements.__len__() > 1

    def common_with(self, other):
        if None in [self.fc_face, other.fc_face]:
            return False

        if self.is_same(other):
            return True
        if self.is_part(other):
            return True
        if self.common_edges(other).shape[0] > 1:
            return True

        return surfaces_in_contact(self.fc_face, other.fc_face)

    def extrude(self, dist, direction=None, dist2=None):

        non_regular = False

        if direction is None:
            direction = self.normal

        vertices = self.vertices
        if set(vertices).__len__() == 3:
            # vertices = list(set(self.vertices))
            vertices = [ii for n, ii in enumerate(self.vertices) if ii not in self.vertices[:n]]
            non_regular = True

        if (dist2 is None) or (abs(dist2) < 1):
            v_1 = vertices
            _1 = self.edges
            _2 = [x.translated_copy(direction * dist) for x in self.edges]
        else:
            v_1 = np.array([x + (dist2 * direction) for x in vertices])
            _1 = [x.translated_copy(direction * dist2) for x in self.edges]
            _2 = [x.translated_copy(direction * dist) for x in self.edges]

        v_2 = np.array([x + (dist * direction) for x in self.vertices])
        _3 = [BlockMeshEdge(vertices=[v_1[i], v_2[i]], type='line') for i in range(v_1.__len__())]

        # export_objects([x.fc_edge for x in [*_1, *_2, *_3]], '/tmp/extruded_edges.FCStd')

        # create quad blocks:

        if vertices.__len__() == 4:

            new_block = Block(vertices=[*v_1, *v_2],
                              name=f'Extruded Block',
                              block_edges=[*_1, *_2, *_3],
                              auto_cell_size=True,
                              non_regular=non_regular,
                              extruded=False)
        elif vertices.__len__() == 3:
            v_1 = [*v_1, v_1[0]]
            v_2 = [*v_2, v_2[0]]

            new_block = Block(vertices=[*v_1, *v_2],
                              name=f'Extruded Block',
                              block_edges=[*_1, *_2, *_3],
                              auto_cell_size=True,
                              extruded=False,
                              non_regular=True)

        return new_block

    def __repr__(self):
        return f'Face {self.id} (type={self.boundary}, Area={self.area}, Vertices={self.vertices})'

    def __eq__(self, other):
        return sorted(self.vertices) == sorted(other.vertices)

    def __hash__(self):
        return id(self)

    # if type(edge.Curve) is FCPart.Line:
    #     return [BlockMeshEdge(vertices=[p1[i], p2[i]], type='line') for i in range(4)]
    # else:
    #     edges = [None] * 4
    #     for i in range(4):
    #         # move center point
    #         center = np.array(edge.Curve.Center) + (p1[i].position - edge.Vertexes[0].Point) * face_normal


class Block(object, metaclass=BlockMetaMock):

    face_map = {
        'inlet': (0, 1, 2, 3),      # 0
        'outlet': (4, 5, 6, 7),     # 1
        'left': (4, 0, 3, 7),       # 2
        'right': (5, 1, 2, 6),      # 3
        'top': (4, 5, 1, 0),        # 4
        'bottom': (7, 6, 2, 3)      # 5
    }

    parallel_edges_dict = {0: np.array([2, 6, 4]),
                           1: np.array([5, 7, 3]),
                           2: np.array([6, 4, 0]),
                           3: np.array([1, 5, 7]),
                           4: np.array([6, 2, 0]),
                           5: np.array([7, 3, 1]),
                           6: np.array([4, 0, 2]),
                           7: np.array([3, 1, 5]),
                           8: np.array([9, 10, 11]),
                           9: np.array([10, 11, 8]),
                           10: np.array([11, 8, 9]),
                           11: np.array([8, 9, 10])}

    all_parallel_edges = np.array([[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 11]])

    _edge_vertex_map = [[0, 1],     # 0
                        [1, 2],     # 1
                        [2, 3],     # 2
                        [3, 0],     # 3
                        [4, 5],     # 4
                        [5, 6],     # 5
                        [6, 7],     # 6
                        [7, 4],     # 7
                        [0, 4],     # 8
                        [1, 5],     # 9
                        [2, 6],     # 10
                        [3, 7]]     # 11

    _comp_solid = None

    id_iter = itertools.count()
    doc = App.newDocument()

    @classmethod
    def update_parallel_edges(cls):
        for instance in cls.instances:
            instance._parallel_edges_sets = []
            for p_edges in Block.all_parallel_edges:
                instance._parallel_edges_sets.append(ParallelEdgesSet.add_set(instance.block_edges[p_edges]))

    @classmethod
    def block_mesh_entry(cls):

        block_entries = ['\t' + x.dict_entry for x in BlockMetaMock.instances if x.dict_entry is not None]
        # block_entries = [None] * cls.instances.__len__()
        # for i, block in enumerate(BlockMetaMock.instances):
        #     block_entries[i] = '\t' + block.dict_entry
        return f'blocks\n(\n' + "\n".join(block_entries) + '\n);\n'

    @classmethod
    def save_fcstd(cls, filename):
        """
        save as freecad document
        :param filename: full filename; example: '/tmp/test.FCStd'
        :param shape_type: 'solid', 'faces'
        """
        doc = App.newDocument(f"Blocks")
        for i, block in enumerate(cls.instances):
            try:
                __o__ = doc.addObject("Part::Feature", f'Block {block.name} {block.id}')
                __o__.Shape = block.fc_solid
            except Exception as e:
                logger.error(f'Error saving Block {i} {block.id} {block.name} {block}:\n{e}')
                raise e
        doc.recompute()
        doc.saveCopy(filename)

    @classmethod
    def search_merge_patch_pairs(cls):
        # https://forum.freecadweb.org/viewtopic.php?t=15699&start=130
        blocks_to_check = [x for x in cls.instances if x.check_merge_patch_pairs]

        combinations = list(itertools.combinations(blocks_to_check, 2))
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)

        merge_faces = []

        i_block = 0
        while blocks_to_check:
            block = blocks_to_check[0]
            blocks_to_check = blocks_to_check[1:]
            # for i_block, block in enumerate(blocks_to_check):
            logger.debug(f'Processing merge patch pairs for block {i_block} of {blocks_to_check.__len__()}')
            if block.check_merge_patch_pairs:
                for io_block, other_block in enumerate(blocks_to_check):
                    # logger.debug(f'\tCheckig with other block {io_block} of {blocks_to_check.__len__()}')
                    block_dist = block.fc_solid.distToShape(other_block.fc_solid)[0]
                    if block_dist > 0.1:
                        continue

                    for s_face in block.faces:
                        if s_face.fc_face is None:
                            continue
                        # if s_face.blocks.__len__() == 2:
                        #     continue
                        for o_face in other_block.faces:
                            if o_face.id == s_face.id:
                                continue
                            if o_face in s_face.contacts:
                                continue
                            if o_face.fc_face is None:
                                continue

                            dist = s_face.fc_face.distToShape(o_face.fc_face)[0]
                            if dist < 0.1:
                                if surfaces_in_contact(s_face.fc_face, o_face.fc_face):
                                    merge_faces.append([s_face, o_face])
                                    s_face.blocks.update([block, other_block])
                                    s_face.contacts.add(o_face)
                                    o_face.contacts.add(s_face)
                                # export_objects([s_face.fc_face, o_face.fc_face], '/tmp/test.FCStd')
            i_block += 1

        return merge_faces

    def __init__(self, *args, **kwargs):
        """
        create a block from vertices

        layer1:

    try:
        shell = FCPart.makeShell(faces)
        shell.sewShape()
        try:
            shell.fix(1e-5, 1e-5, 1e-5)
        except Exception as e:
            logger.warning(f'Could not fix block shell')
        solid = FCPart.Solid(shell)
    except Exception as e:
        logger.error(f'Error creating geometric volume for {block.__repr__()}: {e}')
        save_fcstd(faces, f'/tmp/error_shape_{block.id}.FCStd')
        raise e

        layer2:


        :param args:
        :param kwargs:
        """

        self._num_cells = None
        self._block_edges = None
        self._fc_solid = None
        self._faces = None
        self._edge = None
        self._dirty_center = None

        self.name = kwargs.get('name', 'unnamed_block')
        self.id = next(Block.id_iter)
        self.vertices = kwargs.get('vertices', [])
        self.assigned_feature = kwargs.get('assigned_feature', None)
        self.edge = kwargs.get('edge', None)
        # self.num_cells = kwargs.get('num_cells', None)
        self.block_edges = kwargs.get('block_edges', None)
        self.cell_zone = kwargs.get('cell_zone', None)
        self.auto_cell_size = kwargs.get('auto_cell_size', True)
        self.cell_size = kwargs.get('cell_size', 100)

        self.non_regular = kwargs.get('non_regular', False)
        self.extruded = kwargs.get('extruded', False)
        self.check_merge_patch_pairs = kwargs.get('check_merge_patch_pairs', True)

        self._face0 = kwargs.get('face0', None)
        self._face1 = kwargs.get('face1', None)
        self._face2 = kwargs.get('face2', None)
        self._face3 = kwargs.get('face3', None)
        self._face4 = kwargs.get('face4', None)
        self._face5 = kwargs.get('face5', None)

        self._edge0 = kwargs.get('edge0', None)
        self._edge1 = kwargs.get('edge1', None)
        self._edge2 = kwargs.get('edge2', None)
        self._edge3 = kwargs.get('edge3', None)
        self._edge4 = kwargs.get('edge4', None)
        self._edge5 = kwargs.get('edge5', None)
        self._edge6 = kwargs.get('edge6', None)
        self._edge7 = kwargs.get('edge7', None)
        self._edge8 = kwargs.get('edge8', None)
        self._edge9 = kwargs.get('edge9', None)
        self._edge10 = kwargs.get('edge10', None)
        self._edge11 = kwargs.get('edge11', None)

        _ = self.block_edges
        _ = self.faces

        self._parallel_edges_sets = None

        self._parallel_edges_sets = []
        for p_edges in Block.all_parallel_edges:
            self._parallel_edges_sets.append(ParallelEdgesSet.add_set(self.block_edges[p_edges]))

        self.pipe_layer_top = kwargs.get('pipe_layer_top', False)
        self.pipe_layer_extrude_top = kwargs.get('pipe_layer_extrude_top', None)
        self.pipe_layer_bottom = kwargs.get('pipe_layer_bottom', False)
        self.pipe_layer_extrude_bottom = kwargs.get('pipe_layer_extrude_bottom', None)

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def dict_entry(self):

        # export_objects([self.fc_solid, *[x.fc_vertex.toShape() for x in self.vertices]], '/tmp/test_export.FCStd')

        if self.non_regular:
            v0 = self.vertices[2].fc_vertex.toShape().Point - self.vertices[1].fc_vertex.toShape().Point
            v1 = self.vertices[0].fc_vertex.toShape().Point - self.vertices[1].fc_vertex.toShape().Point
            v2 = self.vertices[5].fc_vertex.toShape().Point - self.vertices[1].fc_vertex.toShape().Point

            v2_ref = v0.normalize().cross(v1.normalize()).normalize()

            if np.allclose(v2_ref, v2.normalize()):
                corrected_vertices = np.array(self.vertices)[np.array([1, 2, 3, 0, 5, 6, 7, 4])].tolist()
            elif np.allclose(v2_ref, -v2.normalize()):
                corrected_vertices = np.array(self.vertices)[np.array([5, 6, 7, 4, 1, 2, 3, 0])].tolist()
            else:
                if abs(angle_between_vectors(v2_ref, v2, v0)) < 90:
                    corrected_vertices = np.array(self.vertices)[np.array([1, 2, 3, 0, 5, 6, 7, 4])].tolist()
                else:
                    corrected_vertices = np.array(self.vertices)[np.array([5, 6, 7, 4, 1, 2, 3, 0])].tolist()

        else:
            v0 = self.vertices[1].fc_vertex.toShape().Point - self.vertices[0].fc_vertex.toShape().Point
            v1 = self.vertices[3].fc_vertex.toShape().Point - self.vertices[0].fc_vertex.toShape().Point
            v2 = self.vertices[4].fc_vertex.toShape().Point - self.vertices[0].fc_vertex.toShape().Point

            try:
                v2_ref = v0.normalize().cross(v1.normalize()).normalize()
            except Exception as e:
                raise e

            if np.allclose(v2_ref, v2.normalize()):
                corrected_vertices = self.vertices
            elif np.allclose(v2_ref, -v2.normalize()):
                corrected_vertices = [*self.vertices[4:], *self.vertices[0:4]]
            else:
                if abs(angle_between_vectors(v2_ref, v2, v0)) < 90:
                    corrected_vertices = self.vertices
                else:
                    corrected_vertices = [*self.vertices[4:], *self.vertices[0:4]]

        if self.cell_zone is not None:
            cell_zone = self.cell_zone.txt_id
        else:
            cell_zone = None

        return f"hex ({' '.join(['v' + str(x.id) for x in corrected_vertices])}) {cell_zone} " \
               f"({self.num_cells[0]} {self.num_cells[1]} {self.num_cells[2]}) simpleGrading (1 1 1)   " \
               f"// block {self.id}"

    # @property
    # def center_line(self):
    #     return self.faces[0].C
    @property
    def face0(self):
        if self._face0 is None:
            self._face0 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['inlet']])
            self._face0.blocks.add(self)
        return self._face0

    @property
    def face1(self):
        if self._face1 is None:
            self._face1 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['outlet']])
            self._face1.blocks.add(self)
        return self._face1

    @property
    def face2(self):
        if self._face2 is None:
            if self.extruded:
                extruded = [self.edge3, self.edge7, self.edge8, self.edge11]
            else:
                extruded = False

            self._face2 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['left']],
                                        extruded=extruded)
            self._face2.blocks.add(self)
        return self._face2

    @property
    def face3(self):
        if self._face3 is None:
            if self.extruded:
                extruded = [self.edge1, self.edge5, self.edge9, self.edge10]
            else:
                extruded = False

            self._face3 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['right']],
                                        extruded=extruded)
            self._face3.blocks.add(self)
        return self._face3

    @property
    def face4(self):
        if self._face4 is None:
            if self.extruded:
                extruded = [self.edge0, self.edge4, self.edge8, self.edge9]
            else:
                extruded = False

            self._face4 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['top']],
                                        extruded=extruded)
            self._face4.blocks.add(self)
        return self._face4

    @property
    def face5(self):
        if self._face5 is None:
            if self.extruded:
                extruded = [self.edge2, self.edge6, self.edge10, self.edge11]
            else:
                extruded = False

            self._face5 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['bottom']],
                                        extruded=extruded)
            self._face5.blocks.add(self)
        return self._face5

    @property
    def faces(self):
        return [self.face0, self.face1, self.face2, self.face3, self.face4, self.face5]

    @property
    def edge0(self):
        if self._edge0 is None:
            v_map = self._edge_vertex_map[0]
            self._edge0 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge0.blocks.add(self)
        return self._edge0

    @property
    def edge1(self):
        if self._edge1 is None:
            v_map = self._edge_vertex_map[1]
            self._edge1 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge1.blocks.add(self)
        return self._edge1

    @property
    def edge2(self):
        if self._edge2 is None:
            v_map = self._edge_vertex_map[2]
            self._edge2 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge2.blocks.add(self)
        return self._edge2

    @property
    def edge3(self):
        if self._edge3 is None:
            v_map = self._edge_vertex_map[3]
            self._edge3 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge3.blocks.add(self)
        return self._edge3

    @property
    def edge4(self):
        if self._edge4 is None:
            v_map = self._edge_vertex_map[4]
            self._edge4 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge4.blocks.add(self)
        return self._edge4

    @property
    def edge5(self):
        if self._edge5 is None:
            v_map = self._edge_vertex_map[5]
            self._edge5 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge5.blocks.add(self)
        return self._edge5

    @property
    def edge6(self):
        if self._edge6 is None:
            v_map = self._edge_vertex_map[6]
            self._edge6 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge6.blocks.add(self)
        return self._edge6

    @property
    def edge7(self):
        if self._edge7 is None:
            v_map = self._edge_vertex_map[7]
            self._edge7 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge7.blocks.add(self)
        return self._edge7

    @property
    def edge8(self):
        if self._edge8 is None:
            v_map = self._edge_vertex_map[8]
            self._edge8 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge8.blocks.add(self)
        return self._edge8

    @property
    def edge9(self):
        if self._edge9 is None:
            v_map = self._edge_vertex_map[9]
            self._edge9 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge9.blocks.add(self)
        return self._edge9

    @property
    def edge10(self):
        if self._edge10 is None:
            v_map = self._edge_vertex_map[10]
            self._edge10 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                           self.vertices[v_map[1]].id])),
                                                  create=True)
            self._edge10.blocks.add(self)
        return self._edge10

    @property
    def edge11(self):
        if self._edge11 is None:
            v_map = self._edge_vertex_map[11]
            self._edge11 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                           self.vertices[v_map[1]].id])),
                                                  create=True)
            self._edge11.blocks.add(self)
        return self._edge11

    @property
    def block_edges(self):
        return np.array([self.edge0, self.edge1, self.edge2,
                         self.edge3, self.edge4, self.edge5,
                         self.edge6, self.edge7, self.edge8,
                         self.edge9, self.edge10, self.edge11])

    @block_edges.setter
    def block_edges(self, value):
        if value.__len__() != 12:
            raise ValueError(f'Error setting block edges of block {self.name}. Value length must be 12, but value is {value}')

        self._edge0 = value[0]
        self._edge1 = value[1]
        self._edge2 = value[2]
        self._edge3 = value[3]
        self._edge4 = value[4]
        self._edge5 = value[5]
        self._edge6 = value[6]
        self._edge7 = value[7]
        self._edge8 = value[8]
        self._edge9 = value[9]
        self._edge10 = value[10]
        self._edge11 = value[11]

    @block_edges.setter
    def block_edges(self, value):
        self._block_edges = value

    @property
    def num_cells(self):
        if self._num_cells is None:
            if self.auto_cell_size:
                self._num_cells = self.calc_cell_size()
        return self._num_cells

    @num_cells.setter
    def num_cells(self, value):
        self._num_cells = value

    @property
    def fc_solid(self):
        if self._fc_solid is None:
            logger.debug(f'Creating fc_solid for block {self.id}')
            self._fc_solid = self.create_fc_solid()
        return self._fc_solid

    @property
    def dirty_center(self):
        if self._dirty_center is None:
            self._dirty_center = np.mean(np.array([x.position for x in self.vertices]), axis=0)
        return self._dirty_center

    @property
    def parallel_edges_sets(self):
        if self._parallel_edges_sets is None:
            self._parallel_edges_sets = self.get_parallel_edges_sets()
        return self._parallel_edges_sets

    # @property
    # def edge(self):
    #     if self._edge is None:
    #         self._edge = self.generate_edge()
    #     return self._edge

    # @edge.setter
    # def edge(self, value):
    #     self._edge = value

    def create_fc_solid(self):
        faces = []
        _ = [faces.extend(face.fc_face.Faces) for face in self.faces if face.fc_face is not None]

        try:
            shell = FCPart.makeShell(faces)
            shell.sewShape()
            try:
                shell.fix(1e-3, 1e-3, 1e-3)
            except Exception as e:
                logger.warning(f'Block {self.id}: Could not fix block shell')
            if not shell.isClosed():
                logger.warning(f'Block {self.id} shell is not closed. Trying to fix with larger tolerance')
                try:
                    shell.fix(1, 1, 1)
                except Exception as e:
                    logger.warning(f'Block {self.id}: Could not fix block shell')
            if not shell.isClosed():
                logger.error(f'Block {self.id} shell is not closed.')
                raise Exception(f'Block {self.id} is not closed')

            solid = FCPart.Solid(shell)
        except Exception as e:
            logger.error(f'Error creating geometric volume for {self.__repr__()}: {e}')
            save_fcstd(faces, f'/tmp/block_{self.id}_faces.FCStd')
            raise e

        if solid.Volume < 0:
            solid.complement()

        if solid.Volume < 1e-3:
            logger.warning(f'Block {self.id} volume very small: {solid.Volume}. Please check this block')

        return solid

    def calc_cell_size(self):
        return [self.edge0.parallel_edge_set.num_cells,
                self.edge1.parallel_edge_set.num_cells,
                self.edge8.parallel_edge_set.num_cells]

        # l1 = self.vertices[1].dist_to_point(self.vertices[0])
        # l2 = self.vertices[3].dist_to_point(self.vertices[0])
        # l3 = self.vertices[4].dist_to_point(self.vertices[0])
        #
        # return [int(np.ceil(l1/self.cell_size)),
        #         int(np.ceil(l2/self.cell_size)),
        #         int(np.ceil(l3/self.cell_size))]

    # def generate_edge(self):
    #     p1 = Base.Vector(self.vertices[0].position + 0.5 * (self.vertices[2].position - self.vertices[0].position))
    #     p2 = Base.Vector(self.vertices[4].position + 0.5 * (self.vertices[6].position - self.vertices[4].position))
    #     return FCPart.Edge(FCPart.LineSegment(p1, p2))

    def __repr__(self):
        return f'Block {self.id} ({self.name})'

    def dist_to(self, other):
        return self.fc_solid.distToShape(other.fc_solid)[0]

    def get_parallel_edges(self, edge):

        edge_block_id = self.get_block_edge_id(edge)
        return self.block_edges[self.parallel_edges_dict[edge_block_id]]

    def get_block_edge_id(self, edge):
        return self.block_edges.index(edge)

    def get_parallel_edges_sets(self):
        parallel_edges_sets = []
        for p_edges in Block.all_parallel_edges:
            parallel_edges_sets.append(ParallelEdgesSet.add_set(self.block_edges[p_edges]))
        return parallel_edges_sets


class CompBlock(object, metaclass=CompBlockMetaMock):

    @classmethod
    def search_merge_patch_pairs(cls):

        merge_faces = []

        for comp_block in cls.instances:
            for other_comp_block in cls.instances:

                if comp_block is other_comp_block:
                    continue

                export_objects([comp_block.fc_solid, other_comp_block.fc_solid], '/tmp/blocks.FCStd')

                # split = BOPTools.SplitAPI.booleanFragments([comp_block.fc_solid, other_comp_block.fc_solid], 'Standard', 0.1)
                #
                # common = comp_block.fc_solid.common(other_comp_block.fc_solid)
                # # export_objects([split], '/tmp/split.FCStd')
                # comp_block.fc_solid.common(other_comp_block.fc_solid)
                # # export_objects([comp_block.fc_solid, other_comp_block.fc_solid, common], '/tmp/common.FCStd')
                # # export_objects([comp_block.fc_solid], '/tmp/comp_block.FCStd')
                # # c_block_dist = comp_block.fc_solid.distToShape(other_comp_block.fc_solid)[0]
                # # if c_block_dist > 1:
                # #     continue
                # total_faces = comp_block.hull_faces.__len__() * other_comp_block.hull_faces.__len__()
                # ii = -1

                for i, face in enumerate(comp_block.hull_faces):
                    if face.fc_face is None:
                        continue

                    for j, other_face in enumerate(other_comp_block.hull_faces):
                        if face.common_with(other_face):
                            merge_faces.append([face, other_face])
                            face.blocks.update([*face.blocks, *other_face.blocks])
                            face.contacts.add(other_face)
                            other_face.contacts.add(face)

        return merge_faces

    def __init__(self, *args, **kwargs):
        self._id = kwargs.get('_id', kwargs.get('id', uuid.uuid4()))
        self._name = kwargs.get('_name', kwargs.get('name', 'BlockMesh {}'.format(self._id)))

        self._blocks = None
        self._faces = None
        self._hull_faces = None
        self._fc_solid = None

        self.blocks = kwargs.get('blocks', [])

    @property
    def blocks(self):
        return self._blocks

    @blocks.setter
    def blocks(self, value):
        self._blocks = value
        self._faces = None
        self._hull_faces = None
        self._fc_solid = None

    @property
    def hull_faces(self):
        if self._hull_faces is None:
            faces_list = set()
            [faces_list.update([face for face in block.faces if face.blocks.__len__() == 1]) for block in self.blocks]
            self._hull_faces = list(faces_list)
        return self._hull_faces

    @property
    def faces(self):
        if self._faces is None:
            faces_list = set()
            [faces_list.update([face for face in block.faces]) for block in self.blocks]
            self._faces = list(faces_list)
        return self._faces

    @property
    def fc_solid(self):
        if self._fc_solid is None:
            self._fc_solid = self.create_hull_solid()
        return self._fc_solid

    def create_hull_solid(self):
        fc_faces_list = []
        _ = [fc_faces_list.extend(face.fc_face.Faces) for face in self.hull_faces if face.fc_face is not None]

        shell = FCPart.makeShell(fc_faces_list)
        shell.sewShape()
        shell.fix(1e-7, 1e-7, 1e-7)
        solid = FCPart.Solid(shell)

        return solid


class BlockMesh(object):

    default_path = '/tmp/'

    def __init__(self, *args, **kwargs):
        self._id = kwargs.get('_id', kwargs.get('id', uuid.uuid4()))
        self._name = kwargs.get('_name', kwargs.get('name', 'BlockMesh {}'.format(self._id)))
        self._case_dir = kwargs.get('_case_dir', kwargs.get('case_dir', None))

        self.template = pkg_resources.read_text(msh_resources, 'block_mesh_dict')
        self._block_mesh_dict = None
        self._control_dict = None
        self._fvschemes = None
        self._fvsolution = None

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        if value == self._id:
            return
        self._id = value

    @property
    def control_dict(self):
        if self._control_dict is None:
            self._control_dict = pkg_resources.read_text(msh_resources, 'controlDict')
        return self._control_dict

    @control_dict.setter
    def control_dict(self, value):
        if value == self._control_dict:
            return
        self._control_dict = value

    @property
    def block_mesh_dict(self):
        if self._block_mesh_dict is None:
            self._block_mesh_dict = self.create_block_mesh_dict()
        return self._block_mesh_dict

    @block_mesh_dict.setter
    def block_mesh_dict(self, value):
        if value == self._block_mesh_dict:
            return
        self._block_mesh_dict = value

    @property
    def fvschemes(self):
        if self._fvschemes is None:
            self._fvschemes = pkg_resources.read_text(msh_resources, 'fvSchemes')
        return self._fvschemes

    @fvschemes.setter
    def fvschemes(self, value):
        if value == self._fvschemes:
            return
        self._fvschemes = value

    @property
    def fvsolution(self):
        if self._fvsolution is None:
            self._fvsolution = pkg_resources.read_text(msh_resources, 'fvSolution')
        return self._fvsolution

    @fvsolution.setter
    def fvsolution(self, value):
        if value == self._fvsolution:
            return
        self._fvsolution = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value == self._name:
            return
        self._name = value

    @property
    def case_dir(self):
        if self._case_dir is None:
            self._case_dir = os.path.join(self.default_path, 'case_' + str(self.id.hex))
        return self._case_dir

    @case_dir.setter
    def case_dir(self, value):
        if value == self._case_dir:
            return
        self._case_dir = value

    def init_case(self):
        logger.info('Initializing case...')

        os.makedirs(self.case_dir, exist_ok=True)
        os.makedirs(os.path.join(self.case_dir, '0'), exist_ok=True)
        os.makedirs(os.path.join(self.case_dir, 'constant'), exist_ok=True)
        os.makedirs(os.path.join(self.case_dir, 'system'), exist_ok=True)

        if not os.path.isfile(os.path.join(self.case_dir, 'system', "controlDict")):
            self.write_control_dict()
        if not os.path.isfile(os.path.join(self.case_dir, 'system', "fvSchemes")):
            self.write_fv_schemes()
        if not os.path.isfile(os.path.join(self.case_dir, 'system', "fvSolution")):
            self.write_fv_solution()
        self.write_block_mesh_dict()

        logger.info('Case successfully initialized')

    def write_control_dict(self):

        with open(os.path.join(self.case_dir, 'system', "controlDict"), mode="w") as f:
            f.write(self.control_dict)

    def write_fv_schemes(self):
        with open(os.path.join(self.case_dir, 'system', "fvSchemes"), "w") as f:
            f.write(self.fvschemes)

    def write_fv_solution(self):
        with open(os.path.join(self.case_dir, 'system', "fvSolution"), "w") as f:
            f.write(self.fvsolution)

    @staticmethod
    def create_block_mesh_dict():

        # contacts = CompBlock.search_merge_patch_pairs()
        Block.update_parallel_edges()
        ParallelEdgesSet.merge_sets()

        # export_objects([FCPart.Compound([x.fc_edge for x in y.edges]) for y in ParallelEdgesSet.instances],
        #                '/tmp/parallel_edges.FCStd')

        logger.info('Creating blockMeshDict...')

        template = pkg_resources.read_text(msh_resources, 'block_mesh_dict')

        vertices_entry = BlockMeshVertex.block_mesh_entry()
        template = template.replace('<vertices>', vertices_entry)

        edges_entry = BlockMeshEdge.block_mesh_entry()
        template = template.replace('<edges>', edges_entry)

        block_entry = Block.block_mesh_entry()
        template = template.replace('<blocks>', block_entry)

        boundary_entry = BlockMeshBoundary.block_mesh_entry()
        template = template.replace('<boundary>', boundary_entry)

        template = template.replace('<faces>', '')

        template = template.replace('<merge_patch_pairs>', '')

        # export_objects([Block.instances[15].fc_solid, Block.instances[1148].fc_solid], '/tmp/error_blocks.FCStd')

        return template

    def write_block_mesh_dict(self):
        with open(os.path.join(self.case_dir, 'system', "blockMeshDict"), "w") as f:
            f.write(self.block_mesh_dict)


# def get_position(vertex: FCPart.Vertex):
#     return np.array([vertex.X, vertex.Y, vertex.Z])


def unit_vector(vec):
    return vec / np.linalg.norm(vec)


def create_o_grid_blocks(edge,
                         reference_face,
                         n_cell=10,
                         outer_pipe: bool = True,
                         inlet: bool = False,
                         outlet: bool = False) -> list[Block]:
    """
    Generate o-grid for a tube extruded along a edge
    :param edge: edge along which the block is extruded
    :param reference_face:  reference face
    :param n_cell: number of cells for a quarter circle
    :param outer_pipe: generate blocks around the pipe (for a rectangular section)
    :param inlet: first layer pipe block face is inlet
    :param outlet: second layer pipe block face is outlet
    :return: generated blocks
    """
    start_point = get_position(edge.Vertexes[0])
    end_point = get_position(edge.Vertexes[1])
    dist = reference_face.tube_diameter / 4
    face_normal = vector_to_np_array(reference_face.normal)
    direction = vector_to_np_array(edge.tangentAt(edge.FirstParameter))
    perp_vec = perpendicular_vector(face_normal, direction)
    direction2 = vector_to_np_array(edge.tangentAt(edge.LastParameter))
    perp_vec2 = perpendicular_vector(face_normal, direction2)

    if type(edge.Curve) is FCPart.Line:

        # create vertices
        layer1_vertices = create_layer_vertices(start_point, face_normal, perp_vec, dist, outer_pipe=outer_pipe)
        layer2_vertices = [x + (end_point - start_point) for x in layer1_vertices]

        # create edges
        # -------------------------------------------------------------------------------------------------------------
        layer1_edges = create_layer_edges(layer1_vertices,
                                          start_point,
                                          face_normal,
                                          reference_face,
                                          perp_vec,
                                          outer_pipe=outer_pipe)

        layer2_edges = create_layer_edges(layer2_vertices,
                                          end_point,
                                          face_normal,
                                          reference_face,
                                          perp_vec2,
                                          outer_pipe=outer_pipe)

    else:
        center = np.array(edge.Curve.Center)

        rot_angle = np.rad2deg(DraftVecUtils.angle(Base.Vector(start_point - center),
                                                   Base.Vector(end_point - center),
                                                   Base.Vector(face_normal)))

        layer1_vertices = create_layer_vertices(start_point, face_normal, perp_vec, dist, outer_pipe=outer_pipe)

        vertex_wire = FCPart.Wire(
            [FCPart.makeLine(Base.Vector(layer1_vertices[i].position),
                             Base.Vector(layer1_vertices[i+1].position)) for i in range(layer1_vertices.__len__() - 1)]
        )

        layer2_vertices = [BlockMeshVertex(position=np.array(x.Point)) for x in vertex_wire.rotate(edge.Curve.Center,
                                                                                                   reference_face.normal,
                                                                                                   rot_angle).Vertexes]

        layer1_edges = create_layer_edges(layer1_vertices,
                                          start_point,
                                          face_normal,
                                          reference_face,
                                          perp_vec,
                                          outer_pipe=outer_pipe)

        layer2_edges = create_layer_edges(layer2_vertices,
                                          end_point,
                                          face_normal,
                                          reference_face,
                                          perp_vec2,
                                          outer_pipe=outer_pipe)

    # create blocks:
    # -------------------------------------------------------------------------------------------------------------
    blocks = create_blocks(layer1_vertices,
                           layer2_vertices,
                           layer1_edges,
                           layer2_edges,
                           face_normal,
                           [n_cell, n_cell, int(np.ceil(edge.Length / 50))],
                           edge,
                           outer_pipe=outer_pipe)

    if inlet:
        for i in [0, 1, 2, 3, 4]:
            blocks[i].faces[0].boundary = inlet_patch
    if outlet:
        for i in [0, 1, 2, 3, 4]:
            blocks[i].faces[1].boundary = outlet_patch

    return blocks


def create_layer_vertices(start_point, face_normal, perp_vec, dist, outer_pipe=True):

    logger.debug('Creating layer vertices')

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
    if outer_pipe:
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
    else:
        return [p0, p1, p2, p3, p4, p5, p6, p7]


def create_layer_edges(layer_vertices, center_point, face_normal, reference_face, perp_vec, outer_pipe=True):

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
    if outer_pipe:
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

        return [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22,
                e23, e24, e25, e26, e27, e28, e29, e30, e31]

    else:
        return [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]


def create_edges_between_layers(p1, p2, edge, face_normal):

    try:
        if type(edge.Curve) is FCPart.Line:
            return [BlockMeshEdge(vertices=[p1[i], p2[i]], type='line') for i in range(4)]
        else:
            edges = [None] * 4
            for i in range(4):
                # move center point
                center = np.array(edge.Curve.Center) + (p1[i].position - edge.Vertexes[0].Point) * face_normal

                v1 = p1[i].position - Base.Vector(center)
                v2 = p2[i].position - Base.Vector(center)

                d_vec = unit_vector(v1 + v2)
                radius = np.linalg.norm(p1[i].position - center)

                mid_point = center + radius * d_vec

                # mid_point = unit_vector(p1[i].position - edge.Curve.Center + p2[i].position - edge.Curve.Center) * \
                #             np.linalg.norm(p1[i].position - edge.Curve.Center) + edge.Curve.Center

                new_edge = BlockMeshEdge(vertices=[p1[i], p2[i]],
                                         type='arc',
                                         interpolation_points=[mid_point])
                edges[i] = new_edge
            return edges
    except Exception as e:
        raise e


def create_blocks(layer1_vertices,
                  layer2_vertices,
                  layer1_edges,
                  layer2_edges,
                  face_normal,
                  num_cells,
                  edge,
                  outer_pipe=True):

    # 'inlet': (0, 1, 2, 3),  # 0
    # 'outlet': (4, 5, 6, 7),  # 1
    # 'left': (4, 0, 3, 7),  # 2
    # 'right': (5, 1, 2, 6),  # 3
    # 'top': (4, 5, 1, 0),  # 4
    # 'bottom': (7, 6, 2, 3)  # 5

    vertex_indices = [[0, 1, 2, 3],
                      [0, 4, 5, 1],
                      [1, 5, 6, 2],
                      [2, 6, 7, 3],
                      [3, 7, 4, 0],
                      ]

    edge_indices = [[0, 1, 2, 3],
                    [4, 8, 5, 0],
                    [5, 9, 6, 1],
                    [6, 10, 7, 2],
                    [7, 11, 4, 3],
                    ]

    block_names = [f'Center Block edge {edge}',
                   f'Pipe Block 1, edge {edge}',
                   f'Pipe Block 2, edge {edge}',
                   f'Pipe Block 3, edge {edge}',
                   f'Pipe Block 4, edge {edge}',
                   ]

    cell_zones = ['pipe', 'pipe', 'pipe', 'pipe', 'pipe']

    blocks = []

    for i in range(5):
        vertices = [*[layer1_vertices[x] for x in vertex_indices[i]], *[layer2_vertices[x] for x in vertex_indices[i]]]
        connect_edges = create_edges_between_layers(vertices[0:4], vertices[4:], edge, face_normal)
        block_edges = [*[layer1_edges[x] for x in edge_indices[i]], *[layer2_edges[x] for x in edge_indices[i]], *connect_edges]
        new_block = Block(name=block_names[i],
                          vertices=vertices,
                          edge=edge,
                          block_edges=block_edges,
                          num_cells=num_cells,
                          cell_zone=cell_zones[i],
                          extruded=True,
                          check_merge_patch_pairs=False)

        blocks.append(new_block)

    blocks[1].faces[5].boundary = pipe_wall_patch
    blocks[2].faces[3].boundary = pipe_wall_patch
    blocks[3].faces[4].boundary = pipe_wall_patch
    blocks[4].faces[2].boundary = pipe_wall_patch

    if outer_pipe:

        outer_vertex_indices = [[4, 19, 8, 9],
                                [4, 9, 10, 5],
                                [5, 10, 11, 12],
                                [5, 12, 13, 6],
                                [6, 13, 14, 15],
                                [6, 15, 16, 7],
                                [7, 16, 17, 18],
                                [7, 18, 19, 4]]

        outer_edge_indices = [[12, 31, 20, 13],
                              [13, 21, 14, 8],
                              [14, 22, 23, 15],
                              [15, 24, 16, 9],
                              [16, 25, 26, 17],
                              [17, 27, 18, 10],
                              [18, 28, 29, 19],
                              [19, 30, 12, 11]]

        outer_block_names = [f'Outer Block 5, edge {edge}',
                             f'Outer Block 6, edge {edge}',
                             f'Outer Block 7, edge {edge}',
                             f'Outer Block 8, edge {edge}',
                             f'Outer Block 9, edge {edge}',
                             f'Outer Block 10, edge {edge}',
                             f'Outer Block 11, edge {edge}',
                             f'Outer Block 12, edge {edge}']

        outer_cell_zones = [None, None, None, None, None, None, None, None]

        for i in range(outer_vertex_indices.__len__()):
            vertices = [*[layer1_vertices[x] for x in outer_vertex_indices[i]],
                        *[layer2_vertices[x] for x in outer_vertex_indices[i]]]
            connect_edges = create_edges_between_layers(vertices[0:4], vertices[4:], edge, face_normal)
            block_edges = [*[layer1_edges[x] for x in outer_edge_indices[i]], *[layer2_edges[x] for x in outer_edge_indices[i]],
                           *connect_edges]
            new_block = Block(name=outer_block_names[i],
                              vertices=vertices,
                              edge=edge,
                              block_edges=block_edges,
                              num_cells=num_cells,
                              cell_zone=outer_cell_zones[i],
                              extruded=True,
                              check_merge_patch_pairs=True
                              )
            # try:
            #     print(f'Block {new_block.id} volume: {new_block.fc_box.Volume}')
            # except Exception as e:
            #     logger.error(f'Block error')
            blocks.append(new_block)

    return blocks


def create_box_from_points(block, vertices=None, edge=None, block_edges=None):

    vertices = block.vertices
    edge = block.edge
    block_edges = block.block_edges

    # points = np.array([x.position for x in vertices])
    # dir_edge_wire = FCPart.Wire(edge)

    solid = None

    face_map = {
        'inlet': (0, 1, 2, 3),  # 0
        'outlet': (4, 5, 6, 7),  # 1
        'left': (4, 0, 3, 7),  # 2
        'right': (5, 1, 2, 6),  # 3
        'top': (4, 5, 1, 0),  # 4
        'bottom': (7, 6, 2, 3)  # 5
    }

    faces = []
    for key, loc_vertices in face_map.items():
        # face_wire = FCPart.makePolygon([Base.Vector(row) for row in points[value, :]])
        # create edges:
        face = None

        e0 = BlockMeshEdge.get_edge(vertex_ids=tuple([vertices[loc_vertices[0]].id, vertices[loc_vertices[1]].id]), create=True)
        e1 = BlockMeshEdge.get_edge(vertex_ids=tuple([vertices[loc_vertices[1]].id, vertices[loc_vertices[2]].id]), create=True)
        e2 = BlockMeshEdge.get_edge(vertex_ids=tuple([vertices[loc_vertices[2]].id, vertices[loc_vertices[3]].id]), create=True)
        e3 = BlockMeshEdge.get_edge(vertex_ids=tuple([vertices[loc_vertices[3]].id, vertices[loc_vertices[0]].id]), create=True)

        block_edges = [e0, e1, e2, e3]
        edges = [None] * 4
        for i in range(block_edges.__len__()):
            if block_edges[i].type == 'arc':
                edges[i] = FCPart.Edge(FCPart.Arc(Base.Vector(block_edges[i].vertices[0].position),
                                                  Base.Vector(block_edges[i].interpolation_points[0]),
                                                  Base.Vector(block_edges[i].vertices[1].position))
                                       )
            elif block_edges[i].type in ['line', None]:
                edges[i] = FCPart.Edge(FCPart.LineSegment(Base.Vector(block_edges[i].vertices[0].position),
                                                          Base.Vector(block_edges[i].vertices[1].position)))

        face_wire = FCPart.Wire(edges)

        try:
            face = make_complex_face_from_edges(face_wire.OrderedEdges)
        except Exception as e:
            logger.error(f'Error creating geometric face {key} for {block.__repr__()}: {e}')
            save_fcstd(face_wire.OrderedEdges, f'/tmp/fghedges_shape_{block.id}.FCStd')
            save_fcstd([face_wire.OrderedEdges], f'/tmp/error_shape_{block.id}.FCStd')
            save_fcstd(faces, f'/tmp/error_faces_{block.id}.FCStd')
            raise e
        faces.extend(face.Faces)

    try:
        shell = FCPart.makeShell(faces)
        shell.sewShape()
        try:
            shell.fix(1e-5, 1e-5, 1e-5)
        except Exception as e:
            logger.warning(f'Could not fix block shell')
        solid = FCPart.Solid(shell)
    except Exception as e:
        logger.error(f'Error creating geometric volume for {block.__repr__()}: {e}')
        save_fcstd(faces, f'/tmp/error_shape_{block.id}.FCStd')
        raise e

    if solid.Volume < 1e-3:
        logger.warning(f'{block.__repr__()} volume very small: {solid.Volume}. Please check this block')

    return solid

    # if all([True if x.type == 'line' else False for x in block_edges]):
    #     w0 = FCPart.makePolygon([Base.Vector(row) for row in points[[0, 1, 2, 3, 0], :]])
    # else:
    #     edges = [None] * 4
    #     for i in range(4):
    #         p1 = Base.Vector(points[i, :])
    #         if i <=2:
    #             p2 = Base.Vector(points[i+1, :])
    #         else:
    #             p2 = Base.Vector(points[0, :])
    #
    #         if block_edges[i].type == 'arc':
    #             edges[i] = FCPart.Edge(FCPart.Arc(p1, Base.Vector(block_edges[i].interpolation_points[0]), p2))
    #         elif block_edges[i].type == 'line':
    #             edges[i] = FCPart.Edge(FCPart.LineSegment(p1, p2))
    #
    #     w0 = FCPart.Wire(edges)
    #
    # base_face = FCPart.Face(w0)
    # return dir_edge_wire.makePipe(base_face)


def angle_between_vertices(p1, p2, p3, deg=True):
    """
    Calculate the angle between three vertices

        p1  * ------------ * p2
            |           |
            |    ALPHA  |
            | ----------|
            |
        p3  *


    :param p1:  coordinates vertex 1 (center)
    :param p2:  coordinates vertex 2
    :param p3:  coordinates vertex 3
    """

    v1 = p2 - p1
    v2 = p3 - p1

    u_v1 = v1 / np.linalg.norm(v1)
    u_v2 = v2 / np.linalg.norm(v2)

    angle = np.arccos(np.dot(u_v1, u_v2))

    if deg:
        return np.rad2deg(angle)
    else:
        return angle


def create_blocks_from_2d_mesh(meshes, reference_face):

    blocks = []

    # calculate vector which translates vertices to lower points of the pipe-block-section:
    trans_base_vec = np.array(reference_face.normal * (reference_face.tube_diameter / 2 / np.sqrt(2) + reference_face.tube_diameter / 4 ))

    for mesh in meshes:
        # create_vertices:
        mesh.points = mesh.points - trans_base_vec
        vertices = np.array([BlockMeshVertex(position=x) for x in mesh.points])

        # create quad blocks:
        if 'quad' in mesh.cells_dict.keys():
            for quad in mesh.cells_dict['quad']:

                v_1 = vertices[quad]
                v_2 = np.array([x + 2 * trans_base_vec for x in v_1])
                new_block = Block(vertices=[*v_1, *v_2],
                                  name=f'Free Block',
                                  auto_cell_size=True,
                                  extruded=False)
                blocks.append(new_block)

        if 'triangle' in mesh.cells_dict.keys():
            for tri in mesh.cells_dict['triangle']:
                v_1 = vertices[[*tri, tri[0]]]
                v_2 = np.array([x + 2 * trans_base_vec for x in v_1])

                new_block = Block(vertices=[*v_1, *v_2],
                                  name=f'Free Block',
                                  auto_cell_size=True,
                                  extruded=False,
                                  non_regular=True)

                blocks.append(new_block)
    return blocks


def save_fcstd(objects: list, filename: str):
    """
    save as freecad document
    :param filename: full filename; example: '/tmp/test.FCStd'
    :param objects: 'solid', 'faces'
    """
    doc = App.newDocument(f"Blocks")
    for obj in objects:
        __o__ = doc.addObject("Part::Feature", f'obj {obj}')
        __o__.Shape = obj
    doc.recompute()
    doc.saveCopy(filename)


def make_complex_face_from_edges(edges):
    # https://forum.freecadweb.org/viewtopic.php?t=21636

    edge_types = [type(x.Curve) for x in edges]
    # all_lines = all([x is FCPart.Line for x in edge_types])

    try:
        w0 = FCPart.Wire(edges)
        return FCPart.Face(w0)
    except:
        if type(edges[0].Curve) is type(edges[2].Curve):
            return FCPart.Wire(FCPart.Wire(edges[0])).makePipeShell([FCPart.Wire(edges[1]), FCPart.Wire(edges[3])],
                                                                    False,
                                                                    True)
        else:
            return FCPart.makeFilledFace(FCPart.Wire(edges).OrderedEdges)

    # ps = FCPart.BRepOffsetAPI.MakePipeShell(FCPart.Wire(edges[0]))
    # ps.setFrenetMode(True)
    # ps.setSpineSupport(FCPart.Wire(edges[0]))
    # ps.setAuxiliarySpine(FCPart.Wire(edges[2]), True, False)
    # ps.add(FCPart.Wire(edges[1]), True, True)
    # ps.add(FCPart.Wire(edges[3]), True, True)
    # if ps.isReady():
    #     ps.build()
    # return ps.shape()

import copy
import uuid
import itertools
import numpy as np
from functools import lru_cache, wraps
from ..logger import logger

import FreeCAD
import Part as FCPart
import Draft
from FreeCAD import Base
import DraftVecUtils
import PartDesign

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

    instances = {}

    @staticmethod
    # @np_cache
    def get_vertex(position):
        logger.debug('Getting vertex...')
        # return next((x for x in VertexMetaMock.instances if np.allclose(x.position, position, atol=1e-3)), None)
        vert = VertexMetaMock.instances.get(tuple(position), None)
        if vert is not None:
            logger.debug('Vertex already existing')
        return vert

    def __call__(cls, *args, **kwargs):
        position = kwargs.get('position', np.array([0, 0, 0]))
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
        logger.debug('Getting edge...')
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
        if edge is not None:
            logger.debug('Edge already existing')
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


class FaceMetaMock(type):

    instances = {}

    @staticmethod
    def get_face(vertices):
        logger.debug('Getting face...')
        # raise NotImplementedError
        # return next((x for x in EdgeMetaMock.instances if np.array_equal(x.vertices, vertices)), None)
        face = FaceMetaMock.instances.get(tuple(sorted(vertices)), None)
        if face is not None:
            logger.debug('Face already existing')
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
            [solids.extend(x.fc_colids.Solids) for x in self.instances]
            self._comp_solid = FCPart.CompSolid(solids)
        return self._comp_solid

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        cls._comp_solid = None
        return obj


class BlockMeshVertex(object, metaclass=VertexMetaMock):
    id_iter = itertools.count()

    @classmethod
    def block_mesh_entry(cls):
        return ''

    def __init__(self, *args, **kwargs):
        self.id: int = next(BlockMeshVertex.id_iter)
        self.position = kwargs.get('position', np.array([0, 0, 0]))
        self._fc_vertex = None

    def __add__(self, vec):
        return BlockMeshVertex(position=self.position + vec)

    def __sub__(self, vec):
        return BlockMeshVertex(position=self.position - vec)

    def __repr__(self):
        return f'Vertex {self.id} (position={self.position[0], self.position[1], self.position[2]})'

    @property
    def fc_vertex(self):
        if self._fc_vertex is None:
            self._fc_vertex = FCPart.Point(Base.Vector(self.position[0], self.position[1], self.position[2]))
        return self._fc_vertex

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
        self.center = kwargs.get('center', None)
        self.interpolation_points = kwargs.get('interpolation_points', None)

        self._fc_edge = None

    @property
    def fc_edge(self):
        if self._fc_edge is None:
            self._fc_edge = self.create_fc_edge()
        return self._fc_edge

    def create_fc_edge(self):
        if self.type == 'arc':
            return FCPart.Edge(FCPart.Arc(Base.Vector(self.vertices[0].position),
                                          Base.Vector(self.interpolation_points[0]),
                                          Base.Vector(self.vertices[1].position))
                               )
        elif self.type in ['line', None]:
            return FCPart.Edge(FCPart.LineSegment(Base.Vector(self.vertices[0].position),
                                                  Base.Vector(self.vertices[1].position)))

    def __repr__(self):
        return f'Edge {self.id} (type={self.type}, v1={self.vertices[0].id}, v2={self.vertices[1].id} interpolation_points={self.interpolation_points})'


class BlockMeshBoundary(object, metaclass=BoundaryMetaMock):

    id_iter = itertools.count()

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
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, value):
        self._faces = value
        [x.set_boundary(self) for x in self._faces]

    def __repr__(self):
        return f'Boundary {self.id} (name={self.name}, type={self.type}, faces={self.faces})'


inlet_patch = BlockMeshBoundary(name='inlet', type='patch')
outlet_patch = BlockMeshBoundary(name='outlet', type='patch')
wall_patch = BlockMeshBoundary(name='wall', type='wall')


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

        self._boundary = kwargs.get('boundary', None)
        self._fc_face = None

        self._edge0 = None
        self._edge1 = None
        self._edge2 = None
        self._edge3 = None

        self.extruded = kwargs.get('extruded', False)   # false or: [base_profile, top_profile, path1, path2]

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
        return self._edge0

    @property
    def edge1(self):
        if self._edge1 is None:
            self._edge1 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[1].id,
                                                                          self.vertices[2].id])),
                                                 create=True)
        return self._edge1

    @property
    def edge2(self):
        if self._edge2 is None:
            self._edge2 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[2].id,
                                                                          self.vertices[3].id])),
                                                 create=True)
        return self._edge2

    @property
    def edge3(self):
        if self._edge3 is None:
            self._edge3 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[3].id,
                                                                          self.vertices[0].id])),
                                                 create=True)
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

        if isinstance(self.extruded, list):
            # https://forum.freecadweb.org/viewtopic.php?t=21636

            save_fcstd([x.fc_edge for x in self.extruded], f'/tmp/extrude_edges_face{self.id}')

            ex_wire = FCPart.Wire([x.fc_edge for x in self.extruded])

            save_fcstd([ex_wire], f'/tmp/extrude_wire{self.id}')

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
            edges = [x.fc_edge for x in self.edges]
            face_wire = FCPart.Wire(edges)
            face = make_complex_face_from_edges(face_wire.OrderedEdges)

        if face.Area < 1e-3:
            logger.warning(f'Face {self.id} area very small: {face.Area}')

        return face

    def __repr__(self):
        return f'Face {self.id} (type={self.boundary}, Vertices={self.vertices})'

    def __eq__(self, other):
        return sorted(self.vertices) == sorted(other.vertices)

    def __hash__(self):
        return id(self)


class Block(object, metaclass=BlockMetaMock):

    face_map = {
        'inlet': (0, 1, 2, 3),      # 0
        'outlet': (4, 5, 6, 7),     # 1
        'left': (4, 0, 3, 7),       # 2
        'right': (5, 1, 2, 6),      # 3
        'top': (4, 5, 1, 0),        # 4
        'bottom': (7, 6, 2, 3)      # 5
    }

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
        for i, block in enumerate(cls.instances):
            try:
                __o__ = doc.addObject("Part::Feature", f'Block {block.name} {block.id}')
                __o__.Shape = block.fc_solid
            except Exception as e:
                logger.error(f'Error saving Block {i} {block.id} {block.name} {block}:\n{e}')
                raise e
        doc.recompute()
        doc.saveCopy(filename)

    def __init__(self, *args, **kwargs):
        """
        create a block from vertices

        layer1:


        layer2:


        :param args:
        :param kwargs:
        """

        self._num_cells = None
        self._block_edges = None
        self._fc_solid = None
        self._faces = None
        self._edge = None

        self.name = kwargs.get('name', 'unnamed_block')
        self.id = next(Block.id_iter)
        self.vertices = kwargs.get('vertices', [])
        self.assigned_feature = kwargs.get('assigned_feature', None)
        self.edge = kwargs.get('edge', None)
        self.num_cells = kwargs.get('num_cells', None)
        self.block_edges = kwargs.get('block_edges', None)
        self.cell_zone = kwargs.get('cell_zone', None)
        self.auto_cell_size = kwargs.get('auto_cell_size', True)
        self.cell_size = kwargs.get('cell_size', 100)

        self.extruded = kwargs.get('extruded', False)

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

    # @property
    # def center_line(self):
    #     return self.faces[0].C
    @property
    def face0(self):
        if self._face0 is None:
            self._face0 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['inlet']])
        return self._face0

    @property
    def face1(self):
        if self._face1 is None:
            self._face1 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['outlet']])
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
        return self._edge0

    @property
    def edge1(self):
        if self._edge1 is None:
            v_map = self._edge_vertex_map[1]
            self._edge1 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge1

    @property
    def edge2(self):
        if self._edge2 is None:
            v_map = self._edge_vertex_map[2]
            self._edge2 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge2

    @property
    def edge3(self):
        if self._edge3 is None:
            v_map = self._edge_vertex_map[3]
            self._edge3 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge3

    @property
    def edge4(self):
        if self._edge4 is None:
            v_map = self._edge_vertex_map[4]
            self._edge4 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge4

    @property
    def edge5(self):
        if self._edge5 is None:
            v_map = self._edge_vertex_map[5]
            self._edge5 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge5

    @property
    def edge6(self):
        if self._edge6 is None:
            v_map = self._edge_vertex_map[6]
            self._edge6 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge6

    @property
    def edge7(self):
        if self._edge7 is None:
            v_map = self._edge_vertex_map[7]
            self._edge7 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge7

    @property
    def edge8(self):
        if self._edge8 is None:
            v_map = self._edge_vertex_map[8]
            self._edge8 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge8

    @property
    def edge9(self):
        if self._edge9 is None:
            v_map = self._edge_vertex_map[9]
            self._edge9 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
        return self._edge9

    @property
    def edge10(self):
        if self._edge10 is None:
            v_map = self._edge_vertex_map[10]
            self._edge10 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                           self.vertices[v_map[1]].id])),
                                                  create=True)
        return self._edge10

    @property
    def edge11(self):
        if self._edge11 is None:
            v_map = self._edge_vertex_map[11]
            self._edge11 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                           self.vertices[v_map[1]].id])),
                                                  create=True)
        return self._edge11

    @property
    def block_edges(self):
        return [self.edge0, self.edge1, self.edge2,
                self.edge3, self.edge4, self.edge5,
                self.edge6, self.edge7, self.edge8,
                self.edge9, self.edge10, self.edge11]

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
            self._fc_solid = self.create_fc_solid()
        return self._fc_solid

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
        _ = [faces.extend(face.fc_face.Faces) for face in self.faces]

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
        l1 = self.vertices[1].dist_to_point(self.vertices[0])
        l2 = self.vertices[3].dist_to_point(self.vertices[0])
        l3 = self.vertices[4].dist_to_point(self.vertices[0])

        return [l1/self.cell_size, l2/self.cell_size, l3/self.cell_size]

    # def generate_edge(self):
    #     p1 = Base.Vector(self.vertices[0].position + 0.5 * (self.vertices[2].position - self.vertices[0].position))
    #     p2 = Base.Vector(self.vertices[4].position + 0.5 * (self.vertices[6].position - self.vertices[4].position))
    #     return FCPart.Edge(FCPart.LineSegment(p1, p2))

    def __repr__(self):
        return f'Block {self.id} ({self.name})'


def get_position(vertex: FCPart.Vertex):
    return np.array([vertex.X, vertex.Y, vertex.Z])


def vector_to_np_array(vector):
    return np.array([vector.x, vector.y, vector.z])


def perpendicular_vector(x, y):
    return np.cross(x, y)


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
    face_normal = vector_to_np_array(reference_face.normal)
    direction = vector_to_np_array(edge.tangentAt(edge.FirstParameter))
    perp_vec = perpendicular_vector(face_normal, direction)
    direction2 = vector_to_np_array(edge.tangentAt(edge.LastParameter))
    perp_vec2 = perpendicular_vector(face_normal, direction2)

    if type(edge.Curve) is FCPart.Line:
        direction = vector_to_np_array(edge.Curve.Direction)

        # create vertices
        # -------------------------------------------------------------------------------------------------------------
        # center block
        start_point = get_position(edge.Vertexes[0])
        end_point = get_position(edge.Vertexes[1])
        dist = reference_face.tube_diameter/4

        layer1_vertices = create_layer_vertices(start_point, face_normal, perp_vec, dist, outer_pipe=outer_pipe)
        layer2_vertices = [x + (end_point - start_point) for x in layer1_vertices]

        # create edges
        # -------------------------------------------------------------------------------------------------------------
        i = 0
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

    else:
        start_point = get_position(edge.Vertexes[0])
        end_point = get_position(edge.Vertexes[1])
        center = np.array(edge.Curve.Center)
        # perp_vec = perpendicular_vector(face_normal, direction)
        dist = reference_face.tube_diameter / 4

        # dir1 = edge.tangentAt(edge.FirstParameter)
        # dir2 = edge.tangentAt(edge.LastParameter)
        # ctr = FCPart.calcCentroid(dir1, dir2)
        #
        # dv = dir2.sub(dir1)
        # rot_placement = FreeCAD.Placement(edge.Curve.Center,
        #                                   FreeCAD.Rotation(FreeCAD.Vector(face_normal[0],
        #                                                                   face_normal[1],
        #                                                                   face_normal[2]),
        #                                                    dv)
        #                                   )

        # rot_angle = angle_between_vertices(edge.Curve.Center,
        #                                    get_position(edge.Vertexes[0]),
        #                                    get_position(edge.Vertexes[1]))

        rot_angle = np.rad2deg(DraftVecUtils.angle(Base.Vector(start_point - center),
                                        Base.Vector(end_point - center),
                                        Base.Vector(face_normal)))

        layer1_vertices = create_layer_vertices(start_point, face_normal, perp_vec, dist, outer_pipe=outer_pipe)

        vertex_wire = FCPart.Wire(
            [FCPart.makeLine(Base.Vector(layer1_vertices[i].position),
                             Base.Vector(layer1_vertices[i+1].position)) for i in range(layer1_vertices.__len__() - 1)]
        )

        # vertex_wire.Placement = rot_placement
        # layer2_vertices = [BlockMeshVertex(position=np.array(x.Point)) for x in vertex_wire.Vertexes]

        # [x.Point for x in vertex_wire.rotate(edge.Curve.Center, reference_face.normal,  rot_angle).Vertexes]
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
                                          perp_vec2)

        blocks = create_blocks(layer1_vertices,
                               layer2_vertices,
                               layer1_edges,
                               layer2_edges,
                               face_normal,
                               [n_cell, n_cell, int(np.ceil(edge.Length / 50))],
                               edge,
                               outer_pipe=outer_pipe)

    if inlet:
        for i in [0, 1, 2, 3]:
            blocks[i].faces[0].boundary = inlet_patch
    if outlet:
        for i in [0, 1, 2, 3]:
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

    if type(edge.Curve) is FCPart.Line:
        return [BlockMeshEdge(vertices=[p1[i], p2[i]], type='line') for i in range(4)]
    else:
        edges = [None] * 4
        for i in range(4):
            # move center point
            center = np.array(edge.Curve.Center) + (p1[i].position - edge.Vertexes[0].Point) * face_normal

            v1 = p1[i].position - center
            v2 = p2[i].position - center

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
                          extruded=True)
        # try:
        #     print(f'Block {new_block.id} volume: {new_block.fc_box.Volume}')
        # except Exception as e:
        #     logger.error(f'Block error')
        #     # raise e
        blocks.append(new_block)

    blocks[1].faces[5].boundary = wall_patch
    blocks[2].faces[3].boundary = wall_patch
    blocks[3].faces[4].boundary = wall_patch
    blocks[4].faces[2].boundary = wall_patch

    # index = [0, 1, 2, 3]
    # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    # edge_index = [0, 1, 2, 3]
    # connect_edges = create_edges_between_layers(vertices[0:4], vertices[4:], edge)
    # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index], *connect_edges]
    # b0 = Block(name=f'Center Block edge {edge}',
    #            vertices=vertices,
    #            edge=edge,
    #            block_edges=block_edges,
    #            num_cells=num_cells,
    #            cell_zone='pipe')
    #
    # index = [0, 4, 5, 1]
    # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    # edge_index = [4, 8, 5, 0]
    # block_edges = [*[layer1_edges[x] for x in edge_index],
    #                *[layer2_edges[x] for x in edge_index],
    #                *create_edges_between_layers(vertices[0:4], vertices[4:], edge)]
    # b1 = Block(name=f'Pipe Block 1, edge {edge}',
    #            vertices=vertices,
    #            edge=edge,
    #            block_edges=block_edges,
    #            num_cells=num_cells,
    #            cell_zone='pipe')
    # b1.faces[5].boundary = wall_patch

    # index = [1, 5, 6, 2]
    # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    # edge_index = [5, 9, 6, 1]
    # block_edges = [*[layer1_edges[x] for x in edge_index],
    #                *[layer2_edges[x] for x in edge_index],
    #                *create_edges_between_layers(vertices[0:4], vertices[4:], edge)]
    # b2 = Block(name=f'Pipe Block 2, edge {edge}',
    #            vertices=vertices,
    #            edge=edge,
    #            block_edges=block_edges,
    #            num_cells=num_cells,
    #            cell_zone='pipe')
    # b2.faces[3].boundary = wall_patch

    # index = [2, 6, 7, 3]
    # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    # edge_index = [6, 10, 7, 2]
    # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    # b3 = Block(name=f'Pipe Block 3, edge {edge}',
    #            vertices=vertices,
    #            edge=edge,
    #            block_edges=block_edges,
    #            num_cells=num_cells,
    #            cell_zone='pipe')
    # b3.faces[4].boundary = wall_patch

    # index = [3, 7, 4, 0]
    # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
    # edge_index = [7, 11, 4, 3]
    # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
    # b4 = Block(name=f'Pipe Block 4, edge {edge}',
    #            vertices=vertices,
    #            edge=edge,
    #            block_edges=block_edges,
    #            num_cells=num_cells,
    #            cell_zone='pipe')
    # b4.faces[2].boundary = wall_patch

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
                              extruded=True)
            # try:
            #     print(f'Block {new_block.id} volume: {new_block.fc_box.Volume}')
            # except Exception as e:
            #     logger.error(f'Block error')
            blocks.append(new_block)

        # index = [4, 19, 8, 9]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [12, 31, 20, 13]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b5 = Block(name=f'Outer Block 5, edge {edge}',
        #            vertices=vertices,
        #            edge=edge,
        #            block_edges=block_edges,
        #            num_cells=num_cells)

        # index = [4, 9, 10, 5]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [13, 21, 14, 8]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b6 = Block(name=f'Outer Block 6, edge {edge}',
        #            vertices=vertices,
        #            edge=edge,
        #            block_edges=block_edges,
        #            num_cells=num_cells)

        # index = [5, 10, 11, 12]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [14, 22, 23, 15]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b7 = Block(name=f'Outer Block 7, edge {edge}',
        #            vertices=vertices,
        #            edge=edge,
        #            block_edges=block_edges,
        #            num_cells=num_cells)

        # index = [5, 12, 13, 6]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [15, 24, 16, 9]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b8 = Block(name=f'Outer Block 8, edge {edge}',
        #            vertices=vertices,
        #            edge=edge,
        #            block_edges=block_edges,
        #            num_cells=num_cells)

        # index = [6, 13, 14, 15]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [16, 25, 26, 17]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b9 = Block(name=f'Outer Block 9, edge {edge}',
        #            vertices=vertices,
        #            edge=edge,
        #            block_edges=block_edges,
        #            num_cells=num_cells)

        # index = [6, 15, 16, 7]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [17, 27, 18, 10]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b10 = Block(name=f'Outer Block 10, edge {edge}',
        #             vertices=vertices,
        #             edge=edge,
        #             block_edges=block_edges,
        #             num_cells=num_cells)

        # index = [7, 16, 17, 18]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [18, 28, 29, 19]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b11 = Block(name=f'Outer Block 11, edge {edge}',
        #             vertices=vertices,
        #             edge=edge,
        #             block_edges=block_edges,
        #             num_cells=num_cells)

        # index = [7, 18, 19, 4]
        # vertices = [*[layer1_vertices[x] for x in index], *[layer2_vertices[x] for x in index]]
        # edge_index = [19, 30, 12, 11]
        # block_edges = [*[layer1_edges[x] for x in edge_index], *[layer2_edges[x] for x in edge_index]]
        # b12 = Block(name=f'Outer Block 12, edge {edge}',
        #             vertices=vertices,
        #             edge=edge,
        #             block_edges=block_edges,
        #             num_cells=num_cells)

        # return [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12]

    # else:
    #     return [b0, b1, b2, b3, b4]

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

        # try:
        #     face_wire = FCPart.Wire(edges)
        #     face = FCPart.Face(face_wire)
        # except Exception as e:
        #     try:
        #         ace = make_complex_face_from_edges(edges)
        #         face = make_complex_face_from_edges(face_wire.OrderedEdges)
        #         # face = FCPart.makeFilledFace(face_wire.OrderedEdges)
        #     except Exception as e:
        #         logger.error(f'Error creating geometric face {key} for {block.__repr__()}: {e}')
        #         save_fcstd(face_wire.OrderedEdges, f'/tmp/fghedges_shape_{block.id}.FCStd')
        #         save_fcstd([face_wire.OrderedEdges], f'/tmp/error_shape_{block.id}.FCStd')
        #         save_fcstd(faces, f'/tmp/error_faces_{block.id}.FCStd')
        #         raise e
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
        blocks = []
        for quad in mesh.cells_dict['quad']:

            v_1 = vertices[quad]
            v_2 = np.array([x + 2 * trans_base_vec for x in v_1])
            new_block = Block(vertices=[*v_1, *v_2],
                              name=f'Free Block',
                              auto_cell_size=True)
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
    all_lines = all([x is FCPart.Line for x in edge_types])

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

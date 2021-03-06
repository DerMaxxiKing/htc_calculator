import copy
import posix
import re
from copy import deepcopy
from collections.abc import Iterable
import subprocess
import functools
import operator
import os.path
import uuid
import itertools
import numpy as np
from re import findall, MULTILINE
from functools import lru_cache, wraps
from ..config import use_ssh, work_dir
if use_ssh:
    from ..ssh import shell_handler
from ..logger import logger
from ..tools import vector_to_np_array, perpendicular_vector, export_objects, angle_between_vectors, array_row_intersection
from ..geo_tools import search_contacts, surfaces_in_contact, get_position
from ..construction import Solid, Fluid
from ..case.boundary_conditions import *
from ..case.boundary_conditions.user_bcs import *
from ..case.function_objects import WallHeatFlux, PressureDifferencePatch, TemperatureDifferencePatch, FOMetaMock
from ..face import Face
# import trimesh

import FreeCAD
import Part as FCPart
# import Draft
from FreeCAD import Base
import DraftVecUtils
# import PartDesign
# import BOPTools.SplitFeatures

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import meshing_resources as msh_resources
from ..case import case_resources

App = FreeCAD


default_cell_size = 25
default_arc_cell_size = 5


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


class CustomID(object):

    def __next__(self):
        return uuid.uuid4().int


class MeshMetaMock(type):

    instances = dict()

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        # cls.instances.append(obj)
        cls.instances[obj.id] = obj
        return obj


class Mesh(object, metaclass=MeshMetaMock):

    instances = {}

    def __init__(self, *args, **kwargs):

        self.default_cell_size = kwargs.get('default_cell_size', 25)
        self.default_arc_cell_size = kwargs.get('default_arc_cell_size', 10)

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', 'unnamed_mesh')

        self.vertices = kwargs.get('vertices', {})
        self.vertex_ids = kwargs.get('vertex_ids', {})
        self.edges = kwargs.get('edges', {})
        self.edge_ids = kwargs.get('edge_ids', {})
        self.parallel_edges = kwargs.get('parallel_edges', [])
        self.faces = kwargs.get('faces', {})
        self.face_ids = kwargs.get('face_ids', {})
        self.face_pos = kwargs.get('face_pos', {})
        self.patch_pairs = kwargs.get('patch_pairs', {})
        self.boundaries = kwargs.get('boundaries', {})
        self.boundary_ids = kwargs.get('boundary_ids', {})
        self.blocks = kwargs.get('blocks', [])
        self.cell_zones = kwargs.get('cell_zones', [])
        self.cell_zone_ids = kwargs.get('cell_zone_ids', {})
        self.comp_blocks = kwargs.get('comp_blocks', [])
        self.mesh_contacts = kwargs.get('mesh_contacts', {})
        self.function_objects = kwargs.get('function_objects', [])
        self.function_object_ids = kwargs.get('function_object_ids', {})

        self.vertex_id_counter = CustomID()
        self.edge_id_counter = CustomID()
        self.parallel_edge_id_counter = CustomID()
        self.faces_id_counter = CustomID()
        self.patch_pairs_id_counter = CustomID()
        self.boundary_id_counter = CustomID()
        self.block_id_counter = CustomID()
        self.cell_zone_id_counter = CustomID()

        self.dict_vertex_id_counter = itertools.count()
        self.dict_edge_id_counter = itertools.count()
        self.dict_parallel_edge_id_counter = itertools.count()
        self.dict_faces_id_counter = itertools.count()
        self.dict_patch_pairs_id_counter = itertools.count()
        self.dict_boundary_id_counter = itertools.count()
        self.dict_block_id_counter = itertools.count()
        self.dict_cell_zone_id_counter = itertools.count()

    def add_mesh_contact(self, face1, face2):
        if face1.mesh is self:
            if face2.mesh.id not in self.mesh_contacts.keys():
                self.mesh_contacts[face2.mesh.id] = {(face1, face2)}
            else:
                self.mesh_contacts[face2.mesh.id].add((face1, face2))

        elif face2.mesh is self:
            if face2.mesh.id not in self.mesh_contacts.keys():
                self.mesh_contacts[face1.mesh.id] = {(face2, face1)}
            else:
                self.mesh_contacts[face1.mesh.id].add((face2, face1))

    def activate(self):
        VertexMetaMock.current_mesh = self
        EdgeMetaMock.current_mesh = self
        ParallelEdgeSetMetaMock.current_mesh = self
        FaceMetaMock.current_mesh = self
        PatchPairMetaMock.current_mesh = self
        BoundaryMetaMock.current_mesh = self
        BlockMetaMock.current_mesh = self
        CellZoneMetaMock.current_mesh = self
        CompBlockMetaMock.current_mesh = self
        FOMetaMock.current_mesh = self

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'mesh_' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def mesh_contact_dict_entry(self):

        dict_entries = []

        for key, contacts in self.mesh_contacts.items():

            faces_entries = []
            for contact in contacts:
                faces_entries.append(f"\t\t\t({' '.join([x.txt_id for x in contact[0].vertices])})")

            faces_entries = '\n'.join(faces_entries)

            entry = f"\tmesh_{key.hex}\n" \
                    f"\t{'{'}\n" \
                    f"\t\ttype patch;\n" \
                    f"\t\tfaces\n" \
                    f"\t\t(\n" \
                    f"{faces_entries}\n" \
                    f"\t\t);\n" \
                    f"\t{'}'}"
            dict_entries.append(entry)

        return dict_entries

        #
        #     for contact in other_mesh:
        #
        #
        #
        #         dict_entries.append(f"\tcontact_face_{contact[0].txt_id}\n"
        #                             f"\t{'{'}\n"
        #                             f"\t\ttype patch;\n"
        #                             f"\t\tfaces\n"
        #                             f"\t\t(\n"
        #                             f"\t\t\t({' '.join([x.txt_id for x in contact[0].vertices])})\n"
        #                             f"\t\t);\n"
        #                             f"\t{'}'}")
        #
        # return dict_entries

    @classmethod
    def join_meshes(cls, meshes, mesh_name):
        last_activated_mesh = VertexMetaMock.current_mesh
        merged_mesh = cls(name=mesh_name)
        merged_mesh.activate()

        vertex_lookup_dict = dict()
        edge_lookup_dict = dict()
        face_lookup_dict = dict()
        boundary_lookup_dict = dict()
        block_lookup_dict = dict()

        for mesh in meshes:
            new_vertices = BlockMeshVertex.copy_to_mesh(list(mesh.vertex_ids.values()),
                                                        mesh=merged_mesh,
                                                        check_existing=False)

            vertex_lookup_dict.update(dict(zip([x.id for x in mesh.vertices.values()], new_vertices)))

            def copy_edges(edges, v_lookup_dict):
                in_mesh_edges = [None] * edges.__len__()
                for ii, edge in enumerate(edges):
                    init_dict = copy.copy(edge.__dict__)
                    del init_dict['id']
                    del init_dict['dict_id']

                    init_dict['num_cells'] = init_dict['_num_cells']

                    init_dict['vertices'] = [v_lookup_dict[edge.vertices[0].id], v_lookup_dict[edge.vertices[1].id]]
                    init_dict['mesh'] = merged_mesh
                    in_mesh_edge = BlockMeshEdge(**init_dict)
                    in_mesh_edges[ii] = in_mesh_edge

                return in_mesh_edges

            def copy_faces(instances, e_lookup_dict, v_lookup_dict):
                in_mesh_instances = [None] * instances.__len__()
                for ii, instance in enumerate(instances):
                    init_dict = copy.copy(instance.__dict__)
                    delete_keys = ['id', 'dict_id', 'boundary', 'merge', 'merge_patch_pairs', 'contacts']
                    for key in delete_keys:
                        if key in init_dict.keys():
                            del init_dict[key]

                    init_dict['vertices'] = [v_lookup_dict[x.id] for x in instance.vertices]
                    init_dict['_edge0'] = e_lookup_dict[instance.edge0.id]
                    init_dict['_edge1'] = e_lookup_dict[instance.edge1.id]
                    init_dict['_edge2'] = e_lookup_dict[instance.edge2.id]
                    init_dict['_edge3'] = e_lookup_dict[instance.edge3.id]
                    init_dict['mesh'] = merged_mesh

                    in_mesh_instances[ii] = BlockMeshFace(**init_dict)

                return in_mesh_instances

            def copy_blocks(instances, f_lookup_dict, e_lookup_dict, v_lookup_dict):
                in_mesh_instances = [None] * instances.__len__()
                for ii, instance in enumerate(instances):
                    init_dict = copy.copy(instance.__dict__)
                    delete_keys = ['id', 'dict_id', 'patch_pairs', 'merge_patch_pairs']
                    for key in delete_keys:
                        if key in init_dict.keys():
                            del init_dict[key]

                    init_dict['vertices'] = [v_lookup_dict[x.id] for x in instance.vertices]
                    init_dict['num_cells'] = init_dict['_num_cells']

                    init_dict['edge0'] = e_lookup_dict[instance.edge0.id]
                    init_dict['edge1'] = e_lookup_dict[instance.edge1.id]
                    init_dict['edge2'] = e_lookup_dict[instance.edge2.id]
                    init_dict['edge3'] = e_lookup_dict[instance.edge3.id]
                    init_dict['edge4'] = e_lookup_dict[instance.edge4.id]
                    init_dict['edge5'] = e_lookup_dict[instance.edge5.id]
                    init_dict['edge6'] = e_lookup_dict[instance.edge6.id]
                    init_dict['edge7'] = e_lookup_dict[instance.edge7.id]
                    init_dict['edge8'] = e_lookup_dict[instance.edge8.id]
                    init_dict['edge9'] = e_lookup_dict[instance.edge9.id]
                    init_dict['edge10'] = e_lookup_dict[instance.edge10.id]
                    init_dict['edge11'] = e_lookup_dict[instance.edge11.id]

                    init_dict['face0'] = f_lookup_dict[instance.face0.id]
                    init_dict['face1'] = f_lookup_dict[instance.face1.id]
                    init_dict['face2'] = f_lookup_dict[instance.face2.id]
                    init_dict['face3'] = f_lookup_dict[instance.face3.id]
                    init_dict['face4'] = f_lookup_dict[instance.face4.id]
                    init_dict['face5'] = f_lookup_dict[instance.face5.id]

                    init_dict['mesh'] = merged_mesh

                    in_mesh_instances[ii] = Block(**init_dict)

                return in_mesh_instances

            def copy_boundaries(instances, face_lookup_dict):
                in_mesh_instances = [None] * instances.__len__()
                for ii, instance in enumerate(instances):
                    init_dict = copy.copy(instance.__dict__)
                    delete_keys = ['id', 'alt_id', 'dict_id', 'boundary', 'merge', 'merge_patch_pairs', 'contacts']
                    for key in delete_keys:
                        if key in init_dict.keys():
                            del init_dict[key]

                    init_dict['faces'] = set([face_lookup_dict[x.id] for x in instance.faces])
                    init_dict['mesh'] = merged_mesh

                    in_mesh_instances[ii] = BlockMeshBoundary(**init_dict)

                return in_mesh_instances

            new_edges = copy_edges(list(mesh.edges.values()), vertex_lookup_dict)
            edge_lookup_dict.update(dict(zip([x.id for x in mesh.edges.values()], new_edges)))

            new_faces = copy_faces(list(mesh.faces.values()), edge_lookup_dict, vertex_lookup_dict)
            face_lookup_dict.update(dict(zip([x.id for x in mesh.faces.values()], new_faces)))

            new_blocks = copy_blocks(mesh.blocks, face_lookup_dict, edge_lookup_dict, vertex_lookup_dict)
            block_lookup_dict.update(dict(zip([x.id for x in mesh.blocks], new_blocks)))

            new_boundaries = copy_boundaries(list(mesh.boundaries.values()), face_lookup_dict)
            boundary_lookup_dict.update(dict(zip([x.id for x in list(mesh.boundaries.values())], new_boundaries)))

        last_activated_mesh.activate()
        return merged_mesh, vertex_lookup_dict, edge_lookup_dict, face_lookup_dict, block_lookup_dict, \
               boundary_lookup_dict

    def add_mesh_copy(self, mesh):
        last_activated_mesh = VertexMetaMock.current_mesh
        self.activate()
        vertex_lookup_dict = dict()
        edge_lookup_dict = dict()
        face_lookup_dict = dict()
        boundary_lookup_dict = dict()
        block_lookup_dict = dict()

        new_vertices = BlockMeshVertex.copy_to_mesh(list(mesh.vertex_ids.values()),
                                                    mesh=self,
                                                    check_existing=False)

        vertex_lookup_dict.update(dict(zip([x.id for x in mesh.vertex_ids.values()], new_vertices)))

        def copy_edges(edges, v_lookup_dict):
            in_mesh_edges = [None] * edges.__len__()
            for ii, edge in enumerate(edges):
                init_dict = copy.copy(edge.__dict__)
                del init_dict['id']
                del init_dict['dict_id']

                init_dict['num_cells'] = init_dict['_num_cells']

                init_dict['vertices'] = [v_lookup_dict[edge.vertices[0].id], v_lookup_dict[edge.vertices[1].id]]
                init_dict['mesh'] = self
                in_mesh_edge = BlockMeshEdge(**init_dict)
                in_mesh_edges[ii] = in_mesh_edge

            return in_mesh_edges

        def copy_faces(instances, e_lookup_dict, v_lookup_dict):
            in_mesh_instances = [None] * instances.__len__()
            for ii, instance in enumerate(instances):
                init_dict = copy.copy(instance.__dict__)
                delete_keys = ['id', 'dict_id', 'boundary', 'merge', 'merge_patch_pairs', 'contacts']
                for key in delete_keys:
                    if key in init_dict.keys():
                        del init_dict[key]

                init_dict['vertices'] = [v_lookup_dict[x.id] for x in instance.vertices]
                init_dict['_edge0'] = e_lookup_dict[instance.edge0.id]
                init_dict['_edge1'] = e_lookup_dict[instance.edge1.id]
                init_dict['_edge2'] = e_lookup_dict[instance.edge2.id]
                init_dict['_edge3'] = e_lookup_dict[instance.edge3.id]
                init_dict['mesh'] = self

                in_mesh_instances[ii] = BlockMeshFace(**init_dict)

            return in_mesh_instances

        def copy_blocks(instances, f_lookup_dict, e_lookup_dict, v_lookup_dict):
            in_mesh_instances = [None] * instances.__len__()
            for ii, instance in enumerate(instances):
                init_dict = copy.copy(instance.__dict__)
                delete_keys = ['id', 'dict_id', 'patch_pairs', 'merge_patch_pairs']
                for key in delete_keys:
                    if key in init_dict.keys():
                        del init_dict[key]

                init_dict['vertices'] = [v_lookup_dict[x.id] for x in instance.vertices]
                init_dict['num_cells'] = init_dict['_num_cells']

                init_dict['edge0'] = e_lookup_dict[instance.edge0.id]
                init_dict['edge1'] = e_lookup_dict[instance.edge1.id]
                init_dict['edge2'] = e_lookup_dict[instance.edge2.id]
                init_dict['edge3'] = e_lookup_dict[instance.edge3.id]
                init_dict['edge4'] = e_lookup_dict[instance.edge4.id]
                init_dict['edge5'] = e_lookup_dict[instance.edge5.id]
                init_dict['edge6'] = e_lookup_dict[instance.edge6.id]
                init_dict['edge7'] = e_lookup_dict[instance.edge7.id]
                init_dict['edge8'] = e_lookup_dict[instance.edge8.id]
                init_dict['edge9'] = e_lookup_dict[instance.edge9.id]
                init_dict['edge10'] = e_lookup_dict[instance.edge10.id]
                init_dict['edge11'] = e_lookup_dict[instance.edge11.id]

                init_dict['face0'] = f_lookup_dict[instance.face0.id]
                init_dict['face1'] = f_lookup_dict[instance.face1.id]
                init_dict['face2'] = f_lookup_dict[instance.face2.id]
                init_dict['face3'] = f_lookup_dict[instance.face3.id]
                init_dict['face4'] = f_lookup_dict[instance.face4.id]
                init_dict['face5'] = f_lookup_dict[instance.face5.id]

                init_dict['mesh'] = self

                in_mesh_instances[ii] = Block(**init_dict)

            return in_mesh_instances

        def copy_boundaries(instances, face_lookup_dict):
            in_mesh_instances = [None] * instances.__len__()
            for ii, instance in enumerate(instances):
                init_dict = copy.copy(instance.__dict__)
                delete_keys = ['id', 'alt_id', 'dict_id', 'boundary', 'merge', 'merge_patch_pairs', 'contacts']
                for key in delete_keys:
                    if key in init_dict.keys():
                        del init_dict[key]

                init_dict['faces'] = set([face_lookup_dict[x.id] for x in instance.faces])
                init_dict['mesh'] = self

                in_mesh_instances[ii] = BlockMeshBoundary(**init_dict)
                _ = [x.set_boundary(in_mesh_instances[ii]) for x in init_dict['faces']]
                if instance.function_objects:
                    in_mesh_instances[ii].function_objects = copy.deepcopy(instance.function_objects)

            return in_mesh_instances

        new_edges = copy_edges(list(mesh.edges.values()), vertex_lookup_dict)
        edge_lookup_dict.update(dict(zip([x.id for x in mesh.edges.values()], new_edges)))

        new_faces = copy_faces(list(mesh.faces.values()), edge_lookup_dict, vertex_lookup_dict)
        face_lookup_dict.update(dict(zip([x.id for x in mesh.faces.values()], new_faces)))

        new_blocks = copy_blocks(mesh.blocks, face_lookup_dict, edge_lookup_dict, vertex_lookup_dict)
        block_lookup_dict.update(dict(zip([x.id for x in mesh.blocks], new_blocks)))

        new_boundaries = copy_boundaries(list(mesh.boundaries.values()), face_lookup_dict)
        boundary_lookup_dict.update(dict(zip([x.id for x in list(mesh.boundaries.values())], new_boundaries)))

        last_activated_mesh.activate()

        return vertex_lookup_dict, edge_lookup_dict, face_lookup_dict, boundary_lookup_dict, block_lookup_dict

    def __repr__(self):
        return f'Mesh {self.id} (name={self.name}, verts: {self.vertices.__len__()} ' \
               f'edges: {self.edges.__len__()} ' \
               f'faces: {self.faces.__len__()} ' \
               f'blocks: {self.blocks.__len__()} ' \
               f'boundaries: {self.boundaries.__len__()} )'


default_mesh = Mesh()


class VertexMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.vertex_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_vertex_id_counter

        # @np_cache

    def get_vertex(cls, position, mesh=None):
        # logger.debug('Getting vertex...')
        # return next((x for x in VertexMetaMock.instances if np.allclose(x.position, position, atol=1e-3)), None)
        if mesh is None:
            mesh = cls.current_mesh

        # vert = cls.instances.get(tuple(position), None)

        vert = mesh.vertices.get(tuple(position), None)
        # if vert is not None:
        #     logger.debug('Vertex already existing')
        return vert

    def get_duplicate(cls, duplicate):
        vert = cls.instances.get(duplicate, None)
        # if vert is not None:
        #     logger.debug('Vertex already existing')
        return vert

    def __call__(cls, *args, **kwargs):
        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        position = np.round(kwargs.get('position', np.array([0, 0, 0])), 3)
        duplicate = kwargs.get('duplicate', False)
        obj = None
        check_existing = kwargs.get('check_existing', True)
        if check_existing:
            obj = cls.get_vertex(position, mesh)
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            # cls.instances.append(obj)
            if duplicate:
                mesh.vertices[duplicate] = obj
            else:
                if tuple(position) in mesh.vertices.keys():
                    logger.debug('Vertex mit position already existing')
                    if isinstance(mesh.vertices[tuple(position)], list):
                        mesh.vertices[tuple(position)].append(obj)
                    else:
                        mesh.vertices[tuple(position)] = [obj, mesh.vertices[tuple(position)]]
                else:
                    mesh.vertices[tuple(position)] = obj
            mesh.vertex_ids[obj.id] = obj
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.vertices

    @property
    def vertex_ids(cls):
        return cls.current_mesh.vertex_ids

    @staticmethod
    def copy_to_mesh(instances, mesh=None, check_existing=False):

        if not hasattr(instances, '__contains__'):  # make it iterable if is not
            instances = [instances]

        if mesh is None:
            mesh = VertexMetaMock.current_mesh

        in_mesh_instances = [None] * instances.__len__()
        for i, vertex in enumerate(instances):
            if vertex.id in mesh.vertex_ids.keys():
                in_mesh_vertex = vertex
            else:
                in_mesh_vertex = BlockMeshVertex(position=vertex.position,
                                                 mesh=mesh,
                                                 check_existing=check_existing)
            in_mesh_instances[i] = in_mesh_vertex

        if in_mesh_instances.__len__() == 1:
            return in_mesh_instances[0]
        else:
            return in_mesh_instances


class EdgeMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.edge_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_edge_id_counter

    @property
    def edge_ids(cls):
        return cls.current_mesh.edge_ids

    def get_edge(self,
                 vertices=None,
                 vertex_ids: tuple[int] = None,
                 create=False,
                 mesh=None):
        # logger.debug('Getting edge...')
        # return next((x for x in EdgeMetaMock.instances if np.array_equal(x.vertices, vertices)), None)

        if mesh is None:
            mesh = self.current_mesh

        if vertices is not None:
            edge = mesh.edges.get(tuple(sorted([vertices[0].id, vertices[1].id])), None)
        elif vertex_ids is not None:
            edge = mesh.edges.get(vertex_ids, None)
        # if edge is None:
        #     edge = EdgeMetaMock.instances.get((vertices[1].id, vertices[0].id), None)
        if edge is None:
            if create:
                # edge = self(vertices=[list(BlockMeshVertex.instances.values())[vertex_ids[0]],
                #                       list(BlockMeshVertex.instances.values())[vertex_ids[1]]])

                edge = BlockMeshEdge(vertices=[BlockMeshVertex.vertex_ids[vertex_ids[0]],
                                               BlockMeshVertex.vertex_ids[vertex_ids[1]]],
                                     create=True,
                                     mesh=mesh)

        # if edge is not None:
        #     logger.debug('Edge already existing')
        return edge

    def __call__(cls, *args, **kwargs):
        vertices = kwargs.get('vertices', np.array([0, 0, 0]))

        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        if not kwargs.get('create', False):
            obj = cls.get_edge(kwargs.get('vertices'), mesh=mesh)
        else:
            obj = None

        if obj is None:
            logger.debug(f'No existing edge found. Creating new one...')
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            # cls.instances.append(obj)
            mesh.edges[tuple(sorted([vertices[0].id, vertices[1].id]))] = obj
            mesh.edge_ids[obj.id] = obj
        else:
            # check same type
            if obj.type != kwargs.get('type', 'line'):
                if kwargs.get('type', 'line'):
                    obj.type = 'line'
                    obj.interpolation_points = [kwargs.get('interpolation_points')]
                    obj.fixed_num_cells = kwargs.get('fixed_num_cells')
                    obj.num_cells = kwargs.get('num_cells')
                    obj._fc_edge = None
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.edges

    @staticmethod
    def copy_to_mesh(edges, mesh=None):

        if mesh is None:
            mesh = EdgeMetaMock.current_mesh

        if not hasattr(edges, '__contains__'):  # make it iterable if is not
            edges = [edges]

        in_mesh_edges = [None] * edges.__len__()
        for ii, edge in enumerate(edges):
            if edge.id in mesh.edge_ids.keys():
                in_mesh_edge = edge
            else:
                init_dict = copy.copy(edge.__dict__)
                del init_dict['id']
                del init_dict['dict_id']

                init_dict['vertices'] = [BlockMeshVertex(position=edge.vertices[0].position, mesh=mesh),
                                         BlockMeshVertex(position=edge.vertices[1].position, mesh=mesh)]
                init_dict['mesh'] = mesh
                in_mesh_edge = BlockMeshEdge(**init_dict)
            in_mesh_edges[ii] = in_mesh_edge

        # in_mesh_edge = None
        # if edge.id in cls.edge_ids.keys():
        #     in_mesh_edge = edge
        # else:
        #     for mesh in Mesh.instances.values():
        #         if mesh is BlockMeshEdge.current_mesh:
        #             continue
        #         if edge.id in mesh.edge_ids.keys():
        #             init_dict = edge.__dict__
        #             init_dict['vertices'] = [BlockMeshVertex(position=edge.vertices[0].position),
        #                                      BlockMeshVertex(position=edge.vertices[1].position)]
        #             in_mesh_edge = BlockMeshEdge(**init_dict)
        if in_mesh_edges.__len__() == 1:
            return in_mesh_edges[0]
        else:
            return in_mesh_edges


class ParallelEdgeSetMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.parallel_edge_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_parallel_edge_id_counter

    def __call__(cls, *args, **kwargs):
        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        # cls.instances.append(obj)
        cls.instances.append(obj)
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.parallel_edges


class FaceMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.faces_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_faces_id_counter

    def get_face(cls, vertices):
        # logger.debug('Getting face...')
        # raise NotImplementedError
        # return next((x for x in EdgeMetaMock.instances if np.array_equal(x.vertices, vertices)), None)
        face = cls.instances.get(tuple(sorted(vertices)), None)
        # if face is not None:
        #     logger.debug('Face already existing')
        return face

    @classmethod
    def get_face_by_vertex_positions(cls, vertices, mesh=None):

        if mesh is None:
            mesh = cls.current_mesh

        key = tuple(sorted([tuple(x.position.tolist()) for x in vertices], key=lambda tup: (tup[1], tup[2])))
        if key in mesh.face_pos.keys():
            return mesh.face_pos[key]
        else:
            return None

    def __call__(cls, *args, **kwargs):

        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        vertices = kwargs.get('vertices')
        obj = cls.get_face(kwargs.get('vertices'))
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            # cls.instances.append(obj)
            mesh.faces[tuple(sorted(vertices))] = obj
            mesh.face_ids[obj.id] = obj

            mesh.face_pos[tuple(sorted([tuple(x.position.tolist()) for x in vertices], key=lambda tup: (tup[1], tup[2])))] = obj
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.faces

    @staticmethod
    def copy_to_mesh(faces, mesh=None, merge_meshes=False):
        if mesh is None:
            mesh = EdgeMetaMock.current_mesh

        if not hasattr(faces, '__contains__'):  # make it iterable if is not
            faces = [faces]

        in_mesh_faces = []

        for face in faces:

            if face.id in mesh.face_ids.keys():
                in_mesh_face = face
            else:
                vertices = [BlockMeshVertex(position=vertex.position, mesh=mesh) for vertex in face.vertices]
                edges = BlockMeshEdge.copy_to_mesh(edges=face.edges, mesh=mesh)

                init_dict = copy.copy(face.__dict__)
                init_dict['name'] = f"Copy of {init_dict['name']}"

                delete_keys = ['id', 'dict_id', 'boundary', 'merge', 'merge_patch_pairs', ]
                for key in delete_keys:
                    if key in init_dict.keys():
                        del init_dict[key]

                init_dict['vertices'] = vertices
                init_dict['_edge0'] = edges[0]
                init_dict['_edge1'] = edges[1]
                init_dict['_edge2'] = edges[2]
                init_dict['_edge3'] = edges[3]
                init_dict['mesh'] = mesh
                init_dict['contacts'] = set([face])
                in_mesh_face = BlockMeshFace(**init_dict)

                face.contacts.add(in_mesh_face)

                if merge_meshes:
                    mesh.add_mesh_contact(in_mesh_face, face)
                    face.mesh.add_mesh_contact(face, in_mesh_face)

            in_mesh_faces.append(in_mesh_face)

        if in_mesh_faces.__len__() == 1:
            return in_mesh_faces[0]
        else:
            return in_mesh_faces


class PatchPairMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.patch_pairs_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_patch_pairs_id_counter

    def get_patch_pair(cls, face_id):
        # logger.debug('Getting face...')
        # raise NotImplementedError
        # return next((x for x in EdgeMetaMock.instances if np.array_equal(x.vertices, vertices)), None)
        patch_pair = cls.instances.get(face_id, None)
        # if face is not None:
        #     logger.debug('Face already existing')
        return patch_pair

    def __call__(cls, *args, **kwargs):
        face_id = kwargs.get('surface').id
        obj = cls.get_patch_pair(face_id)
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            cls.instances[face_id] = obj
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.patch_pairs


class BoundaryMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.boundary_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_boundary_id_counter

    def get_boundary_by_name(cls, name):
        return next((x for x in cls.instances if x.name == name), None)

    def get_boundary_by_txt_id(cls, txt_id, mesh=None):
        if mesh is None:
            mesh = cls.current_mesh
        boundary = next((mesh.boundaries[key] for key, value in mesh.boundaries.items()
                         if value.txt_id == txt_id), None)
        if boundary is None:
            return next((mesh.boundaries[value.name] for key,
                                                         value in mesh.boundaries.items()
                         if value.alt_txt_id == txt_id), None)
        else:
            return boundary

    def __call__(cls, *args, **kwargs):
        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        if isinstance(mesh, list):
            for sub_mesh in mesh:
                sub_mesh.boundaries[kwargs.get('name')] = obj
        else:
            mesh.boundaries[kwargs.get('name')] = obj
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.boundaries

    def copy_to_mesh(cls, instances=None, mesh: Mesh = None):

        if instances is None:
            return

        instance_ids = mesh.boundary_ids

        if mesh is None:
            mesh = cls.current_mesh

        if not hasattr(instances, '__contains__'):  # make it iterable if is not
            instances = [instances]

        in_mesh_instances = [None] * instances.__len__()
        for ii, instance in enumerate(instances):
            if instance in instance_ids.keys():
                in_mesh_instance = instance
                in_mesh_instances[ii] = in_mesh_instance
            else:
                init_dict = copy.copy(instance.__dict__)
                delete_keys = ['id', 'dict_id', 'alt_id']
                for key in delete_keys:
                    if key in init_dict.keys():
                        del init_dict[key]

                init_dict['faces'] = set()

                init_dict['mesh'] = mesh
                in_mesh_instance = cls(**init_dict)
                in_mesh_instances[ii] = in_mesh_instance

                # if instance.function_objects:
                #     new_fos = []
                #     for fo in instance.function_objects:
                #         if fo.mesh is not mesh:
                #
                #
                #
                #         init_dict = fo.__dict__
                #         del init_dict['id']
                #
                #         new_fo = type(fo)(**init_dict)
                #         if instance in fo.patches:
                #             new_fo.patches.remove(instance)
                #             new_fo.patches = in_mesh_instances[ii]
                #         new_fos.append(new_fo)
                #
                #     in_mesh_instances[ii].function_objects = new_fos

        if in_mesh_instances.__len__() == 1:
            return in_mesh_instances[0]
        else:
            return in_mesh_instances


class BlockMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.block_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_block_id_counter

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

        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        cls._comp_solid = None
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.blocks

    def copy_to_mesh(cls, instances=None, mesh: Mesh = None):

        instance_ids = mesh.boundary_ids

        if mesh is None:
            mesh = cls.current_mesh

        if not hasattr(instances, '__contains__'):  # make it iterable if is not
            instances = [instances]

        in_mesh_instances = [None] * instances.__len__()
        for ii, instance in enumerate(instances):
            if instance.id in instance_ids.keys():
                in_mesh_instance = instance
            else:
                init_dict = copy.copy(instance.__dict__)
                delete_keys = ['id', 'dict_id', 'alt_id']
                for key in delete_keys:
                    if key in init_dict.keys():
                        del init_dict[key]

                init_dict['vertices'] = BlockMeshVertex.copy_to_mesh(instance.faces, mesh=mesh)

                init_dict['mesh'] = mesh
                in_mesh_instance = cls(**init_dict)
            in_mesh_instances[ii] = in_mesh_instance

        if in_mesh_instances.__len__() == 1:
            return in_mesh_instances[0]
        else:
            return in_mesh_instances


class CellZoneMetaMock(type):

    current_mesh = default_mesh

    @property
    def id_iter(cls):
        return cls.current_mesh.cell_zone_id_counter

    @property
    def dict_id_iter(cls):
        return cls.current_mesh.dict_cell_zone_id_counter

    def get_cell_zone(cls, material):
        # logger.debug('Getting CellZone...')
        return next((x for x in cls.instances if x.material is material), None)

    def __call__(cls, *args, **kwargs):
        material = kwargs.get('material', None)
        new = kwargs.get('new', False)

        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        obj = None
        if not new:
            obj = cls.get_cell_zone(material)
        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            mesh.cell_zones.append(obj)
            mesh.cell_zone_ids[obj.id] = obj
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.cell_zones

    def copy_to_mesh(cls, instances=None, mesh: Mesh = None):

        instance_ids = mesh.cell_zone_ids

        if mesh is None:
            mesh = cls.current_mesh

        if not hasattr(instances, '__contains__'):  # make it iterable if is not
            instances = [instances]

        in_mesh_instances = [None] * instances.__len__()
        for ii, instance in enumerate(instances):
            if instance.id in instance_ids.keys():
                in_mesh_instance = instance
            else:
                init_dict = copy.copy(instance.__dict__)
                delete_keys = ['id', 'dict_id', 'alt_id']
                for key in delete_keys:
                    if key in init_dict.keys():
                        del init_dict[key]

                init_dict['mesh'] = mesh
                in_mesh_instance = cls(**init_dict)
            in_mesh_instances[ii] = in_mesh_instance

        if in_mesh_instances.__len__() == 1:
            return in_mesh_instances[0]
        else:
            return in_mesh_instances


class CompBlockMetaMock(type):

    current_mesh = default_mesh

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        return obj

    @property
    def instances(cls):
        return cls.current_mesh.comp_blocks


class BlockMeshVertex(object, metaclass=VertexMetaMock):

    # id_iter = VertexMetaMock.current_mesh.vertex_id_counter

    @classmethod
    def block_mesh_entry(cls, vertices=None):
        if vertices is None:
            vertices = cls.instances.values()

        if vertices.__len__() > 0:
            vertex_entries = [None] * vertices.__len__()
            for i, vertex in enumerate(vertices):
                vertex_entries[i] = '\t' + vertex.dict_entry

        else:
            vertex_entries = []

        return f'vertices\n(\n' + "\n".join(vertex_entries) + '\n);\n'

    def __init__(self, *args, **kwargs):
        self.id: int = next(BlockMeshVertex.id_iter)
        self.dict_id = next(BlockMeshVertex.dict_id_iter)
        self.position = kwargs.get('position', np.array([0, 0, 0]))
        self._fc_vertex = None

        self.mesh = kwargs.get('mesh', None)

        self.edges = set()
        self.faces = set()
        self.blocks = set()

        self.duplicated = kwargs.get('duplicated', False)
        self.duplicates = set()

    def __add__(self, vec):
        return BlockMeshVertex(position=self.position + vec)

    def __sub__(self, vec):
        return BlockMeshVertex(position=self.position - vec)

    def __repr__(self):
        return f'Vertex {self.id} (position={self.position[0], self.position[1], self.position[2]}), ' \
               f'duplicate: {[x.id for x in self.duplicates]}'

    @property
    def txt_id(self):
        if isinstance(self.dict_id, uuid.UUID):
            vertex_id = str(self.dict_id.hex)
        else:
            vertex_id = str(self.dict_id)

        if self.duplicated:
            return 'dv' + vertex_id
        else:
            return 'v' + vertex_id
        # if isinstance(self.id, uuid.UUID):
        #     return 'a' + str(self.id.hex)
        # else:
        #     return str(self.id)

    @property
    def fc_vertex(self):
        if self._fc_vertex is None:
            self._fc_vertex = FCPart.Point(Base.Vector(self.position[0], self.position[1], self.position[2]))
        return self._fc_vertex

    @property
    def dict_entry(self):

        return f'name {self.txt_id} ({self.position[0]:16.6f} {self.position[1]:16.6f} {self.position[2]:16.6f}) ' \
               f'// vertex {self.id}'

        # if self.duplicated:
        #     return f'name dv{list(self.duplicates)[0].id} ' \
        #            f'({self.position[0]:16.6f} {self.position[1]:16.6f} {self.position[2]:16.6f}) ' \
        #            f'// vertex {self.id}'
        # else:
        #     return f'name v{self.id} ({self.position[0]:16.6f} {self.position[1]:16.6f} {self.position[2]:16.6f}) ' \
        #            f'// vertex {self.id}'

    def dist_to_point(self, vertex):
        return np.linalg.norm(self.position - vertex.position)

    def duplicate(self):

        if self.duplicated:
            return self

        try:
            duplicate = BlockMeshVertex(position=self.position,
                                        duplicate=self,
                                        duplicated=True)

            self.duplicates.add(duplicate)
            duplicate.duplicates.add(self)

            # duplicate edges:
            for edge in self.edges:
                vertices = copy.copy(edge.vertices)
                if duplicate not in vertices:
                    vertices[vertices.index(self)] = duplicate
                    edge.duplicate(vertices=vertices)
        except Exception as e:
            logger.error(f'Error duplicating vertex {self}:\n{e}')
            raise e

        return duplicate

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

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_fc_vertex']
        return d

    def __setstate__(self, d):
        d['_fc_vertex'] = None
        self.__dict__.update(d)


class BlockMeshEdge(object, metaclass=EdgeMetaMock):
    # id_iter = itertools.count()

    arc_cell_factor = 2

    @classmethod
    def from_fc_edge(cls, fc_edge, mesh=None, num_cells=None, fixed_num_cells=False, translate=np.array([0, 0, 0])):
        if mesh is None:
            mesh = cls.current_mesh

        if isinstance(fc_edge.Curve, FCPart.Circle) or isinstance(fc_edge.Curve, FCPart.Arc):
            interpolation_points = [np.array(((fc_edge.Vertex1.Point - fc_edge.Curve.Center)
                                              + (fc_edge.Vertex2.Point - fc_edge.Curve.Center)).normalize()
                                             * fc_edge.Curve.Radius + fc_edge.Curve.Center) + translate]
            edge_type = 'arc'
        else:
            edge_type = 'line'
            interpolation_points = None

        p0 = BlockMeshVertex(position=np.array(fc_edge.Vertex1.Point) + translate)
        p1 = BlockMeshVertex(position=np.array(fc_edge.Vertex2.Point) + translate)

        return cls(mesh=mesh,
                   vertices=[p0, p1],
                   type=edge_type,
                   fixed_num_cells=fixed_num_cells,
                   num_cells=num_cells,
                   interpolation_points=interpolation_points
                   )

    @classmethod
    def block_mesh_entry(cls, edges=None):

        if edges is None:
            edges = cls.instances.values()

        edge_entries = ['\t' + x.dict_entry for x in edges if x.dict_entry is not None]

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
                        line	        Straight line	    ???
        """
        self.id = next(BlockMeshEdge.id_iter)
        self.dict_id = next(BlockMeshEdge.dict_id_iter)
        self.vertices = kwargs.get('vertices')

        self.type = kwargs.get('type', 'line')      # arc or line
        self.center = kwargs.get('center', None)
        self.interpolation_points = kwargs.get('interpolation_points', None)

        self.mesh = kwargs.get('mesh')

        self._num_cells = None
        self._fc_edge = None
        self._collapsed = False
        self._direction = None

        # number cells:
        self.fixed_num_cells = kwargs.get('fixed_num_cells', False)
        self.num_cells = kwargs.get('num_cells', None)
        self.cell_size = kwargs.get('cell_size', self.mesh.default_cell_size)

        self.vertices[0].edges.add(self)
        self.vertices[1].edges.add(self)

        self.faces = set()
        self.blocks = set()

        self._parallel_edge_set = None
        self.duplicates = set()

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
            if (self.length is not None) and (self.length != 0):
                if isinstance(self.fc_edge.Curve, FCPart.Line):
                    self._num_cells = int(np.ceil(self.length / self.cell_size))
                if isinstance(self.fc_edge.Curve, FCPart.Arc):
                    self._num_cells = int(np.ceil(self.length / self.cell_size)) * self.arc_cell_factor
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
            return f'arc {self.vertices[0].txt_id} {self.vertices[1].txt_id} ' \
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

    def duplicate(self, vertices):

        init_dict = self.__dict__
        init_dict['vertices'] = vertices
        duplicate = BlockMeshEdge(**init_dict)
        duplicate.duplicates.add(self)
        self.duplicates.add(duplicate)
        return duplicate

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_fc_edge']
        return d

    def __setstate__(self, d):
        d['_fc_edge'] = None
        self.__dict__.update(d)


class ParallelEdgesSet(object, metaclass=ParallelEdgeSetMetaMock):

    # id_iter = itertools.count()

    @classmethod
    def get_edges_set(cls, edge):
        for edge_set in ParallelEdgesSet.instances:
            if edge in edge_set.edges:
                return edge_set
        return None

    @classmethod
    def merge_sets(cls, pe_sets=None):

        if pe_sets is None:
            pe_sets = cls.instances

        do_merge = True

        new_pe_set = copy.copy(pe_sets)

        while do_merge:
            do_merge = False
            current_pe_set = new_pe_set
            for instance in current_pe_set:
                for next_instance in current_pe_set:
                    if instance == next_instance:
                        continue
                    if (instance.edges & next_instance.edges).__len__() > 0:
                        instance.edges.update(next_instance.edges)
                        _ = [setattr(x, 'parallel_edge_set', instance) for x in next_instance.edges]
                        instance._cell_size = instance.mesh.default_cell_size
                        instance._num_cells = None
                        new_pe_set = copy.copy(current_pe_set)
                        new_pe_set.remove(next_instance)
                        do_merge = True
                        break

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
        self.dict_id = next(ParallelEdgesSet.dict_id_iter)
        self.mesh = kwargs.get('mesh')

        self.edges = kwargs.get('edges', set())
        self._cell_size = kwargs.get('cell_size', self.mesh.default_cell_size)
        self._num_cells = kwargs.get('num_cells', None)

        self.mesh = kwargs.get('mesh')

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
            fixed_num_cells = [x.num_cells for x in self.edges if (x.fixed_num_cells and (x.num_cells is not None))]

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
        self._num_cells = value

    def add_edges(self, edges: set[BlockMeshEdge]):
        self.edges.update(edges)
        _ = [setattr(x, 'parallel_edge_set', self) for x in edges]
        self.num_cells = None


class BlockMeshBoundary(object, metaclass=BoundaryMetaMock):

    # id_iter = itertools.count()

    @classmethod
    def block_mesh_entry(cls, boundaries=None, mesh=None):
        # boundary_entries = [None] * cls.instances.values().__len__()

        if boundaries is None:
            boundaries = cls.instances.values()

        if mesh is None:
            mesh = cls.current_mesh

        boundary_entries = [x.dict_entry for x in boundaries.values() if x.dict_entry is not None]

        merge_patch_entries = [x.boundary_dict_entry for x in PatchPair.instances.values()
                               if x.boundary_dict_entry is not None]

        mesh_contact_entries = mesh.mesh_contact_dict_entry

        # for i, boundary in enumerate(BoundaryMetaMock.instances.values()):
        #     entry = boundary.dict_entry
        #     if entry is not None:
        #         boundary_entries[i] = '\t' + boundary.dict_entry
        return f'boundary\n(\n' + "\n".join([*boundary_entries, *merge_patch_entries, *mesh_contact_entries]) + '\n);\n'

    @classmethod
    def create_patch_dict_entries(cls, boundaries=None, mesh=None):
        # boundary_entries = [None] * cls.instances.values().__len__()

        if mesh is None:
            mesh = cls.current_mesh

        if boundaries is None:
            boundaries = mesh.boundaries.values()

        boundary_entries = [x.create_patch_dict_entry for x in boundaries if x.create_patch_dict_entry is not None]

        return "\n".join([*boundary_entries]) + '\n'

    def __init__(self, *args, **kwargs):

        self._txt_id = None
        self._alt_txt_id = None
        self._function_objects = None
        self._fc_face = None
        self._geo_face = None

        self.id = next(BlockMeshBoundary.id_iter)
        self.alt_id = next(BlockMeshBoundary.id_iter)       # alternative id, needed in changePatchDict as existing patch type will remain the same
        self.use_alt_id = False
        self.dict_id = next(BlockMeshBoundary.dict_id_iter)

        self.mesh = kwargs.get('mesh')

        self.name = kwargs.get('name')
        self.type = kwargs.get('type')
        self._faces = kwargs.get('faces', set())

        self.txt_id = kwargs.get('txt_id', None)

        self.n_faces = kwargs.get('n_faces', None)
        self.start_face = kwargs.get('start_face', None)

        self.user_bc = kwargs.get('user_bc', None)
        self.solid_user_bc = kwargs.get('solid_user_bc', None)
        self.fluid_user_bc = kwargs.get('fluid_user_bc', None)

        self.case = kwargs.get('case', None)
        self._cell_zone = kwargs.get('cell_zone', None)

        self.function_objects = kwargs.get('function_objects', [])
        self.geo_face = kwargs.get('geo_face', None)

    def add_face(self, face):
        self._faces.add(face)

    def add_faces(self, faces):
        self._faces.update(faces)

    @property
    def cell_zone(self):
        return self._cell_zone

    @cell_zone.setter
    def cell_zone(self, value):
        self._cell_zone = value
        for fo in self._function_objects:
            fo.cell_zone = self.cell_zone

    @property
    def txt_id(self):
        if self._txt_id is None:
            if self.geo_face is None:
                if isinstance(self.id, uuid.UUID):
                    self._txt_id = re.sub('\W+','', 'a' + str(self.id))
                else:
                    self._txt_id = re.sub('\W+','', 'a' + str(self.id))
            else:
                self._txt_id = self.geo_face.txt_id
        return self._txt_id

    @txt_id.setter
    def txt_id(self, value):
        self._txt_id = value

    @property
    def alt_txt_id(self):
        if self._alt_txt_id is None:
            if isinstance(self.alt_id, uuid.UUID):
                self._alt_txt_id = 'bc' + str(self.alt_id.hex)
            else:
                self._alt_txt_id = 'bc' + str(self.alt_id)
        return self._alt_txt_id

    @alt_txt_id.setter
    def alt_txt_id(self, value):
        self._alt_txt_id = value

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

        if self.faces.__len__() == 0:
            return None

        faces_entry = "\n".join(['\t\t\t(' + ' '.join([str(y.txt_id) for y in x.vertices]) +
                                 ')' +
                                 f'\t//{x.id}' for x in self.faces])

        # return (f"\t{self.txt_id + '_' + self.name}\n"
        #         f"\t{'{'}\n"
        #         f"\t\ttype {self.type};\n"
        #         f"\t\tfaces\n"
        #         f"\t\t(\n"
        #         f"{faces_entry}\n"
        #         f"\t\t);\n"
        #         f"\t{'}'}")

        return (f"\t{self.txt_id}\n"
                f"\t{'{'}\n"
                f"\t\ttype {self.type};\n"
                f"\t\tfaces\n"
                f"\t\t(\n"
                f"{faces_entry}\n"
                f"\t\t);\n"
                f"\t{'}'}")

    @property
    def create_patch_dict_entry(self):

        return None

    @property
    def function_objects(self):
        return self._function_objects

    @function_objects.setter
    def function_objects(self, value):
        self._function_objects = value
        for fo in self._function_objects:
            fo.patches.add(self)
            fo.cell_zone = self.cell_zone

    @property
    def fc_face(self):
        raise NotImplementedError

    @property
    def geo_face(self):
        return self._geo_face

    @geo_face.setter
    def geo_face(self, value):
        self._geo_face = value
        self._txt_id = None

    def __repr__(self):
        return f'Boundary {self.id} (name={self.name}, type={self.type}, faces={self.faces})'

    def __eq__(self, other):
        if not isinstance(other, BlockMeshBoundary):
            return False
        return self.id == other.id

    def __hash__(self):
        return id(self)


class CyclicAMI(BlockMeshBoundary):

    def __init__(self, *args, **kwargs):
        BlockMeshBoundary.__init__(self, *args, **kwargs)

        faces = kwargs.get('faces', None)
        if faces is not None:
            _ = [x.set_boundary(self) for x in faces]

        self.neighbour_patch = kwargs.get('neighbour_patch', None)
        self.type = 'mappedWall'
        self.transform = kwargs.get('transform', 'none')
        self.match_tolerance = kwargs.get('match_tolerance', 0.1)

    @property
    def dict_entry(self):

        if self.type == 'interface':
            return None

        if self.faces.__len__() == 0:
            return None

        faces_entry = "\n".join(['\t\t\t(' + ' '.join([str(y.txt_id) for y in x.vertices]) +
                                 ')' +
                                 f'\t//{x.id}'for x in self.faces])

        # return (f"\t{self.txt_id + '_' + self.name}\n"
        #         f"\t{'{'}\n"
        #         f"\t\ttype {self.type};\n"
        #         f"\t\tfaces\n"
        #         f"\t\t(\n"
        #         f"{faces_entry}\n"
        #         f"\t\t);\n"
        #         f"\t{'}'}")

        return (f"\t{self.txt_id}\n"
                f"\t{'{'}\n"
                f"\t\ttype {self.type};\n"
                f"\t\tsampleRegion {list(self.faces[0].blocks)[0].cell_zone.txt_id};\n"
                f"\t\tsamplePatch {self.neighbour_patch.txt_id};\n"
                f"\t\tsampleMode nearestPatchFaceAMI;\n"
                f"\t\tfaces\n"
                f"\t\t(\n"
                f"{faces_entry}\n"
                f"\t\t);\n"
                f"\t{'}'}")

    @property
    def create_patch_dict_entry(self):
        """
        createPatchDict entry
        :return:
        """
        if self.type == 'interface':
            return None

        if self.faces.__len__() == 0:
            return None

        return (f"{'{'}\n"
                f"\tname {self.txt_id};\n"
                f"\tpatchInfo\n"
                f"\t{'{'}\n"
                f"\t\tsampleRegion {list(self.faces[0].blocks)[0].cell_zone.txt_id};\n"
                f"\t\tsamplePatch {self.neighbour_patch.txt_id};\n"
                f"\t\ttype mappedWall;\n"
                f"\t\tsampleMode nearestPatchFaceAMI;\n"
                f"\t\toffsetMode uniform;\n"
                f"\t\toffset (0 0 0);\n"
                f"\t{'}'}\n"
                f"\tconstructFrom patches;\n"
                f"\tpatches ({self.txt_id});\n"
                f"{'}'}")


        # return (f"{'{'}\n"
        #         f"\tname {self.alt_txt_id};\n"
        #         f"\tpatchInfo\n"
        #         f"\t{'{'}\n"
        #         f"\t\ttype cyclicAMI;\n"
        #         f"\t\tneighbourPatch {self.neighbour_patch.alt_txt_id};\n"
        #         f"\t\ttransform {self.transform};\n"
        #         f"\t\tmatchTolerance  {self.match_tolerance};\n"
        #         f"\t{'}'}\n"
        #         f"\tconstructFrom patches;\n"
        #         f"\tpatches ({self.txt_id});\n"
        #         f"{'}'}")


class NearestPatchFaceAMI(BlockMeshBoundary):

    def __init__(self, *args, **kwargs):
        BlockMeshBoundary.__init__(self, *args, **kwargs)

        faces = kwargs.get('faces', None)
        if faces is not None:
            _ = [x.set_boundary(self) for x in faces]

        self.neighbour_patch = kwargs.get('neighbour_patch', None)
        self.type = 'mappedWall'

    @property
    def dict_entry(self):

        if self.faces.__len__() == 0:
            return None

        faces_entry = "\n".join(['\t\t\t(' + ' '.join([str(y.txt_id) for y in x.vertices]) + ')' for x in self.faces])

        return (f"\t{self.txt_id}\n"
                f"\t{'{'}\n"
                f"\t\ttype {self.type};\n"
                f"\t\tsampleRegion {list(self.faces[0].blocks)[0].cell_zone.txt_id};\n"
                f"\t\tsamplePatch {self.neighbour_patch.txt_id};\n"
                f"\t\tsampleMode nearestPatchFaceAMI;\n"
                f"\t\tfaces\n"
                f"\t\t(\n"
                f"{faces_entry}\n"
                f"\t\t);\n"
                f"\t{'}'}")


class CreatePatchDict(object):
    default_path = '/tmp/'

    def __init__(self, *args, **kwargs):
        self._id = kwargs.get('_id', kwargs.get('id', uuid.uuid4()))
        self._name = kwargs.get('_name', kwargs.get('name', 'BlockMesh {}'.format(self._id)))
        self._case_dir = kwargs.get('_case_dir', kwargs.get('case_dir', None))

        self._create_patch_dict = None
        self.template = pkg_resources.read_text(case_resources, 'createPatchDict')

        self.boundaries = kwargs.get('boundaries', [])

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        if value == self._id:
            return
        self._id = value

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

    @property
    def create_patch_dict(self):
        if self._create_patch_dict is None:
            self._create_patch_dict = self.create_patch_dict_entry()
        return self._create_patch_dict

    @create_patch_dict.setter
    def create_patch_dict(self, value):
        if value == self._create_patch_dict:
            return
        self._create_patch_dict = value

    def create_patch_dict_entry(self):

        if self.boundaries.__len__() == 0:
            return None

        template = copy.copy(self.template)
        boundary_entries = BlockMeshBoundary.create_patch_dict_entries(boundaries=self.boundaries)
        template = template.replace('<patches>', boundary_entries)
        return template

    def write_create_patch_dict(self, case_dir=None):
        if case_dir is None:
            case_dir = self.case_dir

        filepath = os.path.join(case_dir, 'system', "createPatchDict")
        logger.info(f'Writing createPatchDict to {filepath}')
        if self.create_patch_dict is not None:
            with open(filepath, "w") as f:
                f.write(self.create_patch_dict)
        else:
            logger.info(f'Any boundaries in createPatchDict. Did not write createPatchDict')

    def run(self, case_dir=None, overwrite=True):

        if case_dir is None:
            case_dir = self.case_dir

        if overwrite:
            ow_str = '-overwrite'
        else:
            ow_str = ''

        res = subprocess.run(
            ["/bin/bash", "-i", "-c", f"createPatch -noFunctionObjects {ow_str}"],
            capture_output=True,
            cwd=case_dir,
            user='root')

        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            logger.info(f"Successfully created Patches \n\n{output}")
        else:
            logger.error(f"Error creating Patches:\n{res.stderr.decode('ascii')}")

        for boundary in self.boundaries:
            boundary.use_alt_id = True


inlet_patch = BlockMeshBoundary(name='inlet', type='patch', user_bc=VolumeFlowInlet())

outlet_patch = BlockMeshBoundary(name='outlet', type='patch', user_bc=Outlet())

wall_patch = BlockMeshBoundary(name='wall', type='wall', user_bc=SolidWall())

fluid_wall_patch = BlockMeshBoundary(name='fluid_wall', type='wall', user_bc=FluidWall())

pipe_wall_patch = BlockMeshBoundary(name='pipe_wall',
                                    type='interface',
                                    solid_user_bc=SolidFluidInterface(),
                                    fluid_user_bc=FluidSolidInterface())

top_side_patch = BlockMeshBoundary(name='top_side',
                                   type='wall',
                                   user_bc=SolidConvection()
                                   )

bottom_side_patch = BlockMeshBoundary(name='bottom_side',
                                      type='wall',
                                      user_bc=SolidConvection()
                                      )

# PressureDifferencePatch(patch1=inlet_patch,
#                         patch2=outlet_patch,
#                         name='Pressure Difference Patch')
#
# TemperatureDifferencePatch(patch1=inlet_patch,
#                            patch2=outlet_patch,
#                            name='Pressure Difference Patch')


NoPlane = object()
NoNormal = object()


class BlockMeshFace(object, metaclass=FaceMetaMock):
    # id_iter = itertools.count()

    def __init__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :keyword type:  arc	            Circular arc	    Single interpolation point
                        simpleSpline	Spline curve	    List of interpolation points
                        polyLine	    Set of lines	    List of interpolation points
                        polySpline	    Set of splines	    List of interpolation points
                        line	        Straight line	    ???
        """
        self.name = kwargs.get('name', 'unnamed face')
        self.id = next(BlockMeshFace.id_iter)
        self.dict_id = next(BlockMeshFace.dict_id_iter)
        self.vertices = kwargs.get('vertices')

        self.mesh = kwargs.get('mesh')

        self.merge = kwargs.get('merge', True)
        self.merge_patch_pairs = kwargs.get('merge_patch_pairs', True)
        self.contacts = kwargs.get('contacts', set())

        self._boundary = kwargs.get('boundary', None)
        self._fc_face = kwargs.get('_fc_face', None)

        self._edge0 = kwargs.get('_edge0', None)
        self._edge1 = kwargs.get('_edge1', None)
        self._edge2 = kwargs.get('_edge2', None)
        self._edge3 = kwargs.get('_edge3', None)

        # self.edges = kwargs.get('edges', None)

        self._plane = None
        self._dirty_center = None
        self._normal = None
        self._planar = None

        self.extruded = kwargs.get('extruded', False)   # false or: [base_profile, top_profile, path1, path2]
        self.patch_pair = kwargs.get('patch_pair', set())

        _ = [x.faces.add(self) for x in self.vertices]

        self.blocks = set()
        _ = self.edges

    @property
    def face_pos(self):
        return tuple(sorted(self.vertices))

    @property
    def face_vertex_pos(self):
        return tuple(tuple(x.position) for x in sorted(self.vertices))

    @property
    def plane(self):
        if self._plane is None:
            pts_set = list(set(self.vertices))
            if pts_set.__len__() < 3:
                return NoPlane
            self._plane = FCPart.Plane(Base.Vector(pts_set[0].position),
                                       Base.Vector(pts_set[1].position),
                                       Base.Vector(pts_set[2].position))
        return self._plane

    @property
    def planar(self):
        if self._planar is None:
            if self.plane is NoPlane:
                self._planar = False
            elif self.plane.toShape().distToShape(self.vertices[3].fc_vertex.toShape())[0] < 1e-3:
                self._planar = True
            else:
                self._planar = False
        return self._planar

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
    def dirty_center(self):
        if self._dirty_center is None:
            self._dirty_center = np.mean(np.array([x.position for x in self.vertices]), axis=0)
        return self._dirty_center

    @property
    def normal(self):
        if self._normal is None:
            if self.planar:
                self._normal = np.array(self.plane.Axis)
            else:
                self._normal = NoNormal
        return self._normal

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, value: BlockMeshBoundary):
        if not value == self._boundary:
            if value.mesh != self.mesh:
                value = BlockMeshBoundary.copy_to_mesh(value, self.mesh)
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

    @edges.setter
    def edges(self, value):
        if value is None:
            self._edge0 = None
            self._edge1 = None
            self._edge2 = None
            self._edge3 = None
        else:
            if value.__len__() != 4:
                raise ValueError(f'Error setting face edges of face {self.name}. Value length must be 4, but value is {value}')
            self._edge0 = value[0]
            self._edge1 = value[1]
            self._edge2 = value[2]
            self._edge3 = value[3]

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
                try:
                    face_wire = FCPart.Wire(edges)
                except Exception as e:
                    export_objects(edges, f'/tmp/face{self.id}_edges.FCStd')
                    logger.info(f'Could not generate face wire for face:\n{self.id}')
                    raise e

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

    def extrude(self, dist, direction=None, dist2=None, mesh=None, merge_meshes=False):

        if mesh is None:
            mesh = FaceMetaMock.current_mesh

        non_regular = False

        if direction is None:
            direction = self.normal

        if self.mesh is not mesh:
            face = FaceMetaMock.copy_to_mesh(face=self,
                                             mesh=mesh,
                                             merge_meshes=False)
        else:
            face = self

        vertices = face.vertices

        if set(face.vertices).__len__() == 3:
            # vertices = list(set(self.vertices))
            vertices = [ii for n, ii in enumerate(face.vertices) if ii not in face.vertices[:n]]
            non_regular = True

        if (dist2 is None) or (abs(dist2) < 1):
            v_1 = vertices
            _1 = face.edges
            _2 = [x.translated_copy(direction * dist) for x in face.edges]
        else:
            v_1 = np.array([x + (dist2 * direction) for x in vertices])
            _1 = [x.translated_copy(direction * dist2) for x in face.edges]
            _2 = [x.translated_copy(direction * dist) for x in face.edges]

        v_2 = np.array([x + (dist * direction) for x in face.vertices])
        _3 = [BlockMeshEdge(vertices=[v_1[i], v_2[i]], type='line') for i in range(v_1.__len__())]

        # export_objects([x.fc_edge for x in [*_1, *_2, *_3]], '/tmp/extruded_edges.FCStd')

        # create quad blocks:

        new_block = None
        if vertices.__len__() == 4:

            new_block = Block(vertices=[*v_1, *v_2],
                              name=f'Extruded Block',
                              block_edges=[*_1, *_2, *_3],
                              auto_cell_size=True,
                              non_regular=non_regular,
                              extruded=False,
                              mesh=mesh)
        elif vertices.__len__() == 3:
            v_1 = [*v_1, v_1[0]]
            v_2 = [*v_2, v_2[0]]

            new_block = Block(vertices=[*v_1, *v_2],
                              name=f'Extruded Block',
                              block_edges=[*_1, *_2, *_3],
                              auto_cell_size=True,
                              extruded=False,
                              non_regular=True,
                              mesh=mesh
                              )

        if merge_meshes:
            if self.mesh is not mesh:

                common_face_ids = set(mesh.face_pos.keys()).intersection(self.mesh.face_pos.keys())
                for common_face_id in common_face_ids:
                    face1 = mesh.face_pos[common_face_id]
                    face2 = self.mesh.face_pos[common_face_id]

                    mesh.add_mesh_contact(face1, face2)
                    self.mesh.add_mesh_contact(face2, face1)

            # for face in new_block.faces:
            #     other_face = FaceMetaMock.get_face_by_vertex_positions(vertices=face.vertices, mesh=self.mesh)
            #     if other_face is not None:
            #         mesh.add_mesh_contact(other_face, face)
            #         face.mesh.add_mesh_contact(face, other_face)

                # set(mesh.face_pos.keys()).intersection(self.mesh.face_pos.keys())

        return new_block

    def __repr__(self):
        return f'Face {self.id} (type={self.boundary}, Area={self.area}, Vertices={self.vertices})'

    def __eq__(self, other):
        return sorted(self.vertices) == sorted(other.vertices)

    def __hash__(self):
        return id(self)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_fc_face']
        return d

    def __setstate__(self, d):
        d['_fc_face'] = None
        self.__dict__.update(d)

    # if type(edge.Curve) is FCPart.Line:
    #     return [BlockMeshEdge(vertices=[p1[i], p2[i]], type='line') for i in range(4)]
    # else:
    #     edges = [None] * 4
    #     for i in range(4):
    #         # move center point
    #         center = np.array(edge.Curve.Center) + (p1[i].position - edge.Vertexes[0].Point) * face_normal


class PatchPair(object, metaclass=PatchPairMetaMock):
    # id_iter = itertools.count()

    @classmethod
    def block_mesh_entry(cls, patch_pairs=None):
        if patch_pairs is None:
            patch_pairs = cls.instances.values()

        patch_pairs_entries = [x.dict_entry for x in patch_pairs if x.dict_entry is not None]
        return f'mergePatchPairs\n(\n' + "\n".join(patch_pairs_entries) + '\n);\n'

    def __init__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :keyword type:  arc	            Circular arc	    Single interpolation point
                        simpleSpline	Spline curve	    List of interpolation points
                        polyLine	    Set of lines	    List of interpolation points
                        polySpline	    Set of splines	    List of interpolation points
                        line	        Straight line	    ???
        """

        self._surface = None
        self._patches = set()

        self.name = kwargs.get('name', 'unnamed PatchPair')
        self.id = uuid.uuid4()
        self.dict_id = uuid.uuid4()
        self.surface = kwargs.get('surface', None)

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, value):
        self._surface = value

    @property
    def patches(self):
        return self._patches

    def add_patch(self, patch):
        self._patches.add(patch)

    @property
    def boundary_dict_entry(self):

        if self.surface.blocks.__len__() == 1:
            return None

        if self.patches.__len__() == 1:
            dict_entry = f'\tmerge_{self.surface.txt_id}\n' \
                         f"\t{'{'}\n" \
                         f'\t\ttype patch;\n' \
                         f"\t\tfaces\n" \
                         f"\t\t\t(({' '.join([str(x.txt_id) for x in self.surface.vertices])}));\n" \
                         f"\t{'}'}\n" \
                         f'\tmerge_{list(self.patches)[0].txt_id}\n' \
                         f"\t{'{'}\n" \
                         f'\t\ttype patch;\n' \
                         f"\t\tfaces\n" \
                         f"\t\t\t(({' '.join([str(x.txt_id) for x in list(self.patches)[0].vertices])}));\n" \
                         f"\t{'}'}\n"

        elif self.patches.__len__() == 2:
            dict_entry = f'\tmerge_{list(self.patches)[0].txt_id}\n' \
                         f"\t{'{'}\n" \
                         f'\t\ttype patch;\n' \
                         f"\t\tfaces\n" \
                         f"\t\t\t(({' '.join([str(x.txt_id) for x in list(self.patches)[0].vertices])}));\n" \
                         f"\t{'}'}\n" \
                         f'\tmerge_{list(self.patches)[1].txt_id}\n' \
                         f"\t{'{'}\n" \
                         f'\t\ttype patch;\n' \
                         f"\t\tfaces\n" \
                         f"\t\t\t(({' '.join([str(x.txt_id) for x in list(self.patches)[1].vertices])}));\n" \
                         f"\t{'}'}\n"
        else:
            raise Exception(
                f'Error in PatchPair {self.id}: number of patches must be 1 or 2.\nPatches are: {self.patches}')
        return dict_entry

    @property
    def dict_entry(self):
        if self.surface.blocks.__len__() == 1:
            return None
        elif self.patches.__len__() == 1:
            return f'\t(merge_{self.surface.txt_id} merge_{list(self.patches)[0].txt_id})'
        elif self.patches.__len__() == 2:
            return f'\t(merge_{list(self.patches)[0]} merge_{list(self.patches)[1].txt_id})'
        else:
            raise Exception(
                f'Error in PatchPair {self.id}: number of patches must be 1 or 2.\nPatches are: {self.patches}')

    def __repr__(self):
        return f'PatchPair {self.id})'

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
        'right': (5, 6, 2, 1),      # 3
        'bottom': (4, 5, 1, 0),        # 4
        'top': (7, 3, 2, 6)      # 5
    }

    edge_map = {'inlet': (0, 1, 2, 3),      # 0
                'outlet': (4, 7, 6, 5),     # 1
                'left': (8, 3, 11, 7),       # 2
                'right': (9, 5, 10, 1),      # 3
                'bottom': (4, 9, 0, 8),        # 4
                'top': (6, 11, 2, 10)      # 5
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

    # face_pe_sets = {0: [0, 1],
    #                 1: [0, 1],
    #                 2: [1, 2],
    #                 3: [1, 2],
    #                 4: [0, 2],
    #                 5: [0, 2]}

    face_adj = {0: [2, 3, 4, 5],
                1: [2, 3, 4, 5],
                2: [0, 1, 4, 5],
                3: [0, 1, 4, 5],
                4: [0, 1, 2, 3],
                5: [0, 1, 2, 3]}

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

    # id_iter = itertools.count()

    doc = App.newDocument()

    @classmethod
    def update_parallel_edges(cls, blocks=None):
        if blocks is None:
            blocks = cls.instances

        for instance in blocks:
            if instance.merge_patch_pairs:
                continue
            instance._parallel_edges_sets = []
            instance.num_cells = None
            for p_edges in Block.all_parallel_edges:
                instance._parallel_edges_sets.append(ParallelEdgesSet.add_set(instance.block_edges[p_edges]))

        logger.debug('Updated parallel edges')

    @classmethod
    def block_mesh_entry(cls, blocks=None):

        if blocks is None:
            blocks = cls.instances

        block_entries = ['\t' + x.dict_entry + f' Block {i}' for i, x in
                         enumerate(blocks) if ((not x.duplicate) and (x.dict_entry is not None))]
        # block_entries = [None] * cls.instances.__len__()
        # for i, block in enumerate(BlockMetaMock.instances):
        #     block_entries[i] = '\t' + block.dict_entry
        return f'blocks\n(\n' + "\n".join(block_entries) + '\n);\n'

    @classmethod
    def save_fcstd(cls, filename, blocks=None):
        """
        save as freecad document
        :param filename: full filename; example: '/tmp/test.FCStd'
        :param shape_type: 'solid', 'faces'
        :param blocks: block to save; default is all
        """
        if blocks is None:
            blocks = cls.instances

        doc = App.newDocument(f"Blocks")
        for i, block in enumerate(blocks):
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

        self._vertices = None
        self._num_cells = kwargs.get('_num_cells', None)
        self._block_edges = None
        self._fc_solid = None
        self._faces = None
        self._edge = None
        self._dirty_center = kwargs.get('_dirty_center', None)

        self._merge_patch_pairs = False
        self._duplicate = False
        self._original = None
        self._duplicated_block = None

        self._face0 = None
        self._face1 = None
        self._face2 = None
        self._face3 = None
        self._face4 = None
        self._face5 = None

        self._edge0 = None
        self._edge1 = None
        self._edge2 = None
        self._edge3 = None
        self._edge4 = None
        self._edge5 = None
        self._edge6 = None
        self._edge7 = None
        self._edge8 = None
        self._edge9 = None
        self._edge10 = None
        self._edge11 = None

        self.face0 = kwargs.get('face0', None)
        self.face1 = kwargs.get('face1', None)
        self.face2 = kwargs.get('face2', None)
        self.face3 = kwargs.get('face3', None)
        self.face4 = kwargs.get('face4', None)
        self.face5 = kwargs.get('face5', None)

        self.edge0 = kwargs.get('edge0', None)
        self.edge1 = kwargs.get('edge1', None)
        self.edge2 = kwargs.get('edge2', None)
        self.edge3 = kwargs.get('edge3', None)
        self.edge4 = kwargs.get('edge4', None)
        self.edge5 = kwargs.get('edge5', None)
        self.edge6 = kwargs.get('edge6', None)
        self.edge7 = kwargs.get('edge7', None)
        self.edge8 = kwargs.get('edge8', None)
        self.edge9 = kwargs.get('edge9', None)
        self.edge10 = kwargs.get('edge10', None)
        self.edge11 = kwargs.get('edge11', None)

        self.name = kwargs.get('name', 'unnamed_block')
        self.id = next(Block.id_iter)
        self.dict_id = next(Block.dict_id_iter)
        self.vertices = kwargs.get('vertices', None)
        self.patch_pairs = set()
        self.assigned_feature = kwargs.get('assigned_feature', None)
        self.edge = kwargs.get('edge', None)
        self.num_cells = kwargs.get('num_cells', None)
        self.block_edges = kwargs.get('block_edges', None)
        self.cell_zone = kwargs.get('cell_zone', None)
        self.auto_cell_size = kwargs.get('auto_cell_size', True)
        self.cell_size = kwargs.get('cell_size', 100)
        self.grading = kwargs.get('grading', [1, 1, 1])

        self.mesh = kwargs.get('mesh')

        self.non_regular = kwargs.get('non_regular', False)
        self.extruded = kwargs.get('extruded', False)

        self.original = kwargs.get('original', None)
        self.duplicate = kwargs.get('duplicate', False)
        self.duplicated_block = kwargs.get('duplicated_block', None)
        self.merge_patch_pairs = kwargs.get('merge_patch_pairs', False)

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
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        if isinstance(value, list):
            value = np.array(value)
        self._vertices = value

    @property
    def duplicate(self):
        return self._duplicate

    @duplicate.setter
    def duplicate(self, value):
        self._duplicate = value

    @property
    def duplicated_block(self):
        return self._duplicated_block

    @duplicated_block.setter
    def duplicated_block(self, value):
        self._duplicated_block = value

    @property
    def original(self):
        return self._original

    @original.setter
    def original(self, value):
        self._original = value

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def dict_entry(self):

        if self.duplicated_block is None:
            vertices = self.vertices
        elif isinstance(self.duplicated_block, Block):
            vertices = self.duplicated_block.vertices
        else:
            raise Exception(f'Error creating dict entry for Block {self.id}:\n'
                            f'DuplicatedBlock must be None or instance of Block but is {type(self.duplicated_block)}')

        # export_objects([self.fc_solid, *[x.fc_vertex.toShape() for x in self.vertices]], '/tmp/test_export.FCStd')

        num_cells = self.num_cells

        if self.non_regular:
            v0 = vertices[2].fc_vertex.toShape().Point - vertices[1].fc_vertex.toShape().Point
            v1 = vertices[0].fc_vertex.toShape().Point - vertices[1].fc_vertex.toShape().Point
            v2 = vertices[5].fc_vertex.toShape().Point - vertices[1].fc_vertex.toShape().Point

            v2_ref = v0.normalize().cross(v1.normalize()).normalize()

            if np.allclose(v2_ref, v2.normalize()):
                corrected_vertices = np.array(vertices)[np.array([1, 2, 3, 0, 5, 6, 7, 4])].tolist()
                num_cells = [num_cells[1], num_cells[0], num_cells[2]]
            elif np.allclose(v2_ref, -v2.normalize()):
                corrected_vertices = np.array(vertices)[np.array([5, 6, 7, 4, 1, 2, 3, 0])].tolist()
                num_cells = [num_cells[1], num_cells[0], num_cells[2]]
            else:
                if abs(angle_between_vectors(v2_ref, v2, v0)) < 90:
                    corrected_vertices = np.array(vertices)[np.array([1, 2, 3, 0, 5, 6, 7, 4])].tolist()
                    num_cells = [num_cells[1], num_cells[0], num_cells[2]]
                else:
                    corrected_vertices = np.array(vertices)[np.array([5, 6, 7, 4, 1, 2, 3, 0])].tolist()
                    num_cells = [num_cells[1], num_cells[0], num_cells[2]]

        else:
            v0 = vertices[1].fc_vertex.toShape().Point - vertices[0].fc_vertex.toShape().Point
            v1 = vertices[3].fc_vertex.toShape().Point - vertices[0].fc_vertex.toShape().Point
            v2 = vertices[4].fc_vertex.toShape().Point - vertices[0].fc_vertex.toShape().Point

            try:
                v2_ref = v0.normalize().cross(v1.normalize()).normalize()
            except Exception as e:
                raise e

            if np.allclose(v2_ref, v2.normalize()):
                corrected_vertices = vertices
            elif np.allclose(v2_ref, -v2.normalize()):
                corrected_vertices = [*vertices[4:], *vertices[0:4]]
            else:
                if abs(angle_between_vectors(v2_ref, v2, v0)) < 90:
                    corrected_vertices = vertices
                else:
                    corrected_vertices = [*vertices[4:], *vertices[0:4]]

        if self.cell_zone is not None:
            cell_zone = self.cell_zone.txt_id
        else:
            cell_zone = None

        return f"hex ({' '.join([x.txt_id for x in corrected_vertices])}) {cell_zone} " \
               f"({num_cells[0]} {num_cells[1]} {num_cells[2]}) " \
               f"simpleGrading ({self.grading[0]} {self.grading[1]} {self.grading[2]})" \
               f"// block {self.id}"

    # @property
    # def center_line(self):
    #     return self.faces[0].C
    @property
    def face0(self):
        if self._face0 is None:
            self._face0 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['inlet']],
                                        edges=self.block_edges[np.array(self.edge_map['inlet'])])
            self._face0.blocks.add(self)
        return self._face0

    @face0.setter
    def face0(self, value):
        if self._face0 is not None:
            try:
                self._face0.blocks.remove(self)
            except Exception as e:
                pass
        self._face0 = value
        if value is not None:
            self._face0.blocks.add(self)

    @property
    def face1(self):
        if self._face1 is None:
            self._face1 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['outlet']],
                                        edges=self.block_edges[np.array(self.edge_map['outlet'])])
            self._face1.blocks.add(self)
        return self._face1

    @face1.setter
    def face1(self, value):
        try:
            self._face1.blocks.remove(self)
        except Exception as e:
            pass
        self._face1 = value
        if value is not None:
            self._face1.blocks.add(self)

    @property
    def face2(self):
        if self._face2 is None:
            if self.extruded:
                extruded = self.block_edges[np.array(self.edge_map['left'])][np.array([0, 2, 1, 3])]
            else:
                extruded = False

            self._face2 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['left']],
                                        extruded=extruded,
                                        edges=self.block_edges[np.array(self.edge_map['left'])])
            self._face2.blocks.add(self)
        return self._face2

    @face2.setter
    def face2(self, value):
        try:
            self._face2.blocks.remove(self)
        except Exception as e:
            pass
        self._face2 = value
        if value is not None:
            self._face2.blocks.add(self)

    @property
    def face3(self):
        if self._face3 is None:
            if self.extruded:
                extruded = self.block_edges[np.array(self.edge_map['right'])][np.array([0, 2, 1, 3])]
            else:
                extruded = False

            self._face3 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['right']],
                                        extruded=extruded,
                                        edges=self.block_edges[np.array(self.edge_map['right'])])
            self._face3.blocks.add(self)
        return self._face3

    @face3.setter
    def face3(self, value):
        try:
            self._face3.blocks.remove(self)
        except Exception as e:
            pass
        self._face3 = value
        if value is not None:
            self._face3.blocks.add(self)

    @property
    def face4(self):
        if self._face4 is None:
            if self.extruded:
                extruded = self.block_edges[np.array(self.edge_map['top'])][np.array([0, 2, 1, 3])]
            else:
                extruded = False

            self._face4 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['top']],
                                        extruded=extruded,
                                        edges=self.block_edges[np.array(self.edge_map['top'])])
            self._face4.blocks.add(self)
        return self._face4

    @face4.setter
    def face4(self, value):
        try:
            self._face4.blocks.remove(self)
        except Exception as e:
            pass
        self._face4 = value
        if value is not None:
            self._face4.blocks.add(self)

    @property
    def face5(self):
        if self._face5 is None:
            if self.extruded:
                extruded = self.block_edges[np.array(self.edge_map['bottom'])][np.array([0, 2, 1, 3])]
            else:
                extruded = False

            self._face5 = BlockMeshFace(vertices=[self.vertices[x] for x in Block.face_map['bottom']],
                                        extruded=extruded,
                                        edges=self.block_edges[np.array(self.edge_map['bottom'])])
            self._face5.blocks.add(self)
        return self._face5

    @face5.setter
    def face5(self, value):
        try:
            self._face5.blocks.remove(self)
        except Exception as e:
            pass
        self._face5 = value
        if value is not None:
            self._face5.blocks.add(self)

    @property
    def faces(self):
        return [self.face0, self.face1, self.face2, self.face3, self.face4, self.face5]

    @faces.setter
    def faces(self, value):

        if isinstance(value, list):
            if value.__len__() != 6:
                raise IndexError('Expected 6 faces')
        elif isinstance(value, np.ndarray):
            if value.shape.__len__() != 1:
                raise IndexError('Expected 6 faces')
            if value.shape[0] != 6:
                raise IndexError('Expected 6 faces')

        self._face0 = value[0]
        self._face1 = value[1]
        self._face2 = value[2]
        self._face3 = value[3]
        self._face4 = value[4]
        self._face5 = value[5]

    @property
    def edge0(self):
        if self._edge0 is None:
            v_map = self._edge_vertex_map[0]
            self._edge0 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge0.blocks.add(self)
        return self._edge0

    @edge0.setter
    def edge0(self, value):
        try:
            self._edge0.blocks.remove(self)
        except Exception as e:
            pass
        self._edge0 = value
        if value is not None:
            self._edge0.blocks.add(self)

    @property
    def edge1(self):
        if self._edge1 is None:
            v_map = self._edge_vertex_map[1]
            self._edge1 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge1.blocks.add(self)
        return self._edge1

    @edge1.setter
    def edge1(self, value):
        try:
            self._edge1.blocks.remove(self)
        except Exception as e:
            pass
        self._edge1 = value
        if value is not None:
            self._edge1.blocks.add(self)

    @property
    def edge2(self):
        if self._edge2 is None:
            v_map = self._edge_vertex_map[2]
            self._edge2 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge2.blocks.add(self)
        return self._edge2

    @edge2.setter
    def edge2(self, value):
        try:
            self._edge2.blocks.remove(self)
        except Exception as e:
            pass
        self._edge2 = value
        if value is not None:
            self._edge2.blocks.add(self)

    @property
    def edge3(self):
        if self._edge3 is None:
            v_map = self._edge_vertex_map[3]
            self._edge3 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge3.blocks.add(self)
        return self._edge3

    @edge3.setter
    def edge3(self, value):
        try:
            self._edge3.blocks.remove(self)
        except Exception as e:
            pass
        self._edge3 = value
        if value is not None:
            self._edge3.blocks.add(self)

    @property
    def edge4(self):
        if self._edge4 is None:
            v_map = self._edge_vertex_map[4]
            self._edge4 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge4.blocks.add(self)
        return self._edge4

    @edge4.setter
    def edge4(self, value):
        try:
            self._edge4.blocks.remove(self)
        except Exception as e:
            pass
        self._edge4 = value
        if value is not None:
            self._edge4.blocks.add(self)

    @property
    def edge5(self):
        if self._edge5 is None:
            v_map = self._edge_vertex_map[5]
            self._edge5 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge5.blocks.add(self)
        return self._edge5

    @edge5.setter
    def edge5(self, value):
        try:
            self._edge5.blocks.remove(self)
        except Exception as e:
            pass
        self._edge5 = value
        if value is not None:
            self._edge5.blocks.add(self)

    @property
    def edge6(self):
        if self._edge6 is None:
            v_map = self._edge_vertex_map[6]
            self._edge6 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge6.blocks.add(self)
        return self._edge6

    @edge6.setter
    def edge6(self, value):
        try:
            self._edge6.blocks.remove(self)
        except Exception as e:
            pass
        self._edge6 = value
        if value is not None:
            self._edge6.blocks.add(self)

    @property
    def edge7(self):
        if self._edge7 is None:
            v_map = self._edge_vertex_map[7]
            self._edge7 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge7.blocks.add(self)
        return self._edge7

    @edge7.setter
    def edge7(self, value):
        try:
            self._edge7.blocks.remove(self)
        except Exception as e:
            pass
        self._edge7 = value
        if value is not None:
            self._edge7.blocks.add(self)

    @property
    def edge8(self):
        if self._edge8 is None:
            v_map = self._edge_vertex_map[8]
            self._edge8 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge8.blocks.add(self)
        return self._edge8

    @edge8.setter
    def edge8(self, value):
        try:
            self._edge8.blocks.remove(self)
        except Exception as e:
            pass
        self._edge8 = value
        if value is not None:
            self._edge8.blocks.add(self)

    @property
    def edge9(self):
        if self._edge9 is None:
            v_map = self._edge_vertex_map[9]
            self._edge9 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                          self.vertices[v_map[1]].id])),
                                                 create=True)
            self._edge9.blocks.add(self)
        return self._edge9

    @edge9.setter
    def edge9(self, value):
        try:
            self._edge9.blocks.remove(self)
        except Exception as e:
            pass
        self._edge9 = value
        if value is not None:
            self._edge9.blocks.add(self)

    @property
    def edge10(self):
        if self._edge10 is None:
            v_map = self._edge_vertex_map[10]
            self._edge10 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                           self.vertices[v_map[1]].id])),
                                                  create=True)
            self._edge10.blocks.add(self)
        return self._edge10

    @edge10.setter
    def edge10(self, value):
        try:
            self._edge10.blocks.remove(self)
        except Exception as e:
            pass
        self._edge10 = value
        if value is not None:
            self._edge10.blocks.add(self)

    @property
    def edge11(self):
        if self._edge11 is None:
            v_map = self._edge_vertex_map[11]
            self._edge11 = BlockMeshEdge.get_edge(vertex_ids=tuple(sorted([self.vertices[v_map[0]].id,
                                                                           self.vertices[v_map[1]].id])),
                                                  create=True)
            self._edge11.blocks.add(self)
        return self._edge11

    @edge11.setter
    def edge11(self, value):
        try:
            self._edge11.blocks.remove(self)
        except Exception as e:
            pass
        self._edge11 = value
        if value is not None:
            self._edge11.blocks.add(self)

    @property
    def block_edges(self):
        return np.array([self.edge0, self.edge1, self.edge2,
                         self.edge3, self.edge4, self.edge5,
                         self.edge6, self.edge7, self.edge8,
                         self.edge9, self.edge10, self.edge11])

    @block_edges.setter
    def block_edges(self, value):
        if value is None:
            self._edge0 = None
            self._edge1 = None
            self._edge2 = None
            self._edge3 = None
            self._edge4 = None
            self._edge5 = None
            self._edge6 = None
            self._edge7 = None
            self._edge8 = None
            self._edge9 = None
            self._edge10 = None
            self._edge11 = None
        else:
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

    @property
    def merge_patch_pairs(self):
        return self._merge_patch_pairs

    @merge_patch_pairs.setter
    def merge_patch_pairs(self, value):
        self._merge_patch_pairs = value

        vertices = copy.copy(self.vertices)

        if self._merge_patch_pairs:
            for face_id in self._merge_patch_pairs:
                duplicate_vertex_ids = np.array(list(self.face_map.values())[face_id])
                duplicated_vertices = [x.duplicate() for x in self.vertices[duplicate_vertex_ids]]

                vertices[duplicate_vertex_ids] = duplicated_vertices

            duplicated_block = Block(vertices=vertices,
                                     name=f'Duplicated Block ({self.id})',
                                     duplicate=True,
                                     extruded=self.extruded,
                                     edge=self.edge,
                                     cell_zone=self.cell_zone,
                                     non_regular=self.non_regular)

            for face_id in self._merge_patch_pairs:
                patch_pair = PatchPair(surface=self.faces[face_id])
                patch_pair.add_patch(duplicated_block.faces[face_id])
                for adj_face_id in self.face_adj[face_id]:
                    patch_pair = PatchPair(surface=self.faces[adj_face_id])
                    patch_pair.add_patch(duplicated_block.faces[adj_face_id])

            self.duplicated_block = duplicated_block

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

    def get_face_outside_normal(self, face_id):
        # https://stackoverflow.com/questions/57434507/determing-the-direction-of-face-normals-consistently

        face_normal = self.faces[face_id].normal
        face_center = self.faces[face_id].dirty_center
        cube_center = self.dirty_center

        if np.dot(face_normal, face_center - cube_center) >= 0.0:
            return face_normal
        else:
            return -face_normal

    def extrude_face(self,
                     dist=1,
                     face_id=None,
                     face=None,
                     direction=None,
                     dist2=None,
                     mesh=None,
                     merge_meshes=False):

        if face is None:
            if face_id is not None:
                face = self.faces[face_id]
            else:
                logger.error(f'Error extruding face of block {self}:\nEither face id or face must be given ')
                raise Exception(f'Either face id or face must be given')
        else:
            # check if face is a block face:
            if face not in self.faces:
                block_faces = '\n'.join(self.faces)
                raise Exception(f'Error extruding face of block {self}:\nFace {face} is not a face of the block.'
                                f"Block faces are: {block_faces}")

        if direction is None:
            direction = self.get_face_outside_normal(face_id)

        return face.extrude(dist, direction=direction, dist2=dist2, mesh=mesh, merge_meshes=merge_meshes)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_fc_solid']
        return d

    def __setstate__(self, d):
        d['_fc_solid'] = None
        self.__dict__.update(d)

    def __repr__(self):
        if self.cell_zone is None:
            cz_name = 'Undefined'
        else:
            cz_name = self.cell_zone.material.name

        return f'Block {self.id} ({self.name}, {cz_name})'


class CellZone(object, metaclass=CellZoneMetaMock):
    # id_iter = itertools.count()

    def __init__(self, *args, **kwargs):
        self._name = None
        self.name = kwargs.get('name', None)
        self.id = kwargs.get('id', next(CellZone.id_iter))
        self.dict_id = kwargs.get('dict_id', next(CellZone.dict_id_iter))
        self.material = kwargs.get('material', None)
        self.boundaries = kwargs.get('boundaries', [])
        self.case = kwargs.get('case', None)

        self.alphat = kwargs.get('alphat', Alphat())
        self.epsilon = kwargs.get('epsilon', Epsilon())
        self.k = kwargs.get('k', K())
        self.nut = kwargs.get('nut', Nut())
        self.p = kwargs.get('p', P())
        self.p_rgh = kwargs.get('p_rgh', PRgh())
        self.t = kwargs.get('t', T())
        self.u = kwargs.get('u', U())

        self.mesh = kwargs.get('mesh')

    @property
    def name(self):
        if self._name is None:
            if self.material is not None:
                self._name = 'Cell Zone ' + self.material.name
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def txt_id(self):
        if self.material is None:
            if isinstance(self.dict_id, uuid.UUID):
                return 'a' + str(self.dict_id.hex)
            else:
                return 'a' + str(self.dict_id)
        else:
            return self.material.txt_id

    def init_directories(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'constant', str(self.txt_id)), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        os.makedirs(os.path.join(case_dir, '0', str(self.txt_id)), exist_ok=True)

    def write_decompose_par_dict(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), 'decomposeParDict')
        with open(full_filename, "w") as f:
            f.write(self.material.decompose_par_dict)

    def write_fvschemes(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), 'fvSchemes')
        with open(full_filename, "w") as f:
            f.write(self.material.fvschemes)

    def write_fvsolution(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), 'fvSolution')
        with open(full_filename, "w") as f:
            f.write(self.material.fvsolution)

    def write_thermo_physical_properties(self, case_dir):
        #
        os.makedirs(os.path.join(case_dir, 'constant', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'thermophysicalProperties')
        with open(full_filename, "w") as f:
            f.write(self.material.thermo_physical_properties_entry)

    def write_g(self, case_dir):
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'g')
        with open(full_filename, "w") as f:
            f.write(self.material.g_entry)

    def write_thermo_physic_transport(self, case_dir):
        # https://cpp.openfoam.org/v8/classFoam_1_1turbulenceThermophysicalTransportModels_1_1eddyDiffusivity.html#a10501494309552f678d858cb7de6c1d3
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'thermophysicalTransport')
        with open(full_filename, "w") as f:
            f.write(self.material.thermo_physic_transport_entry)

    def write_momentum_transport(self, case_dir):
        # https://cpp.openfoam.org/v8/classFoam_1_1turbulenceThermophysicalTransportModels_1_1eddyDiffusivity.html#a10501494309552f678d858cb7de6c1d3
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'momentumTransport')
        with open(full_filename, "w") as f:
            f.write(self.material.momentum_transport_entry)

    def write_to_of(self, case_dir):
        self.init_directories(case_dir)
        self.write_thermo_physical_properties(case_dir)
        self.write_decompose_par_dict(case_dir)
        self.write_fvschemes(case_dir)
        self.write_fvsolution(case_dir)

        if isinstance(self.material, Fluid):
            self.write_thermo_physic_transport(case_dir)
            self.write_momentum_transport(case_dir)
            self.write_g(case_dir)

    def update_bcs(self):

        for boundary in self.boundaries:
            logger.debug(f'updating_bcs')

            if boundary.type == 'interface':
                bc_key = boundary.txt_id
            else:
                bc_key = boundary.txt_id

            if isinstance(boundary, CyclicAMI):
                bc_key = boundary.txt_id

            self.t.patches[bc_key] = boundary.user_bc.t

            if isinstance(self.material, Fluid):
                self.alphat.patches[bc_key] = boundary.user_bc.alphat
                self.epsilon.patches[bc_key] = boundary.user_bc.epsilon
                self.k.patches[bc_key] = boundary.user_bc.k
                self.nut.patches[bc_key] = boundary.user_bc.nut
                self.p.patches[bc_key] = boundary.user_bc.p
                self.p_rgh.patches[bc_key] = boundary.user_bc.p_rgh
                self.u.patches[bc_key] = boundary.user_bc.u

        logger.debug(f'updating_bcs')

    def write_bcs(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case.case_dir

        # write T:
        self.t.internal_field_value = self.case.bc.initial_temperature[self.material]
        self.t.write(os.path.join(case_dir, '0', self.txt_id))

        for bc_name in ['alphat', 'epsilon', 'k', 'nut', 'p', 'p_rgh', 'u']:
            bc_file = getattr(self, bc_name)
            bc_file.write(os.path.join(case_dir, '0', self.txt_id))


class CompBlock(object, metaclass=CompBlockMetaMock):

    @classmethod
    def search_merge_patch_pairs(cls):

        merge_faces = []

        for comp_block in cls.instances:
            for other_comp_block in cls.instances:

                if comp_block is other_comp_block:
                    continue

                # export_objects([comp_block.fc_solid, other_comp_block.fc_solid], '/tmp/blocks.FCStd')

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
        self._cell_zones = None

        self.blocks = kwargs.get('blocks', [])

    @property
    def id(self):
        return self._id

    @property
    def txt_id(self):
        return re.sub('\W+', '', 'a' + str(self.id))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def cell_zones(self):
        if self._cell_zones is None:
            self._cell_zones = {x.cell_zone for x in self.blocks}
        return self._cell_zones

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

    def save_fcstd(self, filename, shape_type='solid'):
        """
        save as freecad document
        :param filename: full filename; example: '/tmp/test.FCStd'
        :param shape_type: 'solid', 'faces'
        """
        doc = App.newDocument(f"Solid {self.name}")
        if shape_type == 'solid':
            __o__ = doc.addObject("Part::Feature", f'Solid {self.name} {self.id}')
            __o__.Shape = self.fc_solid.Shape
        elif shape_type == 'face':
            for face in self.faces:
                __o__ = doc.addObject("Part::Face", f'Face {face.name} {face.id}')
                __o__.Shape = face.fc_solid.Shape
        doc.recompute()
        doc.saveCopy(filename)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_fc_solid']
        return d

    def __setstate__(self, d):
        d['_fc_solid'] = None
        self.__dict__.update(d)


class BlockMesh(object):

    default_path = work_dir

    @classmethod
    def join_meshes(cls, meshes, mesh_name):

        logger.info(f'Joining block meshes {[x.name for x in meshes]}')

        join_meshes = [x.mesh for x in meshes]
        joined_mesh, _, _, face_lookup_dict, _, _ = Mesh.join_meshes(join_meshes, mesh_name)

        top_faces = [[face_lookup_dict[y.id] for y in x.top_faces] for x in meshes]
        bottom_faces = [[face_lookup_dict[y.id] for y in x.bottom_faces] for x in meshes]
        interfaces = [[face_lookup_dict[y.id] for y in x.interfaces] for x in meshes]

        merged_block_mesh = cls(
            name='Block Mesh ' + mesh_name,
            top_faces=functools.reduce(operator.iconcat, top_faces, []),
            bottom_faces=functools.reduce(operator.iconcat, bottom_faces, []),
            interfaces=functools.reduce(operator.iconcat, interfaces, []),
            mesh=joined_mesh)

        logger.info(f'Successfully joined block meshes {[x.name for x in meshes]}')

        return merged_block_mesh, face_lookup_dict

    @classmethod
    def add_mesh_contacts(cls, block_meshes):

        logger.info('Adding mesh contacts')
        mesh_contacts = add_mesh_contacts(block_meshes)
        # mesh_contacts = []
        #
        # for block_mesh in block_meshes:
        #     for other_block_mesh in block_meshes:
        #         if block_mesh is other_block_mesh:
        #             continue
        #
        #         common_face_ids = set(block_mesh.mesh.face_pos.keys()).intersection(other_block_mesh.mesh.face_pos.keys())
        #
        #         if common_face_ids:
        #             logger.info(f'Found mesh contacts between:\n'
        #                         f' {block_mesh} and {other_block_mesh}.\n'
        #                         f'Common face ids: {[block_mesh.mesh.face_pos[x].id for x in common_face_ids]}\n')
        #
        #             mesh_contacts.append(
        #                 add_face_contacts([block_mesh.mesh.face_pos[x] for x in common_face_ids],
        #                                   [other_block_mesh.mesh.face_pos[x] for x in common_face_ids],
        #                                   block_mesh.mesh,
        #                                   other_block_mesh.mesh,
        #                                   f'{block_mesh.mesh.txt_id}_to_{other_block_mesh.mesh.txt_id}',
        #                                   f'{other_block_mesh.mesh.txt_id}_to_{block_mesh.mesh.txt_id}')
        #             )
        #
        # logger.info(f'Successfully added mesh contacts')
        logger.info('Successfully added mesh contacts')
        return mesh_contacts

    @classmethod
    def add_face_contacts(cls, faces1, faces2, mesh1, mesh2, name1, name2):
        return add_face_contacts(faces1, faces2, mesh1, mesh2, name1, name2)

    def __init__(self, *args, **kwargs):
        self._id = kwargs.get('_id', kwargs.get('id', uuid.uuid4()))
        self._name = kwargs.get('_name', kwargs.get('name', 'BlockMesh {}'.format(self._id)))
        self._case_dir = kwargs.get('_case_dir', kwargs.get('case_dir', None))

        # self.type = kwargs.get('type')

        self.template = pkg_resources.read_text(msh_resources, 'block_mesh_dict')
        self._block_mesh_dict = None
        self._control_dict = None
        self._fvschemes = None
        self._fvsolution = None

        # self.blocks = kwargs.get('blocks', None)
        self.mesh = kwargs.get('mesh', None)

        self.top_faces = kwargs.get('top_faces', [])
        self.bottom_faces = kwargs.get('bottom_faces', [])
        self.interfaces = kwargs.get('interfaces', [])

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
            self._fvschemes = pkg_resources.read_text(case_resources, 'fvSchemes')
        return self._fvschemes

    @fvschemes.setter
    def fvschemes(self, value):
        if value == self._fvschemes:
            return
        self._fvschemes = value

    @property
    def fvsolution(self):
        if self._fvsolution is None:
            self._fvsolution = pkg_resources.read_text(case_resources, 'fvSolution')
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
    def txt_id(self):
        return re.sub('\W+', '', 'a' + str(self.id))

    @property
    def case_dir(self):
        if self._case_dir is None:
            self._case_dir = os.path.join(os.path.join(self.default_path, self.txt_id))
        return self._case_dir

    @case_dir.setter
    def case_dir(self, value):
        if value == self._case_dir:
            return
        self._case_dir = value

    @property
    def cell_zones(self):
        return self.mesh.cell_zones

    def add_mesh_copy(self, other_block_mesh, copy_feature_faces=True):

        logger.info(f'Adding mesh copy of {other_block_mesh.name} to {self.name}')

        _, _, face_lookup_dict, boundary_lookup_dict, _ = self.mesh.add_mesh_copy(other_block_mesh.mesh)

        if copy_feature_faces:
            top_faces = [face_lookup_dict[y.id] for y in other_block_mesh.top_faces]
            bottom_faces = [face_lookup_dict[y.id] for y in other_block_mesh.bottom_faces]
            interfaces = [face_lookup_dict[y.id] for y in other_block_mesh.interfaces]

            self.top_faces.extend(top_faces)
            self.bottom_faces.extend(bottom_faces)
            self.interfaces.extend(interfaces)

        logger.info(f'Successfully added mesh copy of {other_block_mesh.name}')

        return face_lookup_dict

    def init_case(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        logger.info(f'Initializing Block Mesh in {case_dir}')

        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, '0'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'constant'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'system'), exist_ok=True)

        if not os.path.isfile(os.path.join(case_dir, 'system', "controlDict")):
            self.write_control_dict(case_dir=case_dir)
        if not os.path.isfile(os.path.join(case_dir, 'system', "fvSchemes")):
            self.write_fv_schemes(case_dir=case_dir)
        if not os.path.isfile(os.path.join(case_dir, 'system', "fvSolution")):
            self.write_fv_solution(case_dir=case_dir)
        self.write_block_mesh_dict(case_dir=case_dir)

        logger.info('Case successfully initialized')

    def write_control_dict(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        with open(os.path.join(case_dir, 'system', "controlDict"), mode="w") as f:
            f.write(self.control_dict)

    def write_fv_schemes(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        with open(os.path.join(case_dir, 'system', "fvSchemes"), "w") as f:
            f.write(self.fvschemes)

    def write_fv_solution(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        with open(os.path.join(case_dir, 'system', "fvSolution"), "w") as f:
            f.write(self.fvsolution)

    def create_block_mesh_dict(self):

        def extend(a):
            out = []
            for sublist in a:
                if isinstance(sublist, list):
                    out.extend(sublist)
                else:
                    out.append(sublist)
            return out

        if self.mesh is not None:
            self.mesh.activate()

            vertices = extend(self.mesh.vertices.values())
            edges = extend(self.mesh.edges.values())
            blocks = self.mesh.blocks
            boundaries = self.mesh.boundaries
            patch_pairs = self.mesh.patch_pairs
            pe_sets = set()
            _ = [pe_sets.update(x.parallel_edges_sets) for x in blocks]

        else:
            vertices = extend(BlockMeshVertex.instances.values())
            edges = extend(BlockMeshEdge.instances.values())
            blocks = Block.instances
            pe_sets = ParallelEdgesSet.instances
            boundaries = BlockMeshBoundary.instances
            patch_pairs =PatchPair.instances

            # contacts = CompBlock.search_merge_patch_pairs()
        # logger.info(f'Updating parallel edges')
        # Block.update_parallel_edges(blocks=blocks)

        logger.info(f'Merging parallel edge sets')
        ParallelEdgesSet.merge_sets(pe_sets=pe_sets)

        # export_objects([FCPart.Compound([x.fc_edge for x in y.edges]) for y in ParallelEdgesSet.instances],
        #                '/tmp/parallel_edges.FCStd')

        logger.info('Creating blockMeshDict...')

        template = pkg_resources.read_text(msh_resources, 'block_mesh_dict')

        vertices_entry = BlockMeshVertex.block_mesh_entry(vertices=vertices)
        template = template.replace('<vertices>', vertices_entry)

        edges_entry = BlockMeshEdge.block_mesh_entry(edges=edges)
        template = template.replace('<edges>', edges_entry)

        block_entry = Block.block_mesh_entry(blocks=blocks)
        template = template.replace('<blocks>', block_entry)

        boundary_entry = BlockMeshBoundary.block_mesh_entry(boundaries=boundaries)
        template = template.replace('<boundary>', boundary_entry)

        template = template.replace('<faces>', '')

        merge_patch_pairs_entry = PatchPair.block_mesh_entry(patch_pairs=patch_pairs)
        template = template.replace('<merge_patch_pairs>', merge_patch_pairs_entry)

        # export_objects([Block.instances[15].fc_solid, Block.instances[1148].fc_solid], '/tmp/error_blocks.FCStd')

        return template

    def write_block_mesh_dict(self, case_dir=None):
        if case_dir is None:
            case_dir = self.case_dir

        with open(os.path.join(case_dir, 'system', "blockMeshDict"), "w") as f:
            f.write(self.block_mesh_dict)

    def fix_inconsistent_block_faces(self, block_ids):
        logger.info(f'Fixing inconsistent block faces for Blocks {block_ids}')
        blocks = [Block.instances[x] for x in block_ids]

        for block in Block.instances:
            block.num_cells = None

        edges_set = []
        for edge in blocks[1].block_edges:
            b1_p_edges_set = None
            b2_p_edges_set = None
            num_cells1 = None
            num_cells2 = None

            if edge in blocks[0].block_edges:
                print(f'{edge} in {blocks[0]}')
                edge_index = np.where(blocks[0].block_edges == edge)
                ape_index = np.where(np.any(Block.all_parallel_edges == edge_index, axis=1))[0][0]
                num_cells1 = blocks[0].num_cells[ape_index]
                if edge_index:
                    b1_p_edges_set = blocks[0].parallel_edges_sets[
                        np.where(np.any(Block.all_parallel_edges == edge_index, axis=1))[0][0]]

                edge_index2 = np.where(blocks[1].block_edges == edge)
                ape_index2 = np.where(np.any(Block.all_parallel_edges == edge_index2, axis=1))[0][0]
                num_cells2 = blocks[1].num_cells[ape_index2]
                if edge_index2:
                    b2_p_edges_set = blocks[1].parallel_edges_sets[
                        np.where(np.any(Block.all_parallel_edges == edge_index2, axis=1))[0][0]]

                if num_cells1 != num_cells2:
                    logger.error(f'  ')
            edges_set.append([(b1_p_edges_set, b2_p_edges_set), (num_cells1, num_cells2)])

            # if not b1_p_edges_set is b2_p_edges_set:
            #     edges_set.append([b1_p_edges_set, b2_p_edges_set])
        bmd_filename = os.path.join(self.case_dir, 'system', 'blockMeshDict')
        os.remove(bmd_filename) if os.path.exists(bmd_filename) else None

        self.write_block_mesh_dict()

    def create_block_obj(self):

        if use_ssh:
            shin, shout, sherr = shell_handler.run_block_mesh(self.case_dir, 
                                                              '-blockTopology -noFunctionObjects -noClean')
        else:
            logger.info(f'Exporting blockTopology....')
            res = subprocess.run(["/bin/bash", "-i", "-c", "blockMesh -blockTopology -noFunctionObjects -noClean"],
                                 capture_output=True,
                                 cwd=self.case_dir,
                                 user='root')
            if res.returncode == 0:
                output = res.stdout.decode('ascii')
                logger.info(f"Successfully exported blockTopology: \n\n {output}")
            else:
                logger.error(f"{res.stderr.decode('ascii')}")
                raise Exception(f"Error exporting blockTopology:\n{res.stderr.decode('ascii')}")

    def obj_to_vtk(self, case_dir=None):
        if case_dir is None:
            case_dir = self.case_dir

        logger.info(f'Creating blockTopology vtk....')
        res = subprocess.run(["/bin/bash", "-i", "-c", "objToVTK blockTopology.obj blockTopology.vtk"],
                             capture_output=True,
                             cwd=case_dir,
                             user='root')
        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            logger.info(f"Successfully created blockTopology vtk: \n\n {output}")
        else:
            logger.error(f"{res.stderr.decode('ascii')}")
            raise Exception(f"Error blockTopology vtk:\n{res.stderr.decode('ascii')}")

    def run_check_mesh(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        if use_ssh:
            shin, shout, sherr = shell_handler.run_check_mesh(case_dir)
        else:
            logger.info(f'Checking mesh....')
            res = subprocess.run(
                ["/bin/bash", "-i", "-c", "checkMesh 2>&1 | tee checkMesh.log"],
                capture_output=True,
                cwd=self.case_dir,
                user='root')
            if res.returncode == 0:
                output = res.stdout.decode('ascii')
                if output.find('FOAM FATAL ERROR') != -1:
                    logger.error(f'Error decomposePar:\n\n{output}')
                    raise Exception(f'Error decomposePar:\n\n{output}')
                logger.info(f"Successfully ran checkMesh \n\n{output}")
            else:
                logger.error(f"{res.stderr.decode('ascii')}")

        return True

    def run_parafoam(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        if use_ssh:
            shin, shout, sherr = shell_handler.run_parafoam(case_dir)
        else:
            logger.info(f'Running paraFoam initialization....')
            res = subprocess.run(["/bin/bash", "-i", "-c", "paraFoam -touchAll"],
                                 capture_output=True,
                                 cwd=case_dir,
                                 user='root')
            if res.returncode == 0:
                output = res.stdout.decode('ascii')
                logger.info(f"Successfully ran paraFoam initialization \n\n{output}")
            else:
                logger.error(f"{res.stderr.decode('ascii')}")
        return True

    def merge_mesh(self, block_mesh, case_dir=None, other_case_dir=None, overwrite=True, add_ami=True):

        if case_dir is None:
            case_dir = self.case_dir

        if other_case_dir is None:
            other_case_dir = block_mesh.case_dir

        if overwrite:
            overwrite_txt = ' -overwrite'
        else:
            overwrite_txt = ''

        logger.info(f'Merging meshes {self} and {block_mesh}')

        res = subprocess.run(["/bin/bash", "-i", "-c", f"mergeMeshes {case_dir} {other_case_dir}{overwrite_txt} -noFunctionObjects"],
                             capture_output=True,
                             cwd=case_dir,
                             user='root')

        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            logger.info(f"Successfully merged meshes \n\n{output}")
        else:
            logger.error(f"Error Merging meshes:\n{res.stderr.decode('ascii')}")

        if add_ami:
            amis = []

            for name, boundary in block_mesh.mesh.boundaries.items():
                if not isinstance(boundary, CyclicAMI):
                    continue
            if boundary.neighbour_patch in self.mesh.boundaries.values():
                amis.extend([boundary, boundary.neighbour_patch])

            cpd = CreatePatchDict(case_dir=case_dir,
                                  boundaries=amis)
            cpd.write_create_patch_dict()
            cpd.run(case_dir=self.case_dir)

            return cpd
        else:
            return True

    def stitch_meshes(self, block_meshes):

        logger.info(f'Stitching meshes....')

        for block_mesh in block_meshes:
            if block_mesh.mesh is self.mesh:
                continue

            if block_mesh.mesh.id in self.mesh.mesh_contacts.keys():
                patch1 = 'mesh_' + str(self.mesh.id.hex)
                patch2 = 'mesh_' + str(block_mesh.mesh.id.hex)

                logger.info(f'Stitching mesh patches: {patch1} and {patch2}')

                res = subprocess.run(["/bin/bash", "-i", "-c",
                                      f"stitchMesh {patch1} {patch2} -partial -noFunctionObjects"],
                                     capture_output=True,
                                     cwd=self.case_dir,
                                     user='root')
                if res.returncode == 0:
                    output = res.stdout.decode('ascii')
                    logger.info(f"Successfully merged meshes \n\n{output}")
                else:
                    logger.error(f"Error Merging meshes:\n{res.stderr.decode('ascii')}")

        pass

    def run_split_mesh_regions(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        if use_ssh:
            shin, shout, sherr = shell_handler.run_split_mesh_regions(case_dir, ' -cellZonesOnly -overwrite 2>&1 | tee splitMeshRegions.log')
        else:
            logger.info(f'Splitting Mesh Regions....')
            res = subprocess.run(
                ["/bin/bash", "-i", "-c", "splitMeshRegions -cellZonesOnly -overwrite 2>&1 | tee splitMeshRegions.log"],
                capture_output=True,
                cwd=case_dir,
                user='root')
            if res.returncode == 0:
                output = res.stdout.decode('ascii')
                if output.find('FOAM FATAL ERROR') != -1:
                    logger.error(f'Error splitting Mesh Regions:\n\n{output}')
                    raise Exception(f'Error splitting Mesh Regions:\n\n{output}')
                logger.info(f"Successfully splitted Mesh Regions \n\n{output}")
            else:
                logger.error(f"{res.stderr.decode('ascii')}")

        return True

    def run_block_mesh(self, case_dir=None, retry=False, export_block_topology=False, run_parafoam=False):
        """

        :param case_dir:
        :param retry:
        :param export_block_topology:
        :param run_parafoam:
        :return: True
        """
        if case_dir is None:
            case_dir = self.case_dir

        if export_block_topology:
            self.create_block_obj()
            self.obj_to_vtk()

        logger.info(f'Generating mesh....')

        if use_ssh:
            shin, shout, sherr = shell_handler.run_block_mesh(case_dir, '-noFunctionObjects -noClean')
        else:
            res = subprocess.run(["/bin/bash", "-i", "-c", "blockMesh -noFunctionObjects 2>&1 | tee blockMesh.log"],
                                 capture_output=True,
                                 cwd=case_dir,
                                 user='root')
            if res.returncode == 0:
                output = res.stdout.decode('ascii')
                if output.find('FOAM FATAL ERROR') != -1:
                    logger.error(f'Error Creating block mesh:\n\n{output}')
                    if retry:
                        if output.find('Inconsistent number of faces') != -1:
                            items = findall("Inconsistent number of faces.*$", output, MULTILINE)
                            inconsistent_blocks = [int(x) for x in findall(r'\d+', items[0])]
                            self.fix_inconsistent_block_faces(inconsistent_blocks)
                            self.run_block_mesh(case_dir=case_dir, retry=False)
                    else:
                        raise Exception(f'Error Creating block mesh:\n\n{output}')

                elif output.find('command not found') != -1:
                    logger.error(f'Error Creating block mesh:\n\n{output}')
                    raise Exception(f'Error Creating block mesh:\n\n{output}')

                if output.find('FOAM FATAL IO ERROR') != -1:
                    logger.error(f'Error Creating block mesh:\n\n{output}')
                    raise Exception(f'Error Creating block mesh:\n\n{output}')

                logger.info(f"Successfully created block mesh:\n"
                            f"Directory: {case_dir}\n\n "
                            f"{output[output.find('Mesh Information'):]}")

                if run_parafoam:
                    self.run_parafoam(case_dir=case_dir)

            else:
                logger.error(f"{res.stderr.decode('ascii')}")
                raise Exception(f"Error creating block Mesh:\n{res.stderr.decode('ascii')}")
        return True

    def create_mesh_solid(self, name=None, mesh_tool='snappyHexMesh', case_dir=None):

        from ..case.boundary_conditions.face_bcs import FaceBoundaryCondition, Interface

        logger.info(f"Creating mesh solid")

        if case_dir is None:
            case_dir = self.case_dir

        from ..solid import Solid as SolidVolume
        from ..solid import MultiMaterialSolid as MultiMaterialSolidVolume

        if self.cell_zones.__len__() <= 1:
            vol_cls = SolidVolume
        elif self.cell_zones.__len__() > 1:
            vol_cls = MultiMaterialSolidVolume

        if name is None:
            name = self.name

        mesh_solid = vol_cls(name=name,
                             mesh_tool=mesh_tool,
                             id=self.id,
                             case_dir=case_dir)
        mesh_solid.mesh = self
        mesh_solid.mesh.case_dir = mesh_solid.case_dir
        mesh_solid._faces = []

        assigned_faces = {*self.bottom_faces, *self.top_faces, *self.interfaces}

        logger.info(f"Creating boundary faces")

        def get_face_mat(fc, return_all=False):
            if isinstance(fc, Iterable):
                materials = {get_face_mat(x) for x in fc}
                if materials.__len__() > 1:
                    if not return_all:
                        logger.warning(f'Multiple materials for faces {fc} found: {materials}')
                if return_all:
                    list(materials)
                else:
                    return list(materials)[0]

            if fc.blocks.__len__() == 1:
                return list(fc.blocks)[0].cell_zone.material
            else:
                if return_all:
                    raise NotImplementedError
                else:
                    raise Exception(f'Face {fc} belongs to multiple blocks')

        for boundary in self.mesh.boundaries.values():
            # Todo: add interfaces
            assigned_faces.update(boundary.faces)

            face_mat = get_face_mat(boundary.faces)
            face_bc = FaceBoundaryCondition.from_block_mesh_boundary(boundary)

            face = Face(name='face_' + boundary.name,
                        fc_face=FCPart.makeShell([x.fc_face for x in boundary.faces]),
                        block_mesh_faces=boundary.faces,
                        block_mesh_boundary_condition=boundary,
                        boundary_condition=face_bc,
                        solid=mesh_solid,
                        material=face_mat)
            boundary.geo_face = face
            mesh_solid._faces.append(face)
            mesh_solid.features[boundary.name] = face

        wall_patches = []

        logger.info(f"Creating interfaces")

        interfaces_dict = {i: [] for i in list(itertools.combinations(self.mesh.cell_zones, 2))}

        for face in set(self.mesh.faces.values()) - assigned_faces:
            if face.blocks.__len__() == 1:
                face.boundary = wall_patch
                wall_patches.append(face)
            elif face.blocks.__len__() == 2:
                if list(face.blocks)[0].cell_zone == list(face.blocks)[1].cell_zone:
                    continue
                else:
                    for i_interface, i_faces in interfaces_dict.items():
                        if (list(face.blocks)[0].cell_zone in i_interface) and \
                                (list(face.blocks)[1].cell_zone in i_interface):
                            i_faces.append(face)

        interfaces_face_dict = {key: (FCPart.makeShell([x.fc_face for x in i_faces]),
                                      FCPart.makeShell([x.fc_face for x in i_faces])) for key, i_faces in
                                interfaces_dict.items() if i_faces}

        local_wall_patch = BlockMeshBoundary.copy_to_mesh(wall_patch, self.mesh)
        face_bc = FaceBoundaryCondition.from_block_mesh_boundary(local_wall_patch)
        face_mat = get_face_mat(wall_patches)

        face = Face(name='Wall Face',
                    fc_face=FCPart.makeShell([x.fc_face for x in wall_patches]),
                    block_mesh_faces=wall_patches,
                    block_mesh_boundary_condition=local_wall_patch,
                    boundary_condition=face_bc,
                    material=face_mat)
        local_wall_patch.geo_face = face
        mesh_solid._faces.append(face)
        mesh_solid.features['walls'] = face

        internal_interfaces = []
        for interface, i_faces in interfaces_dict.items():
            if i_faces:
                fc_face = interfaces_face_dict[interface][0]
                face1 = Face(fc_face=fc_face,
                             name=f'{interface[0].txt_id}_to_{interface[1].txt_id}',
                             material=interface[0].material)
                mesh_solid._faces.append(face1)
                mesh_solid.features[f'{interface[0].txt_id}_to_{interface[1].txt_id}'] = face
                internal_interfaces.append(face1)

                fc_face = interfaces_face_dict[interface][1]
                face2 = Face(fc_face=fc_face,
                            name=f'{interface[1].txt_id}_to_{interface[0].txt_id}',
                             material=interface[1].material)
                mesh_solid._faces.append(face2)
                mesh_solid.features[f'{interface[1].txt_id}_to_{interface[0].txt_id}'] = face
                internal_interfaces.append(face2)

                face1.boundary_condition = Interface(interface_type='mapped_wall',
                                                     face_1=face1,
                                                     face_2=face2)

                face2.boundary_condition = Interface(interface_type='mapped_wall',
                                                     face_1=face2,
                                                     face_2=face1)

        mesh_solid.features['internal_interfaces'] = internal_interfaces
        interface_block_mesh_bc = BlockMeshBoundary(name='Interface to Layer Material',
                                                    type='patch',
                                                    user_bc=None)

        # add interfaces
        face_mat = get_face_mat([*self.bottom_faces, *self.top_faces, *self.interfaces])
        face = Face(fc_face=FCPart.makeShell([x.fc_face for x in [*self.bottom_faces,
                                                                  *self.top_faces,
                                                                  *self.interfaces]]),
                    block_mesh_faces=[*self.bottom_faces,
                                      *self.top_faces,
                                      *self.interfaces],
                    boundary_condition=Interface(),
                    block_mesh_boundary_condition=interface_block_mesh_bc,
                    material=face_mat)

        interface_block_mesh_bc.geo_face = face
        mesh_solid._faces.append(face)
        mesh_solid.features['interfaces'] = face

        # add default_faces
        mesh_solid.features['interfaces'].block_mesh_boundary_condition = interface_block_mesh_bc

        mesh_solid.generate_solid_from_faces()

        logger.info(f"Successfully created mesh solid")

        return mesh_solid

    def create_shm_mesh_solid(self, name=None, mesh_tool='snappyHexMesh'):

        logger.info(f"Creating snappyHexMesh solid from Block Mesh")

        solid = self.create_mesh_solid(name=name, mesh_tool=mesh_tool)
        solid.base_block_mesh = self
        solid.mesh = None

        # for face in solid.faces:
        #     face.export_stl(f'/simulations/{face.name}.stl')

        geo_dir = os.path.join(solid.case_dir, 'constant', 'triSurface')
        os.makedirs(solid.case_dir, exist_ok=True)

        solid.write_of_geo(geo_dir, separate_interface=False)

        # do not do mesh refinement
        # solid.mesh.castellated_mesh = False

        for face in solid.faces:
            face.surface_mesh_setup.min_refinement_level = 0
            face.surface_mesh_setup.max_refinement_level = 0

        logger.info(f"Successfully created snappyHexMesh solid from Block Mesh")

        return solid

    def __repr__(self):
        return f'BlockMesh {self.id} ({self.name}, ' \
               f'Blocks: {self.mesh.blocks.__len__()}, ' \
               f'Faces: {self.mesh.faces.__len__()}, ' \
               f'Edges: {self.mesh.edges.__len__()}, ' \
               f'Vertices: {self.mesh.vertices.__len__()})'


class PipeMesh(BlockMesh):
    pass


class LayerMesh(BlockMesh):
    pass


class ConstructionMesh(BlockMesh):
    pass


class UpperPipeLayerMesh(BlockMesh):
    pass


class LowerPipeLayerMesh(BlockMesh):
    pass


class PipeLayerMesh(BlockMesh):
    pass


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


def extrude_2d_mesh(mesh, distance, direction, block_name='Extruded Block', mesh_to_add=None, grading=None):

    if grading is None:
        grading = [1, 1, 1]

    if mesh_to_add is None:
        mesh_to_add = Block.current_mesh

    blocks = []
    # create_vertices:
    vertices = np.array([BlockMeshVertex(position=x) for x in mesh.points])

    # create quad blocks:
    if 'quad' in mesh.cells_dict.keys():
        for quad in mesh.cells_dict['quad']:
            v_1 = vertices[quad]
            v_2 = np.array([x + distance * direction for x in v_1])
            new_block = Block(vertices=[*v_1, *v_2],
                              name=block_name,
                              auto_cell_size=True,
                              extruded=False,
                              grading=grading,
                              mesh=mesh_to_add)
            blocks.append(new_block)

    if 'triangle' in mesh.cells_dict.keys():
        for tri in mesh.cells_dict['triangle']:
            v_1 = vertices[[*tri, tri[0]]]
            v_2 = np.array([x + distance * direction for x in v_1])

            new_block = Block(vertices=[*v_1, *v_2],
                              name=block_name,
                              auto_cell_size=True,
                              extruded=False,
                              non_regular=True,
                              grading=grading,
                              mesh=mesh_to_add)

            blocks.append(new_block)
    return blocks


def create_blocks_from_2d_mesh(meshes, reference_face, mesh_to_add):
    """

    :param meshes: meshio-meshes
    :param reference_face:
    :param mesh_to_add:  Mesh to add the blocks
    :return:
    """
    blocks = []

    # calculate vector which translates vertices to lower points of the pipe-block-section:
    # trans_base_vec = np.array(reference_face.normal * (reference_face.tube_diameter / 2 / np.sqrt(2) + reference_face.tube_diameter / 4 ))

    for mesh in meshes:
        # create_vertices:
        mesh.points = mesh.points - np.array(reference_face.normal * (reference_face.tube_diameter / 2 / np.sqrt(2) + reference_face.tube_diameter / 4))
        dist = 2 * (reference_face.tube_diameter / 2 / np.sqrt(2) + reference_face.tube_diameter / 4)
        direction = np.array(reference_face.normal)

        blocks.extend(extrude_2d_mesh(mesh, dist, direction, mesh_to_add))
        # vertices = np.array([BlockMeshVertex(position=x) for x in mesh.points])

        # create quad blocks:
        # if 'quad' in mesh.cells_dict.keys():
        #     for quad in mesh.cells_dict['quad']:
        #
        #         v_1 = vertices[quad]
        #         v_2 = np.array([x + 2 * trans_base_vec for x in v_1])
        #         new_block = Block(vertices=[*v_1, *v_2],
        #                           name=f'Free Block',
        #                           auto_cell_size=True,
        #                           extruded=False)
        #         blocks.append(new_block)
        #
        # if 'triangle' in mesh.cells_dict.keys():
        #     for tri in mesh.cells_dict['triangle']:
        #         v_1 = vertices[[*tri, tri[0]]]
        #         v_2 = np.array([x + 2 * trans_base_vec for x in v_1])
        #
        #         new_block = Block(vertices=[*v_1, *v_2],
        #                           name=f'Free Block',
        #                           auto_cell_size=True,
        #                           extruded=False,
        #                           non_regular=True)
        #
        #         blocks.append(new_block)
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


def add_face_contacts(faces1, faces2, mesh1, mesh2, name1, name2):

    c_ami1 = CyclicAMI(name=name1,
                       faces=faces1,
                       mesh=mesh1,
                       user_bc=SolidCyclicAMI())

    c_ami2 = CyclicAMI(name=name2,
                       faces=faces2,
                       mesh=mesh2,
                       user_bc=SolidCyclicAMI())

    c_ami1.neighbour_patch = c_ami2
    c_ami2.neighbour_patch = c_ami1

    return c_ami1, c_ami2


def add_mesh_contacts(block_meshes):

    mesh_contacts = []

    for block_mesh in block_meshes:
        for other_block_mesh in block_meshes:
            if block_mesh is other_block_mesh:
                continue

            common_face_ids = set(block_mesh.mesh.face_pos.keys()).intersection(
                other_block_mesh.mesh.face_pos.keys())

            if common_face_ids:
                logger.info(f'Found mesh contacts between:\n'
                            f' {block_mesh} and {other_block_mesh}.\n'
                            f'Common face ids: {[block_mesh.mesh.face_pos[x].id for x in common_face_ids]}\n')

                mesh_contacts.append(
                    add_face_contacts([block_mesh.mesh.face_pos[x] for x in common_face_ids],
                                      [other_block_mesh.mesh.face_pos[x] for x in common_face_ids],
                                      block_mesh.mesh,
                                      other_block_mesh.mesh,
                                      f'{block_mesh.mesh.txt_id}_to_{other_block_mesh.mesh.txt_id}',
                                      f'{other_block_mesh.mesh.txt_id}_to_{block_mesh.mesh.txt_id}')
                )

    return mesh_contacts

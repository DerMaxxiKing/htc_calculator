import os
import sys
import uuid
import tempfile
import time
import operator
from io import StringIO
import re
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from math import ceil
from tqdm import tqdm, trange
import functools

from .meshing.block_mesh import create_blocks_from_2d_mesh, Mesh, BlockMesh, \
    CompBlock, NoNormal, bottom_side_patch, top_side_patch, CellZone, wall_patch, extrude_2d_mesh, Block, \
    BlockMeshEdge, BlockMeshFace, PipeMesh, ConstructionMesh, LayerMesh, UpperPipeLayerMesh, LowerPipeLayerMesh, \
    add_face_contacts, PipeLayerMesh


from .meshing.surface_mesh_parameters import default_surface_mesh_parameter
from .face import Face
from .tools import export_objects, add_radius_to_edges, vector_to_np_array, perpendicular_vector, extrude, create_pipe, create_pipe_wire, add_radius_to_edges
from .config import work_dir
from .meshing import meshing_resources
from random import random

from .logger import logger

import FreeCAD
import Part as FCPart
import BOPTools.SplitAPI
from FreeCAD import Base
from Draft import make_fillet
from Arch import makePipe
import Arch


App = FreeCAD


class Solid(object):

    def __init__(self, *args, **kwargs):

        self._mesh = None
        self._case_dir = None
        self._material = None
        self._cell_zones = None
        self._faces = None

        self.id = kwargs.get('id', uuid.uuid4())
        self.type = kwargs.get('type', None)
        # logger.debug(f'initializing solid {self.id}')
        self.name = kwargs.get('name', None)
        self.normal = kwargs.get('normal', None)
        self._fc_solid = kwargs.get('fc_solid', None)
        self.faces = kwargs.get('faces', kwargs.get('_faces', None))
        self.interfaces = kwargs.get('interfaces', [])
        self.features = kwargs.get('features', {})
        self.surface_mesh_setup = kwargs.get('_surface_mesh_setup',
                                             kwargs.get('surface_mesh_setup', default_surface_mesh_parameter))
        self._base_block_mesh = kwargs.get('base_block_mesh', None)
        self.layer = kwargs.get('layer', None)

        self.state = kwargs.get('state', 'solid')
        self._obb = kwargs.get('obb', None)
        self._location_in_mesh = kwargs.get('location_in_mesh', None)

        self.enlarge_obb = kwargs.get('enlarge_obb', 10)     #

        self.mesh = kwargs.get('mesh', None)     #
        self.case_dir = kwargs.get('case_dir', None)     #
        self.mesh_tool = kwargs.get('mesh_tool', 'snappyHexMesh')     # Mesh tool: 'snappyHexMesh' or 'blockMesh'

        self.material = kwargs.get('material', None)
        self.cell_zones = kwargs.get('cell_zones', None)

        self.base_block_mesh_ok = False
        self.mesh_ok = False

    @property
    def case_dir(self):
        if self._case_dir is None:
            self._case_dir = os.path.join(os.path.join(work_dir, self.txt_id))
        return self._case_dir

    @case_dir.setter
    def case_dir(self, value):
        self._case_dir = value

    @property
    def mesh(self):
        if self._mesh is None:
            if self.mesh_tool == 'snappyHexMesh':
                self._mesh = self.create_shm_mesh()
            elif self.mesh_tool == 'blockMesh':
                self._mesh = self.create_block_mesh()
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def cell_zones(self):
        if self._cell_zones is None:
            if self.mesh is not None:
                self._cell_zones = self.mesh.cell_zones
        return self._cell_zones

    @cell_zones.setter
    def cell_zones(self, value):
        self._cell_zones = value

    @property
    def obb(self):
        if self._obb is None:
            self.calc_obb()
        return self._obb

    @property
    def location_in_mesh(self):
        if self._location_in_mesh is None:
            self._location_in_mesh = self.get_location_in_mesh()
        return self._location_in_mesh

    @property
    def locations_in_mesh(self):
        s = '\n\t(\n'
        s += f"\t\t(({self.point_in_mesh[0]} {self.point_in_mesh[1]} {self.point_in_mesh[2]}) {self.txt_id})\n"
        s += '\t)'

        return s

    @property
    def txt_id(self):
        return re.sub('\W+','', 'a' + str(self.id))

    @property
    def faces(self):
        if self._faces is None:
            if self._fc_solid is not None:
                self._faces = [Face(fc_face=x) for x in self._fc_solid.Shape.Faces]
                [setattr(x, 'solid', self) for x in self._faces]
            else:
                self._faces = []
        return self._faces

    @faces.setter
    def faces(self, value):
        self._faces = value
        if self._faces is not None:
            [setattr(x, 'solid', self) for x in self._faces]
            self.generate_solid_from_faces()

    @property
    def fc_solid(self):
        if self._fc_solid is None:
            self.generate_solid_from_faces()
        return self._fc_solid

    @fc_solid.setter
    def fc_solid(self, value):
        self._fc_solid = value

    @property
    def Volume(self):
        return self.fc_solid.Shape.Volume

    @property
    def stl(self):
        return ''.join([x.stl for x in self.faces])

    @property
    def face_names(self):
        return [str(x.id) for x in self.faces]

    @property
    def base_block_mesh(self):
        if self._base_block_mesh is None:
            logger.info(f'Solid: {self.name}\n'
                        f'  ID: {self.txt_id}'
                        f'  Base mesh missing.'
                        f'  Creating base mesh in {self.case_dir}')
            self._base_block_mesh = self.create_base_block_mesh(self.case_dir)
            logger.info(f'Successfully created base block mesh for {self.name} {self.txt_id}')
        return self._base_block_mesh

    @base_block_mesh.setter
    def base_block_mesh(self, value):
        self._base_block_mesh = value

    @staticmethod
    def update_face(face, new_face):

        face.fc_face = new_face.fc_face
        face.normal = None

        # if new_face.name is None:
        #     new_face.name = face.name
        #
        # if face in self.faces:
        #     offset = self.faces.index(face, 0)
        #     self.faces[offset] = new_face
        #
        # if face in self.interfaces:
        #     offset = self.interfaces.index(face, 0)
        #     self.interfaces[offset] = new_face
        #
        # for key in self.features.keys():
        #     feature_faces = self.features[key]
        #     if face in feature_faces:
        #         offset = feature_faces.index(face, 0)
        #         feature_faces[offset] = new_face

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        if self._material is not None:
            if self._material == value:
                return

        self._material = value
        if self._material is not None:
            self.material.solids.add(self)

    def export_stl(self, filename):
        try:
            logger.debug(f'exporting .stl for solid {self.id}')
            new_file = open(filename, 'w')
            new_file.writelines(self.stl)
            new_file.close()
            logger.debug(f'    finished exporting .stl for solid {self.id}')
        except Exception as e:
            logger.error(f'error while exporting .stl for solid {self.id}: {e}')

    def export_step(self, filename):
        logger.debug(f'exporting .stp for solid {self.id}')

        path = os.path.dirname(filename)
        os.makedirs(path, exist_ok=True)

        try:
            from .tools import name_step_faces

            __objs__ = [self.fc_solid]

            fd, tmp_file = tempfile.mkstemp(suffix='.stp')
            os.close(fd)

            FCPart.export(__objs__, tmp_file)
            names = self.face_names
            names_dict = {k: v for k, v in zip(range(names.__len__()), names)}
            name_step_faces(fname=tmp_file, name=names_dict, new_fname=filename)
        except Exception as e:
            logger.error(f'error while exporting .stp for solid {self.id}:\n{e}')
        finally:
            try:
                os.remove(tmp_file)
            except FileNotFoundError as e:
                pass

        logger.debug(f'    finished exporting .stp for solid {self.id}')

    def get_location_in_mesh(self):

        vec0 = self.fc_solid.Shape.CenterOfMass
        vec = vec0

        while not self.fc_solid.Shape.isInside(vec, 0, True):
            vec = Base.Vector(random() * (self.fc_solid.Shape.BoundBox.XMax - self.fc_solid.Shape.BoundBox.XMin) + self.fc_solid.Shape.BoundBox.XMin,
                              random() * (self.fc_solid.Shape.BoundBox.YMax - self.fc_solid.Shape.BoundBox.YMin) + self.fc_solid.Shape.BoundBox.YMin,
                              random() * (self.fc_solid.Shape.BoundBox.ZMax - self.fc_solid.Shape.BoundBox.ZMin) + self.fc_solid.Shape.BoundBox.ZMin)

        return np.array(vec)

    def generate_solid_from_faces(self):

        # logger.debug(f'generating solid from faces: {self.id}')
        # start_time = time.time()

        logger.info(f'Generating solid from faces: {self.name} {self.id}')

        faces = []
        [faces.extend(x.fc_face.Faces) for x in self.faces]
        shell = FCPart.makeShell(faces)
        shell.sewShape()
        shell.fix(1e-7, 1e-7, 1e-7)
        solid = FCPart.Solid(shell)

        if not solid.isClosed():
            logger.error(f'Solid {self.id}: solid is not closed')

        # doc_start_time = time.time()
        doc = App.newDocument()
        __o__ = doc.addObject("Part::Feature", f'{str(self.id)}')
        __o__.Label = f'{str(self.id)}'
        __o__.Shape = solid
        # logger.debug(f'        doc time: {time.time() - doc_start_time} s')

        self.fc_solid = __o__

        logger.debug(f'Successfully finished generation of solid from faces: {self.id}')

        return self.fc_solid

    def generate_faces_from_solid(self, solid):
        pass

    def write_of_geo(self, directory, separate_interface=True, write_internal_interfaces=False):

        logger.info(f'Writing OF geometry for solid: {self.txt_id} to {directory}')

        faces = set(self.faces)

        if separate_interface:
            faces = faces - set(self.interfaces)

        if not write_internal_interfaces:
            if 'internal_interfaces' in self.features.keys():
                faces = faces - set(self.features['internal_interfaces'])

        stl_str = ''.join([x.create_stl_str(of=True) for x in faces])

        # if separate_interface:
        #     stl_str = ''.join([x.create_stl_str(of=True) for x in set(self.faces) - set(self.interfaces)])
        # else:
        #     stl_str = ''.join([x.create_stl_str(of=True) for x in set(self.faces)])

        os.makedirs(directory, exist_ok=True)
        new_file = open(os.path.join(directory, str(self.txt_id) + '.stl'), 'w')
        new_file.writelines(stl_str)
        new_file.close()

        if separate_interface:
            for interface in self.interfaces:
                interface_path = os.path.join(directory, str(interface.txt_id) + '.stl')
                if not os.path.exists(interface_path):
                    new_file = open(interface_path, 'w')
                    new_file.writelines(interface.create_stl_str(of=True))
                    new_file.close()

        logger.info(f'Successfully written OF geometry for solid: {self.txt_id} to {directory}')

    @property
    def shm_geo_entry(self, offset=0, write_internal_interfaces=False):

        local_offset = 4
        offset = offset + local_offset

        solid_faces = set(self.faces)

        if not write_internal_interfaces:
            if 'internal_interfaces' in self.features.keys():
                solid_faces = solid_faces - set(self.features['internal_interfaces'])

        buf = StringIO()

        buf.write(f"{' ' * offset}{str(self.txt_id)}\n")
        buf.write(f"{' ' * offset}{'{'}\n")
        buf.write(f"{' ' * (offset + 4)}type            triSurfaceMesh;\n")
        buf.write(f"{' ' * (offset + 4)}file            \"{str(self.txt_id)}.stl\";\n")
        buf.write(f"{' ' * (offset + 4)}scale            1;\n")
        buf.write(f"{' ' * (offset + 4)}regions\n")
        buf.write(f"{' ' * (offset + 4)}{'{'}\n")

        for face in solid_faces:
            buf.write(f"{' ' * (offset + 8)}{str(face.txt_id)}\t{'{'} name {str(face.txt_id)};\t{'}'}\n")

        buf.write(f"{' ' * (offset + 4)}{'}'}\n")
        buf.write(f"{' ' * offset}{'}'}\n")

        return buf.getvalue()

    @property
    def shm_refinement_entry(self, offset=0, write_internal_interfaces=False):

        local_offset = 4
        offset = offset + local_offset

        faces = set(self.faces)

        if not write_internal_interfaces:
            if 'internal_interfaces' in self.features.keys():
                faces = faces - set(self.features['internal_interfaces'])

        buf = StringIO()

        buf.write(f"{' ' * offset}{str(self.txt_id)}\n")
        buf.write(f"{' ' * offset}{'{'}\n")
        buf.write(f"{' ' * (offset + 4)}level           ({self.surface_mesh_setup.min_refinement_level} {self.surface_mesh_setup.max_refinement_level});\n")
        buf.write(f"{' ' * (offset + 4)}regions\n")
        buf.write(f"{' ' * (offset + 4)}{'{'}\n")

        for face in faces:
            face_level = f"({face.surface_mesh_setup.min_refinement_level} {face.surface_mesh_setup.max_refinement_level})"
            buf.write(f"{' ' * (offset + 8)}{str(face.txt_id)}           {'{'} level {face_level}; patchInfo {'{'} type patch; {'}'} {'}'}\n")

        buf.write(f"{' ' * (offset + 4)}{'}'}\n")
        buf.write(f"{' ' * offset}{'}'}\n")

        return buf.getvalue()

    @property
    def point_in_mesh(self):

        if self.fc_solid is None:
            return None

        solid = self.fc_solid.Shape
        b_box = self.fc_solid.Shape.BoundBox

        location_in_mesh = np.array([b_box.Center.x + np.random.uniform(-0.1 + 0.1),
                                     b_box.Center.y + np.random.uniform(-0.1 + 0.1),
                                     b_box.Center.z + np.random.uniform(-0.1 + 0.1)])

        if solid.isInside(Base.Vector(location_in_mesh), 0, True):
            return location_in_mesh
        else:
            while not solid.isInside(Base.Vector(location_in_mesh), 0, True):
                location_in_mesh = np.array([np.random.uniform(b_box.XMin, b_box.XMax),
                                             np.random.uniform(b_box.YMin, b_box.YMax),
                                             np.random.uniform(b_box.ZMin, b_box.ZMax)])
            return location_in_mesh

    def calc_obb(self):
        from pyobb.obb import OBB
        pts = np.stack([np.array([x.X, x.Y, x.Z]) for x in self.fc_solid.Shape.Vertexes], axis=0)
        hull = ConvexHull(pts)
        hullpts = [pts[i] for i in hull.vertices]

        obb = OBB.build_from_points(hullpts)

        center = np.mean(np.array(obb.points), axis=0)
        vecs = obb.points - center
        unit_vecs = vecs / np.linalg.norm(vecs, axis=0)
        scaled_points = np.array(obb.points) + unit_vecs * self.enlarge_obb

        # obbvec = [FreeCAD.Vector(p)for p in obb.points]
        obbvec = [FreeCAD.Vector(p) for p in scaled_points]

        faces = []
        idx = [[0, 1, 2, 3, 0],
               [4, 5, 6, 7, 4],
               [0, 1, 4, 5, 0],
               [2, 3, 6, 7, 2],
               [1, 2, 7, 4, 1],
               [0, 5, 6, 3, 0]]

        for ix in idx:
            wire = FCPart.makePolygon([obbvec[i] for i in ix])
            faces.append(FCPart.Face(wire))

        shell = FCPart.makeShell(faces)
        FCPart.show(shell)

        p = trimesh.points.PointCloud(pts)
        self._obb = p.bounding_box_oriented

        return self._obb

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

    def is_inside(self, vec: Base.Vector):
        return self.fc_solid.Shape.isInside(vec, 0, True)

    def create_block_mesh(self):
        raise NotImplementedError('create_block_mesh is not implemented yet')

    def create_shm_mesh(self,
                        directory=None,
                        create_block_mesh=True,
                        normal=None,
                        block_mesh_size=200,
                        parallel=True,
                        feature_edges_level=0,
                        refine_normal_direction=True):

        logger.info(f'creating snappyHexMesh for {self.name} {self.id}')

        from .meshing.snappy_hex_mesh import SnappyHexMesh

        if directory is not None:
            self.case_dir = directory

        if 'side_faces' in self.features.keys():
            side_faces = self.features['side_faces']
            if not isinstance(side_faces, list):
                side_faces = [side_faces]
            for surf in side_faces:
                setattr(surf.surface_mesh_setup, 'max_refinement_level', 0)
                setattr(surf.surface_mesh_setup, 'min_refinement_level', 0)

        # for surf in self.faces:
        #     setattr(surf.surface_mesh_setup, 'min_refinement_level', 2)
        #     setattr(surf.surface_mesh_setup, 'max_refinement_level', 2)

        _ = self.base_block_mesh

        if self.base_block_mesh is None:
            self.create_base_block_mesh(directory=self.case_dir,
                                        normal=normal,
                                        cell_size=block_mesh_size,
                                        refine_normal_direction=refine_normal_direction)

        geo_dir = os.path.join(self.case_dir, 'constant', 'triSurface')
        os.makedirs(self.case_dir, exist_ok=True)

        self.write_of_geo(geo_dir, separate_interface=False)

        shm = SnappyHexMesh(name='SHM ' + self.name,
                            assembly=self,
                            allow_free_standing_zone_faces=False,
                            feature_edges_level=feature_edges_level,
                            case_dir=self.case_dir)
        # shm.create_surface_feature_extract_dict(case_dir=self.case_dir)
        # shm.run_surface_feature_extract(case_dir=self.case_dir)
        # shm.write_snappy_hex_mesh(case_dir=self.case_dir)
        # shm.run(case_dir=self.case_dir, parallel=parallel)

        shm.cell_zones = self.base_block_mesh.mesh.cell_zones

        logger.info(f'Successfully created snappyHexMesh for {self.name} {self.id}')

        return shm

    def create_base_block_mesh(self,
                               directory=None,
                               normal=None,
                               cell_size=200,
                               scale=10,
                               refine_normal_direction=True):

        if directory is not None:
            self.case_dir = directory

        logger.info(f'Creating block mesh for solid: {self.txt_id}')

        from .meshing.block_mesh import BlockMesh, Mesh, Block, BlockMeshVertex
        block_mesh = BlockMesh(case_dir=directory,
                               name='Block Mesh ' + self.txt_id,
                               mesh=Mesh(name=self.txt_id,
                                         default_cell_size=cell_size))

        block_mesh.mesh.activate()

        block_vertices = np.array([
            [self.obb.bounds[0, 0], self.obb.bounds[0, 1], self.obb.bounds[0, 2]],
            [self.obb.bounds[0, 0], self.obb.bounds[1, 1], self.obb.bounds[0, 2]],
            [self.obb.bounds[1, 0], self.obb.bounds[1, 1], self.obb.bounds[0, 2]],
            [self.obb.bounds[1, 0], self.obb.bounds[0, 1], self.obb.bounds[0, 2]],
            [self.obb.bounds[0, 0], self.obb.bounds[0, 1], self.obb.bounds[1, 2]],
            [self.obb.bounds[0, 0], self.obb.bounds[1, 1], self.obb.bounds[1, 2]],
            [self.obb.bounds[1, 0], self.obb.bounds[1, 1], self.obb.bounds[1, 2]],
            [self.obb.bounds[1, 0], self.obb.bounds[0, 1], self.obb.bounds[1, 2]]])

        center = np.mean(block_vertices, axis=0)

        # translate block vertices
        block_vertices = block_vertices + (block_vertices - center) / np.linalg.norm(block_vertices - center, axis=1)[:, None] * scale

        vertices = [BlockMeshVertex(position=x) for x in block_vertices]

        if self.material is not None:
            cell_zone = CellZone(material=self.material,
                                 mesh=block_mesh.mesh)
        else:
            cell_zone = CellZone(material=None,
                                 mesh=block_mesh.mesh)

        new_block = Block(vertices=vertices,
                          name=self.txt_id + 'base block',
                          auto_cell_size=True,
                          extruded=False,
                          grading=[1, 1, 1],
                          mesh=block_mesh.mesh,
                          cell_zone=cell_zone)

        num_cells = [ceil(new_block.edge0.length / cell_size),
                     ceil(new_block.edge3.length / cell_size),
                     ceil(new_block.edge8.length / cell_size)
                     ]

        if refine_normal_direction:
            if normal is not None:
                if np.allclose(new_block.edge0.direction, normal) or np.allclose(new_block.edge0.direction, -normal):
                    num_cells[0] = ceil(new_block.edge0.length / 50)
                if np.allclose(new_block.edge3.direction, normal) or np.allclose(new_block.edge3.direction, -normal):
                    num_cells[1] = ceil(new_block.edge3.length / 50)
                if np.allclose(new_block.edge8.direction, normal) or np.allclose(new_block.edge8.direction, -normal):
                    num_cells[2] = ceil(new_block.edge8.length / 50)

        for i, num in enumerate(num_cells):
            if num < 3:
                num_cells[i] = 3

        new_block.num_cells = num_cells

        # self.run_base_block_mesh(base_block_mesh=block_mesh,
        #                          case_dir=self.case_dir)

        return block_mesh

    def run_base_block_mesh(self,
                            base_block_mesh=None,
                            case_dir=None):

        logger.info(f'Running blockMesh for base mesh of {self.name} {self.id}')

        if case_dir is None:
            case_dir = self.case_dir

        if base_block_mesh is None:
            base_block_mesh = self.base_block_mesh

        os.makedirs(case_dir, exist_ok=True)
        base_block_mesh.init_case()
        base_block_mesh.run_block_mesh(run_parafoam=True)
        logger.info(f'Successfully created base block mesh for solid: {self.txt_id} in {case_dir}')

        logger.info(f'Checking base mesh')
        base_block_mesh.run_check_mesh(case_dir=case_dir)

        self.base_block_mesh_ok = True
        return base_block_mesh

    def common(self, other):
        """
        Find common with other solid
        :param other:
        :return: common faces
        """

        return self.fc_solid.Shape.Shells[0].common(other.fc_solid.Shape.Shells[0])

    def run_meshing(self,
                    parallel=True,
                    init_case=True,
                    split_mesh_regions=True):

        logger.info(f'Running meshing for solid {self.name} in {self.case_dir}')
        st_time = time.time()

        from .meshing.snappy_hex_mesh import SnappyHexMesh

        if self.mesh is None:
            raise Exception(f'No mesh found for solid {self.name}, {self.txt_id}')

        if isinstance(self.mesh, SnappyHexMesh):
            if not self.base_block_mesh_ok:
                self.run_base_block_mesh()
            self.mesh.create_surface_feature_extract_dict(case_dir=self.case_dir)
            self.mesh.run_surface_feature_extract(case_dir=self.case_dir)
            self.mesh.write_snappy_hex_mesh(case_dir=self.case_dir)
            self.mesh.run(case_dir=self.case_dir, parallel=parallel)

        elif isinstance(self.mesh, BlockMesh):
            if init_case:
                self.mesh.init_case(case_dir=self.case_dir)
            self.mesh.run_block_mesh(case_dir=self.case_dir)
            if split_mesh_regions:
                self.mesh.run_split_mesh_regions(case_dir=self.case_dir)
            # self.mesh.run_check_mesh(case_dir=self.case_dir)
            # self.mesh.run_parafoam(case_dir=self.case_dir)

        et_time = time.time()
        logger.info(f'Successfully ran meshing for solid {self.name} in {self.case_dir}\n'
                    f'Mesh creation took {et_time - st_time} seconds')

        logger.info(f"Checking mesh for {self.name}, in {self.case_dir}")
        self.mesh.run_check_mesh(case_dir=self.case_dir)
        logger.info(f'Successfully checked mesh for solid {self.name} in {self.case_dir}')

        self.mesh.run_parafoam(case_dir=self.case_dir)

        self.mesh_ok = True

    def run_check_mesh(self):
        logger.info(f'Checking mesh for {self.name} in {self.case_dir}')
        self.mesh.run_check_mesh(case_dir=self.case_dir)
        logger.info(f'Successfully checked mesh for {self.name} in {self.case_dir}')

    def __repr__(self):
        rep = f'Solid {self.name} {self.id} {self.Volume}'
        return rep


class MultiMaterialSolid(Solid):

    def __init__(self, *args, **kwargs):
        Solid.__init__(self, *args, **kwargs)


class PipeSolid(Solid):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)

        self.reference_face = kwargs.get('reference_face', None)
        self.reference_edge_id = kwargs.get('reference_edge_id', 0)

        self.tube_diameter = kwargs.get('tube_diameter', 0.02)
        self.tube_inner_diameter = kwargs.get('tube_inner_diameter', 0.016)
        self.tube_distance = kwargs.get('tube_distance', 0.50)
        self.tube_side_1_offset = kwargs.get('tube_side_1_offset', 0.085)
        self.tube_edge_distance = kwargs.get('tube_edge_distance', 0.50)
        self.bending_radius = kwargs.get('bending_radius', 0.05)

        self.reference_edge = None
        self._initial_pipe_wire = None  # pipe wire without radius
        self._pipe_wire = None          # pipe wire with radius
        self._horizontal_lines = None
        self._comp_solid = None
        self._mesh_solid = None

        self.pipe_mesh = PipeMesh(name='Block Mesh ' + 'pipe_layer_mesh',
                                  mesh=Mesh(name='pipe_layer_mesh'))

        self.comp_solid = kwargs.get('comp_solid', None)
        self.mesh_solid = kwargs.get('mesh_solid', None)

        integrate_pipe = kwargs.get('integrate_pipe', True)

        if integrate_pipe:
            try:
                pipe = self.generate_solid()

                kwargs['fc_solid'] = pipe.fc_solid
                kwargs['faces'] = pipe.faces
                kwargs['interfaces'] = pipe.interfaces
                kwargs['features'] = pipe.features
                kwargs['type'] = 'pipe'

            except Exception as e:
                logger.error(f'Error generating pipe solid for {self.name} {self.id}')

        Solid.__init__(self, *args, **kwargs)

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = self.create_mesh()
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def initial_pipe_wire(self):
        if self._initial_pipe_wire is None:
            self._initial_pipe_wire, _ = self.generate_initial_pipe_wire()
        return self._initial_pipe_wire

    @property
    def pipe_wire(self):
        if self._pipe_wire is None:
            self.pipe_wire = self.generate_pipe_wire()
        return self._pipe_wire

    @pipe_wire.setter
    def pipe_wire(self, value):
        self._pipe_wire = value

    @property
    def pipe_length(self):
        return self.pipe_wire.Length

    @property
    def comp_solid(self):
        if self._comp_solid is None:
            self._comp_solid = CompBlock(name='Pipe Blocks', blocks=self.pipe_mesh.mesh.blocks)
        return self._comp_solid

    @comp_solid.setter
    def comp_solid(self, value):
        self._comp_solid = value

    @property
    def mesh_solid(self):
        if self._mesh_solid is None:
            self._mesh_solid = self.create_mesh_solid()
        return self._mesh_solid

    @mesh_solid.setter
    def mesh_solid(self, value):
        self._mesh_solid = value

    def generate_initial_pipe_wire(self):
        pipe_wire, self._horizontal_lines = create_pipe_wire(self.reference_face.reference_face,
                                                             self.reference_edge_id,
                                                             self.tube_distance,
                                                             self.tube_edge_distance,
                                                             self.bending_radius,
                                                             self.tube_diameter
                                                             )

        reference_face = self.reference_face.reference_face
        self.reference_edge = reference_face.Edges[self.reference_edge_id]
        normal = self.reference_face.get_normal(Base.Vector(self.reference_edge.Vertexes[0].X,
                                                            self.reference_edge.Vertexes[0].Y,
                                                            self.reference_edge.Vertexes[0].Z))
        pipe_wire.Placement.move(self.reference_face.layer_dir * normal *
                                 (- self.reference_face.component_construction.side_1_offset +
                                  self.reference_face.tube_side_1_offset))

        return pipe_wire, self._horizontal_lines

    def generate_pipe_wire(self):

        return add_radius_to_edges(self.initial_pipe_wire, self.bending_radius)

    def generate_solid(self):

        logger.info(f'Creating solid for pipe {self.name} {self.id}')

        hull = self.reference_face.plain_reference_face_solid.assembly.hull

        pipe_shape = create_pipe(self.pipe_wire.Edges, self.tube_inner_diameter, self.reference_face.normal)
        initial_tube_wall = create_pipe(self.pipe_wire.Edges, self.tube_diameter, self.reference_face.normal)
        tube_wall = initial_tube_wall.cut(pipe_shape).common(hull.fc_solid.Shape)

        # export_objects([tube_wall, pipe_shape], '/tmp/tube_wall.FCStd')

        inlet_outlet = hull.fc_solid.Shape.Shells[0].common(pipe_shape)

        if inlet_outlet.SubShapes.__len__() == 2:
            inlet = Face(fc_face=inlet_outlet.SubShapes[0].removeSplitter(),
                         name='Pipe_Inlet')
            outlet = Face(fc_face=inlet_outlet.SubShapes[1].removeSplitter(),
                          name='Pipe_Outlet')
        else:
            raise Exception('can not identify inlet and outlet')

        pipe_faces = BOPTools.SplitAPI.slice(pipe_shape.Shells[0], hull.fc_solid.Shape.Shells, "Split",
                                             1e-3)

        faces = [*inlet.fc_face.Faces, *outlet.fc_face.Faces, *pipe_faces.Faces]
        shell = FCPart.makeShell(faces)
        shell.sewShape()
        shell.fix(1e-3, 1e-3, 1e-3)
        pipe = FCPart.Solid(shell)

        layer_pipe_interfaces = []

        logger.info(f'Updating layer solids reference face {self.reference_face.name} {self.reference_face.id}')

        for i, solid in enumerate(self.reference_face.solids.values()):

            logger.info(f'Updating layer solid {i+1} of {self.reference_face.solids.values().__len__()}: '
                        f'{solid.name} {solid.id}')

            common = solid.fc_solid.Shape.common(initial_tube_wall)
            if common.Faces:
                new_faces = []
                for face in solid.faces:
                    new_face = Face(fc_face=face.fc_face.cut(initial_tube_wall))
                    new_faces.append(new_face)
                    solid.update_face(face, new_face)
                solid.generate_solid_from_faces()
                # export_objects([solid.fc_solid.Shape], '/tmp/solid.FCStd')
                # export_objects([common], '/tmp/common.FCStd')

                pipe_interface = Face(fc_face=solid.fc_solid.Shape.common(initial_tube_wall).Shells[0],
                                      name=f'Pipe interface',
                                      linear_deflection=0.5,
                                      angular_deflection=0.5)
                layer_pipe_interfaces.append(pipe_interface)
                solid.faces.append(pipe_interface)
                solid.interfaces.append(pipe_interface)
                solid.generate_solid_from_faces()

                solid.features['pipe_faces'] = pipe_interface

        # generate pipe solid
        pipe_solid = Solid(faces=[inlet, outlet, *[Face(fc_face=x) for x in pipe_faces.Faces]],
                           name='PipeSolid')
        pipe_solid.generate_solid_from_faces()

        pipe_wall_solid = Solid(faces=[Face(fc_face=x) for x in tube_wall.Solids[0].Faces],
                                name='PipeWallSolid')
        pipe_wall_solid.generate_solid_from_faces()

        export_objects(tube_wall, '/tmp/pipe_wall_solid.FCStd')
        export_objects(pipe_solid.fc_solid.Shape, '/tmp/pipe_solid.FCStd')

        pipe_solid.features['inlet'] = inlet
        pipe_solid.features['outlet'] = outlet
        pipe_solid.features['layer_interfaces'] = layer_pipe_interfaces
        pipe_solid.interfaces = layer_pipe_interfaces

        logger.info(f'Updating assembly of reference face {self.reference_face.name} {self.reference_face.id}')

        self.reference_face.assembly.solids.append(pipe_solid)
        self.reference_face.assembly.solids.append(pipe_wall_solid)
        self.reference_face.assembly.features['pipe'] = pipe_solid
        self.reference_face.assembly.features['pipe_wall_solid'] = pipe_wall_solid

        self.reference_face.assembly.faces.extend([inlet, outlet, *layer_pipe_interfaces])
        self.reference_face.assembly.interfaces.extend(layer_pipe_interfaces)

        logger.info(f'Successfully created solid for pipe {self.name} {self.id}')

        return pipe_solid

    def export_tube_wire(self, filename):

        __objs__ = [self.pipe_wire]

        try:
            FCPart.export(__objs__, filename)
        except Exception as e:
            print(e)
            raise e

    def print_info(self):
        print(f'\nPipe info:\n'
              f'----------------------------------\n\n'
              f'Tube diameter: {self.tube_diameter} mm\n'
              f'Distance between tubes: {self.tube_distance} mm\n'
              f'Bending radius: {self.bending_radius} mm\n'
              f'Tube length: {self.pipe_length / 1000} m\n\n')

    def create_mesh(self):

        self.pipe_mesh.mesh.activate()

        logger.info(f'Generation o-grid blocks for pipe...')

        wire = self.pipe_wire
        blocks = []

        for i, edge in enumerate(tqdm(wire.Edges, desc='creating o-grid', colour="green")):

            if i == 0:
                outer_pipe = False
                inlet = True
                outlet = False
            elif i == wire.Edges.__len__() - 1:
                outer_pipe = False
                inlet = False
                outlet = True
            else:
                outer_pipe = True
                inlet = False
                outlet = False

            # logger.info(f'creating block {i} of {wire.Edges.__len__()}')

            new_blocks = self.reference_face.pipe_section.create_block(edge=edge,
                                                                       face_normal=self.normal,
                                                                       tube_inner_diameter=self.tube_inner_diameter,
                                                                       tube_diameter=self.tube_diameter,
                                                                       outer_pipe=outer_pipe,
                                                                       inlet=inlet,
                                                                       outlet=outlet)
            blocks.append(new_blocks)

            if outer_pipe:

                def get_side_faces(items):
                    side_faces = []
                    for block_id, face_ids in items.items():
                        for face_id in face_ids:
                            side_faces.append(new_blocks[block_id].faces[face_id])
                    return side_faces

                self.pipe_mesh.top_faces.extend(get_side_faces(self.reference_face.pipe_section.top_side))
                self.pipe_mesh.bottom_faces.extend(get_side_faces(self.reference_face.pipe_section.bottom_side))
                self.pipe_mesh.interfaces.extend(get_side_faces(self.reference_face.pipe_section.interface_side))

            logger.info(f'Successfully generated o-grid blocks for pipe\n\n')
            block_list = functools.reduce(operator.iconcat, blocks, [])
            # pipe_comp_block = CompBlock(name='Pipe Blocks',
            #                             blocks=block_list)

            self.reference_face.update_cell_zone(blocks=block_list)

            self.pipe_mesh.init_case()
            self.pipe_mesh.run_block_mesh()
            self.pipe_mesh.run_split_mesh_regions()
            self.pipe_mesh.run_check_mesh()
            self.pipe_mesh.run_parafoam()

            # Block.save_fcstd('/tmp/blocks.FCStd')
            # export_objects([pipe_comp_block.fc_solid], '/tmp/pipe_comp_block.FCStd')
            return self.pipe_mesh

    def __repr__(self):
        rep = f'Solid {self.name} {self.id}'
        return rep

    def create_mesh_solid(self):

        mesh_solid = None
        self.comp_solid
        return mesh_solid

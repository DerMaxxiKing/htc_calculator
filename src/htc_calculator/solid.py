import os
import sys
import uuid
import tempfile
import time
from io import StringIO
import re
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from .meshing.surface_mesh_parameters import default_surface_mesh_parameter
from .face import Face
from .tools import export_objects, add_radius_to_edges, vector_to_np_array, perpendicular_vector, extrude, create_pipe, create_pipe_wire, add_radius_to_edges
from .config import work_dir
from .meshing import meshing_resources

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
        self.id = kwargs.get('id', uuid.uuid4())
        self.type = kwargs.get('type', None)
        self.base_directory = kwargs.get('base_directory', os.path.join(work_dir, self.txt_id))
        # logger.debug(f'initializing solid {self.id}')
        self.name = kwargs.get('name', None)
        self.normal = kwargs.get('normal', None)
        self._fc_solid = kwargs.get('fc_solid', None)
        self._faces = kwargs.get('faces', kwargs.get('_faces', None))
        self.interfaces = kwargs.get('interfaces', [])
        self.features = kwargs.get('features', {})
        self.surface_mesh_setup = kwargs.get('_surface_mesh_setup',
                                             kwargs.get('surface_mesh_setup', default_surface_mesh_parameter))
        self._base_block_mesh = kwargs.get('base_block_mesh', None)
        self.layer = kwargs.get('layer', None)

        self.state = kwargs.get('state', 'solid')
        self._obb = kwargs.get('obb', None)

    @property
    def obb(self):
        if self._obb is None:
            self.calc_obb()
        return self._obb

    @property
    def txt_id(self):
        return re.sub('\W+','', 'a' + str(self.id))

    @property
    def faces(self):
        if self._faces is None:
            if self._fc_solid is not None:
                self._faces = [Face(fc_face=x) for x in self._fc_solid.Shape.Faces]
            else:
                self._faces = []
        return self._faces

    @faces.setter
    def faces(self, value):
        self._faces = value
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
            self._base_block_mesh = self.create_base_block_mesh(self.base_directory)

    def update_face(self, face, new_face):

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

    def generate_solid_from_faces(self):

        # logger.debug(f'generating solid from faces: {self.id}')
        start_time = time.time()

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

        # logger.debug(f'    finished generation of solid from faces: {self.id} in {time.time() - start_time} s')

        return self.fc_solid

    def generate_faces_from_solid(self, solid):
        pass

    def write_of_geo(self, directory, separate_interface=True):

        if separate_interface:
            stl_str = ''.join([x.create_stl_str(of=True) for x in set(self.faces) - set(self.interfaces)])
        else:
            stl_str = ''.join([x.create_stl_str(of=True) for x in set(self.faces)])

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

    @property
    def shm_geo_entry(self, offset=0):

        local_offset = 4
        offset = offset + local_offset

        solid_faces = set(self.faces) - set(self.interfaces)

        buf = StringIO()

        buf.write(f"{' ' * offset}{str(self.txt_id)}\n")
        buf.write(f"{' ' * offset}{'{'}\n")
        buf.write(f"{' ' * (offset + 4)}type            triSurfaceMesh;\n")
        buf.write(f"{' ' * (offset + 4)}file            \"{str(self.txt_id)}.stl\";\n")
        buf.write(f"{' ' * (offset + 4)}regions\n")
        buf.write(f"{' ' * (offset + 4)}{'{'}\n")

        for face in solid_faces:
            buf.write(f"{' ' * (offset + 8)}{str(face.txt_id)}\t{'{'} name {str(face.txt_id)};\t{'}'}\n")

        buf.write(f"{' ' * (offset + 4)}{'}'}\n")
        buf.write(f"{' ' * offset}{'}'}\n")

        return buf.getvalue()

    @property
    def shm_refinement_entry(self, offset=0):

        local_offset = 4
        offset = offset + local_offset

        hull_faces = set(self.faces) - set(self.interfaces)

        buf = StringIO()

        buf.write(f"{' ' * offset}{str(self.txt_id)}\n")
        buf.write(f"{' ' * offset}{'{'}\n")
        buf.write(f"{' ' * (offset + 4)}level           ({self.surface_mesh_setup.min_refinement_level} {self.surface_mesh_setup.max_refinement_level});\n")
        buf.write(f"{' ' * (offset + 4)}regions\n")
        buf.write(f"{' ' * (offset + 4)}{'{'}\n")

        for face in hull_faces:
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
        obbvec = [FreeCAD.Vector(p)for p in obb.points]

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

    def create_shm_mesh(self, directory=None):

        if directory is None:
            directory = self.base_directory

        self.create_base_block_mesh(directory=directory)

        geo_dir = os.path.join(directory, 'constant', 'geometry')
        os.makedirs(directory, exist_ok=True)
        self.write_of_geo(geo_dir, separate_interface=False)



    def create_base_block_mesh(self, directory=None):

        if directory is None:
            directory = self.base_directory

        logger.info(f'Creating block mesh for solid: {self.txt_id}')

        from .meshing.block_mesh import BlockMesh, Mesh, Block, BlockMeshVertex
        block_mesh = BlockMesh(case_dir=directory,
                               name='Block Mesh ' + self.txt_id,
                               mesh=Mesh(name=self.txt_id))

        block_mesh.mesh.activate()

        vertices = [BlockMeshVertex(position=x) for x in self.obb.vertices]

        new_block = Block(vertices=vertices,
                          name=self.txt_id + 'base block',
                          auto_cell_size=True,
                          extruded=False,
                          grading=[1, 1, 1],
                          mesh=block_mesh.mesh)

        os.makedirs(directory, exist_ok=True)
        block_mesh.init_case()
        block_mesh.run_block_mesh()
        logger.info(f'Successfully created block mesh for solid: {self.txt_id}')

    def __repr__(self):
        rep = f'Solid {self.name} {self.id} {self.Volume}'
        return rep




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

        # from .tools import project_point_on_line
        #
        # reference_face = self.reference_face.reference_face
        #
        # self.reference_edge = reference_face.Edges[self.reference_edge_id]
        # normal = self.reference_face.get_normal(Base.Vector(self.reference_edge.Vertexes[0].X,
        #                                                     self.reference_edge.Vertexes[0].Y,
        #                                                     self.reference_edge.Vertexes[0].Z))
        # tube_main_dir = self.reference_edge.Curve.Direction.cross(normal)
        #
        # offset = -self.tube_edge_distance
        # wires = []
        # offset_possible = True
        #
        # while offset_possible:
        #     try:
        #         wire = reference_face.OuterWire.makeOffset2D(offset, join=1, openResult=False, intersection=False)
        #
        #         # check if another is possible (return wire)
        #         try:
        #             reference_face.OuterWire.makeOffset2D(offset - self.tube_distance, join=1, openResult=False,
        #                                                   intersection=False)
        #             wires.append(wire)
        #             offset = offset - 2 * self.tube_distance
        #         except Exception as e:
        #             logger.debug(f'no further wire generation possible{e}')
        #             offset_possible = False
        #     except Exception as e:
        #         logger.debug(f'no further wire generation possible{e}')
        #         offset_possible = False
        #
        # # check if last circle is possible:
        # # try:
        # #     last_wire = reference_face.OuterWire.makeOffset2D(offset, join=1, openResult=False, intersection=False)
        # # except Exception as e:
        # #     last_wire = None
        # #     logger.debug(f'no further wire generation possible{e}')
        #
        # last_wire = None
        #
        # # export_objects([*wires, last_wire], '/tmp/initial_wires2.FCStd')
        #
        # pipe_edges = []
        #
        # if (reference_face.Edges.__len__() - 1) >= (self.reference_edge_id + 1):
        #     start_edge_id = self.reference_edge_id + 1
        # else:
        #     start_edge_id = 0
        #
        # # create inflow
        # V1 = wires[0].Edges[start_edge_id].Vertex1.Point + tube_main_dir * 2 * self.tube_edge_distance
        # V2 = wires[0].Edges[start_edge_id].Vertex1.Point + tube_main_dir * 1 * self.tube_edge_distance
        # pipe_edges.append(FCPart.LineSegment(V1, V2).toShape())
        #
        # V1 = wires[0].Edges[start_edge_id].Vertex1.Point + tube_main_dir * 1 * self.tube_edge_distance
        # V2 = wires[0].Edges[start_edge_id].Vertex1.Point
        # pipe_edges.append(FCPart.LineSegment(V1, V2).toShape())
        #
        # # add edges except the start_edge
        # pipe_edges.extend(wires[0].Edges[self.reference_edge_id + 1:])
        # pipe_edges.extend(wires[0].Edges[0:self.reference_edge_id:])
        #
        # # modify reference_edge_id edge
        # p1 = wires[0].Edges[self.reference_edge_id].Vertex1.Point
        # p2 = wires[0].Edges[self.reference_edge_id].Vertex2.Point
        # v1 = p1
        # v2 = p2 - 2 * (p2 - p1).normalize() * self.tube_distance
        # pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())
        #
        # # export_objects(pipe_edges, '/tmp/pipe_edges7.FCStd')
        # # export_wire([self.reference_face.OuterWire, *pipe_edges])
        # # export_objects(wires, '/tmp/wires.FCStd')
        #
        # i = 1
        # while i <= (wires.__len__() - 1):
        #     # create connection from previous wire to current wire:
        #     dir1 = (wires[i].Edges[start_edge_id].Vertex1.Point - pipe_edges[-1].Vertex2.Point).normalize()
        #     dir2 = (wires[i].Edges[start_edge_id].Vertexes[1].Point - pipe_edges[-1].Vertex2.Point).normalize()
        #
        #     if sum(abs(abs(dir1) - abs(dir2))) < 1e-10:
        #         # export_objects([wires[i].Edges[start_edge_id]], '/tmp/pipe_edges6.FCStd')
        #         pipe_edges.append(FCPart.LineSegment(pipe_edges[-1].Vertex2.Point,
        #                                              wires[i].Edges[start_edge_id].Vertexes[0].Point).toShape())
        #         pipe_edges.append(wires[i].Edges[start_edge_id])
        #         # pipe_edges.append(FCPart.LineSegment(pipe_edges[-1].Vertex2.Point,
        #         #                                      wires[i].Edges[start_edge_id].Vertexes[1].Point).toShape())
        #     else:
        #         projected_point = FreeCAD.Base.Vector(
        #             project_point_on_line(point=wires[i].Edges[start_edge_id].Vertex1.Point, line=pipe_edges[-1]))
        #
        #         # change_previous end edge:
        #         pipe_edges[-1] = FCPart.LineSegment(pipe_edges[-1].Vertex1.Point, projected_point).toShape()
        #
        #         pipe_edges.append(FCPart.LineSegment(wires[i].Edges[start_edge_id].Vertex1.Point,
        #                                              projected_point).toShape())
        #         pipe_edges.append(wires[i].Edges[start_edge_id])
        #
        #     # #pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())
        #     #
        #     # pipe_edges.append(FCPart.LineSegment(pipe_edges[-1].Vertex2.Point,
        #     #                                      wires[i].Edges[start_edge_id].Vertexes[1].Point).toShape())
        #
        #     # add other edges except start_edge
        #     pipe_edges.extend(wires[i].Edges[self.reference_edge_id + 2:])
        #     pipe_edges.extend(wires[i].Edges[0:self.reference_edge_id:])
        #
        #     # modify reference_edge_id edge
        #     p1 = wires[i].Edges[self.reference_edge_id].Vertex1.Point
        #     p2 = wires[i].Edges[self.reference_edge_id].Vertex2.Point
        #     v1 = p1
        #     v2 = p2 - 2 * (p2 - p1).normalize() * self.tube_distance
        #     pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())
        #
        #     i = i + 1
        #
        # # export_objects(pipe_edges, '/tmp/all_edges_io4.FCStd')
        # # export_objects(wire_out_edges, '/tmp/all_edges_io2.FCStd')
        # # export_objects([wire_in], '/tmp/wire_in.FCStd')
        # # export_objects([wire_out], '/tmp/wire_out9.FCStd')
        # # export_objects([last_wire], '/tmp/last_wire.FCStd')
        # # export_objects([wire_in, wire_out], '/tmp/wires.FCStd')
        #
        # # create
        # succeeded = False
        # while not succeeded:
        #     wire_in = FCPart.Wire(pipe_edges)
        #     wire_out = wire_in.makeOffset2D(-self.tube_distance,
        #                                     join=0,
        #                                     openResult=True,
        #                                     intersection=True,
        #                                     fill=False)
        #     # wire_in.distToShape(wire_out)
        #
        #     if last_wire is not None:
        #         wire_in_edges = pipe_edges
        #
        #         dir1 = (last_wire.Edges[start_edge_id].Vertex1.Point - pipe_edges[-1].Vertex2.Point).normalize()
        #         dir2 = (last_wire.Edges[start_edge_id].Vertexes[1].Point - pipe_edges[-1].Vertex2.Point).normalize()
        #
        #         if sum(abs(abs(dir1) - abs(dir2))) < 1e-10:
        #             wire_in_edges.append(FCPart.LineSegment(wire_in_edges[-1].Vertex2.Point,
        #                                                     last_wire.Edges[start_edge_id].Vertexes[1].Point).toShape())
        #         else:
        #             projected_point = FreeCAD.Base.Vector(
        #                 project_point_on_line(point=last_wire.Edges[start_edge_id].Vertex1.Point, line=wire_in_edges[-1]))
        #
        #             # change_previous end edge:
        #             wire_in_edges[-1] = FCPart.LineSegment(wire_in_edges[-1].Vertex1.Point, projected_point).toShape()
        #
        #             wire_in_edges.append(FCPart.LineSegment(last_wire.Edges[start_edge_id].Vertex1.Point,
        #                                                  projected_point).toShape())
        #             wire_in_edges.append(wires[i].Edges[start_edge_id])
        #
        #         last_wire_edges = last_wire.Edges
        #         start_edge = last_wire.Edges[start_edge_id - 1]
        #         # del last_wire_edges[start_edge_id - 1]
        #         # del last_wire_edges[start_edge_id - 1]
        #         last_wire_edges.append(start_edge.split(start_edge.LastParameter - self.tube_distance).SubShapes[0])
        #         # wire_in_edges.extend(last_wire_edges)
        #         wire_in_edges.extend(last_wire_edges[self.reference_edge_id + 1:])
        #         wire_in_edges.extend(last_wire_edges[0:self.reference_edge_id:])
        #         wire_in = FCPart.Wire(wire_in_edges)
        #
        #         # cut last wire out edge:
        #         wire_out_edges = wire_out.Edges
        #         wire_out_edges[-1] = wire_out_edges[-1].split(wire_out_edges[-1].LastParameter -
        #                                                       self.tube_distance).SubShapes[0]
        #
        #         wire_out = FCPart.Wire(wire_out_edges)
        #
        #     # create connection between wire_in and wire_out:
        #     v1 = wire_in.Edges[-1].Vertex2.Point
        #     v2 = wire_out.Edges[-1].Vertex2.Point
        #     connection_edge = FCPart.LineSegment(v1, v2).toShape()
        #
        #     edges_out = wire_out.Edges
        #     edges_out.reverse()
        #
        #     all_edges = [*wire_in.Edges, connection_edge, *edges_out]
        #
        #     try:
        #         FCPart.Wire(all_edges, intersection=False)
        #         succeeded = True
        #     except Exception as e:
        #         succeeded = False
        #         del pipe_edges[-1]
        #
        # if self.bending_radius is not None:
        #     all_edges = add_radius_to_edges(FCPart.Wire(all_edges).OrderedEdges, self.bending_radius)
        #
        # pipe_wire = FCPart.Wire(all_edges)
        #
        # pipe_wire.Placement.move(
        #     self.reference_face.layer_dir * normal * (- self.reference_face.component_construction.side_1_offset + self.reference_face.tube_side_1_offset))
        #
        # pipe_wire.Edges[0].reverse()
        #
        # return FCPart.Wire(pipe_wire.OrderedEdges)

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

        for i, solid in enumerate(self.reference_face.assembly.solids):

            logger.info(f'Updating layer solid {i+1} of {self.reference_face.assembly.solids.__len__()}: '
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

    def __repr__(self):
        rep = f'Solid {self.name} {self.id}'
        return rep

import os
import sys
import uuid
import tempfile
import time
from io import StringIO
import re
import numpy as np
import trimesh
from pyobb.obb import OBB
from scipy.spatial import ConvexHull

from .meshing.surface_mesh_parameters import default_surface_mesh_parameter
from .face import Face

from .logger import logger

import FreeCAD
import Part as FCPart
import BOPTools.SplitAPI
from FreeCAD import Base
from Draft import make_fillet
from Arch import makePipe


App = FreeCAD


class Solid(object):

    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', uuid.uuid4())
        self.type = kwargs.get('type', None)
        logger.debug(f'initializing solid {self.id}')
        self.name = kwargs.get('name', None)
        self.normal = kwargs.get('normal', None)
        self._fc_solid = kwargs.get('fc_solid', None)
        self._faces = kwargs.get('faces', kwargs.get('_faces', None))
        self.interfaces = kwargs.get('interfaces', [])
        self.features = kwargs.get('features', {})
        self.surface_mesh_setup = kwargs.get('_surface_mesh_setup',
                                             kwargs.get('surface_mesh_setup', default_surface_mesh_parameter))
        self.state = kwargs.get('state', 'solid')
        self._obb = kwargs.get('obb', None)

        print(self.Volume)

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

        logger.debug(f'generating solid from faces: {self.id}')
        start_time = time.time()

        faces = []
        [faces.extend(x.fc_face.Faces) for x in self.faces]
        shell = FCPart.makeShell(faces)
        shell.sewShape()
        shell.fix(1e-7, 1e-7, 1e-7)
        solid = FCPart.Solid(shell)

        if not solid.isClosed():
            logger.error(f'Solid {self.id}: solid is not closed')

        # fuse_start_time = time.time()
        # face0 = self.faces[0].fc_face
        # faces = [x.fc_face for x in self.faces[1:]]
        # shell = face0.multiFuse((faces))
        # solid = FCPart.Solid(shell)
        # logging.debug(f'        fuse time: {time.time() - fuse_start_time} s')

        doc_start_time = time.time()
        doc = App.newDocument()
        __o__ = doc.addObject("Part::Feature", f'{str(self.id)}')
        __o__.Label = f'{str(self.id)}'
        __o__.Shape = solid
        logger.debug(f'        doc time: {time.time() - doc_start_time} s')

        self.fc_solid = __o__

        logger.debug(f'    finished generation of solid from faces: {self.id} in {time.time() - start_time} s')

        # # hull = self.solids[0].fc_solid.Shape
        # # hull = hull.multiFuse([x.fc_solid.Shape for x in self.solids[1:]]).removeSplitter()
        # # hull_solid = Solid(faces=[Face(fc_face=x) for x in hull.Faces])
        #
        #
        #
        # doc = App.newDocument()
        # doc_shapes = []
        # for i, face in enumerate(self.faces):
        #     new_obj = doc.addObject("Part::Feature", f'{face.id}')
        #     new_obj.Shape = face.fc_face
        #     doc.recompute()
        #     doc_shapes.append(new_obj)
        #
        # doc.addObject("Part::MultiFuse", "Fusion")
        # doc.Fusion.Shapes = doc_shapes
        # doc.recompute()
        #
        # __s__ = doc.Fusion.Shape.Faces
        # __s__ = FCPart.Solid(FCPart.Shell(__s__))
        # __o__ = doc.addObject("Part::Feature", f'{str(self.id)}')
        # __o__.Label = f'{str(self.id)}'
        # __o__.Shape = __s__
        #
        # self.fc_solid = __o__
        #
        # logging.debug(f'    finished generating solid from faces: {self.id}')

        return self.fc_solid

    def generate_faces_from_solid(self, solid):
        pass

    def write_of_geo(self, directory):
        stl_str = ''.join([x.create_stl_str(of=True) for x in set(self.faces) - set(self.interfaces)])
        new_file = open(os.path.join(directory, str(self.txt_id) + '.stl'), 'w')
        new_file.writelines(stl_str)
        new_file.close()

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

    def __repr__(self):
        rep = f'Solid {self.name} {self.id} {self.area}'
        return rep


class PipeSolid(Solid):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)

        self.reference_face = kwargs.get('reference_face', None)
        self.reference_edge_id = kwargs.get('reference_edge_id', 0.02)

        self.tube_diameter = kwargs.get('tube_diameter', 0.02)
        self.tube_distance = kwargs.get('tube_distance', 0.50)
        self.tube_side_1_offset = kwargs.get('tube_side_1_offset', 0.085)
        self.tube_edge_distance = kwargs.get('tube_edge_distance', 0.50)
        self.bending_radius = kwargs.get('bending_radius', 0.05)

        self.reference_edge = None
        self._pipe_wire = None

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
    def pipe_wire(self):
        if self._pipe_wire is None:
            self.pipe_wire = self.generate_pipe_wire()
        return self._pipe_wire

    @pipe_wire.setter
    def pipe_wire(self, value):
        self._pipe_wire = value

    @property
    def length(self):
        return self.pipe_wire.Length

    def generate_pipe_wire(self):

        from .tools import project_point_on_line

        reference_face = self.reference_face.reference_face

        self.reference_edge = reference_face.Edges[self.reference_edge_id]
        normal = self.reference_face.get_normal(Base.Vector(self.reference_edge.Vertexes[0].X, self.reference_edge.Vertexes[0].Y,
                                                            self.reference_edge.Vertexes[0].Z))
        tube_main_dir = self.reference_edge.Curve.Direction.cross(normal)

        offset = -self.tube_edge_distance
        wires = []
        offset_possible = True

        while offset_possible:
            try:
                wire = reference_face.OuterWire.makeOffset2D(offset, join=1, openResult=False, intersection=False)

                # check if another
                try:
                    reference_face.OuterWire.makeOffset2D(offset - self.tube_distance, join=1, openResult=False,
                                                          intersection=False)
                    wires.append(wire)
                    offset = offset - 2 * self.tube_distance
                except Exception as e:
                    print(e)
                    offset_possible = False
            except Exception as e:
                print(e)
                offset_possible = False

        pipe_edges = []

        if (reference_face.Edges.__len__() - 1) >= (self.reference_edge_id + 1):
            start_edge_id = self.reference_edge_id + 1
        else:
            start_edge_id = 0

        # create inflow
        V1 = wires[0].Edges[start_edge_id].Vertex1.Point + tube_main_dir * 2 * self.tube_edge_distance
        V2 = wires[0].Edges[start_edge_id].Vertex1.Point
        pipe_edges.append(FCPart.LineSegment(V1, V2).toShape())

        # add edges except the start_edge
        pipe_edges.extend(wires[0].Edges[self.reference_edge_id + 1:])
        pipe_edges.extend(wires[0].Edges[0:self.reference_edge_id:])

        # modify reference_edge_id edge
        p1 = wires[0].Edges[self.reference_edge_id].Vertex1.Point
        p2 = wires[0].Edges[self.reference_edge_id].Vertex2.Point
        v1 = p1
        v2 = p2 - 2 * (p2 - p1).normalize() * self.tube_distance
        pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())

        # export_wire([self.reference_face.OuterWire, *pipe_edges])

        i = 1
        while i <= (wires.__len__() - 1):
            # create connection from previous wire to current wire:
            dir1 = (wires[i].Edges[start_edge_id].Vertex1.Point - pipe_edges[-1].Vertex2.Point).normalize()
            dir2 = (wires[i].Edges[start_edge_id].Vertexes[1].Point - pipe_edges[-1].Vertex2.Point).normalize()

            if sum(abs(abs(dir1) - abs(dir2))) < 1e-10:
                pipe_edges.append(FCPart.LineSegment(pipe_edges[-1].Vertex2.Point,
                                                     wires[i].Edges[start_edge_id].Vertexes[1].Point).toShape())
            else:
                projected_point = FreeCAD.Base.Vector(
                    project_point_on_line(point=wires[i].Edges[start_edge_id].Vertex1.Point, line=pipe_edges[-1]))

                # change_previous end edge:
                pipe_edges[-1] = FCPart.LineSegment(pipe_edges[-1].Vertex1.Point, projected_point).toShape()

                pipe_edges.append(FCPart.LineSegment(wires[i].Edges[start_edge_id].Vertex1.Point,
                                                     projected_point).toShape())
                pipe_edges.append(wires[i].Edges[start_edge_id])

            # #pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())
            #
            # pipe_edges.append(FCPart.LineSegment(pipe_edges[-1].Vertex2.Point,
            #                                      wires[i].Edges[start_edge_id].Vertexes[1].Point).toShape())

            # add other edges except start_edge
            pipe_edges.extend(wires[i].Edges[self.reference_edge_id + 2:])
            pipe_edges.extend(wires[i].Edges[0:self.reference_edge_id:])

            # modify reference_edge_id edge
            p1 = wires[i].Edges[self.reference_edge_id].Vertex1.Point
            p2 = wires[i].Edges[self.reference_edge_id].Vertex2.Point
            v1 = p1
            v2 = p2 - 2 * (p2 - p1).normalize() * self.tube_distance
            pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())

            i = i + 1

        # create
        succeeded = False
        while not succeeded:
            wire_in = FCPart.Wire(pipe_edges)
            wire_out = wire_in.makeOffset2D(-self.tube_distance, join=1, openResult=True, intersection=False)
            wire_in.distToShape(wire_out)

            # create connection between wire_in and wire_out:
            v1 = wire_in.Edges[-1].Vertex2.Point
            v2 = wire_out.Edges[-1].Vertex2.Point
            connection_edge = FCPart.LineSegment(v1, v2).toShape()

            edges_out = wire_out.Edges
            edges_out.reverse()

            all_edges = [*wire_in.Edges, connection_edge, *edges_out]

            try:
                FCPart.Wire(all_edges, intersection=False)
                succeeded = True
            except Exception as e:
                succeeded = False
                del pipe_edges[-1]

        if self.bending_radius is not None:
            edges_with_radius = all_edges[0:1]

            for i in range(1, all_edges.__len__()):
                if self.bending_radius > min([edges_with_radius[-1].Length * 0.5, all_edges[i].Length * 0.5]):
                    bending_radius = min([edges_with_radius[-1].Length * 0.5, all_edges[i].Length * 0.5])
                else:
                    bending_radius = self.bending_radius

                new_edges = make_fillet([edges_with_radius[-1], all_edges[i]], radius=bending_radius)
                if new_edges is not None:
                    edges_with_radius[-1] = new_edges.Shape.OrderedEdges[0]
                    edges_with_radius.extend(new_edges.Shape.OrderedEdges[1:])
                else:
                    edges_with_radius.append(all_edges[i])

            pipe_wire = FCPart.Wire(edges_with_radius)
        else:
            pipe_wire = FCPart.Wire(all_edges)

        pipe_wire.Placement.move(
            self.reference_face.layer_dir * normal * (- self.reference_face.component_construction.side_1_offset + self.reference_face.tube_side_1_offset))

        return pipe_wire

    def generate_solid(self):

        logger.info(f'Creating solid for pipe {self.name} {self.id}')

        doc = App.newDocument()
        __o__ = doc.addObject("Part::Feature", f'pipe_wire')
        __o__.Shape = self.pipe_wire
        initial_pipe = makePipe(__o__, self.tube_diameter)
        doc.recompute()
        # self.pipe = pipe.Shape

        hull = self.reference_face.plain_reference_face_solid.assembly.hull
        # export_objects([hull.fc_solid.Shape], 'hull_solid_test.stp')
        # export_objects([x.fc_solid.Shape for x in self.assembly.solids], 'assembly_solid_test.stp')
        inlet_outlet = hull.fc_solid.Shape.Shells[0].common(initial_pipe.Shape)

        if inlet_outlet.SubShapes.__len__() == 2:
            inlet = Face(fc_face=inlet_outlet.SubShapes[0].removeSplitter(),
                         name='Pipe_Inlet')
            outlet = Face(fc_face=inlet_outlet.SubShapes[1].removeSplitter(),
                          name='Pipe_Outlet')
        else:
            raise Exception('can not identify inlet and outlet')

        # BOPTools.SplitAPI.booleanFragments([self.pipe.fc_solid.Shape, inlet.fc_face, outlet.fc_face], "Split", tolerance=1e-5)

        pipe_faces = BOPTools.SplitAPI.slice(initial_pipe.Shape.Shells[0], hull.fc_solid.Shape.Shells, "Split",
                                             1e-3)

        faces = [*inlet.fc_face.Faces, *outlet.fc_face.Faces, *pipe_faces.Faces]
        shell = FCPart.makeShell(faces)
        shell.sewShape()
        shell.fix(1e-7, 1e-7, 1e-7)
        pipe = FCPart.Solid(shell)

        layer_pipe_interfaces = []

        logger.info(f'Updating layer solids reference face {self.reference_face.name} {self.reference_face.id}')

        for i, solid in enumerate(self.reference_face.assembly.solids):

            logger.info(f'Updating layer solid {i} of {self.reference_face.assembly.solids.__len__()}: '
                        f'{solid.name} {solid.id}')
            # side_faces = FCPart.makeShell([x.fc_face for x in solid.faces[2:]])
            # check if common:
            # export_objects([solid.fc_solid.Shape, self.pipe], 'second_intersection.stp')

            common = solid.fc_solid.Shape.common(pipe)
            if common.Faces:
                new_faces = []
                for face in solid.faces:
                    new_face = Face(fc_face=face.fc_face.cut(pipe))
                    new_faces.append(new_face)
                    solid.update_face(face, new_face)
                solid.generate_solid_from_faces()
                pipe_interface = Face(fc_face=solid.fc_solid.Shape.common(pipe).Shells[0],
                                      name=f'Pipe interface',
                                      linear_deflection=0.5,
                                      angular_deflection=0.5)
                layer_pipe_interfaces.append(pipe_interface)
                solid.faces.append(pipe_interface)
                solid.interfaces.append(pipe_interface)
                solid.generate_solid_from_faces()
                # solid.save_fcstd(f'/tmp/Layer{i}.FCStd')

                # cut = self.pipe.Shells[0].cut(solid.fc_solid.Shape)
                # common = pipe.Shape.Shells[0].common(solid.fc_solid.Shape)
                # pipe_faces = Face(fc_face=common.Shells[0])
                # # generate new layer solid with pipe faces:
                # new_faces.append(pipe_faces)
                # solid.faces = new_faces
                # solid.generate_solid_from_faces()

                solid.features['pipe_faces'] = pipe_interface

        # generate pipe solid
        pipe_solid = Solid(faces=[inlet, outlet, *layer_pipe_interfaces],
                           name='PipeSolid')
        pipe_solid.generate_solid_from_faces()

        pipe_solid.features['inlet'] = inlet
        pipe_solid.features['outlet'] = outlet
        pipe_solid.features['layer_interfaces'] = layer_pipe_interfaces
        pipe_solid.interfaces = layer_pipe_interfaces

        logger.info(f'Updating assembly of reference face {self.reference_face.name} {self.reference_face.id}')

        self.reference_face.assembly.solids.append(pipe_solid)
        self.reference_face.assembly.features['pipe'] = pipe_solid

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

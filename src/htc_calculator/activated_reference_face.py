import sys
import pathlib
from .reference_face import ReferenceFace
from .tools import project_point_on_line, export_objects
from .face import Face
from .solid import Solid, PipeSolid
from .meshing.block_mesh import BlockMeshVertex, Block, create_o_grid_blocks

import FreeCAD
import Part as FCPart
import BOPTools.SplitAPI
# import Mesh
# import BOPTools
from Draft import make_fillet
from FreeCAD import Base
from Arch import makePipe
from MeshPart import meshFromShape
import ObjectsFem
from femmesh.gmshtools import GmshTools as gt


App = FreeCAD


class ActivatedReferenceFace(ReferenceFace):

    def __init__(self, *args, **kwargs):

        ReferenceFace.__init__(self, *args, **kwargs)

        self.plain_reference_face_solid = ReferenceFace(*args, **kwargs)

        self.tube_diameter = kwargs.get('tube_diameter', 0.02)
        self.tube_distance = kwargs.get('tube_distance', 0.50)
        self.tube_side_1_offset = kwargs.get('tube_side_1_offset', 0.085)
        self.tube_edge_distance = kwargs.get('tube_edge_distance', 0.50)
        self.bending_radius = kwargs.get('bending_radius', 0.05)

        self.reference_edge_id = kwargs.get('start_edge', 0)

        self.reference_edge = None
        # self.pipe_wire = None
        self._pipe = None

        self.integrate_pipe()

    def integrate_pipe(self):
        self.pipe = PipeSolid(reference_face=self,
                              reference_edge_id=self.reference_edge_id,
                              tube_diameter=self.tube_diameter,
                              tube_distance=self.tube_distance,
                              tube_side_1_offset=self.tube_side_1_offset,
                              tube_edge_distance=self.tube_edge_distance,
                              bending_radius=self.bending_radius)

    @property
    def pipe(self):
        if self._pipe is None:
            self.integrate_pipe()
        return self._pipe

    @pipe.setter
    def pipe(self, value):
        self._pipe = value

    # def generate_tube_spline(self):
    #
    #     # doc = App.newDocument("tube_spline")
    #
    #     self.reference_edge = self.reference_face.Edges[self.reference_edge_id]
    #     normal = self.get_normal(Base.Vector(self.reference_edge.Vertexes[0].X, self.reference_edge.Vertexes[0].Y, self.reference_edge.Vertexes[0].Z))
    #     tube_main_dir = self.reference_edge.Curve.Direction.cross(normal)
    #
    #     offset = -self.tube_edge_distance
    #     wires = []
    #     offset_possible = True
    #
    #     while offset_possible:
    #         try:
    #             wire = self.reference_face.OuterWire.makeOffset2D(offset, join=1, openResult=False, intersection=False)
    #
    #             # check if another
    #             try:
    #                 self.reference_face.OuterWire.makeOffset2D(offset - self.tube_distance, join=1, openResult=False, intersection=False)
    #                 wires.append(wire)
    #                 offset = offset - 2 * self.tube_distance
    #             except Exception as e:
    #                 print(e)
    #                 offset_possible = False
    #         except Exception as e:
    #             print(e)
    #             offset_possible = False
    #
    #     pipe_edges = []
    #
    #     if (self.reference_face.Edges.__len__() - 1) >= (self.reference_edge_id + 1):
    #         start_edge_id = self.reference_edge_id + 1
    #     else:
    #         start_edge_id = 0
    #
    #     # create inflow
    #     V1 = wires[0].Edges[start_edge_id].Vertex1.Point + tube_main_dir * 2 * self.tube_edge_distance
    #     V2 = wires[0].Edges[start_edge_id].Vertex1.Point
    #     pipe_edges.append(FCPart.LineSegment(V1, V2).toShape())
    #
    #     # add edges except the start_edge
    #     pipe_edges.extend(wires[0].Edges[self.reference_edge_id+1:])
    #     pipe_edges.extend(wires[0].Edges[0:self.reference_edge_id:])
    #
    #     # modify reference_edge_id edge
    #     p1 = wires[0].Edges[self.reference_edge_id].Vertex1.Point
    #     p2 = wires[0].Edges[self.reference_edge_id].Vertex2.Point
    #     v1 = p1
    #     v2 = p2 - 2 * (p2 - p1).normalize() * self.tube_distance
    #     pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())
    #
    #     # export_wire([self.reference_face.OuterWire, *pipe_edges])
    #
    #     i = 1
    #     while i <= (wires.__len__()-1):
    #         # create connection from previous wire to current wire:
    #         dir1 = (wires[i].Edges[start_edge_id].Vertex1.Point - pipe_edges[-1].Vertex2.Point).normalize()
    #         dir2 = (wires[i].Edges[start_edge_id].Vertexes[1].Point - pipe_edges[-1].Vertex2.Point).normalize()
    #
    #         if sum(abs(abs(dir1) - abs(dir2))) < 1e-10:
    #             pipe_edges.append(FCPart.LineSegment(pipe_edges[-1].Vertex2.Point,
    #                                                  wires[i].Edges[start_edge_id].Vertexes[1].Point).toShape())
    #         else:
    #             projected_point = FreeCAD.Base.Vector(project_point_on_line(point=wires[i].Edges[start_edge_id].Vertex1.Point, line=pipe_edges[-1]))
    #
    #             # change_previous end edge:
    #             pipe_edges[-1] = FCPart.LineSegment(pipe_edges[-1].Vertex1.Point, projected_point).toShape()
    #
    #             pipe_edges.append(FCPart.LineSegment(wires[i].Edges[start_edge_id].Vertex1.Point,
    #                                                  projected_point).toShape())
    #             pipe_edges.append(wires[i].Edges[start_edge_id])
    #
    #
    #         # #pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())
    #         #
    #         # pipe_edges.append(FCPart.LineSegment(pipe_edges[-1].Vertex2.Point,
    #         #                                      wires[i].Edges[start_edge_id].Vertexes[1].Point).toShape())
    #
    #         # add other edges except start_edge
    #         pipe_edges.extend(wires[i].Edges[self.reference_edge_id + 2:])
    #         pipe_edges.extend(wires[i].Edges[0:self.reference_edge_id:])
    #
    #         # modify reference_edge_id edge
    #         p1 = wires[i].Edges[self.reference_edge_id].Vertex1.Point
    #         p2 = wires[i].Edges[self.reference_edge_id].Vertex2.Point
    #         v1 = p1
    #         v2 = p2 - 2 * (p2 - p1).normalize() * self.tube_distance
    #         pipe_edges.append(FCPart.LineSegment(v1, v2).toShape())
    #
    #         i = i + 1
    #
    #     # create
    #     succeeded = False
    #     while not succeeded:
    #         wire_in = FCPart.Wire(pipe_edges)
    #         wire_out = wire_in.makeOffset2D(-self.tube_distance, join=1, openResult=True, intersection=False)
    #         wire_in.distToShape(wire_out)
    #
    #         # create connection between wire_in and wire_out:
    #         v1 = wire_in.Edges[-1].Vertex2.Point
    #         v2 = wire_out.Edges[-1].Vertex2.Point
    #         connection_edge = FCPart.LineSegment(v1, v2).toShape()
    #
    #         edges_out = wire_out.Edges
    #         edges_out.reverse()
    #
    #         all_edges = [*wire_in.Edges, connection_edge, *edges_out]
    #
    #         try:
    #             FCPart.Wire(all_edges, intersection=False)
    #             succeeded = True
    #         except Exception as e:
    #             succeeded = False
    #             del pipe_edges[-1]
    #
    #     if self.bending_radius is not None:
    #         edges_with_radius = all_edges[0:1]
    #
    #         for i in range(1, all_edges.__len__()):
    #             if self.bending_radius > min([edges_with_radius[-1].Length * 0.5, all_edges[i].Length * 0.5]):
    #                 bending_radius = min([edges_with_radius[-1].Length * 0.5, all_edges[i].Length * 0.5])
    #             else:
    #                 bending_radius = self.bending_radius
    #
    #             new_edges = make_fillet([edges_with_radius[-1], all_edges[i]], radius=bending_radius)
    #             if new_edges is not None:
    #                 edges_with_radius[-1] = new_edges.Shape.OrderedEdges[0]
    #                 edges_with_radius.extend(new_edges.Shape.OrderedEdges[1:])
    #             else:
    #                 edges_with_radius.append(all_edges[i])
    #
    #         self.pipe_wire = FCPart.Wire(edges_with_radius)
    #     else:
    #         self.pipe_wire = FCPart.Wire(all_edges)
    #
    #     self.pipe_wire.Placement.move(self.layer_dir * normal * (- self.component_construction.side_1_offset + self.tube_side_1_offset))

    # def generate_solid_pipe(self):
    #
    #     doc = App.newDocument()
    #     __o__ = doc.addObject("Part::Feature", f'pipe_wire')
    #     __o__.Shape = self.pipe_wire
    #     pipe = makePipe(__o__, self.tube_diameter)
    #     doc.recompute()
    #     # self.pipe = pipe.Shape
    #
    #     self.pipe = Solid(name=f'Pipe_{self.tube_diameter}mm',
    #                       type='Pipe',
    #                       fc_solid=pipe)
    #     print(self.pipe.faces)
        
    def export_solid_pipe(self, filename):
        doc = App.newDocument()
        __o__ = doc.addObject("Part::Feature", f'pipe_solid')
        __o__.Shape = self.pipe
        FCPart.export(doc.Objects, filename)

    def export_solids(self, filename):

        doc = App.newDocument()
        for i, solid in enumerate(self.assembly.solids):
            __o__ = doc.addObject("Part::Feature", f'Layer {i} solid: {solid.name} {solid.id}')
            __o__.Shape = solid.fc_solid.Shape

        # # add pipe:
        # __o__ = doc.addObject("Part::Feature", f'Pipe solid')
        # __o__.Shape = self.pipe

        file_suffix = pathlib.Path(filename).suffix

        if file_suffix == '.FCStd':
            doc.recompute()
            doc.saveCopy(filename)
        else:
            FCPart.export(doc.Objects, filename)

    def create_o_grid(self):

        wire = self.pipe.pipe_wire
        blocks = [None] * wire.Edges.__len__()

        for i, edge in enumerate(wire.Edges):
            blocks[i] = create_o_grid_blocks(edge, self)



        pass


    # def generate_hole_part(self):
    #
    #     hull = self.assembly.hull
    #     # export_objects([hull.fc_solid.Shape], 'hull_solid_test.stp')
    #     # export_objects([x.fc_solid.Shape for x in self.assembly.solids], 'assembly_solid_test.stp')
    #     inlet_outlet = hull.fc_solid.Shape.Shells[0].common(self.pipe.fc_solid.Shape)
    #
    #     if inlet_outlet.SubShapes.__len__() == 2:
    #         inlet = Face(fc_face=inlet_outlet.SubShapes[0].removeSplitter(),
    #                      name='Pipe_Inlet')
    #         outlet = Face(fc_face=inlet_outlet.SubShapes[1].removeSplitter(),
    #                       name='Pipe_Outlet')
    #     else:
    #         raise Exception('can not identify inlet and outlet')
    #
    #     # BOPTools.SplitAPI.booleanFragments([self.pipe.fc_solid.Shape, inlet.fc_face, outlet.fc_face], "Split", tolerance=1e-5)
    #
    #     pipe_faces = BOPTools.SplitAPI.slice(self.pipe.fc_solid.Shape.Shells[0], hull.fc_solid.Shape.Shells, "Split", 1e-3)
    #
    #     Solid(faces=[Face(fc_face=x) for x in pipe_faces.SubShapes[1].Faces]).save_fcstd('/tmp/pipe_hull.FCStd')
    #
    #     self.pipe.fc_solid.Shape.Area - (inlet.fc_face.Area + outlet.fc_face.Area)
    #
    #
    #     splitted_pipe_faces = []
    #     for solid in self.assembly.solids:
    #         # side_faces = FCPart.makeShell([x.fc_face for x in solid.faces[2:]])
    #         # check if common:
    #         # export_objects([solid.fc_solid.Shape, self.pipe], 'second_intersection.stp')
    #
    #         common = solid.fc_solid.Shape.common(self.pipe.fc_solid.Shape.Shells[0])
    #         if common.Faces:
    #             new_faces = []
    #             for face in solid.faces:
    #                 new_face = Face(fc_face=face.fc_face.cut(self.pipe.fc_solid.Shape))
    #                 new_faces.append(new_face)
    #                 solid.update_face(face, new_face)
    #             # cut = self.pipe.Shells[0].cut(solid.fc_solid.Shape)
    #             common = self.pipe.fc_solid.Shape.Shells[0].common(solid.fc_solid.Shape)
    #
    #             pipe_faces = Face(fc_face=common.Shells[0])
    #
    #             # generate new layer solid with pipe faces:
    #             new_faces.append(pipe_faces)
    #             solid.faces = new_faces
    #             solid.generate_solid_from_faces()
    #
    #             solid.features['pipe_faces'] = pipe_faces
    #             splitted_pipe_faces.append(pipe_faces)
    #
    #     # generate pipe solid
    #     pipe_solid = Solid(faces=[inlet, outlet, *splitted_pipe_faces],
    #                        name='PipeSolid')
    #     pipe_solid.generate_solid_from_faces()
    #
    #     pipe_solid.features['inlet'] = inlet
    #     pipe_solid.features['outlet'] = outlet
    #
    #     self.assembly.solids.append(pipe_solid)
    #     self.assembly.features['pipe'] = pipe_solid

    def save_fcstd(self, filename):
        """
        save as freecad document
        :param filename: full filename; example: '/tmp/test.FCStd'
        """
        doc = App.newDocument("MeshTest")
        __o__ = doc.addObject("Part::Feature", f'Activated Reference Face {self.name} {self.id}')
        __o__.Shape = self.assembly.comp_solid
        doc.recompute()
        doc.saveCopy(filename)

    def generate_mesh(self):
        pass

        doc = App.newDocument("MeshTest")
        __o__ = doc.addObject("Part::Feature", f'pipe_solid')
        __o__.Shape = self.assembly.comp_solid
        femmesh_obj = ObjectsFem.makeMeshGmsh(doc, self.name + "_Mesh")
        doc.recompute()
        doc.saveCopy('/tmp/test.FCStd')
        gm = gt(femmesh_obj)
        gm.update_mesh_data()
        gm.get_tmp_file_paths("/tmp/fcgm_" + str(len), True)
        gm.get_gmsh_command()
        gm.write_gmsh_input_files()
        error = gm.run_gmsh_with_geo()
        print(error)
        gm.read_and_set_new_mesh()
        doc.recompute()


def replace(arr, find, replace):
    # fast and readable
    base = 0
    for cnt in range(arr.count(find)):
        offset = arr.index(find, base)
        arr[offset] = replace
        base = offset + 1

    return arr


def test_mesh_creation():
    # https://github.com/berndhahnebach/FreeCAD_bhb/blob/59c470dedf28da2632abfe6e4481ce09aaf6e233/src/Mod/Fem/femmesh/gmshtools.py#L906
    # more sophisticated example which changes the mesh size
    doc = App.newDocument("MeshTest")
    box_obj = doc.addObject("Part::Box", "Box")
    doc.recompute()
    box_obj.ViewObject.Visibility = False
    max_mesh_sizes = [0.5, 1, 2, 3, 5, 10]
    for len in max_mesh_sizes:
        quantity_len = "{}".format(len)
        print("\n\n Start length = {}".format(quantity_len))
        femmesh_obj = ObjectsFem.makeMeshGmsh(doc, box_obj.Name + "_Mesh")
        femmesh_obj.Part = box_obj
        femmesh_obj.CharacteristicLengthMax = "{}".format(quantity_len)
        femmesh_obj.CharacteristicLengthMin = "{}".format(quantity_len)
        doc.recompute()
        gm = GmshTools(femmesh_obj)
        gm.update_mesh_data()
        # set the tmp file path to some user path including the length
        gm.get_tmp_file_paths("/tmp/fcgm_" + str(len), True)
        gm.get_gmsh_command()
        gm.write_gmsh_input_files()
        error = gm.run_gmsh_with_geo()
        print(error)
        gm.read_and_set_new_mesh()
        doc.recompute()
        print("Done length = {}".format(quantity_len))

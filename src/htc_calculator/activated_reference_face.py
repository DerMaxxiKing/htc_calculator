import sys
import numpy as np
import pathlib
import operator
import functools
from .reference_face import ReferenceFace
from .tools import project_point_on_line, export_objects
from .face import Face
from .solid import Solid, PipeSolid
from .meshing.block_mesh import Block, create_o_grid_blocks, create_blocks_from_2d_mesh, BlockMesh, CompBlock
from .logger import logger
from .tools import export_objects, split_wire_by_projected_vertices

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

        self.pipe_section = kwargs.get('pipe_section')
        self.tube_diameter = kwargs.get('tube_diameter', 0.02)
        self.tube_inner_diameter = kwargs.get('tube_inner_diameter', 0.016)
        self.tube_material = kwargs.get('tube_material', None)
        self.tube_distance = kwargs.get('tube_distance', 0.50)
        self.tube_side_1_offset = kwargs.get('tube_side_1_offset', 0.085)
        self.tube_edge_distance = kwargs.get('tube_edge_distance', 0.50)
        self.bending_radius = kwargs.get('bending_radius', 0.05)

        self.reference_edge_id = kwargs.get('start_edge', 0)

        self.reference_edge = None
        # self.pipe_wire = None
        self._pipe = None
        self._pipe_comp_blocks = None
        self._free_comp_blocks = None

        self.integrate_pipe()

    @property
    def pipe_comp_blocks(self):
        if self._pipe_comp_blocks is None:
            self._pipe_comp_blocks = self.create_o_grid_with_section()
        return self._pipe_comp_blocks

    @property
    def free_comp_blocks(self):
        if self._free_comp_blocks is None:
            self._free_comp_blocks = self.create_free_blocks()
        return self._free_comp_blocks

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

    def create_o_grid_with_section(self):
        wire = self.pipe.pipe_wire
        blocks = []

        for i, edge in enumerate(wire.Edges):

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

            logger.info(f'creating block {i} of {wire.Edges.__len__()}')

            new_blocks = self.pipe_section.create_block(edge=edge,
                                                        face_normal=self.normal,
                                                        tube_inner_diameter=self.tube_inner_diameter,
                                                        tube_diameter=self.tube_diameter,
                                                        outer_pipe=outer_pipe,
                                                        inlet=inlet,
                                                        outlet=outlet)
            blocks.append(new_blocks)

        logger.info(f'Finished Pipe Block generation successfully\n\n')
        block_list = functools.reduce(operator.iconcat, blocks, [])
        pipe_comp_block = CompBlock(name='Pipe Blocks',
                                    blocks=block_list)

        # Block.save_fcstd('/tmp/blocks.FCStd')
        # export_objects([pipe_comp_block.fc_solid], '/tmp/pipe_comp_block.FCStd')
        return pipe_comp_block

    # def create_o_grid(self):
    #
    #     wire = self.pipe.pipe_wire
    #     blocks = []
    #
    #     for i, edge in enumerate(wire.Edges):
    #
    #         if i == 0:
    #             outer_pipe = False
    #             inlet = True
    #             outlet = False
    #         elif i == wire.Edges.__len__() - 1:
    #             outer_pipe = False
    #             inlet = False
    #             outlet = True
    #         else:
    #             outer_pipe = True
    #             inlet = False
    #             outlet = False
    #
    #         logger.info(f'creating block {i} of {wire.Edges.__len__()}')
    #
    #         if type(edge.Curve) is FCPart.Line:
    #             logger.debug(f'creating block {i} of {wire.Edges.__len__()} as line')
    #             new_blocks = create_o_grid_blocks(edge, self, outer_pipe=outer_pipe, inlet=inlet, outlet=outlet)
    #             blocks.append(new_blocks)
    #         else:
    #             # split edge:
    #             logger.debug(f'creating block {i} of {wire.Edges.__len__()} as arc')
    #             sub_edges = edge.split((edge.LastParameter-edge.FirstParameter) / 2).SubShapes
    #             # for sub_edge in sub_edges:
    #             #     sub_edge.Placement = edge.Placement
    #             #     new_blocks = create_o_grid_blocks(sub_edge, self, outer_pipe=outer_pipe, inlet=inlet, outlet=outlet)
    #             #     blocks.append(new_blocks)
    #             new_blocks = create_o_grid_blocks(edge, self, outer_pipe=outer_pipe, inlet=inlet, outlet=outlet)
    #             blocks.append(new_blocks)
    #
    #     logger.info(f'Finished Pipe Block generation successfully\n\n')
    #     block_list = functools.reduce(operator.iconcat, blocks, [])
    #     pipe_comp_block = CompBlock(name='Pipe Blocks',
    #                                 blocks=block_list)
    #
    #     # Block.save_fcstd('/tmp/blocks.FCStd')
    #     # export_objects([pipe_comp_block.fc_solid], '/tmp/pipe_comp_block.FCStd')
    #     return pipe_comp_block

    def create_free_blocks(self):

        logger.info(f'Creating free block mesh for {self.name}, {self.id}')

        # comp_solid = Block.comp_solid

        # move reference face in pipe layer:
        mv_vec = self.layer_dir * self.normal * (- self.component_construction.side_1_offset + self.tube_side_1_offset)

        ref_face = self.reference_face.copy()
        ref_face2 = ref_face.translate(mv_vec)
        logger.info(f'Cutting reference face with pipe wire')

        cutted_face = ref_face2.cut(self.pipe_comp_blocks.fc_solid)

        # cutted_face = ref_face2.cut(Block.comp_solid)
        splitted_ref_face_wire = split_wire_by_projected_vertices(ref_face2.OuterWire,
                                                                  cutted_face.SubShapes[0].Vertexes,
                                                                  self.tube_edge_distance)
        cutted_ref_face = FCPart.Face(splitted_ref_face_wire).translate(mv_vec)
        cutted_face = cutted_ref_face.cut(self.pipe_comp_blocks.fc_solid)

        # add points to second (inner) face
        # splitted_inner_face_wire = split_wire_by_projected_vertices(cutted_face.SubShapes[1].OuterWire,
        #                                                             [],
        #                                                             3 * self.tube_diameter,
        #                                                             ensure_closed=True)

        wire = FCPart.Wire(cutted_face.SubShapes[1].OuterWire)
        if not wire.isClosed():
            return FCPart.Wire([*wire.OrderedEdges,
                                FCPart.LineSegment(wire.OrderedVertexes[-1].Point,
                                                   wire.OrderedVertexes[0].Point).toShape()])

        quad_meshes = [Face(fc_face=x).create_hex_g_mesh(lc=9999999999) for x in [cutted_face.SubShapes[0],
                                                                                  FCPart.Face(wire)]]
        free_blocks = create_blocks_from_2d_mesh(quad_meshes, self)

        free_comp_block = CompBlock(name='Free Blocks',
                                    blocks=free_blocks)

        # export_objects([free_comp_block.fc_solid], '/tmp/free_comp_block.FCStd')

        return free_comp_block

        # Block.save_fcstd('/tmp/blocks2.FCStd')
        #
        # export_objects([ref_face, cutted_face], '/tmp/cutted_face.FCStd')
        #
        # print('done')
        #
        # free_blocks = create_blocks_from_2d_mesh(quad_meshes, self)

    def extrude_pipe_layer(self):
        # top side:

        layer_thicknesses = [0, *[x.thickness for x in self.component_construction.layers]]
        layer_interfaces = [self.layer_dir * self.normal * (- self.component_construction.side_1_offset + x) for x in np.cumsum(layer_thicknesses)]

        layer_interface_planes = np.array([FCPart.makePlane(99999,
                                                            99999,
                                                            Base.Vector(self.vertices[0] + x),
                                                            self.normal) for x in layer_interfaces])

        new_blocks = []
        for block in self.pipe_comp_blocks.blocks:
            if not block.pipe_layer_top:
                continue
            logger.debug(f'Extruding block {block}')
            faces_to_extrude = np.array(block.faces)[np.array(block.pipe_layer_extrude_top)]
            for face in faces_to_extrude:
                extrude_to = face.vertices[0].fc_vertex.toShape().distToShape(layer_interface_planes[0])[0] < np.cumsum(layer_thicknesses)
                ext_dist = 0
                for dist in [face.vertices[0].fc_vertex.toShape().distToShape(x)[0] for x in layer_interface_planes[extrude_to]]:
                    new_block = face.extrude(dist, direction=self.normal, dist2=ext_dist)
                    new_blocks.append(new_block)

                export_objects([new_block.fc_solid], '/tmp/extrude_block.FCStd')





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

    def generate_block_mesh_dict(self):

        block_mesh = BlockMesh(name=self.name)
        block_mesh.init_case()

        # vertices_entry = BlockMeshVertex.block_mesh_entry()
        # print(vertices_entry)
        # edges_entry = BlockMeshEdge.block_mesh_entry()
        # print(edges_entry)
        # block_entry = Block.block_mesh_entry()
        # print(block_entry)
        # boundary_entry = BlockMeshBoundary.block_mesh_entry()
        # print(boundary_entry)








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

        Block.save_fcstd('/tmp/blocks2.FCStd')


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

import copy
import os
import sys
from bisect import bisect
import numpy as np
import pathlib
import operator
import functools
from .reference_face import ReferenceFace
from .tools import project_point_on_line, export_objects
from .face import Face
from .solid import Solid, PipeSolid, MultiMaterialSolid
from .assembly import Assembly
from .meshing import block_mesh as imp_block_mesh
from .meshing.block_mesh import create_blocks_from_2d_mesh, Mesh, BlockMesh, \
    CompBlock, NoNormal, bottom_side_patch, top_side_patch, CellZone, wall_patch, extrude_2d_mesh, Block, \
    BlockMeshEdge, BlockMeshFace, PipeMesh, ConstructionMesh, LayerMesh, UpperPipeLayerMesh, LowerPipeLayerMesh, \
    add_face_contacts, PipeLayerMesh
from .meshing.mesh_config import pipe_cut_section_min_refinement_level, pipe_cut_section_max_refinement_level
from .logger import logger
from .tools import export_objects, split_wire_by_projected_vertices
from .case.case import OFCase
from tqdm import tqdm, trange


import FreeCAD
import Part as FCPart
from FreeCAD import Base
import BOPTools.SplitAPI


App = FreeCAD


class ActivatedReferenceFace(ReferenceFace):

    def __init__(self, *args, **kwargs):

        ReferenceFace.__init__(self, *args, **kwargs)

        self.separate_meshes = kwargs.get('separate_meshes', True)

        self.reference_edge = None
        self._pipe = None
        self._pipe_comp_blocks = None
        self._pipe_layer = None
        self._pipe_layer_index = None
        self._free_comp_blocks = None
        self._layer_meshes = None
        self._extruded_comp_blocks = None
        self._comp_blocks = None
        self._case = None
        self._layer_interface_planes = None
        self._cut_pipe_layer_solid = None

        self.pipe_mesh = PipeMesh(name='Block Mesh ' + 'pipe_layer_mesh',
                                  mesh=Mesh(name='pipe_layer_mesh'))
        self.pipe_section = kwargs.get('pipe_section')
        self.pipe_section.block_mesh = self.pipe_mesh

        # if no seperated meshes should be created, add all to self.pipe_mesh
        if self.separate_meshes:
            self.construction_mesh = ConstructionMesh(name='Block Mesh ' + 'pipe_layer_free_mesh',
                                                      mesh=Mesh(name='pipe_layer_free_mesh'))
        else:
            self.construction_mesh = self.pipe_mesh

        self.case = kwargs.get('case', OFCase(reference_face=self))
        self.plain_reference_face_solid = ReferenceFace(*args, **kwargs)
        self.case = kwargs.get('case', None)

        self.default_mesh_size = kwargs.get('default_mesh_size', 100)
        self.default_arc_cell_size = kwargs.get('default_arc_cell_size', 20)
        self.tube_diameter = kwargs.get('tube_diameter', 0.02)
        self.tube_inner_diameter = kwargs.get('tube_inner_diameter', 0.016)
        self.tube_material = kwargs.get('tube_material', None)
        self.tube_distance = kwargs.get('tube_distance', 0.50)
        self.tube_side_1_offset = kwargs.get('tube_side_1_offset', 0.085)
        self.tube_edge_distance = kwargs.get('tube_edge_distance', 0.50)
        self.bending_radius = kwargs.get('bending_radius', 0.05)

        self.reference_edge_id = kwargs.get('start_edge', 0)

        self.pipe_cut_section_min_refinement_level = kwargs.get('pipe_cut_section_min_refinement_level',
                                                                pipe_cut_section_min_refinement_level)

        self.pipe_cut_section_max_refinement_level = kwargs.get('pipe_cut_section_max_refinement_level',
                                                                pipe_cut_section_max_refinement_level)

        # self.integrate_pipe()

    @property
    def case(self):
        if self._case is None:
            self._case = OFCase(reference_face=self)
        return self._case

    @case.setter
    def case(self, value):
        self._case = value

    @property
    def pipe_comp_blocks(self):
        if self._pipe_comp_blocks is None:
            self._pipe_comp_blocks = self.create_o_grid_with_section()
        return self._pipe_comp_blocks

    @property
    def free_comp_blocks(self):
        if self._free_comp_blocks is None:
            if self.separate_meshes:
                mesh = self.construction_mesh.mesh
            else:
                mesh = self.pipe_mesh
            self._free_comp_blocks = self.create_free_blocks(mesh=mesh)
        return self._free_comp_blocks

    @property
    def comp_blocks(self):
        if self._comp_blocks is None:
            self._comp_blocks = CompBlock(name='Final Blocks',
                                          blocks=[*self.pipe_comp_blocks.blocks,
                                                  *self.free_comp_blocks.blocks,
                                                  *self.extruded_comp_blocks.blocks])
            # export_objects([FCPart.Compound([x.fc_face for x in self._comp_blocks.hull_faces])], '/tmp/hull_faces.FCStd')
        return self._comp_blocks

    @property
    def extruded_comp_blocks(self):
        if self._extruded_comp_blocks is None:
            self._extruded_comp_blocks = self.extrude_pipe_layer()
        return self._extruded_comp_blocks

    @extruded_comp_blocks.setter
    def extruded_comp_blocks(self, value):
        self._extruded_comp_blocks = value

    @property
    def pipe_layer_index(self):
        if self._pipe_layer_index is None:
            self._pipe_layer_index, self._pipe_layer = self.get_pipe_layer()
            self.pipe_layer.meshes.add(self.pipe_mesh)
        return self._pipe_layer_index

    @property
    def pipe_layer(self):
        if self._pipe_layer is None:
            self._pipe_layer_index, self._pipe_layer = self.get_pipe_layer()
            self.pipe_layer.meshes.add(self.pipe_mesh)
        return self._pipe_layer

    @property
    def cut_pipe_layer_solid(self):
        if self._cut_pipe_layer_solid is None:
            self._cut_pipe_layer_solid = self.create_cut_pipe_layer_solid()
        return self._cut_pipe_layer_solid

    def get_pipe_layer(self):
        layer_thicknesses = np.array([0, *[x.thickness for x in self.component_construction.layers]])
        offset0 = - self.component_construction.side_1_offset * self.layer_dir
        layer_positions = np.cumsum(layer_thicknesses) + offset0
        layer_index = bisect(
            layer_positions,
            self.tube_side_1_offset - self.component_construction.side_1_offset * self.layer_dir) - 1
        pipe_layer = self.component_construction.layers[layer_index]

        return layer_index, pipe_layer

    def integrate_pipe(self):
        logger.info(f'Creating pipe solid')
        self.pipe = PipeSolid(reference_face=self,
                              reference_edge_id=self.reference_edge_id,
                              tube_diameter=self.tube_diameter,
                              tube_inner_diameter=self.tube_inner_diameter,
                              tube_distance=self.tube_distance,
                              tube_side_1_offset=self.tube_side_1_offset,
                              tube_edge_distance=self.tube_edge_distance,
                              bending_radius=self.bending_radius)
        logger.info(f'Successfully created pipe solid')
        self.pipe.print_info()
        return self.pipe

    @property
    def pipe(self):
        if self._pipe is None:
            self.integrate_pipe()
        return self._pipe

    @pipe.setter
    def pipe(self, value):
        self._pipe = value

    @property
    def layer_interface_planes(self):
        if self._layer_interface_planes is None:
            layer_thicknesses = [0, *[x.thickness for x in self.component_construction.layers]]
            layer_interfaces = [self.layer_dir * self.normal * (- self.component_construction.side_1_offset + x) for x
                                in np.cumsum(layer_thicknesses)]
            self._layer_interface_planes = np.array([FCPart.makePlane(99999,
                                                                99999,
                                                                Base.Vector(self.vertices[0] + x),
                                                                self.normal) for x in layer_interfaces])
        return self._layer_interface_planes

    @property
    def layer_meshes(self):
        if self._layer_meshes is None:
            if self.separate_meshes:
                block_mesh = None
            else:
                block_mesh = self.pipe_mesh
            self._layer_meshes, blocks, self.extruded_comp_blocks = self.extrude_clean_layers(
                master_block_mesh=block_mesh
            )
        return self._layer_meshes

    @layer_meshes.setter
    def layer_meshes(self, value):
        self._layer_meshes = value

    @property
    def pipe_layer_thickness(self):
        return 2 * (self.tube_diameter / 2 / np.sqrt(2) + self.tube_diameter / 4)

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

        self.pipe_mesh.mesh.activate()

        logger.info(f'Generation o-grid blocks for pipe...')

        wire = self.pipe.pipe_wire
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

            new_blocks = self.pipe_section.create_block(edge=edge,
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

                self.pipe_mesh.top_faces.extend(get_side_faces(self.pipe_section.top_side))
                self.pipe_mesh.bottom_faces.extend(get_side_faces(self.pipe_section.bottom_side))
                self.pipe_mesh.interfaces.extend(get_side_faces(self.pipe_section.interface_side))

        logger.info(f'Successfully generated o-grid blocks for pipe\n\n')
        block_list = functools.reduce(operator.iconcat, blocks, [])
        pipe_comp_block = CompBlock(name='Pipe Blocks',
                                    blocks=block_list)

        self.pipe_layer.meshes.add(self.pipe_mesh)

        # Block.save_fcstd('/tmp/blocks.FCStd')
        # export_objects([pipe_comp_block.fc_solid], '/tmp/pipe_comp_block.FCStd')
        return pipe_comp_block

    def create_free_blocks(self, mesh=None):

        logger.info(f'Creating free block mesh for {self.name}, {self.id}')

        if mesh is None:
            mesh = self.construction_mesh.mesh

        mesh.activate()

        # pipe_interface_edges = set()

        # copy edges and faces of interfaces to pipe_mesh to construction_mesh
        # copy edges
        # logger.info(f'Copy interfaces to {mesh}')
        # _ = [pipe_interface_edges.update(x.edges) for x in self.pipe_mesh.interfaces]
        # BlockMeshEdge.copy_to_mesh(edges=pipe_interface_edges, mesh=mesh)

        # copy faces
        # self.construction_mesh.interfaces = BlockMeshFace.copy_to_mesh(faces=self.pipe_mesh.interfaces, mesh=mesh)

        # comp_solid = Block.comp_solid

        # move reference face in pipe layer:

        mv_vec = self.layer_dir * self.normal * (- self.component_construction.side_1_offset + self.tube_side_1_offset)

        ref_face = self.reference_face.copy()
        ref_face2 = ref_face.translate(mv_vec)
        logger.info(f'Cutting reference face with pipe wire')
        cutted_face = ref_face2.cut(self.pipe_comp_blocks.fc_solid)

        logger.info(f'Splitting wire with projected edges')
        # cutted_face = ref_face2.cut(Block.comp_solid)

        jump3_wire0 = ref_face2.OuterWire.makeOffset2D(-self.tube_edge_distance - 0.5 * self.tube_diameter,
                                                       join=1,
                                                       openResult=False,
                                                       intersection=False)

        # export_objects([jump3_wire0, ref_face2.OuterWire], '/tmp/wires.FCStd')

        vertexes = [x for x in cutted_face.SubShapes[0].OuterWire.Vertexes if x.distToShape(jump3_wire0)[0] < self.tube_diameter]

        # export_objects(vertexes, '/tmp/vertexes.FCStd')

        splitted_ref_face_wire = split_wire_by_projected_vertices(cutted_face.SubShapes[0].OuterWire,
                                                                  vertexes,
                                                                  self.tube_edge_distance,
                                                                  ensure_closed=True,
                                                                  add_arc_midpoint=False)
        # splitted_ref_face_wire = cutted_face.SubShapes[0].OuterWire
        # cutted_ref_face = FCPart.Face(splitted_ref_face_wire).translate(mv_vec)
        # cutted_face = cutted_ref_face.cut(self.pipe_comp_blocks.fc_solid)
        # export_objects(splitted_ref_face_wire, '/tmp/splitted_ref_face_wire.FCStd')

        # add points to second (inner) face
        splitted_inner_face_wire = split_wire_by_projected_vertices(cutted_face.SubShapes[1].OuterWire,
                                                                    cutted_face.SubShapes[1].OuterWire.Vertexes,
                                                                    3 * self.tube_diameter,
                                                                    ensure_closed=True,
                                                                    add_arc_midpoint=True)

        # wire = splitted_inner_face_wire
        if not splitted_inner_face_wire.isClosed():
            splitted_inner_face_wire = FCPart.Wire([*splitted_inner_face_wire.OrderedEdges,
                                                    FCPart.LineSegment(splitted_inner_face_wire.OrderedVertexes[-1].Point,
                                                                       splitted_inner_face_wire.OrderedVertexes[0].Point).toShape()])

        # add edges:
        logger.info(f'Adding edges to mesh {self.construction_mesh.mesh.name}')
        for edge in [*splitted_inner_face_wire.Edges, *splitted_ref_face_wire.Edges]:
            if isinstance(edge.Curve, FCPart.Arc) or isinstance(edge.Curve, FCPart.Circle):
                offset0 = -np.array(self.normal * (self.tube_diameter / 2 / np.sqrt(2) + self.tube_diameter / 4))
                dist = 2 * (self.tube_diameter / 2 / np.sqrt(2) + self.tube_diameter / 4)
                direction = np.array(self.normal)
                _ = BlockMeshEdge.from_fc_edge(fc_edge=edge,
                                               mesh=mesh,
                                               translate=offset0)

                _ = BlockMeshEdge.from_fc_edge(fc_edge=edge,
                                               mesh=mesh,
                                               translate=dist * direction + offset0)

        # export_objects([x.fc_edge for x in mesh.edges.values()], '/tmp/edges.FCStd')

        logger.info(f'Creating hex mesh for free faces')
        quad_meshes = [Face(fc_face=x).create_hex_g_mesh_2(lc=99999999) for x in [FCPart.Face(splitted_ref_face_wire),
                                                                             FCPart.Face(splitted_inner_face_wire)]]
        quad_meshes[0].write('/tmp/mesh1.vtk')
        quad_meshes[1].write('/tmp/mesh2.vtk')
        logger.info(f'Extruding blocks for free faces mesh')
        free_blocks = create_blocks_from_2d_mesh(quad_meshes, self, mesh_to_add=mesh)

        for block in free_blocks:
            block.pipe_layer_top = True
            block.pipe_layer_bottom = True
            block.pipe_layer_extrude_top = [1]
            block.pipe_layer_extrude_bottom = [0]

        free_comp_block = CompBlock(name='Free Blocks',
                                    blocks=free_blocks)

        self.construction_mesh.top_faces = [block.face1 for block in free_blocks]
        self.construction_mesh.bottom_faces = [block.face0 for block in free_blocks]

        # export_objects([FCPart.Compound([x.fc_face for x in self.construction_mesh.top_faces]),
        #                 FCPart.Compound([x.fc_face for x in self.construction_mesh.bottom_faces]),
        #                 FCPart.Compound([x.fc_face for x in self.construction_mesh.interfaces])],
        #                '/tmp/construction_mesh_faces.FCStd')

        # export_objects([free_comp_block.fc_solid], '/tmp/free_comp_block.FCStd')

        self.pipe_layer.meshes.add(self.construction_mesh)
        # export_objects([x.fc_solid for x in free_blocks], '/tmp/free_blocks.FCStd')
        logger.info(f'Successfully created free block mesh for {self.name}, {self.id}')
        return free_comp_block

    def extrude_clean_layers(self, master_block_mesh=None):

        logger.info(f'Creating Layer meshes')

        layer_thicknesses = np.array([0, *[x.thickness for x in self.component_construction.layers]])
        new_blocks = []
        offset0 = - self.component_construction.side_1_offset * self.layer_dir
        layer_positions = np.cumsum(layer_thicknesses) + offset0

        layer_meshes = set()

        for i, layer_thickness in enumerate(tqdm(layer_thicknesses[:-1], desc='creating layer meshes', colour="green")):

            layer_name = self.component_construction.layers[i].name

            if i == self.pipe_layer_index:
                # if the tube is in the layer

                #       top face
                # -----------------------------------------                     ---
                #                                                                ↑
                #            Mesh 1
                #
                # ---------|--------|--------------   ①                          L
                #          |        |                                            a
                #          |    ⊙   |      ← pipe | quad_mesh layer              y
                #          |        |                                            e
                # ---------|--------|--------------   ②                          r
                #
                #
                #            Mesh 2
                #                                                                ↓
                # -----------------------------------------                     ---
                #       bottom face

                # extrude pipe layer to bottom of the material layer:
                # -------------------------------------------------------------------------------------------------------
                # create new mesh:
                if master_block_mesh is None:
                    layer_mesh = LowerPipeLayerMesh(name='Block Mesh ' + layer_name + ' 1',
                                                    mesh=Mesh(name='Mesh ' + layer_name + ' 1'))
                else:
                    layer_mesh = master_block_mesh
                self.component_construction.layers[i].meshes.add(layer_mesh)
                layer_meshes.add(layer_mesh)
                layer_mesh.mesh.activate()

                quad_mesh = copy.copy(self.quad_mesh)

                dist0 = (-self.component_construction.side_1_offset +
                         self.tube_side_1_offset - self.pipe_layer_thickness / 2) * self.layer_dir

                quad_mesh.points = quad_mesh.points + dist0 * self.normal       # -> move mesh to ②
                dist = layer_positions[i] - dist0                               # -> distance ② to bottom face
                lower_layer_blocks = extrude_2d_mesh(quad_mesh,
                                                     distance=dist,
                                                     grading=[1, 1, 0.33],
                                                     direction=self.normal * self.layer_dir,
                                                     block_name=f'Pipe Layer ({i}) lower block')
                new_blocks.extend(lower_layer_blocks)
                layer_mesh.top_faces = [block.faces[0] for block in lower_layer_blocks]
                layer_mesh.bottom_faces = [block.faces[1] for block in lower_layer_blocks]

                # extrude pipe layer to top of the material layer:
                # -------------------------------------------------------------------------------------------------------

                # create new block mesh
                if master_block_mesh is None:
                    layer_mesh = UpperPipeLayerMesh(name='Block Mesh ' + layer_name + ' 2',
                                                    mesh=Mesh(name='Mesh ' + layer_name + ' 2'))
                else:
                    layer_mesh = master_block_mesh

                self.component_construction.layers[i].meshes.add(layer_mesh)
                layer_meshes.add(layer_mesh)
                layer_mesh.mesh.activate()

                quad_mesh = copy.copy(self.quad_mesh)
                dist0 = (-self.component_construction.side_1_offset +
                         self.tube_side_1_offset + self.pipe_layer_thickness / 2) * self.layer_dir

                quad_mesh.points = quad_mesh.points + dist0 * self.normal       # -> move mesh to ①
                dist = layer_positions[i+1] - dist0                             # -> distance ① to top face
                upper_layer_blocks = extrude_2d_mesh(quad_mesh,
                                                     distance=dist,
                                                     direction=self.normal,
                                                     grading=[1, 1, 3],
                                                     block_name=f'Pipe Layer ({i}) upper block')
                new_blocks.extend(upper_layer_blocks)
                layer_mesh.bottom_faces = [block.faces[0] for block in upper_layer_blocks]
                layer_mesh.top_faces = [block.faces[1] for block in upper_layer_blocks]

            else:
                if master_block_mesh is None:
                    layer_mesh = LayerMesh(name='Block Mesh ' + layer_name,
                                           mesh=Mesh(name='Mesh ' + layer_name))
                else:
                    layer_mesh = master_block_mesh

                self.component_construction.layers[i].meshes.add(layer_mesh)
                layer_meshes.add(layer_mesh)
                layer_mesh.mesh.activate()

                quad_mesh = copy.copy(self.quad_mesh)
                quad_mesh.points = quad_mesh.points + offset0 + layer_positions[i] * self.normal * self.layer_dir
                layer_blocks = extrude_2d_mesh(quad_mesh,
                                               distance=layer_positions[i+1] - layer_thickness,
                                               direction=self.normal,
                                               block_name=f'Layer {i} block')
                new_blocks.extend(layer_blocks)

                layer_mesh.bottom_faces = [block.faces[0] for block in layer_blocks]
                layer_mesh.top_faces = [block.faces[1] for block in layer_blocks]

        # Block.save_fcstd(filename='/tmp/new_blocks.FCStd', blocks=new_blocks)

        free_comp_block = CompBlock(name='Extruded Blocks',
                                    blocks=new_blocks)

        return layer_meshes, new_blocks, free_comp_block

    def extrude_pipe_layer(self):

        self.construction_mesh.mesh.activate()

        logger.info('Extruding pipe layer')

        # top side:
        # ____________________________________________________________________________________________________________________________________________
        layer_thicknesses = np.array([0, *[x.thickness for x in self.component_construction.layers]])
        layer_interface_planes = self.layer_interface_planes

        # export_objects([x.fc_solid for x in [*self.pipe_comp_blocks.blocks, *self.free_comp_blocks.blocks]], '/tmp/initial_blocks.FCStd')

        new_blocks = []
        for block in tqdm([*self.pipe_comp_blocks.blocks, *self.free_comp_blocks.blocks],
                          desc='extruding layer blocks',
                          colour="green"):
            if block.pipe_layer_top:
                # logger.debug(f'Extruding block top {block}')
                faces_to_extrude = np.array(block.faces)[np.array(block.pipe_layer_extrude_top)]
                for face in faces_to_extrude:
                    extrude_to = face.vertices[0].fc_vertex.toShape().distToShape(layer_interface_planes[0])[0] < np.cumsum(layer_thicknesses)
                    ext_dist = 0
                    for dist in [face.vertices[0].fc_vertex.toShape().distToShape(x)[0] for x in layer_interface_planes[extrude_to]]:
                        new_block = face.extrude(dist, direction=self.normal, dist2=ext_dist)
                        new_blocks.append(new_block)
                        ext_dist = dist
            if block.pipe_layer_bottom:
                # logger.debug(f'Extruding block bottom {block}')
                faces_to_extrude = np.array(block.faces)[np.array(block.pipe_layer_extrude_bottom)]
                # export_objects([block.fc_solid, faces_to_extrude[0].fc_face], '/tmp/test.FCStd')
                for face in faces_to_extrude:
                    extrude_to = face.vertices[0].fc_vertex.toShape().distToShape(layer_interface_planes[0])[0] > np.cumsum(layer_thicknesses)
                    ext_dist = 0
                    for dist in [face.vertices[0].fc_vertex.toShape().distToShape(x)[0] for x in layer_interface_planes[extrude_to]]:
                        try:
                            new_block = face.extrude(dist, direction=-self.normal, dist2=ext_dist)
                        except Exception as e:
                            raise e
                        new_blocks.append(new_block)
                        ext_dist = dist
            # export_objects([x.fc_solid for x in new_blocks], '/tmp/new_blocks.FCStd')

        free_comp_block = CompBlock(name='Extruded Blocks',
                                    blocks=new_blocks)
        return free_comp_block
        # export_objects([x.fc_solid for x in new_blocks], '/tmp/extrude_block.FCStd')

    def update_cell_zone(self, blocks=None, mesh=None):

        last_activated_mesh = BlockMeshEdge.current_mesh

        if mesh is not None:
            mesh.activate()
            foreign_cell_zones = list({x.cell_zone for x in blocks if (x.cell_zone is not None)})
            local_cell_zones = CellZone.copy_to_mesh(instances=foreign_cell_zones,
                                                     mesh=mesh)
            cell_zone_lookup_dict = dict(zip(foreign_cell_zones, local_cell_zones))
            _ = [setattr(x, 'cell_zone', cell_zone_lookup_dict[x.cell_zone]) for x in blocks if (x.cell_zone is not None)]

        logger.info('Updating cell zones...')

        layer_materials = np.array([x.material for x in self.component_construction.layers])

        layer_thicknesses = np.array([0, *[x.thickness for x in self.component_construction.layers]])
        layer_interface_planes = self.layer_interface_planes

        # layer_solids = self.assembly.solids
        # layer_solids.remove(self.assembly.features['pipe'])

        if blocks is None:
            check_blocks = [*self.pipe_comp_blocks.blocks,
                            *self.free_comp_blocks.blocks,
                            *self.extruded_comp_blocks.blocks]
        else:
            check_blocks = blocks

        # _ = [setattr(x, 'cell_zone',
        #     layer_materials[np.argmax(
        #         layer_interface_planes[0].distToShape(
        #     FCPart.Vertex(tuple(x.dirty_center)))[0] < layer_thicknesses) - 1])
        #      for x in check_blocks if x.cell_zone is None]
        cell_zones = set()
        for block in tqdm(check_blocks, desc='Updating cell zones', colour="green"):
            if block.cell_zone is not None:
                cell_zones.add(block.cell_zone)
                if block.cell_zone.material is not None:
                    continue
            try:

                # block.cell_zone = layer_materials[np.argmax(layer_interface_planes[0].distToShape(
                #     FCPart.Vertex(block.fc_solid.CenterOfGravity))[0] < layer_thicknesses) - 1]
                material = layer_materials[np.argmax(layer_interface_planes[0].distToShape(
                    FCPart.Vertex(tuple(block.dirty_center)))[0] < layer_thicknesses) - 1]
                block.cell_zone = CellZone(material=material,
                                           mesh=mesh)
                cell_zones.add(block.cell_zone)
            except Exception as e:
                raise e

        # export_objects([FCPart.Compound([*[x.fc_solid for x in self.pipe_comp_blocks.blocks],
        #                                  *[x.fc_solid for x in self.free_comp_blocks.blocks],
        #                                  *[x.fc_solid for x in self.extruded_comp_blocks.blocks]]),
        #                 block.fc_solid],
        #                '/tmp/update_mat.FCStd')

        logger.info('Cell zones updated successfully')
        last_activated_mesh.activate()
        return cell_zones

    def create_cut_pipe_layer_solid(self):
        pipe_layer_solid = self.pipe_layer.solid
        pipe_mesh_solid = Solid(name='pipe_mesh_solid',
                                faces=[Face(fc_face=x) for x in self.pipe_comp_blocks.fc_solid.Faces])

        cutted_fc_solid = pipe_layer_solid.fc_solid.Shape.cut(pipe_mesh_solid.fc_solid.Shape)

        common = cutted_fc_solid.Shells[0].common(pipe_mesh_solid.fc_solid.Shape.Shells[0])

        base_faces = []
        if isinstance(pipe_layer_solid.features['base_faces'], list):
            _ = [base_faces.extend(x.fc_face.Faces) for x in pipe_layer_solid.features['base_faces']]
        else:
            base_faces.extend(pipe_layer_solid.features['base_faces'].fc_face.Faces)
        base_faces_shell = FCPart.makeShell(base_faces)

        top_faces = []
        if isinstance(pipe_layer_solid.features['top_faces'], list):
            _ = [top_faces.extend(x.fc_face.Faces) for x in pipe_layer_solid.features['top_faces']]
        else:
            top_faces.extend(pipe_layer_solid.features['top_faces'].fc_face.Faces)
        top_faces_shell = FCPart.makeShell(top_faces)

        side_faces = []
        if isinstance(pipe_layer_solid.features['side_faces'], list):
            _ = [side_faces.extend(x.fc_face.Faces) for x in pipe_layer_solid.features['side_faces']]
        else:
            side_faces.extend(pipe_layer_solid.features['side_faces'].fc_face.Faces)
        side_faces_shell = FCPart.makeShell(side_faces)

        base_face = Face(name='base_face', fc_face=cutted_fc_solid.Shells[0].common(base_faces_shell))
        top_face = Face(name='top_face', fc_face=cutted_fc_solid.Shells[0].common(top_faces_shell))
        side_face = Face(name='side_face', fc_face=cutted_fc_solid.Shells[0].common(side_faces_shell))
        common_face = Face(name='pipe_mesh_interface', fc_face=common)

        common_face.surface_mesh_setup.max_refinement_level = 4
        common_face.surface_mesh_setup.min_refinement_level = 4

        cutted_solid = Solid(name='cut_pipe_layer_solid',
                             faces=[base_face, top_face, side_face, common_face])
        cutted_solid.features['base_faces'] = base_face
        cutted_solid.features['top_faces'] = top_face
        cutted_solid.features['side_faces'] = side_face
        cutted_solid.features['pipe_mesh_interfaces'] = common_face

        return cutted_solid

    def combine_meshes(self):
        logger.info('Creating combined mesh...')
        combined_mesh = BlockMesh(name='Combined Block Mesh',
                                  mesh=Mesh(name='Combined Mesh'))

        _ = self.pipe_comp_blocks
        _ = self.free_comp_blocks
        _ = self.layer_meshes

        for i, layer in enumerate(self.component_construction.layers):

            is_pipe_layer = layer is self.pipe_layer

            if is_pipe_layer:

                pipe_lookup_dict = combined_mesh.add_mesh_copy(self.pipe_mesh, copy_feature_faces=False)
                construction_lookup_dict = combined_mesh.add_mesh_copy(self.construction_mesh, copy_feature_faces=False)

                add_face_contacts([pipe_lookup_dict[x.id] for x in self.pipe_mesh.interfaces],
                                  [construction_lookup_dict[x.id] for x in self.construction_mesh.interfaces],
                                  combined_mesh.mesh,
                                  combined_mesh.mesh,
                                  f'pipe_mesh_to_construction_mesh',
                                  f'construction_mesh_to_pipe_mesh'
                                  )

                pipe_layer_bottom_faces = [*[pipe_lookup_dict[x.id] for x in self.pipe_mesh.bottom_faces],
                                *[construction_lookup_dict[x.id] for x in self.construction_mesh.bottom_faces]]

                pipe_layer_top_faces = [*[pipe_lookup_dict[x.id] for x in self.pipe_mesh.top_faces],
                                        *[construction_lookup_dict[x.id] for x in self.construction_mesh.top_faces]]

                # combine with lower pipe_layer_mesh
                bottom_mesh = next(filter(lambda x: type(x) == LowerPipeLayerMesh, layer.meshes))
                top_mesh = next(filter(lambda x: type(x) == UpperPipeLayerMesh, layer.meshes))
                # pipe_layer_mesh = next(filter(lambda x: type(x) == PipeLayerMesh, layer.meshes))

                bottom_face_lookup_dict = combined_mesh.add_mesh_copy(bottom_mesh, copy_feature_faces=False)
                # pipe_layer_face_lookup_dict = combined_mesh.add_mesh_copy(pipe_layer_mesh, copy_feature_faces=False)
                top_face_lookup_dict = combined_mesh.add_mesh_copy(top_mesh, copy_feature_faces=False)

                add_face_contacts([bottom_face_lookup_dict[x.id] for x in bottom_mesh.top_faces],
                                  pipe_layer_bottom_faces,
                                  combined_mesh.mesh,
                                  combined_mesh.mesh,
                                  f'pipe_layer_bottom_to_pipe_layer_middle',
                                  f'pipe_layer_middle_to_pipe_layer_bottom')

                add_face_contacts(pipe_layer_top_faces,
                                  [top_face_lookup_dict[x.id] for x in top_mesh.bottom_faces],
                                  combined_mesh.mesh,
                                  combined_mesh.mesh,
                                  f'pipe_layer_top_to_pipe_layer_middle',
                                  f'pipe_layer_middle_to_pipe_layer_top')
                if i == 0:
                    combined_mesh.bottom_faces = [bottom_face_lookup_dict[x.id] for x in bottom_mesh.bottom_faces]
                else:
                    add_face_contacts(combined_mesh.top_faces,
                                      [bottom_face_lookup_dict[x.id] for x in bottom_mesh.bottom_faces],
                                      combined_mesh.mesh,
                                      combined_mesh.mesh,
                                      f'layer_{i - 1}_to_layer_{i}',
                                      f'layer_{i}_to_layer_{i - 1}')
                combined_mesh.top_faces = [top_face_lookup_dict[x.id] for x in top_mesh.top_faces]
            else:
                layer_block_mesh = list(layer.meshes)[0]
                face_lookup_dict = combined_mesh.add_mesh_copy(layer_block_mesh, copy_feature_faces=False)
                if i == 0:
                    combined_mesh.bottom_faces = [face_lookup_dict[x.id] for x in layer_block_mesh.bottom_faces]
                else:
                    add_face_contacts(combined_mesh.top_faces,
                                      [face_lookup_dict[x.id] for x in layer_block_mesh.bottom_faces],
                                      combined_mesh.mesh,
                                      combined_mesh.mesh,
                                      f'layer_{i-1}_to_layer_{i}',
                                      f'layer_{i}_to_layer_{i-1}')
                combined_mesh.top_faces = [face_lookup_dict[x.id] for x in layer_block_mesh.top_faces]

        # export_objects([FCPart.Compound([x.fc_face for x in combined_mesh.top_faces]),
        #                 FCPart.Compound([x.fc_face for x in combined_mesh.bottom_faces]),
        #                 FCPart.Compound([x.fc_face for x in combined_mesh.mesh.boundaries['inlet'].faces]),
        #                 FCPart.Compound([x.fc_face for x in combined_mesh.mesh.boundaries['outlet'].faces]),
        #                 FCPart.Compound([x.fc_face for x in combined_mesh.mesh.boundaries['fluid_wall'].faces]),
        #                 FCPart.Compound([x.fc_face for x in combined_mesh.mesh.boundaries['layer_0_to_layer_1'].faces])
        #                 ],
        #                '/tmp/boundaries.FCStd'
        #                )
        #
        # export_objects([FCPart.Compound([x.fc_solid for x in combined_mesh.mesh.blocks])],
        #                '/tmp/blocks.FCStd'
        #                )

        logger.info('Successfully created combined mesh')
        return combined_mesh

    def update_boundary_conditions(self, mesh):

        logger.info('Updating boundary conditions...')
        _ = [setattr(x, 'boundary', bottom_side_patch) for x in mesh.bottom_faces]

        # if faces is None:
        #     faces = self.comp_blocks.hull_faces
        #
        # ref_normal = np.array(self.normal)
        #
        # top_side_faces = []
        # bottom_side_faces = []
        #
        # if not self.separate_meshes:
        #
        #     for face in faces:
        #         if face.normal is NoNormal:
        #             if face.blocks.__len__() == 1:
        #                 if face.boundary is None:
        #                     face.boundary = wall_patch
        #             continue
        #         if not (np.allclose(face.normal, ref_normal, 1e-3) or np.allclose(face.normal, -ref_normal, 1e-3)):
        #             if face.blocks.__len__() == 1:
        #                 if face.boundary is None:
        #                     face.boundary = wall_patch
        #             continue
        #         # bottom side:
        #         if self.layer_interface_planes[0].distToShape(FCPart.Vertex(Base.Vector(face.dirty_center)))[0] < 1e-3:
        #             face.boundary = bottom_side_patch
        #             bottom_side_faces.append(face)
        #         elif self.layer_interface_planes[-1].distToShape(FCPart.Vertex(Base.Vector(face.dirty_center)))[0] < 1e-3:
        #             face.boundary = top_side_patch
        #             top_side_faces.append(face)
        # else:
        # add bottom boundary condition:
        # _ = [setattr(x, 'boundary', bottom_side_patch) for x in mesh.bottom_faces]

            # def add_pipe_layer_bcs(layer):
            #     bottom_mesh = next(filter(lambda x: type(x) == LowerPipeLayerMesh, layer.meshes))
            #     top_mesh = next(filter(lambda x: type(x) == UpperPipeLayerMesh, layer.meshes))
            #     pipe_layer_mesh = next(filter(lambda x: type(x) == PipeLayerMesh, layer.meshes))
            #
            #     # connect bottom_mesh with pipe_layer_mesh
            #     add_face_contacts(bottom_mesh.top_faces,
            #                       pipe_layer_mesh.bottom_faces,
            #                       bottom_mesh.mesh,
            #                       pipe_layer_mesh.mesh,
            #                       f'{bottom_mesh.mesh.txt_id}_to_{pipe_layer_mesh.mesh.txt_id}',
            #                       f'{pipe_layer_mesh.mesh.txt_id}_to_{bottom_mesh.mesh.txt_id}')
            #
            #     # connect top_mesh with construction_mesh
            #     add_face_contacts(top_mesh.bottom_faces,
            #                       pipe_layer_mesh.top_faces,
            #                       top_mesh.mesh,
            #                       pipe_layer_mesh.mesh,
            #                       f'{top_mesh.mesh.txt_id}_to_{pipe_layer_mesh.mesh.txt_id}',
            #                       f'{pipe_layer_mesh.mesh.txt_id}_to_{top_mesh.mesh.txt_id}')
            #
            # num_layers = self.component_construction.layers.__len__()
            # for i, layer in enumerate(self.component_construction.layers):
            #
            #     is_pipe_layer = layer is self.pipe_layer
            #
            #     if layer is self.pipe_layer:
            #         add_pipe_layer_bcs(layer)
            #
            #     if i == 0:
            #         if is_pipe_layer:
            #             # bottom boundary
            #             bottom_mesh = next(filter(lambda x: type(x) == LowerPipeLayerMesh, layer.meshes))
            #             _ = [setattr(x, 'boundary', bottom_side_patch) for x in bottom_mesh.bottom_faces]
            #         else:
            #             _ = [setattr(x, 'boundary', bottom_side_patch) for x in list(layer.meshes)[0].bottom_faces]
            #
            #     if i == num_layers - 1:
            #         if is_pipe_layer:
            #             # top boundary
            #             top_mesh = next(filter(lambda x: type(x) == UpperPipeLayerMesh, layer.meshes))
            #             _ = [setattr(x, 'boundary', top_side_patch) for x in top_mesh.top_faces]
            #         else:
            #             _ = [setattr(x, 'boundary', top_side_patch) for x in list(layer.meshes)[0].top_faces]
            #     else:
            #         # connect layer with next layer:
            #         if is_pipe_layer:
            #             top_mesh = next(filter(lambda x: type(x) == UpperPipeLayerMesh, layer.meshes))
            #         else:
            #             top_mesh = list(layer.meshes)[0]
            #
            #         next_layer = self.component_construction.layers[i+1]
            #         if next_layer is self.pipe_layer:
            #             bottom_mesh = next(filter(lambda x: type(x) == LowerPipeLayerMesh, next_layer.meshes))
            #         else:
            #             bottom_mesh = list(next_layer.meshes)[0]
            #
            #         add_face_contacts(top_mesh.top_faces,
            #                           bottom_mesh.bottom_faces,
            #                           top_mesh.mesh,
            #                           bottom_mesh.mesh,
            #                           f'{top_mesh.mesh.txt_id}_to_{bottom_mesh.mesh.txt_id}',
            #                           f'{bottom_mesh.mesh.txt_id}_to_{top_mesh.mesh.txt_id}')

        # export_objects([x.fc_face for x in self._comp_blocks.hull_faces], '/tmp/hull_faces.FCStd')
        # export_objects(FCPart.Compound([x.fc_face for x in self._comp_blocks.hull_faces]), '/tmp/hull_faces.FCStd')
        # export_objects(FCPart.Compound([x.fc_face for x in top_side_faces]), '/tmp/top_side_faces.FCStd')
        # export_objects(FCPart.Compound([x.fc_face for x in bottom_side_faces]), '/tmp/bottom_side_faces.FCStd')

        logger.info('Successfully updated boundary conditions ')

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

        _ = self.pipe_comp_blocks
        _ = self.free_comp_blocks
        _ = self.extruded_comp_blocks
        _ = self.comp_blocks

        self.update_cell_zone()
        self.update_boundary_conditions()

        imp_block_mesh.default_cell_size = self.default_mesh_size
        imp_block_mesh.default_arc_cell_size = self.default_arc_cell_size

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

    def run_case(self, *args, **kwargs):

        self.case.run()

    def generate_3d_geometry(self):

        logger.info(f'Creating 3D geometry for ActivatedReferenceFace {self.name} {self.id}')

        assembly = ReferenceFace.generate_3d_geometry(self)
        self.pipe = PipeSolid(reference_face=self,
                              reference_edge_id=self.reference_edge_id,
                              tube_diameter=self.tube_diameter,
                              tube_inner_diameter=self.tube_inner_diameter,
                              tube_distance=self.tube_distance,
                              tube_side_1_offset=self.tube_side_1_offset,
                              tube_edge_distance=self.tube_edge_distance,
                              bending_radius=self.bending_radius,
                              integrate_pipe=False)

        cutted_solid = self.create_cut_pipe_layer_solid()
        cutted_solid.material = self.pipe_layer.material
        self.pipe_layer.solid = cutted_solid

        self.update_cell_zone(blocks=self.pipe_mesh.mesh.blocks, mesh=self.pipe_mesh.mesh)
        mesh_solid = self.pipe_mesh.create_mesh_solid()
        # mesh_solid = self.pipe_mesh.create_shm_mesh_solid()

        solids = [*[x.solid for x in self.component_construction.layers], mesh_solid]

        for solid in solids:
            solid.write_of_geo(f'/simulations', separate_interface=False)

        pipe_assembly = Assembly(solids=solids,
                                 interfaces=[],
                                 faces=None,
                                 topology=assembly.topology,
                                 reference_face=self,
                                 features={'pipe_mesh_solid': mesh_solid,
                                           'pipe_layer_solid': cutted_solid})

        # for i, solid in enumerate([x.solid for x in self.component_construction.layers]):
        #     solid

        self.side_1_face = assembly.solids[0].faces[0]
        self.side_2_face = assembly.solids[-1].faces[1]

        logger.info(f'Successfully created 3D geometry for ActivatedReferenceFace {self.name} {self.id}')

        return pipe_assembly


def replace(arr, find, replace):
    # fast and readable
    base = 0
    for cnt in range(arr.count(find)):
        offset = arr.index(find, base)
        arr[offset] = replace
        base = offset + 1

    return arr

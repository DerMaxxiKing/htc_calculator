import copy
import os
import sys
import numpy as np
import pathlib
import operator
import functools
from .reference_face import ReferenceFace
from .tools import project_point_on_line, export_objects
from .face import Face
from .solid import Solid, PipeSolid
from .meshing import block_mesh as imp_block_mesh
from .meshing.block_mesh import create_blocks_from_2d_mesh, Mesh, BlockMesh, \
    CompBlock, NoNormal, bottom_side_patch, top_side_patch, CellZone, wall_patch, extrude_2d_mesh, Block, BlockMeshEdge
from .logger import logger
from .tools import export_objects, split_wire_by_projected_vertices
from .case.case import OFCase
from tqdm import tqdm, trange

import FreeCAD
import Part as FCPart
from FreeCAD import Base


App = FreeCAD


class ActivatedReferenceFace(ReferenceFace):

    def __init__(self, *args, **kwargs):

        ReferenceFace.__init__(self, *args, **kwargs)

        self.reference_edge = None
        self._pipe = None
        self._pipe_comp_blocks = None
        self._free_comp_blocks = None
        self._layer_meshes = None
        self._extruded_comp_blocks = None
        self._comp_blocks = None
        self._case = None
        self._layer_interface_planes = None

        self.pipe_mesh = BlockMesh(name='Block Mesh ' + 'pipe_layer_mesh',
                                   mesh=Mesh(name='pipe_layer_mesh'))

        self.construction_mesh = BlockMesh(name='Block Mesh ' + 'pipe_layer_free_mesh',
                                           mesh=Mesh(name='pipe_layer_free_mesh'))

        self.case = kwargs.get('case', OFCase(reference_face=self))
        self.plain_reference_face_solid = ReferenceFace(*args, **kwargs)
        self.case = kwargs.get('case', None)

        self.default_mesh_size = kwargs.get('default_mesh_size', 100)
        self.default_arc_cell_size = kwargs.get('default_arc_cell_size', 20)
        self.pipe_section = kwargs.get('pipe_section')
        self.tube_diameter = kwargs.get('tube_diameter', 0.02)
        self.tube_inner_diameter = kwargs.get('tube_inner_diameter', 0.016)
        self.tube_material = kwargs.get('tube_material', None)
        self.tube_distance = kwargs.get('tube_distance', 0.50)
        self.tube_side_1_offset = kwargs.get('tube_side_1_offset', 0.085)
        self.tube_edge_distance = kwargs.get('tube_edge_distance', 0.50)
        self.bending_radius = kwargs.get('bending_radius', 0.05)

        self.reference_edge_id = kwargs.get('start_edge', 0)

        self.integrate_pipe()

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
            self._free_comp_blocks = self.create_free_blocks()
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
            self._layer_meshes = self.extrude_clean_layers()
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

        logger.info(f'Finished pipe o-grid block generation successfully\n\n')
        block_list = functools.reduce(operator.iconcat, blocks, [])
        pipe_comp_block = CompBlock(name='Pipe Blocks',
                                    blocks=block_list)

        # Block.save_fcstd('/tmp/blocks.FCStd')
        # export_objects([pipe_comp_block.fc_solid], '/tmp/pipe_comp_block.FCStd')
        return pipe_comp_block

    def create_free_blocks(self):

        self.construction_mesh.mesh.activate()

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
            wire = FCPart.Wire([*wire.OrderedEdges,
                                FCPart.LineSegment(wire.OrderedVertexes[-1].Point,
                                                   wire.OrderedVertexes[0].Point).toShape()])

        # add edges:
        for edge in [*wire.Edges, *cutted_face.SubShapes[0].Edges]:
            if type(edge.Curve) is FCPart.Arc:
                BlockMeshEdge.from_fc_edge(fc_edge=edge,
                                           mesh=self.construction_mesh.mesh)

        quad_meshes = [Face(fc_face=x).create_hex_g_mesh_2(lc=9999999999) for x in [cutted_face.SubShapes[0],
                                                                                    FCPart.Face(wire)]]
        free_blocks = create_blocks_from_2d_mesh(quad_meshes, self)

        for block in free_blocks:
            block.pipe_layer_top = True
            block.pipe_layer_bottom = True
            block.pipe_layer_extrude_top = [1]
            block.pipe_layer_extrude_bottom = [0]

        free_comp_block = CompBlock(name='Free Blocks',
                                    blocks=free_blocks)

        # export_objects([free_comp_block.fc_solid], '/tmp/free_comp_block.FCStd')

        return free_comp_block

    def extrude_clean_layers(self):

        layer_thicknesses = np.array([0, *[x.thickness for x in self.component_construction.layers]])
        # layer_interface_planes = self.layer_interface_planes

        # quad_mesh = self.quad_mesh

        layer_meshes = []

        new_blocks = []
        offset0 = - self.component_construction.side_1_offset * self.layer_dir
        layer_positions = np.cumsum(layer_thicknesses) + offset0

        for i, layer_thickness in enumerate(layer_thicknesses[:-1]):

            layer_mesh = Mesh(name=f'{self.component_construction.layers[i].name}_mesh')

            if layer_positions[i] < self.tube_side_1_offset < layer_positions[i+1]:
                block_mesh = BlockMesh(name='Block Mesh ' + layer_mesh.name,
                                       mesh=layer_mesh)
                self.layer_meshes.append(block_mesh)
                layer_mesh.activate()

                # extrude pipe layer to bottom of the material layer:
                quad_mesh = copy.copy(self.quad_mesh)

                dist0 = (-self.component_construction.side_1_offset +
                         self.tube_side_1_offset - self.pipe_layer_thickness / 2) * self.layer_dir

                quad_mesh.points = quad_mesh.points + dist0 * self.normal
                dist = layer_positions[i] - dist0
                layer_blocks = extrude_2d_mesh(quad_mesh,
                                               distance=dist,
                                               direction=self.normal * self.layer_dir)
                new_blocks.extend(layer_blocks)

                block_mesh.bottom_faces = [block.faces[1] for block in layer_blocks]

                # extrude pipe layer to top of the material layer:
                quad_mesh = copy.copy(self.quad_mesh)
                dist0 = (-self.component_construction.side_1_offset +
                         self.tube_side_1_offset + self.pipe_layer_thickness / 2) * self.layer_dir

                quad_mesh.points = quad_mesh.points + dist0 * self.normal
                dist = layer_positions[i+1] - dist0
                layer_blocks = extrude_2d_mesh(quad_mesh,
                                               distance=dist,
                                               direction=self.normal)
                new_blocks.extend(layer_blocks)
                block_mesh.bottom_faces = [block.faces[1] for block in layer_blocks]

            else:
                block_mesh = BlockMesh(name='Block Mesh ' + layer_mesh.name,
                                       mesh=layer_mesh)
                self.layer_meshes.append(block_mesh)
                layer_mesh.activate()

                quad_mesh = copy.copy(self.quad_mesh)
                quad_mesh.points = quad_mesh.points + offset0 + layer_positions[i] * self.normal * self.layer_dir
                layer_blocks = extrude_2d_mesh(quad_mesh,
                                               distance=layer_thickness,
                                               direction=self.normal)
                new_blocks.extend(layer_blocks)

                block_mesh.bottom_faces = [block.faces[0] for block in layer_blocks]
                block_mesh.top_faces = [block.faces[1] for block in layer_blocks]

        # Block.save_fcstd(filename='/tmp/new_blocks.FCStd', blocks=new_blocks)

        return layer_meshes

    def extrude_pipe_layer(self):

        self.construction_mesh.activate()

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

    def update_cell_zone(self, blocks=None):

        logger.info('Updating cell zones...')

        layer_materials = np.array([x.material for x in self.component_construction.layers])

        layer_thicknesses = np.array([0, *[x.thickness for x in self.component_construction.layers]])
        layer_interface_planes = self.layer_interface_planes

        layer_solids = self.assembly.solids
        layer_solids.remove(self.assembly.features['pipe'])

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

        for block in tqdm(check_blocks, desc='Updating cell zones', colour="green"):
            if block.cell_zone is not None:
                if block.cell_zone.material is not None:
                    continue
            try:

                # block.cell_zone = layer_materials[np.argmax(layer_interface_planes[0].distToShape(
                #     FCPart.Vertex(block.fc_solid.CenterOfGravity))[0] < layer_thicknesses) - 1]
                material = layer_materials[np.argmax(layer_interface_planes[0].distToShape(
                    FCPart.Vertex(tuple(block.dirty_center)))[0] < layer_thicknesses) - 1]
                block.cell_zone = CellZone(material=material)
            except Exception as e:
                raise e

        # export_objects([FCPart.Compound([*[x.fc_solid for x in self.pipe_comp_blocks.blocks],
        #                                  *[x.fc_solid for x in self.free_comp_blocks.blocks],
        #                                  *[x.fc_solid for x in self.extruded_comp_blocks.blocks]]),
        #                 block.fc_solid],
        #                '/tmp/update_mat.FCStd')

        logger.info('Cell zones updated successfully')

    def update_boundary_conditions(self):
        logger.info('Updating boundary conditions...')

        ref_normal = np.array(self.normal)

        top_side_faces = []
        bottom_side_faces = []

        for face in self.comp_blocks.hull_faces:
            if face.normal is NoNormal:
                if face.blocks.__len__() == 1:
                    if face.boundary is None:
                        face.boundary = wall_patch
                continue
            if not (np.allclose(face.normal, ref_normal, 1e-3) or np.allclose(face.normal, -ref_normal, 1e-3)):
                if face.blocks.__len__() == 1:
                    if face.boundary is None:
                        face.boundary = wall_patch
                continue
            # bottom side:
            if self.layer_interface_planes[0].distToShape(FCPart.Vertex(Base.Vector(face.dirty_center)))[0] < 1e-3:
                face.boundary = bottom_side_patch
                bottom_side_faces.append(face)
            elif self.layer_interface_planes[-1].distToShape(FCPart.Vertex(Base.Vector(face.dirty_center)))[0] < 1e-3:
                face.boundary = top_side_patch
                top_side_faces.append(face)

        # export_objects([x.fc_face for x in self._comp_blocks.hull_faces], '/tmp/hull_faces.FCStd')
        # export_objects(FCPart.Compound([x.fc_face for x in self._comp_blocks.hull_faces]), '/tmp/hull_faces.FCStd')
        # export_objects(FCPart.Compound([x.fc_face for x in top_side_faces]), '/tmp/top_side_faces.FCStd')
        # export_objects(FCPart.Compound([x.fc_face for x in bottom_side_faces]), '/tmp/bottom_side_faces.FCStd')

        logger.info('Updated boundary conditions successfully')

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


def replace(arr, find, replace):
    # fast and readable
    base = 0
    for cnt in range(arr.count(find)):
        offset = arr.index(find, base)
        arr[offset] = replace
        base = offset + 1

    return arr

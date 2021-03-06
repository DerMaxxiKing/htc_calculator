import numpy as np
import copy

from ..logger import logger
from .block_mesh import BlockMeshVertex, BlockMeshEdge, Block, create_edges_between_layers, BlockMeshBoundary
from ..geo_tools import get_position
from ..tools import vector_to_np_array, perpendicular_vector, export_objects
from ..case.boundary_conditions.user_bcs import VolumeFlowInlet, Outlet, FluidWall

import FreeCAD
import Part as FCPart
from FreeCAD import Base
import DraftVecUtils


class PipeSection(object):

    def __init__(self, *args, **kwargs):

        self._block_mesh = None
        self._inlet_patch = None
        self._outlet_patch = None
        self._fluid_wall_patch = None

        self.name = kwargs.get('name')
        self.layer_vertex_gen_function = kwargs.get('layer_vertex_gen_function')    # function which generates vertices for the section
        self.edge_def = kwargs.get('edge_def')                                      # List with vertex-ids for inner and outer edges
        self.vertex_indices = kwargs.get('vertex_indices')                          #
        self.edge_indices = kwargs.get('edge_indices')                              #
        self.cell_zones = kwargs.get('cell_zones')  #
        self.block_cell_zones = kwargs.get('block_cell_zones')                      #
        self.cell_zone_ids = kwargs.get('cell_zone_ids')                            #

        self.n_cell = kwargs.get('n_cell')                                          #
        self.cell_size = kwargs.get('cell_size')
        self.grading = kwargs.get('grading')

        self.pipe_wall_def = kwargs.get('pipe_wall_def')

        self.block_inlet_faces = kwargs.get('block_inlet_faces')
        self.block_outlet_faces = kwargs.get('block_outlet_faces')

        self._materials = kwargs.get('materials', [])

        self.top_side = kwargs.get('top_side', dict())
        self.bottom_side = kwargs.get('bottom_side', dict())
        self.interface_side = kwargs.get('interface_side', dict())
        self.merge_patch_pairs = kwargs.get('merge_patch_pairs', [{}, {}])

        self.block_mesh = kwargs.get('block_mesh', None)

        if kwargs.get('inlet_patch', None) is not None:
            self.inlet_patch = kwargs.get('inlet_patch', None)
        if kwargs.get('outlet_patch', None) is not None:
            self.outlet_patch = kwargs.get('outlet_patch', None)
        if kwargs.get('fluid_wall_patch', None) is not None:
            self.fluid_wall_patch = kwargs.get('fluid_wall_patch', None)

    @property
    def materials(self):
        return self._materials

    @materials.setter
    def materials(self, value):
        if value is not None:
            self._materials = value
        for i, material in enumerate(self._materials):
            if self.cell_zones is not None:
                self.cell_zones[i].material = material

    @property
    def block_mesh(self):
        return self._block_mesh

    @block_mesh.setter
    def block_mesh(self, value):
        if value is self._block_mesh:
            return
        self._block_mesh = value

        self.inlet_patch = BlockMeshBoundary(name='inlet',
                                             type='patch',
                                             user_bc=VolumeFlowInlet(),
                                             mesh=self._block_mesh.mesh)

        self.outlet_patch = BlockMeshBoundary(name='outlet',
                                              type='patch',
                                              user_bc=Outlet(),
                                              mesh=self._block_mesh.mesh)

        self.fluid_wall_patch = BlockMeshBoundary(name='fluid_wall',
                                                  type='wall',
                                                  user_bc=FluidWall(),
                                                  mesh=self._block_mesh.mesh)

    @property
    def inlet_patch(self):
        return self._inlet_patch

    @inlet_patch.setter
    def inlet_patch(self, value):
        self._inlet_patch = value

    @property
    def outlet_patch(self):
        return self._outlet_patch

    @outlet_patch.setter
    def outlet_patch(self, value):
        self._outlet_patch = value

    @property
    def fluid_wall_patch(self):
        return self._fluid_wall_patch

    @fluid_wall_patch.setter
    def fluid_wall_patch(self, value):
        self._fluid_wall_patch = value

    def create_block(self, *args, **kwargs):

        edge = kwargs.get('edge')
        face_normal = kwargs.get('face_normal')
        tube_inner_diameter = kwargs.get('tube_inner_diameter')
        tube_diameter = kwargs.get('tube_diameter')
        outer_pipe = kwargs.get('outer_pipe')
        inlet = kwargs.get('inlet')
        outlet = kwargs.get('outlet')

        layer1_vertices, layer2_vertices, construction_points1, construction_points2 = self.create_layer_vertices(edge,
                                                                                                                  face_normal,
                                                                                                                  tube_inner_diameter,
                                                                                                                  tube_diameter,
                                                                                                                  outer_pipe)

        layer1_edges = self.create_layer_edges(layer1_vertices, construction_points1, outer_pipe)
        layer2_edges = self.create_layer_edges(layer2_vertices, construction_points2, outer_pipe)

        if outer_pipe:
            block_vertex_indices = [*self.vertex_indices[0], *self.vertex_indices[1]]
            block_cell_zones = [*self.block_cell_zones[0], *self.block_cell_zones[1]]
            block_edge_indices = [*self.edge_indices[0], *self.edge_indices[1]]
            merge_patch_pairs = self.merge_patch_pairs[0] | self.merge_patch_pairs[1]
        else:
            block_vertex_indices = self.vertex_indices[0]
            block_cell_zones = self.block_cell_zones[0]
            block_edge_indices = self.edge_indices[0]
            merge_patch_pairs = self.merge_patch_pairs[0]

        blocks = []
        for i in range(block_vertex_indices.__len__()):

            vertex_indices = block_vertex_indices[i]
            cell_zones = block_cell_zones[i]
            edge_indices = block_edge_indices[i]

            vertices = [*[layer1_vertices[x] for x in vertex_indices],
                        *[layer2_vertices[x] for x in vertex_indices]]

            connect_edges = create_edges_between_layers(vertices[0:4], vertices[4:], edge, face_normal)

            block_edges = [*[layer1_edges[x] for x in edge_indices], *[layer2_edges[x] for x in edge_indices],
                           *connect_edges]

            n_cell = copy.deepcopy(self.n_cell)

            if type(edge.Curve) is FCPart.Line:
                cell_multiplier = 1
            else:
                cell_multiplier = 2

            if None in n_cell:
                if n_cell[0] is None:
                    n_cell[0] = int(np.ceil(block_edges[0].length / self.cell_size[0])) * cell_multiplier
                if n_cell[1] is None:
                    n_cell[1] = int(np.ceil(block_edges[3].length / self.cell_size[1])) * cell_multiplier
                if n_cell[2] is None:
                    n_cell[2] = int(np.ceil(edge.Length / self.cell_size[2])) * cell_multiplier

            # top / bottom side
            pipe_layer_top = i in self.top_side.keys()
            pipe_layer_bottom = i in self.bottom_side.keys()

            if pipe_layer_top:
                pipe_layer_extrude_top = self.top_side[i]
            else:
                pipe_layer_extrude_top = None
            if pipe_layer_bottom:
                pipe_layer_extrude_bottom = self.bottom_side[i]
            else:
                pipe_layer_extrude_bottom = None

            if cell_zones is not None:
                block_name = 'Pipe Block ' + cell_zones.material.name
                cell_zone = cell_zones
            else:
                block_name = 'Pipe Outer Block'
                cell_zone = None

            new_block = Block(name=block_name,
                              vertices=vertices,
                              edge=edge,
                              block_edges=block_edges,
                              num_cells=n_cell,
                              grading=self.grading[i],
                              cell_zone=cell_zone,
                              extruded=True,
                              merge_patch_pairs=merge_patch_pairs.get(i, False),
                              pipe_layer_top=pipe_layer_top,
                              pipe_layer_bottom=pipe_layer_bottom,
                              pipe_layer_extrude_top=pipe_layer_extrude_top,
                              pipe_layer_extrude_bottom=pipe_layer_extrude_bottom)
            # try:
            #     export_objects([new_block.fc_solid], '/tmp/blocks.FCStd')
            # except Exception as e:
            #     export_objects([x.fc_edge for x in new_block.face3.edges], '/tmp/face3_edges.FCStd')
            #     export_objects([x.fc_edge for x in block_edges], '/tmp/init_block_edges.FCStd')
            #     export_objects([new_block.face0.fc_face], '/tmp/face0.FCStd')
            #     export_objects([x.fc_face for x in new_block.faces], '/tmp/faces.FCStd')
            #     export_objects([x.fc_edge for x in new_block.block_edges], '/tmp/edges.FCStd')
            #     raise e

            blocks.append(new_block)

        if not outer_pipe:
            _ = [[setattr(blocks[xx[0]].faces[yy], 'boundary', self.fluid_wall_patch)
                  for yy in xx[1]] for xx in self.pipe_wall_def]

        if inlet:
            for block_inlet in self.block_inlet_faces:
                for face_nr in block_inlet[1]:
                    blocks[block_inlet[0]].faces[face_nr].boundary = self.inlet_patch
        if outlet:
            for block_outlet in self.block_outlet_faces:
                for face_nr in block_outlet[1]:
                    blocks[block_outlet[0]].faces[face_nr].boundary = self.outlet_patch
        return blocks

    def create_layer_vertices(self,
                              edge,
                              face_normal,
                              tube_inner_diameter,
                              tube_diameter,
                              outer_pipe):

        start_point = get_position(edge.Vertexes[0])
        end_point = get_position(edge.Vertexes[1])
        direction = vector_to_np_array(edge.tangentAt(edge.FirstParameter))
        perp_vec = perpendicular_vector(face_normal, direction)

        layer1_vertices, construction_points1 = self.layer_vertex_gen_function(start_point,
                                                                               face_normal,
                                                                               perp_vec,
                                                                               tube_inner_diameter,
                                                                               tube_diameter,
                                                                               outer_pipe)

        if type(edge.Curve) is FCPart.Line:
            trans = Base.Vector(end_point - start_point)
            layer2_vertices = [x + trans for x in layer1_vertices]
            construction_points2 = construction_points1 + np.array(trans)

            # export_objects([*[x.fc_vertex.toShape() for x in layer1_vertices],
            #                 *[FCPart.Point(Base.Vector(x)).toShape() for x in construction_points1],
            #                 *[x.fc_vertex.toShape() for x in layer2_vertices],
            #                 *[FCPart.Point(Base.Vector(x)).toShape() for x in construction_points2]],
            #                '/tmp/vertices.FCStd')

        else:
            center = np.array(edge.Curve.Center)

            rot_angle = np.rad2deg(DraftVecUtils.angle(Base.Vector(start_point - center),
                                                       Base.Vector(end_point - center),
                                                       Base.Vector(face_normal)))

            vertex_wire = FCPart.Wire(
                [FCPart.makeLine(Base.Vector(layer1_vertices[i].position),
                                 Base.Vector(layer1_vertices[i + 1].position)) for i in
                 range(layer1_vertices.__len__() - 1)]
            )

            constr_wire = FCPart.Wire(
                [FCPart.makeLine(Base.Vector(construction_points1[i]),
                                 Base.Vector(construction_points1[i + 1])) for i in
                 range(construction_points1.__len__() - 1)]
            )

            layer2_vertices = [BlockMeshVertex(position=np.array(x.Point)) for x in
                               vertex_wire.rotate(edge.Curve.Center,
                                                  face_normal,
                                                  rot_angle).Vertexes]

            construction_points2 = [np.array(x.Point) for x in
                                    constr_wire.rotate(edge.Curve.Center,
                                                       face_normal,
                                                       rot_angle).Vertexes]

        return layer1_vertices, layer2_vertices, construction_points1, construction_points2

    def create_layer_edges(self, layer_vertices, construction_points, outer_pipe):

        if outer_pipe:
            c_edef = [*self.edge_def[0], *self.edge_def[1]]
        else:
            c_edef = self.edge_def[0]

        edges = [None] * c_edef.__len__()
        for i, e_def in enumerate(c_edef):
            if e_def[1] == 'line':
                if e_def.__len__() > 2:
                    fixed_num_cells = True
                    num_cells = e_def[2]
                else:
                    num_cells = None
                    fixed_num_cells = False

                edges[i] = BlockMeshEdge(vertices=[layer_vertices[e_def[0][0]],
                                                   layer_vertices[e_def[0][1]]],
                                         type='line',
                                         fixed_num_cells=fixed_num_cells,
                                         num_cells=num_cells)
            elif e_def[1] == 'arc':
                if e_def.__len__() > 3:
                    fixed_num_cells = True
                    num_cells = e_def[3]
                else:
                    num_cells = None
                    fixed_num_cells = False

                edges[i] = BlockMeshEdge(vertices=[layer_vertices[e_def[0][0]], layer_vertices[e_def[0][1]]],
                                         type='arc',
                                         interpolation_points=[Base.Vector(construction_points[e_def[2][0]])],
                                         fixed_num_cells=fixed_num_cells,
                                         num_cells=num_cells
                                         )
        return edges

import numpy as np

from ..logger import logger
from .block_mesh import BlockMeshVertex, BlockMeshEdge, Block, unit_vector, create_edges_between_layers, pipe_wall_patch, inlet_patch, outlet_patch
from ..geo_tools import get_position
from ..tools import vector_to_np_array, perpendicular_vector, export_objects

import FreeCAD
import Part as FCPart
from FreeCAD import Base
import DraftVecUtils


class PipeSection(object):

    def __init__(self, *args, **kwargs):

        self.name = kwargs.get('name')
        self.layer_vertex_gen_function = kwargs.get('layer_vertex_gen_function')    # function which generates vertices for the section
        self.edge_def = kwargs.get('edge_def')                                      # List with vertex-ids for inner and outer edges
        self.vertex_indices = kwargs.get('vertex_indices')                          #
        self.edge_indices = kwargs.get('edge_indices')                              #
        self.cell_zones = kwargs.get('cell_zones')                                  #

        self.n_cell = kwargs.get('n_cell')                                          #
        self.cell_size = kwargs.get('cell_size')

        self.pipe_wall_def = kwargs.get('pipe_wall_def')

        self.block_inlet_faces = kwargs.get('block_inlet_faces')
        self.block_outlet_faces = kwargs.get('block_outlet_faces')

    def create_block(self,
                     edge,
                     face_normal,
                     tube_inner_diameter,
                     tube_diameter,
                     outer_pipe: bool = True,
                     inlet: bool = False,
                     outlet: bool = False):

        layer1_vertices, layer2_vertices, construction_points1, construction_points2 = self.create_layer_vertices(edge,
                                                                                                                  face_normal,
                                                                                                                  tube_inner_diameter,
                                                                                                                  tube_diameter,
                                                                                                                  outer_pipe)

        layer1_edges = self.create_layer_edges(layer1_vertices, construction_points1, outer_pipe)
        layer2_edges = self.create_layer_edges(layer2_vertices, construction_points2, outer_pipe)

        if outer_pipe:
            block_vertex_indices = [*self.vertex_indices[0], *self.vertex_indices[1]]
            block_cell_zones = [*self.cell_zones[0], *self.cell_zones[1]]
            block_edge_indices = [*self.edge_indices[0], *self.edge_indices[1]]
        else:
            block_vertex_indices = self.vertex_indices[0]
            block_cell_zones = self.cell_zones[0]
            block_edge_indices = self.edge_indices[0]

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

            n_cell = self.n_cell
            if None in n_cell:
                if n_cell[0] is None:
                    n_cell[0] = block_edges[0].length / self.cell_size[0]
                if n_cell[1] is None:
                    n_cell[1] = block_edges[3].length / self.cell_size[1]
                if n_cell[2] is None:
                    n_cell[2] = edge.Length / self.cell_size[2]

            new_block = Block(name=cell_zones,
                              vertices=vertices,
                              edge=edge,
                              block_edges=block_edges,
                              num_cells=n_cell,
                              cell_zone=cell_zones,
                              extruded=True,
                              check_merge_patch_pairs=False)

            blocks.append(new_block)

        _ = [[setattr(blocks[xx[0]].faces[yy], 'boundary', pipe_wall_patch) for yy in xx[1]] for xx in self.pipe_wall_def]

        if inlet:
            for block_inlet in self.block_inlet_faces:
                for face_nr in block_inlet[1]:
                    blocks[block_inlet[0]].faces[face_nr].boundary = inlet_patch
        if outlet:
            for block_outlet in self.block_outlet_faces:
                for face_nr in block_outlet[1]:
                    blocks[block_outlet[0]].faces[face_nr].boundary = inlet_patch

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
                edges[i] = BlockMeshEdge(vertices=[layer_vertices[e_def[0][0]], layer_vertices[e_def[0][1]]], type='line')
            elif e_def[1] == 'arc':
                edges[i] = BlockMeshEdge(vertices=[layer_vertices[e_def[0][0]], layer_vertices[e_def[0][1]]],
                                         type='arc',
                                         interpolation_points=[Base.Vector(construction_points[e_def[2][0]])])

                # export_objects([layer_vertices[e_def[0][0]].fc_vertex.toShape(),
                #                 layer_vertices[e_def[0][1]].fc_vertex.toShape(),
                #                 FCPart.Point(Base.Vector(construction_points[e_def[2][0]])).toShape()], '/tmp/arc_edge.FCStd')
        return edges

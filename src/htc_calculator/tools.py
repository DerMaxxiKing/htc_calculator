import os
import numpy as np
import pathlib
import gmsh
import meshio
import itertools

from .logger import logger

import FreeCAD
import Part as FCPart
import Points
from Draft import make_fillet
from FreeCAD import Base
import BOPTools.SplitFeatures

from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.BRepBndLib import brepbndlib_AddOBB
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_XYZ
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox


App = FreeCAD


def name_step_faces(fname, name=None, new_fname=None, delete=True, debug=False):
    basename, extension = os.path.splitext (fname)
    if new_fname is None:
        new_fname = '{}_named{}'.format(basename, extension)
    new_file = open(new_fname, 'w')

    # replacement string
    repstr = "FACE('{}'"
    # reverse sorted ordinals
    pos = list(name.keys())
    pos.sort(reverse=True)
    counter = 0
    with open(fname) as file:
        for line in file:
            if ('ADVANCED_FACE' in line):
                if counter in pos:
                    face_name = name.pop(pos.pop())
                    line = line.replace(repstr.format(''), repstr.format(face_name))
                    if debug:
                        print(line)
                counter += 1
            new_file.write(line)
    file.close()
    new_file.close()

    if delete:
        try:
            os.remove(fname)
        except Exception as e:
            if debug:
                print(e)


def generate_solid_from_faces(faces, solid_id):

    face0 = faces[0]
    faces = faces[1:]
    shell = face0.multiFuse((faces), 1e-3)
    solid = FCPart.Solid(shell)

    doc = App.newDocument()
    __o__ = doc.addObject("Part::Feature", f'{solid_id}')
    __o__.Label = f'{solid_id}'
    __o__.Shape = solid

    solid = __o__
    return solid


def project_point_on_line(point, line):

    p1 = np.array(line.Vertex1.Point)
    p2 = np.array(line.Vertex2.Point)

    p3 = np.array(point)

    # distance between p1 and p2
    l2 = np.sum((p1 - p2) ** 2)
    if l2 == 0:
        print('p1 and p2 are the same points')

    # The line extending the segment is parameterized as p1 + t (p2 - p1).
    # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

    # if you need the point to project on line extention connecting p1 and p2
    t = np.sum((p3 - p1) * (p2 - p1)) / l2

    # if you need to ignore if p3 does not project onto line segment
    if t > 1 or t < 0:
        print('p3 does not project onto p1-p2 line segment')

    # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

    projection = p1 + t * (p2 - p1)
    return projection


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

    if isinstance(p1, Base.Vector):
        p1 = np.array(p1)

    if isinstance(p2, Base.Vector):
        p2 = np.array(p2)

    if isinstance(p3, Base.Vector):
        p3 = np.array(p3)

    v1 = p2 - p1
    v2 = p3 - p1

    u_v1 = v1 / np.linalg.norm(v1)
    u_v2 = v2 / np.linalg.norm(v2)

    angle = np.arccos(np.dot(u_v1, u_v2))

    if deg:
        return np.rad2deg(angle)
    else:
        return angle


export_iter = itertools.count()


def export_objects(objects, filename, add_suffix=True):

    if not isinstance(objects, list):
        objects = [objects]

    file_suffix = pathlib.Path(filename).suffix

    doc = App.newDocument()

    for i, object in enumerate(objects):

        __o__ = doc.addObject("Part::Feature", f'{type(object).__name__}{i}')
        __o__.Label = f'{type(object).__name__}{i}'
        __o__.Shape = object

    if add_suffix:
        id = next(export_iter)
        p = pathlib.Path(filename)
        filename = "{0}_{2}{1}".format(pathlib.Path.joinpath(p.parent, p.stem), p.suffix, id)

    if file_suffix == '.FCStd':
        doc.recompute()
        doc.saveCopy(filename)
    else:
        FCPart.export(doc.Objects, filename)


def import_file(filename):

    from .face import Face
    from .solid import Solid
    from .assembly import Assembly

    imported_shape = FCPart.Shape()
    imported_shape.read(filename)

    solid0 = imported_shape.Solids[0]
    solids = [x.fc_solid.Shape for x in imported_shape.Solidsolids[1:]]
    hull = solid0.multiFuse((solids), 1e-6)

    # faces = []
    # solids = []
    # assemblies = []

    def import_shape(loaded_shape):

        n_faces = []
        n_solids = []
        n_assemblies = []

        for shape in loaded_shape.SubShapes:
            if isinstance(shape, FCPart.Solid):
                solid_faces = []
                for face in shape.Faces:
                    solid_faces.append(Face(fc_face=face))
                n_faces.extend(solid_faces)
                n_solids.append(Solid(faces=solid_faces))
            elif isinstance(shape, FCPart.Face):
                n_faces.append(Face(fc_face=shape))
            elif isinstance(shape, FCPart.Shell):
                n_faces.append(Face(fc_face=shape))
            elif isinstance(shape, FCPart.CompSolid):
                faces, solids, assemblies = import_shape(shape)
                n_faces.extend(faces)
                n_solids.extend(solids)
                n_assemblies.append(Assembly(solids=solids))

        return n_faces, n_solids, n_assemblies

    faces, solids, assemblies = import_shape(imported_shape)
    return faces, solids, assemblies


def create_obb(points, box_points=True):

    vectors = [FreeCAD.Vector(p)for p in points]
    pts = Points.Points(vectors)
    Points.show(pts)

    obb = Bnd_OBB()
    for p in points:
        pnt = BRepBuilderAPI_MakeVertex(gp_Pnt(float(p[0]), float(p[1]), float(p[2]))).Shape()
        brepbndlib_AddOBB(pnt, obb)

    aXDir = obb.XDirection()
    aYDir = obb.YDirection()
    aZDir = obb.ZDirection()
    aHalfX = obb.XHSize()
    aHalfY = obb.YHSize()
    aHalfZ = obb.ZHSize()

    aBaryCenter = obb.Center()

    if box_points:
        ax = np.array([aXDir.X(), aXDir.Y(), aXDir.Z()])
        ay = np.array([aYDir.X(), aYDir.Y(), aYDir.Z()])
        az = np.array([aZDir.X(), aZDir.Y(), aZDir.Z()])

        center = [aBaryCenter.X(), aBaryCenter.Y(), aBaryCenter.Z()]

        return np.array([center - ax * aHalfX - ay * aHalfY + az * aHalfZ,
                         center - ax * aHalfX + ay * aHalfY + az * aHalfZ,
                         center + ax * aHalfX + ay * aHalfY + az * aHalfZ,
                         center + ax * aHalfX - ay * aHalfY + az * aHalfZ,
                         center - ax * aHalfX - ay * aHalfY - az * aHalfZ,
                         center - ax * aHalfX + ay * aHalfY - az * aHalfZ,
                         center + ax * aHalfX + ay * aHalfY - az * aHalfZ,
                         center + ax * aHalfX - ay * aHalfY - az * aHalfZ,
                         ])

    else:

        ax = gp_XYZ(aXDir.X(), aXDir.Y(), aXDir.Z())
        ay = gp_XYZ(aYDir.X(), aYDir.Y(), aYDir.Z())
        az = gp_XYZ(aZDir.X(), aZDir.Y(), aZDir.Z())
        p = gp_Pnt(aBaryCenter.X(), aBaryCenter.Y(), aBaryCenter.Z())
        anAxes = gp_Ax2(p, gp_Dir(aZDir), gp_Dir(aXDir))
        anAxes.SetLocation(gp_Pnt(p.XYZ() - ax * aHalfX - ay * aHalfY - az * aHalfZ))
        aBox = BRepPrimAPI_MakeBox(anAxes, 2.0 * aHalfX, 2.0 * aHalfY, 2.0 * aHalfZ).Shape()
        return aBox


def extract_to_meshio():
    # extract point coords
    idx, points, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(points).reshape(-1, 3)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    points = points[srt]

    # extract cells
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements()
    cells = []
    for elem_type, elem_tags, node_tags in zip(elem_types, elem_tags, node_tags):
        # `elementName', `dim', `order', `numNodes', `localNodeCoord',
        # `numPrimaryNodes'
        num_nodes_per_cell = gmsh.model.mesh.getElementProperties(elem_type)[3]

        node_tags_reshaped = np.asarray(node_tags).reshape(-1, num_nodes_per_cell) - 1
        node_tags_sorted = node_tags_reshaped[np.argsort(elem_tags)]
        cells.append(
            meshio.CellBlock(
                meshio.gmsh.gmsh_to_meshio_type[elem_type], node_tags_sorted
            )
        )

    cell_sets = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        cell_sets[name] = [[] for _ in range(len(cells))]
        for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            # TODO node_tags?
            # elem_types, elem_tags, node_tags
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, e)
            assert len(elem_types) == len(elem_tags)
            assert len(elem_types) == 1
            elem_type = elem_types[0]
            elem_tags = elem_tags[0]

            meshio_cell_type = meshio.gmsh.gmsh_to_meshio_type[elem_type]
            # make sure that the cell type appears only once in the cell list
            # -- for now
            idx = []
            for k, cell_block in enumerate(cells):
                if cell_block.type == meshio_cell_type:
                    idx.append(k)
            assert len(idx) == 1
            idx = idx[0]
            cell_sets[name][idx].append(elem_tags - 1)

        cell_sets[name] = [
            (None if len(idcs) == 0 else np.concatenate(idcs))
            for idcs in cell_sets[name]
        ]

    # make meshio mesh
    return meshio.Mesh(points, cells, cell_sets=cell_sets)
# axis coming soon


def vector_to_np_array(vector):
    return np.array([vector.x, vector.y, vector.z])


def perpendicular_vector(x, y):
    return np.cross(x, y)


def extrude(path, sections: list, additional_paths=None, occ=False):

    if occ:
        ps = FCPart.BRepOffsetAPI.MakePipeShell(path)
        ps.setFrenetMode(False)
        ps.setSpineSupport(path)
        # ps.setAuxiliarySpine(FCPart.Wire(self.extruded[3].fc_edge), True, False)
        for section in sections:
            ps.add(section, True, True)
            ps.add(section, True, True)
        if ps.isReady():
            ps.build()
        return ps.shape()
    else:
        doc = App.newDocument()
        sweep = doc.addObject('Part::Sweep', 'Sweep')

        sweep_sections = []
        for i, section in enumerate(sections):
            sec = doc.addObject("Part::Feature", f'section{i}')
            sec.Shape = section
            sweep_sections.append(sec)

        spine = doc.addObject("Part::Feature", f'spine')
        spine.Shape = path

        sweep.Sections = sweep_sections
        sweep.Spine = spine
        sweep.Solid = False
        sweep.Frenet = True

        doc.recompute()

        return sweep.Shape


def create_pipe(edges, tube_diameter, face_normal):

    inlet = None
    outlet = None
    faces = []
    for i, edge in enumerate(edges):
        c1 = FCPart.makeCircle(tube_diameter / 2, edge.Vertex1.Point, edge.tangentAt(edge.FirstParameter))
        c2 = FCPart.makeCircle(tube_diameter / 2, edge.Vertex2.Point, edge.tangentAt(edge.LastParameter))
        pipe_profile1 = FCPart.Wire([c1])
        pipe_profile2 = FCPart.Wire([c2])
        if i == 0:
            inlet = FCPart.Face(pipe_profile1)
        if i == edges.__len__() - 1:
            outlet = FCPart.Face(pipe_profile1)

        new_faces = extrude(FCPart.Wire([edge]), [pipe_profile1, pipe_profile2], occ=False)
        faces.extend(new_faces.Faces)
        # export_objects([pipe_profile1, pipe_profile2, edge, new_faces], '/tmp/test3.FCStd')
    # export_objects([*faces, inlet, outlet], '/tmp/test3.FCStd')

    shell = FCPart.makeShell([*faces, inlet, outlet])
    shell.sewShape()
    shell.fix(1e-3, 1e-3, 1e-3)
    solid = FCPart.Solid(shell)

    # export_objects([solid], '/tmp/solid_test.FCStd')

    return solid


def add_radius_to_edges(edges, radius):

    edges_with_radius = FCPart.Wire(edges[0:1])

    i = 1
    while i < edges.__len__():

        current_edges = edges_with_radius.OrderedEdges

        if i == 22:
            print('error')

        dir1 = vector_to_np_array(current_edges[-1].tangentAt(current_edges[-1].LastParameter))
        dir2 = vector_to_np_array(edges[i].tangentAt(edges[i].LastParameter))

        if np.allclose(dir1, dir2, 1e-5) or np.allclose(dir1, -dir2, 1e-5):
            current_edges.append(edges[i])
            edges_with_radius = FCPart.Wire(FCPart.sortEdges(current_edges)[0])
            i += 1
            continue
        try:
            new_edges = make_fillet([current_edges[-1], edges[i]], radius=radius)
            if new_edges is not None:
                current_edges[-1] = new_edges.Shape.OrderedEdges[0]
                current_edges.extend(new_edges.Shape.OrderedEdges[1:])
            else:
                current_edges.append(edges[i])

            FCPart.sortEdges(current_edges)
            edges_with_radius = FCPart.Wire(FCPart.sortEdges(current_edges)[0])

        except Exception as e:
            print(e)

        i += 1

    # export_objects([*current_edges, *new_edges.Shape.Edges], '/tmp/edges2.FCStd')
    # export_objects(current_edges, '/tmp/edges8.FCStd')
    # export_objects(edges_with_radius.OrderedEdges, '/tmp/edges_with_radius7.FCStd')
    # export_objects(edges, '/tmp/next_edge2.FCStd')
    # export_objects([edges_with_radius[-1], edges[i]], '/tmp/next_edge2.FCStd')

    return edges_with_radius


def face_normal(fc_face):

    normals = []
    for vertex in fc_face.Vertexes:
        u, v = fc_face.Surface.parameter(vertex.Point)
        nv = fc_face.normalAt(u, v)
        normals.append(vector_to_np_array(nv.normalize()))

    return normals


def edge_to_line(edge):
    return FCPart.Line(edge.Vertex1.Point, edge.Vertex2.Point)


def intersect_lines(edge1, edge2):
    if not isinstance(edge1, FCPart.Line):
        edge1 = edge_to_line(edge1)

    if not isinstance(edge2, FCPart.Line):
        edge2 = edge_to_line(edge2)

    return edge1.intersect(edge2)


def extrude_edge(edge: FCPart.Edge, side=None, dist: float = 99999999, include=True):

    param1 = edge.FirstParameter
    param2 = edge.LastParameter

    if side is None:
        return FCPart.LineSegment(edge.valueAt(param1) - edge.tangentAt(param1) * dist,
                                  edge.valueAt(param2) + edge.tangentAt(param2) * dist).toShape()
    else:
        if side == 1:
            if include:
                return FCPart.LineSegment(edge.valueAt(param1) - edge.tangentAt(param1) * dist,
                                          edge.valueAt(param2)).toShape()
            else:
                return FCPart.LineSegment(edge.valueAt(param1) - edge.tangentAt(param1) * dist,
                                          edge.valueAt(param1)).toShape()
        else:
            if include:
                return FCPart.LineSegment(edge.valueAt(param1),
                                          edge.valueAt(param2) + edge.tangentAt(param2) * dist).toShape()
            else:
                return FCPart.LineSegment(edge.valueAt(param2),
                                          edge.valueAt(param2) + edge.tangentAt(param2) * dist).toShape()


def connect_edges(edge1, edge2, side1, side2):

    e1_param = [None, edge1.FirstParameter, edge1.LastParameter]
    e2_param = [None, edge2.FirstParameter, edge2.LastParameter]

    return FCPart.LineSegment(edge1.valueAt(e1_param[side1]),
                              edge2.valueAt(e2_param[side2])).toShape()


def create_pipe_wire(reference_face,
                     start_edge=0,
                     tube_distance=225,
                     tube_edge_distance=300,
                     bending_radius=100,
                     tube_diameter=20,
                     layout='DoubleSerpentine'):

    normal = face_normal(reference_face)[0]
    start_edge = reference_face.OuterWire.Edges[start_edge]

    # real_edge_distance = tube_edge_distance + bending_radius + 2.5 * tube_diameter

    if layout == 'DoubleSerpentine':
        logger.debug(f'Creating pipe with DoubleSerpentine layout')

        inflow_dir = perpendicular_vector(normal,
                                          vector_to_np_array(start_edge.tangentAt(
                                              start_edge.LastParameter - tube_edge_distance)
                                          ))

        # create horizontal lines
        # -----------------------------------------------------------------

        # line 0
        horizontal_lines = []

        base_edge = FCPart.LineSegment(
            start_edge.Vertex1.Point - start_edge.tangentAt(start_edge.FirstParameter) * 10000000,
            start_edge.Vertex2.Point + start_edge.tangentAt(start_edge.LastParameter) * 1000000).toShape()

        movement = Base.Vector(inflow_dir) * 0

        cut_wire = reference_face.OuterWire.makeOffset2D(-tube_edge_distance - 0.5 * tube_diameter,
                                                         join=1,
                                                         openResult=False,
                                                         intersection=False)

        jump3_wire = reference_face.OuterWire.makeOffset2D(-tube_edge_distance - 0.5 * tube_diameter,
                                                           join=1,
                                                           openResult=False,
                                                           intersection=False)
        jump3_comp = FCPart.Compound([extrude_edge(x,
                                                   dist=tube_edge_distance + 0.5 * tube_diameter)
                                      for x in jump3_wire.Edges])

        jump1_wire = reference_face.OuterWire.makeOffset2D(-tube_edge_distance - 3 * tube_diameter,
                                                           join=1,
                                                           openResult=False,
                                                           intersection=False)

        jump1_comp = FCPart.Compound([extrude_edge(x,
                                                   dist=tube_edge_distance + 3 * tube_diameter)
                                      for x in jump1_wire.Edges])

        i = 0
        while True:
            if i == 0:
                movement = Base.Vector(inflow_dir) * (tube_edge_distance + 1 + 0.5 * tube_diameter)
            else:
                movement = movement + Base.Vector(inflow_dir) * tube_distance

            e_init = base_edge.copy()
            e_init.Placement.move(movement)
            cut_shapes = e_init.cut(cut_wire)
            if cut_shapes.SubShapes.__len__() > 1:
                e_init2 = cut_shapes.SubShapes[1]
                e0 = FCPart.LineSegment(e_init2.valueAt(e_init2.FirstParameter + 2 * bending_radius + 3.5 * tube_diameter),
                                        e_init2.valueAt(e_init2.LastParameter - 2 * bending_radius - 3.5 * tube_diameter)).toShape()
                horizontal_lines.append(e0)
            else:
                break

            i += 1

        if (horizontal_lines.__len__() % 2) != 0:
            horizontal_lines = horizontal_lines[0:-1]

        # export_objects([reference_face, jump1_wire, jump3_wire, *horizontal_lines], '/tmp/h_lines.FCStd')

        # create pipe edges:
        pipe_edges_in = []
        pipe_edges_out = []

        # create outflow
        h0_edge = horizontal_lines[0]
        lv1 = extrude_edge(h0_edge, side=2, dist=1000)
        lv1_cut = lv1.cut(jump1_comp)
        out_edge0 = FCPart.LineSegment(h0_edge.Vertex2.Point,
                                       lv1_cut.SubShapes[0].Vertex2.Point).toShape()
        out_edge1 = FCPart.LineSegment(out_edge0.Vertex2.Point,
                                       Base.Vector(project_point_on_line(out_edge0.Vertex2.Point, start_edge))).toShape()
        out_edge2 = FCPart.LineSegment(out_edge1.Vertex2.Point,
                                       out_edge1.Vertex2.Point + out_edge1.tangentAt(out_edge1.LastParameter) * 500).toShape()
        pipe_edges_out.extend([out_edge0, out_edge1, out_edge2])

        # create inflow
        h1_edge = horizontal_lines[1]
        lv1 = extrude_edge(h1_edge, side=2, dist=1000)
        lv1_cut = lv1.cut(jump3_comp)
        in_edge0 = FCPart.LineSegment(h1_edge.Vertex2.Point,
                                      lv1_cut.SubShapes[0].Vertex2.Point).toShape()
        in_edge1 = FCPart.LineSegment(in_edge0.Vertex2.Point,
                                      out_edge0.Vertex2.Point + out_edge0.tangentAt(out_edge0.LastParameter) * 2.5 * tube_diameter).toShape()

        in_edge2 = FCPart.LineSegment(in_edge1.Vertex2.Point,
                                      Base.Vector(
                                          project_point_on_line(in_edge1.Vertex2.Point, start_edge))).toShape()

        in_edge3 = FCPart.LineSegment(in_edge2.Vertex2.Point,
                                      in_edge2.Vertex2.Point + in_edge2.tangentAt(in_edge2.LastParameter) * 500).toShape()

        pipe_edges_in.extend([in_edge0, in_edge1, in_edge2, in_edge3])

        pipe_edges_in, end_side_in = connect_h_lines(horizontal_lines,
                                                     bending_radius,
                                                     tube_diameter,
                                                     pipe_edges_in,
                                                     cut_wire,
                                                     normal,
                                                     [jump1_wire, jump3_wire],
                                                     jump=1,
                                                     i=1,
                                                     side=1)

        pipe_edges_out, end_side_out = connect_h_lines(horizontal_lines,
                                                       bending_radius,
                                                       tube_diameter,
                                                       pipe_edges_out,
                                                       cut_wire,
                                                       normal,
                                                       [jump1_wire, jump3_wire],
                                                       jump=3,
                                                       i=0,
                                                       side=1)

        if end_side_in != end_side_out:
            raise Exception(f'Error while connection horizontal lines. End side 1 is not equal to end side 2')

        if end_side_in == 2:
            ce_1 = extrude_edge(pipe_edges_in[-1], end_side_in, 1000, include=False).cut(jump3_wire).SubShapes[0]
            ce_3 = extrude_edge(pipe_edges_out[-1], end_side_in, 1000, include=False).cut(jump3_wire).SubShapes[0]
            ce_2 = get_inner_split(jump3_wire, [ce_1, ce_3]).Edges

        elif end_side_in == 1:
            ce_1 = extrude_edge(pipe_edges_in[-1], end_side_in, 1000, include=False).cut(jump3_wire).SubShapes[1]
            ce_3 = extrude_edge(pipe_edges_out[-1], end_side_in, 1000, include=False).cut(jump3_wire).SubShapes[1]
            ce_2 = get_inner_split(jump3_wire, [ce_1, ce_3]).Edges

        pipe_edges_out.reverse()

        pipe_wire = FCPart.Wire(FCPart.sortEdges([*pipe_edges_in,
                                                  ce_1, *ce_2, ce_3,
                                                  *pipe_edges_out])[0]
                                )

        return pipe_wire

    elif layout == 'Concentric':
        pipe_wire = None
        raise NotImplementedError


def connect_h_lines(horizontal_lines,
                    bending_radius,
                    tube_diameter,
                    pipe_edges,
                    cut_wire,
                    normal,     # reference face normal
                    jump_wires,
                    jump=1,
                    i=1,
                    side=1
                    ):

    pipe_edges.append(horizontal_lines[i])

    while i + jump < horizontal_lines.__len__():

        if jump == 1:
            jump_wire = jump_wires[0]
        else:
            jump_wire = jump_wires[1]

        con_edges = connect_lines(horizontal_lines[i],
                                  horizontal_lines[i + jump],
                                  side,
                                  jump_wire)
        pipe_edges.extend([*con_edges, horizontal_lines[i + jump]])

        # export_objects([*con_edges], f'/tmp/connect_hlines_{i}.FCStd')
        #
        # export_objects([*pipe_edges,
        #                 *jump_wires,
        #                 FCPart.Compound(horizontal_lines)], f'/tmp/connect_hlines_{i}.FCStd')

        i = i + jump

        if side == 1:
            side = 2
        elif side == 2:
            side = 1

        if jump == 1:
            jump = 3
        elif jump == 3:
            jump = 1

    return pipe_edges, side


def connect_lines(line1,
                  line2,
                  side,
                  jump_wire      # reference face normal
                  ):

    if side == 1:
        extr_edge = extrude_edge(line1, side=side, dist=1000, include=False)
        ex_cut1 = extr_edge.cut(jump_wire)
        c_edge1 = ex_cut1.SubShapes[1]

        extr_edge2 = extrude_edge(line2, side=side, dist=1000, include=False)
        ex_cut2 = extr_edge2.cut(jump_wire)
        c_edge3 = ex_cut2.SubShapes[1]

    elif side == 2:
        extr_edge = extrude_edge(line1, side=side, dist=1000, include=False)
        ex_cut1 = extr_edge.cut(jump_wire)
        c_edge1 = ex_cut1.SubShapes[0]

        extr_edge2 = extrude_edge(line2, side=side, dist=1000, include=False)
        ex_cut2 = extr_edge2.cut(jump_wire)
        c_edge3 = ex_cut2.SubShapes[0]

    c_wire2 = get_inner_split(jump_wire, [c_edge1, c_edge3])

    c_edges = FCPart.sortEdges([c_edge1, *c_wire2.Edges, c_edge3])[0]

    return c_edges


def get_inner_split(wire, edges):
    split_shapes = BOPTools.SplitAPI.slice(wire, edges, 'Split', tolerance=0.0)
    sub_wires_length = [x.Length for x in split_shapes.SubShapes]
    return split_shapes.SubShapes[sub_wires_length.index(min(sub_wires_length))]

import itertools
import os
import pathlib
from scipy.optimize import fsolve

import BOPTools.SplitFeatures
import FreeCAD
import Part as FCPart
import Points
import gmsh
import meshio
# import Draft
import numpy as np
from Draft import make_fillet
from FreeCAD import Base
import DraftVecUtils
from OCC.Core.BRepBndLib import brepbndlib_AddOBB
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_XYZ

from .logger import logger

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


def project_point_on_line(point, line, on_segment=False, return_on_segment=False):

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

    point_on_segment = True
    if t > 1 or t < 0:
        point_on_segment = False

    if on_segment:
        if t > 1 or t < 0:
            raise Exception(f'Point does not project onto line segment')

    # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    # t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

    projection = p1 + t * (p2 - p1)

    if return_on_segment:
        return projection, point_on_segment
    else:
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

    export_objects([*edges], '/tmp/tube_edges.FCStd')

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
            # export_objects([pipe_profile1, pipe_profile2, inlet, edge, FCPart.Wire(edges)], '/tmp/test3.FCStd')
        if i == edges.__len__() - 1:
            outlet = FCPart.Face(pipe_profile2)

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

    if isinstance(edges, FCPart.Wire):
        edges = edges.OrderedEdges

    edges_with_radius = FCPart.Wire(edges[0:1])

    # create closed wire
    wire = FCPart.Wire(edges)

    closed_wire = FCPart.Wire([*edges, FCPart.LineSegment(wire.OrderedVertexes[-1].Point,
                                                 wire.OrderedVertexes[0].Point).toShape()])
    closed_wire_face = FCPart.Face(closed_wire)

    def fix_next_problem_edge(c_edges):

        new_edges = c_edges
        ordered_vertexes = FCPart.Wire(c_edges).OrderedVertexes

        edge_dirs = [None] * (ordered_vertexes.__len__() - 1)
        for ii in range(ordered_vertexes.__len__() - 1):
            vec = ordered_vertexes[ii + 1].Point - ordered_vertexes[ii].Point
            edge_dirs[ii] = vec / np.linalg.norm(vec)

        angles = [None] * (ordered_vertexes.__len__() - 2)
        for ii in range(ordered_vertexes.__len__() - 2):
            angles[ii] = angle_between_vertices(ordered_vertexes[ii + 1].Point,
                                                ordered_vertexes[ii].Point,
                                                ordered_vertexes[ii + 2].Point)

        problem_edge_indices = None
        current_test_edge = c_edges[0]
        for i, t_edge in enumerate(c_edges[0:-1]):

            t_angle = angle_between_edges(current_test_edge, t_edge)
            if any(abs(np.array([0, 180]) - abs(t_angle)) < 1e-3):
                current_test_edge = t_edge
                continue

            l_min1 = radius / np.tan(np.deg2rad(abs(angles[i-1] / 2))) - 1e-6
            l_min2 = 0
            if angles.__len__() > 1:
                l_min2 = radius / np.tan(np.deg2rad(abs(angles[i] / 2))) - 1e-6

            cond = [c_edges[i-1].Length < l_min1,
                    c_edges[i].Length < (l_min1 + l_min2),
                    make_fillet([current_test_edge, t_edge], radius=radius) is None]

            if any(cond):
                logger.debug(f'Radius generation not possible')
                problem_edge_indices = i
                logger.debug(f'Fixing edge {i}')
                angle1 = angle_between_edges(c_edges[i - 1], c_edges[i], side1=1, side2=0, deg=True)
                angle2 = angle_between_edges(c_edges[i], c_edges[i + 1], side1=1, side2=0, deg=True)

                if edge_dirs[i - 1] == edge_dirs[i + 1]:
                    d = calc_d(c_edges[i].Length, angle1, radius)
                    new_edges[i - 1] = FCPart.LineSegment(ordered_vertexes[i-1].Point,
                                                          ordered_vertexes[i].Point -
                                                          edge_dirs[i - 1] * float(
                                                              d) / 2).toShape()
                    new_edges[i + 1] = FCPart.LineSegment(
                        ordered_vertexes[i + 1].Point + edge_dirs[i + 1] * float(d) / 2,
                        ordered_vertexes[i + 2].Point).toShape()

                    new_edges[i] = FCPart.LineSegment(
                        ordered_vertexes[i].Point - edge_dirs[i - 1] * float(d) / 2,
                        ordered_vertexes[i + 1].Point + edge_dirs[i + 1] * float(d) / 2).toShape()

                elif edge_dirs[i - 1] == -edge_dirs[i + 1]:
                    rep_wire = FCPart.Wire(c_edges[i-2:i+3])
                    p_proj1, on_segment1 = project_point_on_line(rep_wire.OrderedVertexes[1].Point,
                                                                 rep_wire.OrderedEdges[-2],
                                                                 return_on_segment=True)
                    p_proj2, on_segment2 = project_point_on_line(rep_wire.OrderedVertexes[-2].Point,
                                                                 rep_wire.OrderedEdges[1],
                                                                 return_on_segment=True)

                    if on_segment1:
                        e0 = FCPart.LineSegment(rep_wire.OrderedVertexes[1].Point, Base.Vector(p_proj1)).toShape()
                        e1 = FCPart.LineSegment(Base.Vector(p_proj1), rep_wire.OrderedVertexes[-2].Point).toShape()
                        new_edges = [*c_edges[0:i-1], e0, e1, *c_edges[i + 2:]]

                        # export_objects([*c_edges[0:i-1], e0, e1, *c_edges[i + 2:]], '/tmp/problem_edges.FCStd')

                    elif on_segment2:
                        e0 = FCPart.LineSegment(rep_wire.OrderedVertexes[1].Point, Base.Vector(p_proj2)).toShape()
                        e1 = FCPart.LineSegment(Base.Vector(p_proj2), rep_wire.OrderedVertexes[-1].Point).toShape()
                        # export_objects([*c_edges[0:i - 1], e0, e1, *c_edges[i + 3:]], '/tmp/test_new_edges.FCStd')
                        new_edges = [*c_edges[0:i - 1], e0, e1, *c_edges[i + 3:]]
                break
            current_test_edge = t_edge
        # export_objects([*new_edges], '/tmp/new_edges.FCStd')
        return new_edges, problem_edge_indices,

    problem_edge_indices = [0]
    new_c_edges = edges
    while problem_edge_indices is not None:
        new_c_edges, problem_edge_indices = fix_next_problem_edge(new_c_edges)
    edges = new_c_edges

    i = 1
    while i < edges.__len__():
        current_edges = edges_with_radius.OrderedEdges
        t_angle = angle_between_edges(current_edges[-1], edges[i])
        if any(abs(np.array([0, 180]) - abs(t_angle)) < 1e-3):
            current_edges.append(edges[i])
            i += 1
        else:
            new_edges = make_fillet([current_edges[-1], edges[i]], radius=radius)
            if new_edges is None:
                raise Exception(f'Error adding radius to edges {i-1} - {i}')
            current_edges[-1] = new_edges.Shape.OrderedEdges[0]
            current_edges.extend(new_edges.Shape.OrderedEdges[1:])
            i += 1
        edges_with_radius = FCPart.Wire(current_edges)

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


def intersect_lines(edge1, edge2, as_vector=False):
    if not isinstance(edge1, FCPart.Line):
        edge1 = edge_to_line(edge1)

    if not isinstance(edge2, FCPart.Line):
        edge2 = edge_to_line(edge2)

    ip = edge1.intersect(edge2)

    if not as_vector:
        return ip
    else:
        return [Base.Vector(x.X, x.Y, x.Z) for x in ip]


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

        jump3_wire0 = reference_face.OuterWire.makeOffset2D(-tube_edge_distance - 0.5 * tube_diameter,
                                                            join=1,
                                                            openResult=False,
                                                            intersection=False)

        cut_start_edge = start_edge.copy()
        cut_start_edge.Placement.move(Base.Vector(inflow_dir) * (tube_edge_distance + 0.5 * tube_diameter + 1))
        jump3_wire_edges = FCPart.Wire(get_split(jump3_wire0, [cut_start_edge], inner=False)).OrderedEdges
        jump3_wire_edges[0] = extrude_edge(jump3_wire_edges[0],
                                           side=1,
                                           dist=tube_edge_distance + 0.5 * tube_diameter + 10)
        jump3_wire_edges[-1] = extrude_edge(jump3_wire_edges[-1],
                                            side=2,
                                            dist=tube_edge_distance + 0.5 * tube_diameter + 10)

        jump3_wire = FCPart.Wire(jump3_wire_edges)

        # export_objects([jump3_wire, jump3_wire0, reference_face], '/tmp/test.FCStd')

        jump1_wire0 = reference_face.OuterWire.makeOffset2D(-tube_edge_distance - 3 * tube_diameter,
                                                            join=1,
                                                            openResult=False,
                                                            intersection=False)

        # export_objects([jump1_wire0, cut_start_edge], '/tmp/test.FCStd')

        cut_start_edge = start_edge.copy()
        cut_start_edge.Placement.move(Base.Vector(inflow_dir) * (tube_edge_distance + 3 * tube_diameter + 1))
        jump1_wire_edges = FCPart.Wire(get_split(jump1_wire0, [cut_start_edge], inner=False)).OrderedEdges
        jump1_wire_edges[0] = extrude_edge(jump1_wire_edges[0],
                                           side=1,
                                           dist=tube_edge_distance + 0.5 * tube_diameter + 10)
        jump1_wire_edges[-1] = extrude_edge(jump1_wire_edges[-1],
                                            side=2,
                                            dist=tube_edge_distance + 0.5 * tube_diameter + 10)

        jump1_wire = FCPart.Wire(jump1_wire_edges)

        i = 0
        while True:
            if i == 0:
                movement = Base.Vector(inflow_dir) * (tube_edge_distance + 1 + 0.5 * tube_diameter)
            else:
                movement = movement + Base.Vector(inflow_dir) * tube_distance

            e_init = base_edge.copy()
            e_init.Placement.move(movement)
            cut_shapes = e_init.cut(jump3_wire)
            if cut_shapes.SubShapes.__len__() > 1:
                e_init2 = cut_shapes.SubShapes[1]
                horizontal_lines.append(e_init2)
            else:
                break

            i += 1

        if (horizontal_lines.__len__() % 2) != 0:
            horizontal_lines = horizontal_lines[0:-1]

        # cut h_lines
        # line 0:
        cutted_h_lines = [None] * horizontal_lines.__len__()

        # export_objects([jump3_wire, jump1_wire, FCPart.Compound(horizontal_lines)], '/tmp/test.FCStd')

        p0_1 = horizontal_lines[0].Vertex1.Point
        p0_2 = horizontal_lines[0].cut(jump1_wire).SubShapes[-2].Vertex2.Point
        cutted_h_lines[0] = FCPart.LineSegment(p0_1, p0_2).toShape()

        p1_1 = horizontal_lines[1].cut(jump1_wire).SubShapes[0].Vertex2.Point
        p1_2 = horizontal_lines[1].Vertex2.Point
        cutted_h_lines[1] = FCPart.LineSegment(p1_1, p1_2).toShape()

        for i in range(int((horizontal_lines.__len__()) / 2)):
            if (i % 2) == 0:

                p0_1 = horizontal_lines[i * 2].Vertex1.Point
                p0_2 = horizontal_lines[i * 2].cut(jump1_wire).SubShapes[-2].Vertex2.Point
                cutted_h_lines[i * 2] = FCPart.LineSegment(p0_1, p0_2).toShape()

                if i == (horizontal_lines.__len__() - 2) / 2:
                    p1_1 = horizontal_lines[i * 2 + 1].Vertex1.Point
                else:
                    p1_1 = horizontal_lines[i * 2 + 1].cut(jump1_wire).SubShapes[0].Vertex2.Point

                p1_2 = horizontal_lines[i * 2 + 1].Vertex2.Point
                cutted_h_lines[i * 2 + 1] = FCPart.LineSegment(p1_1, p1_2).toShape()
            else:
                p0_1 = horizontal_lines[i * 2].cut(jump1_wire).SubShapes[0].Vertex2.Point
                p0_2 = horizontal_lines[i * 2].Vertex2.Point
                cutted_h_lines[i * 2] = FCPart.LineSegment(p0_1, p0_2).toShape()

                p1_1 = horizontal_lines[i * 2 + 1].Vertex1.Point

                if i == (horizontal_lines.__len__() - 2) / 2:
                    p1_2 = horizontal_lines[i * 2 + 1].Vertex2.Point
                else:
                    p1_2 = horizontal_lines[i * 2 + 1].cut(jump1_wire).SubShapes[-2].Vertex2.Point

                cutted_h_lines[i * 2 + 1] = FCPart.LineSegment(p1_1, p1_2).toShape()

        # export_objects([reference_face,
        #                 jump1_wire,
        #                 jump3_wire,
        #                 FCPart.Compound(cutted_h_lines)], '/tmp/h_lines.FCStd')

        # create pipe edges:
        pipe_edges_in = []
        pipe_edges_out = []

        # create outflow
        h0_edge = cutted_h_lines[0]
        out_edge1 = FCPart.LineSegment(h0_edge.Vertex2.Point,
                                       Base.Vector(project_point_on_line(h0_edge.Vertex2.Point, start_edge))).toShape()
        out_edge2 = FCPart.LineSegment(out_edge1.Vertex2.Point,
                                       out_edge1.Vertex2.Point + out_edge1.tangentAt(out_edge1.LastParameter) * 500).toShape()
        pipe_edges_out.extend([out_edge1, out_edge2])

        # create inflow
        h1_edge = cutted_h_lines[1]
        e2p1 = Base.Vector(project_point_on_line(out_edge1.Vertex1.Point, h1_edge) +
                           (h1_edge.tangentAt(h1_edge.LastParameter) * 3 * tube_diameter))

        cutted_h_lines[1] = FCPart.LineSegment(h1_edge.Vertex1.Point,
                                               e2p1).toShape()
        h1_edge = cutted_h_lines[1]
        in_edge1 = FCPart.LineSegment(e2p1,
                                      Base.Vector(
                                          project_point_on_line(h1_edge.Vertex2.Point, start_edge))).toShape()

        in_edge2 = FCPart.LineSegment(in_edge1.Vertex2.Point,
                                      in_edge1.Vertex2.Point + in_edge1.tangentAt(in_edge1.LastParameter) * 500).toShape()

        pipe_edges_in.extend([in_edge2, in_edge1])

        # export_objects([reference_face,
        #                 jump1_wire,
        #                 jump3_wire,
        #                 FCPart.Compound(pipe_edges_in)
        #                 ], '/tmp/h_lines.FCStd')

        pipe_edges_in, end_side_in = connect_h_lines(cutted_h_lines,
                                                     [jump1_wire, jump3_wire],
                                                     jump=1,
                                                     i=1,
                                                     side=1)

        # export_objects([reference_face,
        #                 jump1_wire,
        #                 jump3_wire,
        #                 FCPart.Compound(pipe_edges_in)
        #                 ], '/tmp/h_lines.FCStd')

        pipe_edges_out, end_side_out = connect_h_lines(cutted_h_lines,
                                                       [jump1_wire, jump3_wire],
                                                       jump=3,
                                                       i=0,
                                                       side=1)

        # export_objects([reference_face,
        #                 jump1_wire,
        #                 jump3_wire,
        #                 FCPart.Compound(pipe_edges_in),
        #                 FCPart.Compound(pipe_edges_out)
        #                 ], '/tmp/h_lines.FCStd')

        if end_side_in != end_side_out:
            raise Exception(f'Error while connection horizontal lines. End side 1 is not equal to end side 2')

        if end_side_in == 2:
            ce = FCPart.LineSegment(cutted_h_lines[-1].Vertex2.Point, cutted_h_lines[-2].Vertex2.Point).toShape()

        elif end_side_in == 1:
            ce = FCPart.LineSegment(cutted_h_lines[-1].Vertex1.Point, cutted_h_lines[-2].Vertex1.Point).toShape()

        pipe_edges_out.reverse()

        pipe_wire = FCPart.Wire(FCPart.sortEdges([in_edge2,
                                                  in_edge1,
                                                  *pipe_edges_in,
                                                  ce,
                                                  *pipe_edges_out,
                                                  out_edge1,
                                                  out_edge2
                                                  ])[0])

        # export_objects([reference_face,
                        # pipe_wire
                        # ], '/tmp/h_lines.FCStd')

        return pipe_wire, horizontal_lines

    elif layout == 'Concentric':
        pipe_wire = None
        raise NotImplementedError


def connect_h_lines(horizontal_lines,
                    jump_wires,
                    jump=1,
                    i=1,
                    side=1
                    ):

    # pipe_edges.append(horizontal_lines[i])

    pipe_edges = []

    while i + jump < horizontal_lines.__len__():

        if jump == 1:
            jump_wire = jump_wires[0]
        else:
            jump_wire = jump_wires[1]

        con_edges = connect_lines(horizontal_lines[i],
                                  horizontal_lines[i + jump],
                                  side,
                                  jump_wire)

        # pipe_edges.extend([*con_edges, horizontal_lines[i + jump]])

        pipe_edges.extend([horizontal_lines[i], *con_edges])

        # export_objects([*con_edges], f'/tmp/connect_hlines_{i}.FCStd')

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

    pipe_edges.append(horizontal_lines[i])

    return pipe_edges, side


def connect_lines(line1,
                  line2,
                  side,
                  jump_wire      # reference face normal
                  ):

    if side == 1:
        dist = max([jump_wire.distToShape(line1.Vertex1)[0], jump_wire.distToShape(line2.Vertex1)[0]])
        split_edge1 = extrude_vertex(line1.Vertex1, line1.tangentAt(line1.FirstParameter) * (dist+10), mirrored=True)
        split_edge2 = extrude_vertex(line2.Vertex1, line2.tangentAt(line2.FirstParameter) * (dist+10), mirrored=True)
        # return get_split(jump_wire, [split_edge1, split_edge2])

    elif side == 2:
        dist = max([jump_wire.distToShape(line1.Vertex2)[0], jump_wire.distToShape(line2.Vertex2)[0]])
        split_edge1 = extrude_vertex(line1.Vertex2, line1.tangentAt(line1.LastParameter) * (dist + 10), mirrored=True)
        split_edge2 = extrude_vertex(line2.Vertex2, line2.tangentAt(line2.LastParameter) * (dist + 10), mirrored=True)
        # return get_split(jump_wire, [split_edge1, split_edge2])

    try:
        sub_shapes = BOPTools.SplitAPI.slice(jump_wire, [split_edge1, split_edge2], 'Split', tolerance=0.1).SubShapes
        con_edges = sub_shapes[1]
    except Exception as e:
        logger.error(f'{e}')
        export_objects([sub_shapes, line1, line2, split_edge1, split_edge2], '/tmp/test.FCStd')
        raise e

    return con_edges.Edges


def get_split(wire, edges: list, inner=True):

    # export_objects([wire, FCPart.Compound(edges)], '/tmp/split_test.FCStd')

    split_shapes = BOPTools.SplitAPI.slice(wire, edges, 'Split', tolerance=0.1)
    sub_wires_length = [x.Length for x in split_shapes.SubShapes]
    if inner:
        shape = split_shapes.SubShapes[sub_wires_length.index(min(sub_wires_length))]
    else:
        shape = split_shapes.SubShapes[sub_wires_length.index(max(sub_wires_length))]
    if isinstance(shape, FCPart.Shape):
        return shape.Edges


def extrude_vertex(vertex, direction, mirrored=False):

    if mirrored:
        return FCPart.LineSegment(vertex.Point - direction,
                                  vertex.Point + direction).toShape()
    else:
        return FCPart.LineSegment(vertex.Point,
                                  vertex.Point + direction).toShape()


def angle_between_edges(edge1: FCPart.Edge,
                        edge2: FCPart.Edge,
                        side1=0,
                        side2=1,
                        normal: Base.Vector = None,
                        deg=True):

    params1 = [edge1.FirstParameter, edge1.LastParameter]
    params2 = [edge2.FirstParameter, edge2.LastParameter]

    if normal is None:
        angle = DraftVecUtils.angle(edge1.tangentAt(params1[side1]),
                                    edge2.tangentAt(params2[side2])
                                    )
    else:
        angle = DraftVecUtils.angle(edge1.tangentAt(params1[side1]),
                                    edge2.tangentAt(params2[side2]),
                                    normal)
    if not deg:
        return angle
    return np.rad2deg(angle)


def angle_between_vectors(v1: Base.Vector,
                          v2: Base.Vector,
                          normal: Base.Vector = None,
                          deg=True):

    if normal is None:
        angle = DraftVecUtils.angle(v1.normalize(), v2.normalize())
    else:
        angle = DraftVecUtils.angle(v1.normalize(), v2.normalize(), normal)
    if not deg:
        return angle
    return np.rad2deg(angle)


def make_custom_fillet(edge1, edge2, radius, side1=1, side2=0):

    params1 = [edge1.FirstParameter, edge1.LastParameter]
    params2 = [edge2.FirstParameter, edge2.LastParameter]

    angle = angle_between_edges(edge1, edge2, side1=1, side2=0, deg=True)

    if any(abs([0, 180] - abs(angle)) < 1):
        return None
    else:
        ip0 = intersect_lines(edge1, edge2, as_vector=True)[0]
        e1 = FCPart.LineSegment(edge1.valueAt(params1[abs(side1-1)]),
                                ip0).toShape()
        e2 = FCPart.LineSegment(ip0,
                                edge2.valueAt(params2[abs(side2 - 1)])).toShape()

        new_edges = make_fillet([e1, e2], radius=radius)

    return new_edges


def calc_d(l_e, beta, radius):

    def func(d):
        return (2 * radius / (np.tan((np.arcsin(d / np.sqrt(l_e**2 + d**2)) + np.deg2rad(beta)) / 2)) + radius/2)**2 - l_e**2 - d**2

    d = fsolve(func, np.array(0))
    return abs(d)


def split_wire_by_projected_vertices(wire, vertices, dist, ensure_closed=False):
    cutted_ref_face_edges = []
    for edge in wire.OrderedEdges:
        edge_parameters = [edge.FirstParameter, edge.LastParameter]
        split_parameters = []
        for vertex in vertices:
            if vertex.distToShape(edge)[0] <= dist:
                parameter = edge.Curve.parameter(Base.Vector(project_point_on_line(vertex.Point, edge)))
                if (parameter not in edge_parameters) and (edge.FirstParameter < parameter < edge.LastParameter):
                    split_parameters.append(parameter)
        split_parameters = sorted(list(set(split_parameters)))

        try:
            cutted_ref_face_edges.extend(edge.split(split_parameters).Edges)
        except Exception as e:
            raise e

    sorted_edges = FCPart.sortEdges(cutted_ref_face_edges)[0]

    if ensure_closed:
        wire = FCPart.Wire(sorted_edges)
        if not wire.isClosed():
            return FCPart.Wire([*wire.OrderedEdges,
                                FCPart.LineSegment(wire.OrderedVertexes[-1].Point, wire.OrderedVertexes[0].Point).toShape()])

    else:
        return FCPart.Wire(sorted_edges)


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[np.sum(np.cumsum(tmp, axis=0)*tmp == 1, axis=1).astype(bool)]

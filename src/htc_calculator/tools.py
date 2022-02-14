import os
import sys
import numpy as np
import pathlib
import gmsh
import meshio

import FreeCAD
import Part as FCPart
import Points
import numpy as np
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


def export_objects(objects, filename):
    file_suffix = pathlib.Path(filename).suffix

    doc = App.newDocument()

    for i, object in enumerate(objects):
        __o__ = doc.addObject("Part::Feature", f'edge{i}')
        __o__.Label = f'edge{i}'
        __o__.Shape = object

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

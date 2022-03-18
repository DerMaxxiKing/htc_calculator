import os
import sys
import uuid
import tempfile
from io import StringIO
import re
import numpy as np
from copy import deepcopy
# from pyobb.obb import OBB

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from .meshing import meshing_resources as msh_resources

# from .tools import name_step_faces
from .logger import logger
from .face import Face
from .solid import Solid
from .meshing.surface_mesh_parameters import default_surface_mesh_parameter

import FreeCAD
import Part as FCPart
from FreeCAD import Base
from BOPTools import SplitFeatures


App = FreeCAD


class Assembly(object):

    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', uuid.uuid4())
        # logger.debug(f'initializing Assembly {self.id}')

        self.name = kwargs.get('name', None)
        self.normal = kwargs.get('normal', None)

        self.faces = kwargs.get('faces', [])
        self.solids = kwargs.get('solids', [])
        self.features = kwargs.get('features', {})

        self.interfaces = kwargs.get('interfaces', [])
        self.topology = kwargs.get('topology', {})
        self._comp_solid = kwargs.get('comp_solid', None)

        self.surface_mesh_setup = kwargs.get('_surface_mesh_setup',
                                             kwargs.get('surface_mesh_setup', default_surface_mesh_parameter))
        self.reference_face = kwargs.get('reference_face', None)

    @property
    def volume(self):
        return sum([x.Volume for x in self.solids])

    @property
    def stl(self):
        return ''.join([x.stl for x in self.solids])

    @property
    def comp_solid(self):
        if self._comp_solid is None:

            # # doc = App.newDocument("CompSolid")
            # active = App.ActiveDocument
            #
            # features = []
            # for solid in self.solids:
            #     __o__ = active.addObject("Part::Feature", f'Reference Face {solid.name} {solid.id}')
            #     __o__.Shape = solid.fc_solid.Shape
            #     features.append(__o__)
            #
            # active.recompute()
            #
            # j = SplitFeatures.makeBooleanFragments(name='BooleanFragments')
            # j.Objects = active.Objects
            # j.Mode = 'CompSolid'
            # j.Proxy.execute(j)

            self._comp_solid = FCPart.CompSolid([x.fc_solid.Shape for x in self.solids])
        return self._comp_solid

    @property
    def hull(self):

        # FCPart.MultiFuse

        # logging.debug(f'Creating hull for {self.id}...')
        # fd, tmp_file = tempfile.mkstemp(suffix='.stp')
        # os.close(fd)
        #
        # self.export_stp(tmp_file)
        #
        # imported_shape = FCPart.Shape()
        # imported_shape.read(tmp_file)
        #
        # try:
        #     os.remove(tmp_file)
        # except FileNotFoundError as e:
        #     pass
        hull_solid = Solid(faces=[x for x in self.faces if x not in self.interfaces])

        # solid0 = self.solids[0].fc_solid.Shape
        # solids = [x.fc_solid.Shape for x in self.solids[1:]]
        # hull = solid0.multiFuse(solids, 1)
        #
        # # solids = [x.fc_solid.Shape for x in self.solids]
        # # solid_faces = []
        # # [solid_faces.extend(x.Faces) for x in solids]
        # #
        # # set(solid_faces)
        #
        # # hull = self.solids[0].fc_solid.Shape
        # # hull = hull.multiFuse([x.fc_solid.Shape for x in self.solids[1:]]).removeSplitter()
        # hull_solid = Solid(faces=[Face(fc_face=x) for x in hull.Faces])
        # hull_solid.generate_solid_from_faces()
        # logger.debug(f'hull creation finished for {self.id}')

        return hull_solid

    @property
    def txt_id(self):
        return re.sub('\W+','', 'a' + str(self.id))

    @property
    def location_in_mesh(self):
        location_in_mesh = np.array([self.comp_solid.BoundBox.Center.x + np.random.uniform(-0.1 + 0.1),
                                     self.comp_solid.BoundBox.Center.y + np.random.uniform(-0.1 + 0.1),
                                     self.comp_solid.BoundBox.Center.z + np.random.uniform(-0.1 + 0.1)])

        if self.comp_solid.isInside(Base.Vector(location_in_mesh), 0, True):
            return location_in_mesh
        else:
            while not self.comp_solid.isInside(Base.Vector(location_in_mesh), 0, True):
                location_in_mesh = np.array([np.random.uniform(self.comp_solid.BoundBox.XMin, self.comp_solid.BoundBox.XMax),
                                             np.random.uniform(self.comp_solid.BoundBox.YMin, self.comp_solid.BoundBox.YMax),
                                             np.random.uniform(self.comp_solid.BoundBox.ZMin, self.comp_solid.BoundBox.ZMax)])
            return location_in_mesh

    @property
    def locations_in_mesh(self):
        s = '\n\t(\n'

        for solid in self.solids:
            s += f"\t\t(({solid.point_in_mesh[0]} {solid.point_in_mesh[1]} {solid.point_in_mesh[2]}) {solid.txt_id})\n"

        s += '\t)'

        return s

    @property
    def hull_faces(self):
        return set(self.faces) - set(self.interfaces)

    def export_stl(self, filename):
        logger.debug(f'exporting stl for {self.id}')

        new_file = open(filename, 'w')
        new_file.writelines(self.stl)
        new_file.close()

        logger.debug(f'export stl finished for {self.id}')

    def export_stp(self, filename):

        logger.debug(f'export step finished for {self.id}')

        from .tools import name_step_faces

        try:
            __objs__ = [x.fc_solid for x in self.solids]

            fd, tmp_file = tempfile.mkstemp(suffix='.stp')
            os.close(fd)
            FCPart.export(__objs__, tmp_file)
            names = []
            for solid in self.solids:
                names.extend(solid.face_names)
            
            names_dict = {k: v for k, v in zip(range(names.__len__()), names)}
            name_step_faces(fname=tmp_file, name=names_dict, new_fname=filename)
        except Exception as e:
            logger.error(f'error while exporting step finished for {self.id}\n{e}')
        finally:
            try:
                os.remove(tmp_file)
            except FileNotFoundError as e:
                pass

    def interface_shm_geo_entry(self, offset=0):

        local_offset = 4
        offset = offset + local_offset

        buf = StringIO()

        for face in self.interfaces:

            buf.write(f"{' ' * offset}{str(face.txt_id)}\n")
            buf.write(f"{' ' * offset}{'{'}\n")
            buf.write(f"{' ' * (offset + 4)}type            triSurfaceMesh;\n")
            buf.write(f"{' ' * (offset + 4)}file            \"{str(face.txt_id)}.stl\";\n")
            buf.write(f"{' ' * offset}{'}'}\n")

        return buf.getvalue()

    @property
    def interface_shm_refinement_entry(self, offset=0):

        local_offset = 4
        offset = offset + local_offset

        buf = StringIO()

        for face in self.interfaces:
            buf.write(f"{' ' * offset}{str(face.txt_id)}\n")
            buf.write(f"{' ' * offset}{'{'}\n")
            buf.write(f"{' ' * (offset + 4)}level           ({face.surface_mesh_setup.min_refinement_level} {face.surface_mesh_setup.max_refinement_level});\n")
            buf.write(f"{' ' * (offset + 4)}faceZone        {str(face.txt_id)};\n")

            # cell_zone = self.topology[face][1].solid.txt_id
            # buf.write(f"{' ' * (offset + 4)}cellZone        {str(cell_zone)};\n")
            # buf.write(f"{' ' * (offset + 4)}cellZoneInside  inside;;\n")
            # buf.write(f"{' ' * (offset + 4)}cellZoneInside  insidePoint;\n")

            # x = self.topology[face][1].solid.point_in_mesh[0]
            # y = self.topology[face][1].solid.point_in_mesh[1]
            # z = self.topology[face][1].solid.point_in_mesh[2]
            #
            # buf.write(f"{' ' * (offset + 4)}insidePoint     ({x} {y} {z});\n")
            buf.write(f"{' ' * offset}{'}'}\n")

        return buf.getvalue()

    @property
    def baffles_dict(self, offset=0):

        baffle_dict_template = pkg_resources.read_text(msh_resources, 'create_baffles_dict')

        local_offset = 4
        offset = offset + local_offset
        buf = StringIO()

        n = 0

        baffle_template = """
        <baffle_name>
        {
            type        faceZone;
            zoneName    <zone_name>;
    
            patches
            {
                master
                {
                    name            <master_name>;
                    type            mappedWall;
                    sampleMode      nearestPatchFace;
                    sampleRegion    <master_sample_region>;
                    samplePatch     <master_sample_patch>;
                }
                slave
                {
                    name            <slave_name>;
                    type            mappedWall;
                    sampleMode      nearestPatchFace;
                    sampleRegion    <slave_sample_region>;
                    samplePatch     <slave_sample_patch>;
                }
            }
        }
        """

        baffles = ''

        for face, neighbours in self.topology.items():

            new_id = re.sub('\W+','', 'a' + str(uuid.uuid4()))

            s = deepcopy(baffle_template)
            s = s.replace('<baffle_name>', f"baffles{n}")
            s = s.replace('<zone_name>', face.txt_id)

            s = s.replace('<master_name>', face.txt_id)
            s = s.replace('<master_sample_region>', neighbours[0].solid.txt_id)
            s = s.replace('<master_sample_patch>', new_id)

            s = s.replace('<slave_name>', new_id)
            s = s.replace('<slave_sample_region>', neighbours[1].solid.txt_id)
            s = s.replace('<slave_sample_patch>', face.txt_id)

            baffles += s

        print('baffles')

        # "\t{" + '\t'.join(('\n' + '\t' + refinement_str.lstrip()).splitlines(True)) + "\t}\n"

        create_baffle_dict = baffle_dict_template.replace('<baffles>', baffles)

        return create_baffle_dict

    @property
    def regions_dict(self):

        template = """/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  8
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      regionProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

regions
(
    <regions>
);

// ************************************************************************* //
"""

        solids = []
        fluids = []

        for solid in self.solids:
            if solid.state == 'solid':
                solids.append(solid)
            elif solid.state == 'fluid':
                fluids.append(solid)

        fluids_str = ' '.join([x.txt_id for x in fluids])
        solids_str = ' '.join([x.txt_id for x in solids])

        s = f"fluid\t({fluids_str})\n\tsolid\t({solids_str})\n"

        regions_dict = template.replace('<regions>', s)
        return regions_dict

    def write_of_geo(self, directory):

        # write hull stl
        stl_str = ''.join([x.create_stl_str(of=True) for x in self.hull_faces])
        new_file = open(os.path.join(directory, str(self.txt_id) + '.stl'), 'w')
        new_file.writelines(stl_str)
        new_file.close()

        # write interfaces
        for face in self.interfaces:
            new_file = open(os.path.join(directory, str(face.txt_id) + '.stl'), 'w')
            new_file.writelines(face.stl)
            new_file.close()

    def shm_geo_entry(self, offset=0):

        local_offset = 4
        offset = offset + local_offset

        hull_faces = set(self.faces) - set(self.interfaces)

        buf = StringIO()

        buf.write(f"{' ' * offset}{str(self.solids[0].txt_id)}\n")
        buf.write(f"{' ' * offset}{'{'}\n")
        buf.write(f"{' ' * (offset + 4)}type            triSurfaceMesh;\n")
        buf.write(f"{' ' * (offset + 4)}file            \"{str(self.txt_id)}.stl\";\n")
        buf.write(f"{' ' * (offset + 4)}regions\n")
        buf.write(f"{' ' * (offset + 4)}{'{'}\n")

        for face in hull_faces:
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

        buf.write(f"{' ' * offset}{str(self.solids[0].txt_id)}\n")
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

    def write_surface_feature_dict(self, directory):

        template = pkg_resources.read_text(msh_resources, 'surfaceFeaturesDict')

        surf_str = ''

        s_template = """
<surface_name>
{
    surfaces
    (
        "<stl_path>"
    );
    includedAngle   120;
    geometricTestOnly       yes;
    writeFeatureEdgeMesh    yes;
    writeObj                yes;
    verboseObj              no;
}"""

        s = deepcopy(s_template)
        s = s.replace('<surface_name>', str(self.txt_id))
        s = s.replace('<stl_path>', str(self.txt_id) + '.stl')
        surf_str += s

        for face in self.interfaces:
            s = deepcopy(s_template)
            s = s.replace('<surface_name>', str(face.txt_id))
            s = s.replace('<stl_path>', str(face.txt_id) + '.stl')
            surf_str += s

        surf_str = template.replace('<surfaces>', surf_str)

        new_file = open(os.path.join(directory, 'surfaceFeaturesDict'), 'w')
        new_file.writelines(surf_str)
        new_file.close()

    def write_block_mesh(self, directory, planar=True, cell_size=0.5):

        from .tools import create_obb

        reference_face = self.reference_face
        ref_normal = np.array([reference_face.normal.x, reference_face.normal.y, reference_face.normal.z])
        block_offset = 1

        box_points = create_obb(reference_face.vertices)

        v_1 = box_points[1, :] - box_points[0, :]
        v_2 = box_points[3, :] - box_points[0, :]
        v_3 = box_points[4, :] - box_points[0, :]

        l1 = np.linalg.norm(v_1)
        l2 = np.linalg.norm(v_2)
        l3 = np.linalg.norm(v_3)

        n1 = v_1 / l1
        n2 = v_2 / l2
        n3 = v_3 / l3

        thickness_dir = [not np.allclose(x, ref_normal) for x in [n1, n2, n3]]

        n_cell_plane = np.ceil(np.array([l1, l2, l3])[thickness_dir] / reference_face.plane_mesh_size).astype(np.int)

        def scale_face(points, scale):
            center = np.mean(points, axis=0)
            points = (points - center) * scale + center
            return points

        pts = scale_face(box_points, block_offset)

        # create block points
        num_layers = reference_face.component_construction.layers.__len__()
        block_points = np.zeros([(num_layers + 1) * 4, 3])

        offset = - (reference_face.component_construction.side_1_offset + block_offset) * reference_face.layer_dir
        block_points[0:4, :] = pts[0:4, :] + offset
        blocks = []

        for i, layer in enumerate(reference_face.component_construction.layers):
            if i == reference_face.component_construction.layers.__len__():
                offset = (layer.thickness + block_offset) * ref_normal * reference_face.layer_dir
            else:
                offset = layer.thickness * ref_normal * reference_face.layer_dir
            block_points[(i+1)*4:(i+1)*4 + 4, :] = block_points[i*4:i*4+4, :] + offset
            n3 = int(np.ceil(layer.thickness / reference_face.thickness_mesh_size))
            if n3 < 3:
                n3 = 3

            blocks.append(f'\thex ({" ".join(str(x) for x in range(i*4, i*4+8))}) ({n_cell_plane[0]} {n_cell_plane[1]} {n3}) simpleGrading (1 1 1)\n')

        template = pkg_resources.read_text(msh_resources, 'block_mesh_dict')
        s = deepcopy(template)

        vertices = 'vertices\n(\n' + ''.join([f'\t({x[0]:10.5f} {x[1]:10.5f} {x[2]:10.5f})\n' for x in block_points]) + ');\n'
        s = s.replace('<vertices>', vertices)

        blocks = f"blocks\n(\n{''.join(blocks)});\n"
        # print(blocks)
        s = s.replace('<blocks>', blocks)

        edges = 'edges\n(\n);\n'
        s = s.replace('<edges>', edges)

        faces = 'faces\n(\n);\n'
        s = s.replace('<faces>', faces)

        boundary = 'boundary\n(\n);\n'
        s = s.replace('<boundary>', boundary)

        merge_patch_pairs = 'merge_patch_pairs\n(\n);\n'
        s = s.replace('<merge_patch_pairs>', merge_patch_pairs)

        new_file = open(os.path.join(directory, 'system', 'blockMeshDict'), 'w')
        new_file.writelines(s)
        new_file.close()

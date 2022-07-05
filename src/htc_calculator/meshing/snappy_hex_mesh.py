import uuid
import numpy as np
import logging
import os
from datetime import datetime
import re
from copy import deepcopy
import subprocess


try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import meshing_resources as msh_resources
from .mesh_quality import MeshQuality, default_mesh_quality
from .layer_definition import default_layer_definition
from ..logger import logger


def to_of_dict_format(parameter):

    if type(parameter) is bool:
        if parameter:
            return 'true'
        else:
            return 'false'

    elif type(parameter) is int:
        return "{}".format(parameter)

    elif type(parameter) is float:
        return "{}".format(parameter)

    elif type(parameter) is str:
        return parameter

    else:
        return ''


def run_shm(case_dir):
    logger.info(f'Generating mesh....')

    res = subprocess.run(["/bin/bash", "-i", "-c", "snappyHexMesh 2>&1 | tee snappyHexMesh.log"],
                         capture_output=True,
                         cwd=case_dir,
                         user='root')
    if res.returncode == 0:
        output = res.stdout.decode('ascii')
        if output.find('FOAM FATAL ERROR') != -1:
            logger.error(f'Error running snappyHexMesh:\n\n{output}')

        if output.find('FOAM FATAL IO ERROR') != -1:
            logger.error(f'Error running snappyHexMesh:\n\n{output}')
            raise Exception(f'Error running snappyHexMesh:\n\n{output}')

        logger.info(f"Successfully created mesh:\n"
                    f"Directory: {case_dir}\n\n "
                    f"{output[output.find('Mesh Information'):]}")

    else:
        logger.error(f"{res.stderr.decode('ascii')}")
        raise Exception(f"Error running snappyHexMesh:\n{res.stderr.decode('ascii')}")
    return True


class SnappyHexMesh(object):

    visible_class_name = 'SnappyHexMesh'

    def __init__(self, *args, **kwargs):
        """
        :keyword
            * *castellated_mesh* (``str``) -- xxxxx
            * *snap* (``str``) -- xxxxx
            * *add_layers* (``str``) -- xxxxx
            * *merge_tolerance* (``str``) -- xxxxx
            * *location_in_mesh* (``str``) -- xxxxx
            * *max_local_cells* (``str``) -- xxxxx
            * *max_global_cells* (``str``) -- xxxxx
            * *min_refinement_cells* (``str``) -- xxxxx
            * *max_load_unbalance* (``str``) -- xxxxx
            * *n_cells_between_levels* (``str``) -- xxxxx
            * *resolve_feature_angle* (``str``) -- xxxxx
            * *allow_free_standing_zone_faces* (``str``) -- xxxxx
            * *features* (``str``) -- xxxxx
            * *refinement_surfaces* (``str``) -- xxxxx
            * *refinement_regions* (``str``) -- xxxxx
            * *n_smooth_patch* (``str``) -- xxxxx
            * *tolerance* (``str``) -- xxxxx
            * *n_solve_iter* (``str``) -- xxxxx
            * *n_relax_iter* (``str``) -- xxxxx
            * *n_feature_snap_iter* (``str``) -- xxxxx
            * *implicit_n_feature_snap_iter_snap* (``str``) -- xxxxx
            * *explicit_feature_snap* (``str``) -- xxxxx
            * *multi_region_feature_snap* (``str``) -- xxxxx
            * *layer_definition* (``str``) -- xxxxx
        """

        self._assembly = kwargs.get('_assembly', kwargs.get('assembly', None))

        self._id = kwargs.get('_id', kwargs.get('id', uuid.uuid4()))
        self._name = kwargs.get('_name', kwargs.get('name', 'SnappyHexMesh {}'.format(self._id)))

        self.template = pkg_resources.read_text(msh_resources, 'snappy_hex_mesh_dict')

        self._mesh_dir = kwargs.get('_mesh_dir', kwargs.get('mesh_dir', None))
        self._case_dir = kwargs.get('_case_dir', kwargs.get('case_dir', None))

        self._snappy_hex_mesh_dict = kwargs.get('_snappy_hex_mesh_dict', kwargs.get('snappy_hex_mesh_dict', None))

        # -----------------------------------------
        # attribute definition
        # -----------------------------------------

        self._castellated_mesh = kwargs.get('_castellated_mesh', kwargs.get('castellated_mesh', True))
        self._snap = kwargs.get('_snap', kwargs.get('snap', True))
        self._add_layers = kwargs.get('_add_layers', kwargs.get('add_layers', False))
        self._merge_tolerance = kwargs.get('_merge_tolerance', kwargs.get('merge_tolerance', float(1e-6)))

        location_in_mesh = kwargs.get('_location_in_mesh',
                                      kwargs.get('location_in_mesh',
                                                 None)
                                      )
        if location_in_mesh is None:
            if self.assembly is not None:
                # self.volume.add_observer(self.pycfd_dir_changed, attr_dependent=['location_in_mesh'])
                location_in_mesh = self.assembly.location_in_mesh
            else:
                location_in_mesh = np.array([0, 0, 0])

        self._location_in_mesh = location_in_mesh

        self._locations_in_mesh = kwargs.get('_locations_in_mesh', kwargs.get('locations_in_mesh', None))

        self._max_local_cells = kwargs.get('_max_local_cells', kwargs.get('max_local_cells', int(2e+7)))

        self._max_global_cells = kwargs.get('_max_global_cells', kwargs.get('max_global_cells', int(3e+7)))

        self._min_refinement_cells = kwargs.get('_min_refinement_cells', kwargs.get('min_refinement_cells', 100))

        self._max_load_unbalance = kwargs.get('_max_load_unbalance', kwargs.get('max_load_unbalance', 0.2))

        self._n_cells_between_levels = kwargs.get('_n_cells_between_levels', kwargs.get('n_cells_between_levels', 1))

        self._resolve_feature_angle = kwargs.get('_resolve_feature_angle', kwargs.get('resolve_feature_angle', 30))

        self._allow_free_standing_zone_faces = kwargs.get('_allow_free_standing_zone_faces',
                                                          kwargs.get('allow_free_standing_zone_faces', False))

        self._features = kwargs.get('_features', kwargs.get('features', None))

        self._refinement_surfaces = kwargs.get('_refinement_surfaces', kwargs.get('refinement_surfaces', None))

        self._refinement_regions = kwargs.get('_refinement_regions', kwargs.get('refinement_regions', None))

        self._n_smooth_patch = kwargs.get('_n_smooth_patch', kwargs.get('n_smooth_patch', 5))

        self._tolerance = kwargs.get('_tolerance', kwargs.get('tolerance', 10.0))

        self._n_solve_iter = kwargs.get('_n_solve_iter', kwargs.get('n_solve_iter', 20))

        self._n_relax_iter = kwargs.get('_n_relax_iter', kwargs.get('n_relax_iter', 10))

        self._n_feature_snap_iter = kwargs.get('_n_feature_snap_iter', kwargs.get('n_feature_snap_iter', 100))

        self._implicit_feature_snap = kwargs.get('_implicit_feature_snap',
                                                 kwargs.get('implicit_feature_snap', False))

        self._explicit_feature_snap = kwargs.get('_explicit_feature_snap', kwargs.get('explicit_feature_snap', True))

        self._multi_region_feature_snap = kwargs.get('_multi_region_feature_snap',
                                                     kwargs.get('multi_region_feature_snap', False))

        self._layer_definition = kwargs.get('_layer_definition',
                                            kwargs.get('layer_definition', default_layer_definition))

        self._mesh_quality_setup = kwargs.get('_mesh_quality_setup',
                                              kwargs.get('mesh_quality_setup', default_mesh_quality))

        self._layer_setup = kwargs.get('_layer_setup', kwargs.get('layer_setup', None))

        self._decompose_par_dict = kwargs.get('_decompose_par_dict', kwargs.get('decompose_par_dict', None))

        self._poly_mesh = kwargs.get('_poly_mesh', kwargs.get('poly_mesh', None))

        self._up_to_date = kwargs.get('_up_to_date', kwargs.get('up_to_date', False))

        self._num_procs = kwargs.get('_num_procs', kwargs.get('num_procs', None))

        self._num_subdomains = kwargs.get('_num_subdomains', kwargs.get('num_subdomains', np.array([1, 1, 1])))

        self.strict_region_snap = False

    # -----------------------------------------
    # property setup for attributes
    # -----------------------------------------

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        if value == self._id:
            return
        self._id = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value == self._name:
            return
        self._name = value

    @property
    def assembly(self):
        return self._assembly

    @assembly.setter
    def assembly(self, value):
        if value == self._assembly:
            return
        self._assembly = value

    @property
    def case_dir(self):
        return self._case_dir

    @case_dir.setter
    def case_dir(self, value):
        if value == self._case_dir:
            return
        self._case_dir = value

    @property
    def snappy_hex_mesh_dict(self):
        if self._snappy_hex_mesh_dict is None:
            self.create_snappy_hex_mesh_dict()
        return self._snappy_hex_mesh_dict

    @snappy_hex_mesh_dict.setter
    def snappy_hex_mesh_dict(self, value):
        if value == self._snappy_hex_mesh_dict:
            return
        self._snappy_hex_mesh_dict = value

    @property
    def castellated_mesh(self):
        return self._castellated_mesh

    @castellated_mesh.setter
    def castellated_mesh(self, value):
        if self._castellated_mesh == value:
            return
        self._castellated_mesh = value

    @property
    def snap(self):
        return self._snap

    @snap.setter
    def snap(self, value):
        if self._snap == value:
            return
        self._snap = value

    @property
    def add_layers(self):
        return self._add_layers

    @add_layers.setter
    def add_layers(self, value):
        if self._add_layers == value:
            return
        self._add_layers = value

    @property
    def merge_tolerance(self):
        return self._merge_tolerance

    @merge_tolerance.setter
    def merge_tolerance(self, value):
        if self._merge_tolerance == value:
            return
        self._merge_tolerance = value

    @property
    def location_in_mesh(self):
        return self._location_in_mesh

    @location_in_mesh.setter
    def location_in_mesh(self, value):
        if self._location_in_mesh == value:
            return
        self._location_in_mesh = value

    @property
    def locations_in_mesh(self):
        if self._locations_in_mesh is None:
            self._locations_in_mesh = self.assembly.locations_in_mesh
        return self._locations_in_mesh

    @locations_in_mesh.setter
    def locations_in_mesh(self, value):
        if self._locations_in_mesh == value:
            return
        self._locations_in_mesh = value

    @property
    def max_local_cells(self):
        return self._max_local_cells

    @max_local_cells.setter
    def max_local_cells(self, value):
        if self._max_local_cells == value:
            return
        self._max_local_cells = value

    @property
    def max_global_cells(self):
        return self._max_global_cells

    @max_global_cells.setter
    def max_global_cells(self, value):
        if self._max_global_cells == value:
            return
        self._max_global_cells = value

    @property
    def min_refinement_cells(self):
        return self._min_refinement_cells

    @min_refinement_cells.setter
    def min_refinement_cells(self, value):
        if self._min_refinement_cells == value:
            return
        self._min_refinement_cells = value

    @property
    def max_load_unbalance(self):
        return self._max_load_unbalance

    @max_load_unbalance.setter
    def max_load_unbalance(self, value):
        if self._max_load_unbalance == value:
            return
        self._max_load_unbalance = value

    @property
    def n_cells_between_levels(self):
        return self._n_cells_between_levels

    @n_cells_between_levels.setter
    def n_cells_between_levels(self, value):
        if self._n_cells_between_levels == value:
            return
        self._n_cells_between_levels = value

    @property
    def resolve_feature_angle(self):
        return self._resolve_feature_angle

    @resolve_feature_angle.setter
    def resolve_feature_angle(self, value):
        if self._resolve_feature_angle == value:
            return
        self._resolve_feature_angle = value

    @property
    def allow_free_standing_zone_faces(self):
        return self._allow_free_standing_zone_faces

    @allow_free_standing_zone_faces.setter
    def allow_free_standing_zone_faces(self, value):
        if self._allow_free_standing_zone_faces == value:
            return
        self._allow_free_standing_zone_faces = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if self._features == value:
            return
        self._features = value

    @property
    def refinement_surfaces(self):
        if self._refinement_surfaces is None:
            self.create_refinement_surfaces_str()
        return self._refinement_surfaces

    @refinement_surfaces.setter
    def refinement_surfaces(self, value):
        if self._refinement_surfaces == value:
            return
        self._refinement_surfaces = value

    @property
    def layer_setup(self):
        if self._layer_setup is None:
            self.create_layer_setup_str()
        return self._layer_setup

    @layer_setup.setter
    def layer_setup(self, value):
        if self._layer_setup == value:
            return
        self._layer_setup = value

    @property
    def refinement_regions(self):
        return self._refinement_regions

    @refinement_regions.setter
    def refinement_regions(self, value):
        if self._refinement_regions == value:
            return
        self._refinement_regions = value

    @property
    def n_smooth_patch(self):
        return self._n_smooth_patch

    @n_smooth_patch.setter
    def n_smooth_patch(self, value):
        if self._n_smooth_patch == value:
            return
        self._n_smooth_patch = value

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if self._tolerance == value:
            return
        self._tolerance = value

    @property
    def n_solve_iter(self):
        return self._n_solve_iter

    @n_solve_iter.setter
    def n_solve_iter(self, value):
        if self._n_solve_iter == value:
            return
        self._n_solve_iter = value

    @property
    def n_relax_iter(self):
        return self._n_relax_iter

    @n_relax_iter.setter
    def n_relax_iter(self, value):
        if self._n_relax_iter == value:
            return
        self._n_relax_iter = value

    @property
    def n_feature_snap_iter(self):
        return self._n_feature_snap_iter

    @n_feature_snap_iter.setter
    def n_feature_snap_iter(self, value):
        if self._n_feature_snap_iter == value:
            return
        self._n_feature_snap_iter = value

    @property
    def implicit_feature_snap(self):
        return self._implicit_feature_snap

    @implicit_feature_snap.setter
    def implicit_feature_snap(self, value):
        if self._implicit_feature_snap == value:
            return
        self._implicit_feature_snap = value

    @property
    def explicit_feature_snap(self):
        return self._explicit_feature_snap

    @explicit_feature_snap.setter
    def explicit_feature_snap(self, value):
        if self._explicit_feature_snap == value:
            return
        self._explicit_feature_snap = value

    @property
    def multi_region_feature_snap(self):
        return self._multi_region_feature_snap

    @multi_region_feature_snap.setter
    def multi_region_feature_snap(self, value):
        if self._multi_region_feature_snap == value:
            return
        self._multi_region_feature_snap = value

    @property
    def layer_definition(self):
        return self._layer_definition

    @layer_definition.setter
    def layer_definition(self, value):
        if self._layer_definition == value:
            return
        self._layer_definition = value

    @property
    def mesh_quality_setup(self):
        return self._mesh_quality_setup

    @mesh_quality_setup.setter
    def mesh_quality_setup(self, value):
        if self._mesh_quality_setup == value:
            return
        self._mesh_quality_setup = value

    def create_entry(self, attr, key, value=None):

        if value is None:
            parameter_str = ''

            parameter = self.__getattribute__(attr)

            if attr == 'location_in_mesh':
                parameter = parameter / 1000

            if parameter is None:
                return parameter_str

            elif type(parameter) is bool:
                if parameter:
                    parameter = 'true'
                else:
                    parameter = 'false'

            elif type(parameter) is float:
                parameter = "{}".format(parameter)

            elif type(parameter) is str:
                parameter_str = parameter

            elif type(parameter) is np.ndarray:
                if parameter.shape.__len__() == 1:
                    parameter = str(parameter)
                    parameter = '( ' + re.sub('[\[\]]', '', parameter) + ' )'

            elif hasattr(parameter, 'dict_entry'):
                parameter = parameter.dict_entry()
                parameter = '\t'.join(parameter.dict_entry().splitlines(True))

            if parameter is not None:
                parameter_str = '\t{:30s}{:>20}'.format(key, parameter) + ';\n'
        else:
            parameter_str = f'\t{key}{value}'

        return parameter_str

    def write_snappy_hex_mesh(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        if case_dir is None:
            logging.error(f'{self.name}: no case_dir')
            return

        # check if directory exists
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, '0'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'constant'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'system'), exist_ok=True)

        dst = os.path.join(case_dir, 'system', 'snappyHexMeshDict')
        with open(dst, 'w') as shmd:
            shmd.write(self.snappy_hex_mesh_dict)

    def geom_string(self):

        # geo_str1 = ''.join(x.shm_geo_entry for x in self.assembly.solids)
        geo_str1 = self.assembly.shm_geo_entry
        if hasattr(self.assembly, 'interface_shm_geo_entry'):
            geo_str = geo_str1 + self.assembly.interface_shm_geo_entry
        else:
            geo_str = geo_str1

        return geo_str

    def castellated_mesh_str(self):

        mapping_dict = {
            'max_local_cells': 'maxLocalCells',
            'max_global_cells': 'maxGlobalCells',
            'min_refinement_cells': 'minRefinementCells',
            'max_load_unbalance': 'maxLoadUnbalance',
            'n_cells_between_levels': 'nCellsBetweenLevels',
            'resolve_feature_angle': 'resolveFeatureAngle',
            'allow_free_standing_zone_faces': 'allowFreeStandingZoneFaces',
            'features': 'features',
            'refinement_surfaces': {'key': 'refinementSurfaces', 'value': '\t'.join(('\n' + self.refinement_surfaces.lstrip()).splitlines(True))},
            'refinement_regions': 'refinementRegions',
            'location_in_mesh': 'locationInMesh'
        }

        s = '{\n'
        for attr, value in mapping_dict.items():
            if attr == 'features':
                parameter = self.__getattribute__(attr)
                if parameter is None:
                    # s += '\t{:30s}{:>20}'.format('features', '( )') + ';\n'

                    s += (f'\tfeatures\n'
                    f'\t(\n'
                    f"\t    {'{'}\n"
                    f'\t        file "{self.assembly.txt_id}.eMesh";\n'
                    f'\t        level 0;\n'
                    f"\t    {'}'}\n"
                    f"\t);\n")
                    continue
            elif attr in ['refinement_surfaces', 'refinement_regions']:
                parameter = self.__getattribute__(attr)
                if parameter is None:
                    s += f'\t{value}\n' + '\t{\n\t}\n'
                    continue

            try:
                if isinstance(value, str):
                    s += self.create_entry(attr=attr, key=value)
                else:
                    s += self.create_entry(attr=attr, key=value['key'], value=value['value'])
            except Exception as e:
                raise e

        s += '}\n'

        return s

    def snap_controls_str(self):

        mapping_dict = {
            'n_smooth_patch': 'nSmoothPatch',
            'tolerance': 'tolerance',
            'n_solve_iter': 'nSolveIter',
            'n_relax_iter': 'nRelaxIter',
            'n_feature_snap_iter': 'nFeatureSnapIter',
            'implicit_feature_snap': 'implicitFeatureSnap',
            'explicit_feature_snap': 'explicitFeatureSnap',
            'multi_region_feature_snap': 'multiRegionFeatureSnap',
            'strict_region_snap': 'strictRegionSnap'
        }

        s = '{\n'
        for attr, value in mapping_dict.items():
            s += self.create_entry(attr=attr, key=value)
        s += '}\n'

        return s

    def create_refinement_surfaces_str(self):

        # refinement_str1 = ''.join(x.shm_refinement_entry for x in self.assembly.solids)
        refinement_str1 = self.assembly.shm_refinement_entry
        if hasattr(self.assembly, 'interface_shm_refinement_entry'):
            refinement_str = refinement_str1 + self.assembly.interface_shm_refinement_entry
        else:
            refinement_str = refinement_str1

        self.refinement_surfaces = "\t{" + '\t'.join(('\n' + '\t' + refinement_str.lstrip()).splitlines(True)) + "\t}\n"

        return self.refinement_surfaces

        # template_face_str = ("\t\\\"<surf_name>\\\"\n"
        #                      "\t{\n"
        #                      "\t\tlevel            (<min_refinement_level> <max_refinement_level>);\n"
        #                      "\t}\n"
        #                      )
        #
        # s = '{\n'
        # for face in self.assembly.faces:
        #     if face.surface_mesh_setup is not None:
        #         face_str = deepcopy(template_face_str)
        #         face_str = face_str.replace('<surf_name>', str(face.txt_id))
        #         face_str = face_str.replace('<min_refinement_level>',
        #                                     str(int(face.surface_mesh_setup.min_refinement_level)))
        #         face_str = face_str.replace('<max_refinement_level>',
        #                                     str(int(face.surface_mesh_setup.max_refinement_level)))
        #         s += face_str
        #
        # s += '}\n'
        #
        # self.refinement_surfaces = s
        # return s

    def create_snappy_hex_mesh_dict(self):

        s = deepcopy(self.template)

        # create info string:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        info_str = f'created: {now}'
        s = s.replace('<info>', info_str)

        s = s.replace('<stls>', self.geom_string())
        s = s.replace('<refinements>', '')

        s = s.replace('<castellated_mesh>', to_of_dict_format(self.castellated_mesh))
        s = s.replace('<snap>', to_of_dict_format(self.snap))
        s = s.replace('<add_layers>', to_of_dict_format(self.add_layers))

        s = s.replace('<merge_tolerance>', to_of_dict_format(self.merge_tolerance))

        s = s.replace('<castellated_mesh_controls>', self.castellated_mesh_str())

        s = s.replace('<snap_controls>', self.snap_controls_str())

        if self.layer_definition is not None:
            s = s.replace('<layer_setup>', self.layer_setup)
        else:
            s = s.replace('<layer_setup>', '{\n}\n')

        s = s.replace('<mesh_quality_controls>', self.mesh_quality_setup.mesh_quality_dict)

        s = s.replace('<max_local_cells>', to_of_dict_format(self.max_local_cells))
        s = s.replace('<max_global_cells>', to_of_dict_format(self.max_global_cells))
        s = s.replace('<min_refinement_cells>', to_of_dict_format(self.min_refinement_cells))

        self.snappy_hex_mesh_dict = s

    def create_layer_setup_str(self):

        s = (f'layers\n'
             '{\n'
             )

        surface_layer_setup_template = ("\t<surf_name>"
                                        "<layer_setup_dict>"
                                        )
        for face in self.assembly.faces:
            if hasattr(face, 'surface_mesh_setup'):
                if face.surface_mesh_setup is None:
                    continue
                if hasattr(face.surface_mesh_setup, 'layer_definition'):
                    if face.surface_mesh_setup.layer_definition is None:
                        continue
                    if not face.surface_mesh_setup.add_layers:
                        continue

            face_str = deepcopy(surface_layer_setup_template)
            face_str = face_str.replace('<surf_name>', str(face.txt_id))
            face_str = face_str.replace('<layer_setup_dict>',
                                        '\t'.join((
                                                          '\n' + face.surface_mesh_setup.layer_definition.layer_setup_dict.lstrip()
                                                  ).splitlines(True)))
            s += face_str
        s += '}\n'

        lines = self.layer_definition.layer_setup_dict.splitlines()
        lines.insert(lines.__len__() - 1, '\t'.join(('\n' + s.lstrip()).splitlines(True)))

        self.layer_setup = '\n'.join(lines)

    def create_surface_feature_extract_dict(self, case_dir=None):
        if case_dir is None:
            case_dir = self.case_dir

        if case_dir is None:
            logging.error(f'{self.name}: no case_dir')
            return

        template = pkg_resources.read_text(msh_resources, 'surfaceFeatureExtractDict')
        s = template.replace('<stl_file>', self.assembly.txt_id + '.stl')

        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(os.path.join(case_dir, '0'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'constant'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'system'), exist_ok=True)

        dst = os.path.join(case_dir, 'system', 'surfaceFeatureExtractDict')
        with open(dst, 'w') as sfed:
            sfed.write(s)

    def run_surface_feature_extract(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        if case_dir is None:
            logging.error(f'{self.name}: no case_dir')
            return

        logger.info(f'Generating mesh....')

        res = subprocess.run(["/bin/bash", "-i", "-c", "surfaceFeatureExtract 2>&1 | tee surfaceFeatureExtract.log"],
                             capture_output=True,
                             cwd=case_dir,
                             user='root')
        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error running surfaceFeatureExtract:\n\n{output}')

            if output.find('FOAM FATAL IO ERROR') != -1:
                logger.error(f'Error running surfaceFeatureExtract:\n\n{output}')
                raise Exception(f'Error running surfaceFeatureExtract:\n\n{output}')

            logger.info(f"Successfully created mesh:\n"
                        f"Directory: {case_dir}\n\n ")

        else:
            logger.error(f"{res.stderr.decode('ascii')}")
            raise Exception(f"Error running surfaceFeatureExtract:\n{res.stderr.decode('ascii')}")
        return True

    def run(self, case_dir=None):
        if case_dir is None:
            case_dir = self.case_dir

        if case_dir is None:
            logging.error(f'{self.name}: no case_dir')
            return

        run_shm(case_dir)

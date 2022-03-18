import uuid
import os
import stat

from ..logger import logger
from ..meshing.block_mesh import BlockMesh
from ..construction import write_region_properties

from PyFoam.Applications.Decomposer import Decomposer

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import case_resources


class OFCase(object):

    default_path = '/tmp/'

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', 'unnamed case')
        self.reference_face = kwargs.get('reference_face')

        self._control_dict = None
        self._case_dir = None
        self._block_mesh = None

        self._control_dict = None
        self._fvschemes = None
        self._fvsolution = None

        # executables
        self._all_mesh = None
        self._all_clean = None
        self._all_run = None

        self.case_dir = kwargs.get('case_dir', None)

    @property
    def block_mesh(self):
        if self._block_mesh is None:
            self._block_mesh = BlockMesh(name=self.name,
                                         case_dir=self.case_dir)
        return self._block_mesh

    @property
    def case_dir(self):
        if self._case_dir is None:
            self._case_dir = os.path.join(self.default_path, 'case_' + str(self.id.hex))
        return self._case_dir

    @case_dir.setter
    def case_dir(self, value):
        if value == self._case_dir:
            return
        self._case_dir = value

    @property
    def control_dict(self):
        if self._control_dict is None:
            self._control_dict = pkg_resources.read_text(case_resources, 'controlDict')
        return self._control_dict

    @control_dict.setter
    def control_dict(self, value):
        if value == self._control_dict:
            return
        self._control_dict = value

    @property
    def fvschemes(self):
        if self._fvschemes is None:
            self._fvschemes = pkg_resources.read_text(case_resources, 'fvSchemes')
        return self._fvschemes

    @fvschemes.setter
    def fvschemes(self, value):
        if value == self._fvschemes:
            return
        self._fvschemes = value

    @property
    def fvsolution(self):
        if self._fvsolution is None:
            self._fvsolution = pkg_resources.read_text(case_resources, 'fvSolution')
        return self._fvsolution

    @fvsolution.setter
    def fvsolution(self, value):
        if value == self._fvsolution:
            return
        self._fvsolution = value

    @property
    def all_mesh(self):
        if self._all_mesh is None:
            self._all_mesh = pkg_resources.read_text(case_resources, 'Allmesh')
        return self._all_mesh

    @all_mesh.setter
    def all_mesh(self, value):
        if value == self._all_mesh:
            return
        self._all_mesh = value

    @property
    def all_clean(self):
        if self._all_clean is None:
            self._all_clean = pkg_resources.read_text(case_resources, 'Allclean')
        return self._all_clean

    @all_clean.setter
    def all_clean(self, value):
        if value == self._all_clean:
            return
        self._all_clean = value

    @property
    def all_run(self):
        if self._all_run is None:
            self._all_run = pkg_resources.read_text(case_resources, 'Allrun')
        return self._all_run

    @all_run.setter
    def all_run(self, value):
        if value == self._all_run:
            return
        self._all_run = value

    def init_case(self):

        logger.info('Initializing case...')

        os.makedirs(self.case_dir, exist_ok=True)
        os.makedirs(os.path.join(self.case_dir, '0'), exist_ok=True)
        os.makedirs(os.path.join(self.case_dir, 'constant'), exist_ok=True)
        os.makedirs(os.path.join(self.case_dir, 'system'), exist_ok=True)

        self.write_all_mesh()
        self.write_all_run()
        self.write_all_clean()

    def write_all_mesh(self):
        all_mesh_full_filename = os.path.join(self.case_dir, "Allmesh")
        with open(all_mesh_full_filename, "w") as f:
            f.write(self.all_mesh)
        os.chmod(all_mesh_full_filename, stat.S_IEXEC)

    def write_all_clean(self):
        all_clean_full_filename = os.path.join(self.case_dir, "Allclean")
        with open(all_clean_full_filename, "w") as f:
            f.write(self.all_clean)
        os.chmod(all_clean_full_filename, stat.S_IEXEC)

    def write_all_run(self):
        all_run_full_filename = os.path.join(self.case_dir, "Allrun")
        with open(all_run_full_filename, "w") as f:
            f.write(self.all_run)
        os.chmod(all_run_full_filename, stat.S_IEXEC)

    def write_control_dict(self):
        with open(os.path.join(self.case_dir, 'system', "controlDict"), mode="w") as f:
            f.write(self.control_dict)

    def write_fv_schemes(self):
        with open(os.path.join(self.case_dir, 'system', "fvSchemes"), "w") as f:
            f.write(self.fvschemes)

    def write_fv_solution(self):
        with open(os.path.join(self.case_dir, 'system', "fvSolution"), "w") as f:
            f.write(self.fvsolution)

    def run(self):

        _ = self.reference_face.pipe_comp_blocks
        _ = self.reference_face.free_comp_blocks
        _ = self.reference_face.extruded_comp_blocks
        comp_blocks = self.reference_face.comp_blocks

        self.reference_face.update_cell_zone()
        self.reference_face.update_boundary_conditions()

        self.block_mesh.init_case()

        self.write_all_mesh()
        self.write_all_run()

        for cell_zone in comp_blocks.cell_zones:
            cell_zone.write_to_of(self.case_dir)
        # write region properties:
        write_region_properties(comp_blocks.cell_zones, self.case_dir)

        logger.debug('bla bla')

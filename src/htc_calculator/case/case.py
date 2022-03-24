import time
import uuid
import os
import stat
import subprocess
from re import findall, MULTILINE

from ..logger import logger
from ..meshing.block_mesh import BlockMesh, inlet_patch, outlet_patch, wall_patch, pipe_wall_patch, top_side_patch, \
    bottom_side_patch, CellZone, BlockMeshBoundary
from ..construction import write_region_properties, Fluid, Solid
from .boundary_conditions.user_bcs import SolidFluidInterface, FluidSolidInterface
from .. import config

from PyFoam.Applications.Decomposer import Decomposer
from .of_parser import CppDictParser

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import case_resources


class TabsBC(object):

    def __init__(self, *args, **kwargs):

        self.inlet_volume_flow = kwargs.get('inlet_volume_flow', 4.1666e-5)     # volume flow rate in m³/s
        self.inlet_temperature = kwargs.get('inlet_temperature', 323.15)     # Inlet temperature in K

        self.top_ambient_temperature = kwargs.get('top_ambient_temperature', 293.15)    # Ambient temperature in K at the top side
        self.bottom_ambient_temperature = kwargs.get('bottom_ambient_temperature', 293.15)  # Ambient temperature in K at the top side

        # heat transfer coefficient in [W/m^2/K] at the top side of the construction
        self.top_htc = kwargs.get('top_htc',
                                  10)
        # heat transfer coefficient in [W/m^2/K] at the top side of the construction
        self.bottom_htc = kwargs.get('bottom_htc',
                                     5.8884)

        self.initial_temperature = kwargs.get('initial_temperatures', 293.15)
        self.cell_zone_initial_temperatures = kwargs.get('initial_temperatures', None)


class OFCase(object):

    default_path = '/tmp/'

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', 'unnamed case')
        self.reference_face = kwargs.get('reference_face')

        self._n_proc = None
        self.n_proc = kwargs.get('n_proc', config.n_proc)

        self._case_dir = None
        self._block_mesh = None

        self._control_dict = None
        self._decompose_par_dict = None
        self._fvschemes = None
        self._fvsolution = None

        # executables
        self._all_mesh = None
        self._all_clean = None
        self._all_run = None

        self.case_dir = kwargs.get('case_dir', None)
        self.bc = kwargs.get('bc', None)

    @property
    def n_proc(self):
        if self._n_proc is None:
            self._n_proc = config.n_proc
        return self._n_proc

    @n_proc.setter
    def n_proc(self, value):
        self._n_proc = value
        config.n_proc = value

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

    @property
    def decompose_par_dict(self):
        if self._decompose_par_dict is None:
            tp_entry = pkg_resources.read_text(case_resources, 'decomposeParDict')
            tp_entry = tp_entry.replace('<n_proc>', str(self.n_proc))
            self._decompose_par_dict = tp_entry
        return self._decompose_par_dict

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

        # self.write_all_mesh()
        # self.write_all_run()
        # self.write_all_clean()

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

    def write_decompose_par_dict(self):
        with open(os.path.join(self.case_dir, 'system', "decomposeParDict"), "w") as f:
            f.write(self.decompose_par_dict)

    def run_block_mesh(self, retry=False):
        logger.info(f'Generating mesh....')
        time.sleep(0.5)
        res = subprocess.run(["/bin/bash", "-i", "-c", "blockMesh 2>&1 | tee blockMesh.log"],
                             capture_output=True,
                             cwd=self.case_dir,
                             user='root')
        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error Creating block mesh:\n\n{output}')
                if retry:
                    if output.find('Inconsistent number of faces') != -1:
                        items = findall("Inconsistent number of faces.*$", output, MULTILINE)
                        inconsistent_blocks = [int(x) for x in findall(r'\d+', items[0])]
                        self.block_mesh.fix_inconsistent_block_faces(inconsistent_blocks)
                        self.run_block_mesh(retry=False)
                else:
                    raise Exception(f'Error Creating block mesh:\n\n{output}')
            logger.info(f"Successfully created block mesh: \n\n {output[output.find('Mesh Information'):]}")
        else:
            logger.error(f"{res.stderr.decode('ascii')}")
            raise Exception(f"Error creating block Mesh:\n{res.stderr.decode('ascii')}")

        time.sleep(0.5)
        return True

    def run_split_mesh_regions(self):
        logger.info(f'Splitting Mesh Regions....')
        res = subprocess.run(
            ["/bin/bash", "-i", "-c", "splitMeshRegions -cellZones -overwrite 2>&1 | tee splitMeshRegions.log"],
            capture_output=True,
            cwd=self.case_dir,
            user='root')
        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error splitting Mesh Regions:\n\n{output}')
                raise Exception(f'Error splitting Mesh Regions:\n\n{output}')
            logger.info(f"Successfully splitted Mesh Regions \n\n{output}")
        else:
            logger.error(f"{res.stderr.decode('ascii')}")

        return True

    def run_decompose_par(self):
        logger.info(f'Running decompose par....')
        res = subprocess.run(
            ["/bin/bash", "-i", "-c", "decomposePar -allRegions 2>&1 | tee decomposePar.log"],
            capture_output=True,
            cwd=self.case_dir,
            user='root')
        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error decomposePar:\n\n{output}')
                raise Exception(f'Error decomposePar:\n\n{output}')
            logger.info(f"Successfully ran decomposePar \n\n{output}")
        else:
            logger.error(f"{res.stderr.decode('ascii')}")

        return True

    def run(self):

        _ = self.reference_face.pipe_comp_blocks
        _ = self.reference_face.free_comp_blocks
        _ = self.reference_face.extruded_comp_blocks
        comp_blocks = self.reference_face.comp_blocks

        self.reference_face.update_cell_zone()
        self.reference_face.update_boundary_conditions()

        self.block_mesh.init_case()

        self.write_control_dict()
        self.write_decompose_par_dict()
        self.write_all_mesh()
        self.write_all_run()
        self.write_all_clean()

        for cell_zone in comp_blocks.cell_zones:
            cell_zone.write_to_of(self.case_dir)
        # write region properties:
        write_region_properties(comp_blocks.cell_zones, self.case_dir)

        self.run_block_mesh(retry=True)
        self.run_split_mesh_regions()

        for cell_zone in comp_blocks.cell_zones:
            cell_zone.case = self
            of_dict = CppDictParser.from_file(os.path.join(self.case_dir, 'constant',
                                                           cell_zone.txt_id, 'polyMesh', 'boundary'))
            boundary_dict = {k: v for k, v in of_dict.values.items() if (v and (k != 'FoamFile'))}

            boundaries = []
            for key, value in boundary_dict.items():
                boundary = BlockMeshBoundary.get_boundary_by_txt_id(key)
                if boundary is None:
                    if '_to_' in key:
                        # create interface
                        if isinstance(cell_zone.material, Solid):
                            user_bc = SolidFluidInterface()
                        elif isinstance(cell_zone.material, Fluid):
                            user_bc = FluidSolidInterface()
                        else:
                            raise NotImplementedError

                        boundary = BlockMeshBoundary(name=key,
                                                     type='interface',
                                                     n_faces=value['nFaces'],
                                                     start_face=value['startFace'],
                                                     case=self,
                                                     txt_id=key,
                                                     user_bc=user_bc,
                                                     cell_zone=cell_zone)
                    else:
                        raise NotImplementedError()
                else:
                    boundary.n_faces = value['nFaces']
                    boundary.startFace = value['startFace']
                    boundary.case = self
                    boundary.cell_zone = cell_zone

                boundaries.append(boundary)

            cell_zone.boundaries = boundaries
            cell_zone.update_bcs()
            cell_zone.write_bcs(self.case_dir)

        self.run_decompose_par()

        logger.debug('bla bla')

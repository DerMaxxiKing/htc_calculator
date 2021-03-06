import copy
import time
import uuid
import os
import stat
import subprocess
from subprocess import Popen, PIPE, STDOUT
from re import findall, MULTILINE
from typing import Union

from ..config import work_dir, n_proc, use_ssh
from ..ssh import shell_handler
from ..logger import logger
from ..meshing.block_mesh import BlockMesh, inlet_patch, outlet_patch, wall_patch, pipe_wall_patch, top_side_patch, \
    bottom_side_patch, CellZone, BlockMeshBoundary, Mesh, CreatePatchDict, export_objects, add_face_contacts, PipeLayerMesh, CyclicAMI
from .function_objects.function_object import PressureDifferencePatch, TemperatureDifferencePatch, WallHeatFlux
from ..construction import write_region_properties, Fluid, Solid
from .boundary_conditions.user_bcs import SolidFluidInterface, FluidSolidInterface
from .. import config
from .function_objects.function_object import FOMetaMock
from ..solid import MultiMaterialSolid

from .of_parser import CppDictParser

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import case_resources
from .case_resources import fluid as fluid_resources

# import FreeCAD
# import Part as FCPart


class TabsBC(object):

    def __init__(self, *args, **kwargs):

        self._g_entry = None

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

        self.g = kwargs.get('g', None)  # Gravity in x, y, z direction

    @property
    def g_entry(self):
        if self._g_entry is None:
            self._g_entry = self.generate_g_entry()
        return self._g_entry

    def generate_g_entry(self):
        entry = pkg_resources.read_text(fluid_resources, 'g')
        entry = entry.replace('<material_id>', '')
        entry = entry.replace('<material_name>', 'all regions')
        return entry

    def init_directory(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'constant'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'system'), exist_ok=True)
        os.makedirs(os.path.join(case_dir, '0'), exist_ok=True)

    def write_g(self, case_dir):
        self.init_directory(case_dir)
        full_filename = os.path.join(case_dir, 'constant', 'g')
        with open(full_filename, "w") as f:
            f.write(self.g_entry)


class OFCase(object):

    default_path = work_dir

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', 'unnamed case')
        self.reference_face = kwargs.get('reference_face')

        self._n_proc = None
        self.n_proc: int = kwargs.get('n_proc', config.n_proc)

        self._case_dir = None
        self._block_mesh = None
        self._combined_mesh = None

        self._control_dict = None
        self._decompose_par_dict = None
        self._fvschemes = None
        self._fvsolution = None

        # executables
        self._all_mesh = None
        self._all_clean = None
        self._all_run = None

        self.case_dir: Union[str, None] = kwargs.get('case_dir', None)
        self.bc: Union[TabsBC, None] = kwargs.get('bc', None)
        self.create_patch_dict = kwargs.get('create_patch_dict', CreatePatchDict(case_dir=self.case_dir))

        self.function_objects = FOMetaMock.instances
        self.combined_mesh = kwargs.get('combined_mesh', None)

        self.cell_zones = []

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
    def combined_mesh(self):
        if self._combined_mesh is None:
            self._combined_mesh = self.combine_mesh()
        return self._combined_mesh

    @combined_mesh.setter
    def combined_mesh(self, value):
        self._combined_mesh = value

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
            control_dict = pkg_resources.read_text(case_resources, 'controlDict')
            control_dict = control_dict.replace('<function_objects>', '')
            self._control_dict = control_dict
        return self._control_dict

    @control_dict.setter
    def control_dict(self, value):
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
            tp_entry = tp_entry.replace('<n_proc>', str(n_proc))
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

        logger.info(f'Initializing case {self.name}\n, {self.id}'
                    f'Direction {self.case_dir}')

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
        res = subprocess.run(["/bin/bash", "-i", "-c", "blockMesh -noFunctionObjects 2>&1 | tee blockMesh.log"],
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
            res = subprocess.run(["/bin/bash", "-i", "-c", "blockMesh - blockTopology"],
                                 capture_output=True,
                                 cwd=self.case_dir,
                                 user='root')

            res = subprocess.run(["/bin/bash", "-i", "-c", "objToVTK blockTopology.obj blockTopology.vtk"],
                                 capture_output=True,
                                 cwd=self.case_dir,
                                 user='root')

        else:
            logger.error(f"{res.stderr.decode('ascii')}")
            raise Exception(f"Error creating block Mesh:\n{res.stderr.decode('ascii')}")
        return True

    def run_check_mesh(self):
        logger.info(f'Checking mesh....')

        if use_ssh:
            shin, shout, sherr = shell_handler.run_check_mesh(self.case_dir)
        else:
            res = subprocess.run(
                ["/bin/bash", "-i", "-c", "checkMesh 2>&1 | tee checkMesh.log"],
                capture_output=True,
                cwd=self.case_dir,
                user='root')
            if res.returncode == 0:
                output = res.stdout.decode('ascii')
                if output.find('FOAM FATAL ERROR') != -1:
                    logger.error(f'Error decomposePar:\n\n{output}')
                    raise Exception(f'Error decomposePar:\n\n{output}')
                logger.info(f"Successfully ran checkMesh \n\n{output}")
            else:
                logger.error(f"{res.stderr.decode('ascii')}")

        return True

    def run_split_mesh_regions(self):
        logger.info(f'Splitting Mesh Regions....')
        res = subprocess.run(
            ["/bin/bash", "-i", "-c", "splitMeshRegions -cellZonesOnly -overwrite 2>&1 | tee splitMeshRegions.log"],
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

    def run_parafoam(self):
        logger.info(f'Running paraFoam initialization....')
        res = subprocess.run(["/bin/bash", "-i", "-c", "paraFoam -touchAll"],
                             capture_output=True,
                             cwd=self.case_dir,
                             user='root')
        if res.returncode == 0:
            output = res.stdout.decode('ascii')
            logger.info(f"Successfully ran paraFoam initialization \n\n{output}")
        else:
            logger.error(f"{res.stderr.decode('ascii')}")
        return True

    def run_change_dict(self, case_dir=None, regions=None):
        if case_dir is None:
            case_dir = self.case_dir

        shell_handler.run_change_dict(self.case_dir, regions=regions)

    def run_allrun(self):
        logger.info(f'Running solver....')

        execute(["/bin/bash", "-i", "-c", "./Allrun"], cwd=self.case_dir)

        # res = subprocess.run(["/bin/bash", "-i", "-c", "./Allrun"],
        #                      capture_output=True,
        #                      cwd=self.case_dir,
        #                      user='root')
        # if res.returncode == 0:
        #     output = res.stdout.decode('ascii')
        #     logger.info(f"Successfully ran solver \n\n{output}")
        # else:
        #     logger.error(f"{res.stderr.decode('ascii')}")
        # return True

    def add_function_objects(self, mesh=None):

        if mesh is None:
            mesh = FOMetaMock.current_mesh

        control_dict = pkg_resources.read_text(case_resources, 'controlDict')

        fo_dict_entry = ''
        for fo in mesh.function_objects:
            fo_dict_entry += fo.dict_entry + '\n'

        control_dict = control_dict.replace('<function_objects>', fo_dict_entry)
        self.control_dict = control_dict

    def combine_mesh(self):
        # _ = self.reference_face.pipe_comp_blocks
        # _ = self.reference_face.free_comp_blocks
        # # export_objects([self.reference_face.free_comp_blocks.fc_solid], '/tmp/free_comp_blocks.FCStd')
        # _ = self.reference_face.layer_meshes
        #
        # joined_pipe_layer_mesh, face_lookup_dict = PipeLayerMesh.join_meshes(
        #     [self.reference_face.pipe_mesh,
        #      self.reference_face.construction_mesh],
        #     'joined_pipe_layer_mesh',
        # )
        # self.reference_face.pipe_layer.meshes.remove(self.reference_face.pipe_mesh)
        # self.reference_face.pipe_layer.meshes.remove(self.reference_face.construction_mesh)
        # self.reference_face.pipe_layer.meshes.add(joined_pipe_layer_mesh)
        #
        # # add cyclicAMI at interfaces:
        # add_face_contacts([face_lookup_dict[x.id] for x in self.reference_face.pipe_mesh.interfaces],
        #                   [face_lookup_dict[x.id] for x in self.reference_face.construction_mesh.interfaces],
        #                   joined_pipe_layer_mesh.mesh,
        #                   joined_pipe_layer_mesh.mesh,
        #                   f'pipe_mesh_to_construction_mesh',
        #                   f'construction_mesh_to_pipe_mesh'
        #                   )

        combined_mesh = self.reference_face.combine_meshes()
        combined_mesh.case_dir = self.case_dir

        return combined_mesh

    def merge_mesh(self, case_dir=None, other_case_dir=None, overwrite=True, add_ami=True):

        if case_dir is None:
            case_dir = self.case_dir

        if overwrite:
            overwrite_txt = ' -overwrite'
        else:
            overwrite_txt = ''

        logger.info(f'Merging meshes:\n'
                    f'Source: {other_case_dir}\n'
                    f'Target {case_dir}\n')

        if use_ssh:
            shell_handler.run_merge_meshes(case_dir, other_case_dir,
                                           parallel=False,
                                           options=' -noFunctionObjects',
                                           overwrite=overwrite)
        else:
            res = subprocess.run(["/bin/bash", "-i", "-c", f"mergeMeshes {case_dir} {other_case_dir} {overwrite_txt} -noFunctionObjects"],
                                 capture_output=True,
                                 cwd=case_dir,
                                 user='root')

            if res.returncode == 0:
                output = res.stdout.decode('ascii')
                logger.info(f"Successfully merged meshes \n\n{output}")
            else:
                logger.error(f"Error Merging meshes:\n{res.stderr.decode('ascii')}")

        # if add_ami:
        #     amis = []
        #
        #     for name, boundary in mesh.mesh.boundaries.items():
        #         if not isinstance(boundary, CyclicAMI):
        #             continue
        #     if boundary.neighbour_patch in self.mesh.boundaries.values():
        #         amis.extend([boundary, boundary.neighbour_patch])
        #
        #     cpd = CreatePatchDict(case_dir=case_dir,
        #                           boundaries=amis)
        #     cpd.write_create_patch_dict()
        #     cpd.run(case_dir=self.case_dir)
        #
        #     return cpd
        # else:
        #     return True

    def run(self):

        _ = self.reference_face.pipe_comp_blocks
        _ = self.reference_face.free_comp_blocks
        _ = self.reference_face.extruded_comp_blocks
        comp_blocks = self.reference_face.comp_blocks

        cell_zones = self.reference_face.update_cell_zone()
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

        # self.run_block_mesh(retry=True)
        self.block_mesh.run_block_mesh(case_dir=self.case_dir)
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

        self.add_function_objects()
        self.write_control_dict()
        self.run_decompose_par()

        logger.debug('bla bla')

    def run_with_separate_meshes(self, export_geometry=True):

        if export_geometry:
            for solid in self.reference_face.assembly.solids:
                solid.export_step(f'/tmp/cad/{solid.name}.step')
            self.reference_face.assembly.export_stp(f'/tmp/cad/assembly.step')
        # _ = self.reference_face.pipe_comp_blocks
        # _ = self.reference_face.free_comp_blocks
        # # export_objects([self.reference_face.free_comp_blocks.fc_solid], '/tmp/free_comp_blocks.FCStd')
        # _ = self.reference_face.layer_meshes
        # # _ = self.reference_face.extruded_comp_blocks
        #
        # joined_pipe_layer_mesh, face_lookup_dict = PipeLayerMesh.join_meshes(
        #     [self.reference_face.pipe_mesh,
        #      self.reference_face.construction_mesh],
        #     'joined_pipe_layer_mesh',
        # )
        # self.reference_face.pipe_layer.meshes.remove(self.reference_face.pipe_mesh)
        # self.reference_face.pipe_layer.meshes.remove(self.reference_face.construction_mesh)
        # self.reference_face.pipe_layer.meshes.add(joined_pipe_layer_mesh)
        #
        # # add cyclicAMI at interfaces:
        # add_face_contacts([face_lookup_dict[x.id] for x in self.reference_face.pipe_mesh.interfaces],
        #                   [face_lookup_dict[x.id] for x in self.reference_face.construction_mesh.interfaces],
        #                   joined_pipe_layer_mesh.mesh,
        #                   joined_pipe_layer_mesh.mesh,
        #                   f'pipe_mesh_to_construction_mesh',
        #                   f'construction_mesh_to_pipe_mesh'
        #                   )
        #
        # combined_mesh = self.reference_face.combine_meshes()

        cell_zones = self.reference_face.update_cell_zone(blocks=self.combined_mesh.mesh.blocks,
                                                          mesh=self.combined_mesh.mesh)

        # self.reference_face.update_boundary_conditions(self.combined_mesh)

        logger.info('Updating boundary conditions...')
        _ = [setattr(x, 'boundary', bottom_side_patch) for x in self.combined_mesh.bottom_faces]
        _ = [setattr(x, 'boundary', top_side_patch) for x in self.combined_mesh.top_faces]

        hull_faces = [x for x in self.combined_mesh.mesh.faces.values() if (x.blocks.__len__() < 2)]
        wall_faces = [x for x in hull_faces if x.boundary is None]

        # export_objects([FCPart.Compound([x.fc_face for x in wall_faces])], '/tmp/wall_faces.FCStd')
        # export_objects([FCPart.Compound([x.fc_face for x in
        #                                  [*self.combined_mesh.bottom_faces, *self.combined_mesh.top_faces]])],
        #                                  '/tmp/top_bottom.FCStd')
        local_wall_patch = BlockMeshBoundary.copy_to_mesh(wall_patch, self.combined_mesh.mesh)
        _ = [setattr(x, 'boundary', local_wall_patch) for x in wall_faces]

        self.combined_mesh.init_case()

        for cell_zone in cell_zones:
            cell_zone.write_to_of(self.case_dir)

        # write region properties:
        write_region_properties(cell_zones, self.case_dir)

        self.combined_mesh.run_block_mesh(run_parafoam=True)
        self.run_check_mesh()
        # self.create_patch_dict.boundaries = [x for x in self.combined_mesh.mesh.boundaries.values() if (type(x) is CyclicAMI)]
        # self.create_patch_dict.case_dir = self.combined_mesh.case_dir
        # self.create_patch_dict.write_create_patch_dict()
        # self.create_patch_dict.run()

        self.run_split_mesh_regions()
        self.run_parafoam()

        self.write_all_mesh()
        self.write_all_run()
        self.write_all_clean()

        for cell_zone in cell_zones:
            cell_zone.case = self
            of_dict = CppDictParser.from_file(os.path.join(self.case_dir, 'constant',
                                                           cell_zone.txt_id, 'polyMesh', 'boundary'))
            boundary_dict = {k: v for k, v in of_dict.values.items() if (v and (k != 'FoamFile'))}

            boundaries = []
            for key, value in boundary_dict.items():
                boundary = BlockMeshBoundary.get_boundary_by_txt_id(key, mesh=self.combined_mesh.mesh)
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
                                                     mesh=self.combined_mesh.mesh,
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

                WallHeatFlux(name=f'Wall Heat Flux on {boundary.name}',
                             cell_zone=cell_zone,
                             mesh=self.combined_mesh,
                             patches=[boundary])

                boundaries.append(boundary)

            cell_zone.boundaries = boundaries
            cell_zone.update_bcs()
            cell_zone.write_bcs(self.case_dir)
            logger.debug(f'Wrote boundary condition for cell zone {cell_zone}')

        # add function objects:
        PressureDifferencePatch(patch1=self.combined_mesh.mesh.boundaries['inlet'],
                                patch2=self.combined_mesh.mesh.boundaries['outlet'],
                                mesh=self.combined_mesh.mesh,
                                name='Pressure Difference Patch')

        TemperatureDifferencePatch(patch1=self.combined_mesh.mesh.boundaries['inlet'],
                                   patch2=self.combined_mesh.mesh.boundaries['outlet'],
                                   mesh=self.combined_mesh.mesh,
                                   name='Pressure Difference Patch')

        self.add_function_objects(mesh=self.combined_mesh.mesh)
        self.write_control_dict()
        self.write_decompose_par_dict()
        self.run_decompose_par()

        self.run_allrun()

        logger.debug('bla bla')

        # logger.info('Adding cyclicAMI boundary condition to mesh interfaces')
        # block_meshes = [joined_pipe_layer_mesh,
        #                 *self.reference_face.layer_meshes]
        #
        # blocks = []
        # _ = [blocks.extend(x.mesh.blocks) for x in block_meshes]
        # self.reference_face.update_cell_zone(blocks=blocks)
        # self.reference_face.update_boundary_conditions()
        #
        # for block_mesh in block_meshes:
        #     block_mesh.init_case()
        #     block_mesh.run_block_mesh(run_parafoam=True)
        #
        # # create empty case
        #
        # for mesh in block_meshes[1:]:
        #     block_meshes[0].merge_mesh(mesh)
        #
        # block_meshes = []
        # for key, mesh in Mesh.instances.items():
        #     if 0 in [mesh.vertices.__len__(), mesh.blocks.__len__()]:
        #         continue
        #
        #     mesh.activate()
        #     block_mesh = BlockMesh(name='Block Mesh ' + mesh.name,
        #                            case_dir=os.path.join(self.default_path, mesh.txt_id),
        #                            mesh=mesh)
        #     block_meshes.append(block_mesh)
        #     block_mesh.init_case()
        #     block_mesh.run_block_mesh(run_parafoam=True)
        #
        # print('done')

    def run_with_separate_meshes2(self):

        # export CAD files:

        for solid in self.reference_face.assembly.solids:
            solid.export_step(f'/tmp/cad/{solid.name}.step')
        self.reference_face.assembly.export_stp(f'/tmp/cad/assembly.step')

        _ = self.reference_face.pipe_comp_blocks
        _ = self.reference_face.free_comp_blocks
        # export_objects([self.reference_face.free_comp_blocks.fc_solid], '/tmp/free_comp_blocks.FCStd')
        _ = self.reference_face.layer_meshes

        meshes = [self.reference_face.pipe_mesh,
                  self.reference_face.construction_mesh,
                  *self.reference_face.layer_meshes]

        blocks = []
        _ = [blocks.extend(x.mesh.blocks) for x in meshes]
        self.reference_face.update_cell_zone(blocks=blocks)

        for mesh in meshes:
            mesh.init_case()
            mesh.run_block_mesh()
            if mesh is self.reference_face.pipe_mesh:
                mesh.run_split_mesh_regions()
            mesh.run_check_mesh()
            mesh.run_parafoam()

        empty_mesh = BlockMesh(name='combined_mesh',
                               mesh=Mesh(name='combined_mesh'),
                               case_dir=self.case_dir)
        empty_mesh.init_case()
        empty_mesh.run_block_mesh()

        for mesh in meshes:
            empty_mesh.merge_mesh(mesh, overwrite=True, add_ami=False)

        self.run_split_mesh_regions()
        empty_mesh.run_parafoam()

    def run_with_separate_meshes_3(self):

        from .boundary_conditions.face_bcs import bottom_side, top_side, wall, Interface

        self.init_case()
        self.write_control_dict()
        self.write_decompose_par_dict()
        self.write_fv_schemes()
        self.write_fv_solution()
        self.write_all_run()
        self.write_all_clean()
        self.bc.write_g(case_dir=self.case_dir)

        assembly = self.reference_face.assembly

        # assembly.solids[-1].run_meshing(split_mesh_regions=False)

        for i, solid in enumerate(assembly.solids):
            print(f'{solid.name}')
            solid.run_meshing(split_mesh_regions=False)

            if not solid.mesh_ok:
                raise Exception(f'Mesh of{solid.name} is not ok')
            # solid.run_check_mesh()
            # if isinstance(solid, MultiMaterialSolid):
            #     shell_handler.run_split_mesh_regions(workdir=solid.case_dir)

            if i == 0:
                shell_handler.copy_mesh(solid.case_dir, self.case_dir)
            else:
                self.merge_mesh(case_dir=self.case_dir,
                                other_case_dir=solid.case_dir)

        logger.info(f'Successfully created and merged meshes')

        # logger.info(f'Splitting mesh ...')
        shell_handler.run_split_mesh_regions(workdir=self.case_dir)
        shell_handler.run_parafoam(workdir=self.case_dir)

        for i, solid in enumerate(assembly.solids):
            if isinstance(solid, MultiMaterialSolid):
                solid.rename_internal_interfaces(case_dir=self.case_dir)

        # logger.info(f'Successfully split mesh regions')

        materials = set()

        _ = [materials.update([y.material for y in x.cell_zones]) for x in assembly.solids]
        cell_zones = [x.cell_zone for x in materials]

        for material in materials:
            material.write_to_of(self.case_dir)
        # write region properties:
        write_region_properties(cell_zones, self.case_dir)

        # update boundary conditions
        # --------------------------------------------------------------------------------------------------------------

        for i, layer in enumerate(self.reference_face.component_construction.layers):
            logger.info(f'Writing boundary conditions for layer {layer}')
            solid = layer.solid
            if i == 0:
                if 'base_faces' in solid.features:
                    solid.features['base_faces'].boundary_condition = bottom_side
                else:
                    logger.error(f'Could not find base face for layer {i} in features: solid {layer.solid}.\n '
                                 f'Features are: {solid.features.keys()}')
                    raise Exception(f'Could not find base face for layer {i} in features: solid {layer.solid}.\n '
                                    f'Features are: {solid.features.keys()}')
            if i == self.reference_face.component_construction.layers.__len__() - 1:
                if 'top_faces' in solid.features:
                    solid.features['top_faces'].boundary_condition = top_side
                else:
                    logger.error(f'Could not find base face for layer {i} in features: solid {layer.solid}.\n '
                                 f'Features are: {solid.features.keys()}')
                    raise Exception(f'Could not find base face for layer {i} in features: solid {layer.solid}.\n '
                                    f'Features are: {solid.features.keys()}')
            else:
                # add interface between layers
                if 'top_faces' in solid.features:
                    side_1_face = solid.features['top_faces']
                else:
                    logger.error(f'Could not find top faces for layer {i} in features: solid {layer.solid}.\n '
                                 f'Features are: {solid.features.keys()}')
                    raise Exception(f'Could not find top faces for layer {i} in features: solid {layer.solid}.\n '
                                    f'Features are: {solid.features.keys()}')
                next_layer_solid = self.reference_face.component_construction.layers[i+1].solid
                if 'base_faces' in next_layer_solid.features:
                    side_2_face = next_layer_solid.features['base_faces']
                else:
                    logger.error(f'Could not find base faces for layer {i+1} in features: solid {next_layer_solid}.\n '
                                 f'Features are: {next_layer_solid.features.keys()}')
                    raise Exception(f'Could not find base faces for layer {i+1} in features: solid {next_layer_solid}.\n '
                                    f'Features are: {next_layer_solid.features.keys()}')

                side_1_face.boundary_condition = Interface(type='mapped_wall', face_1=side_1_face, face_2=side_2_face)
                side_2_face.boundary_condition = Interface(type='mapped_wall', face_1=side_2_face, face_2=side_1_face)

            if 'side_faces' in solid.features:
                solid.features['side_faces'].boundary_condition = wall

        # add interface between pipe and layer
        side_1_face = assembly.features['pipe_mesh_solid'].features['interfaces']
        side_2_face = assembly.features['pipe_layer_solid'].features['pipe_mesh_interfaces']

        side_1_face.boundary_condition = Interface(type='mapped_wall', face_1=side_1_face, face_2=side_2_face)
        side_2_face.boundary_condition = Interface(type='mapped_wall', face_1=side_2_face, face_2=side_1_face)

        assembly_materials = []
        for solid in assembly.solids:

            if isinstance(solid, MultiMaterialSolid):
                for face in solid.faces:
                    if face.material is None:
                        continue
                    mat = face.material
                    if mat not in assembly_materials:
                        assembly_materials.append(mat)
                    mat.boundaries[face] = face.boundary_condition
            else:
                mat = solid.material
                if mat not in assembly_materials:
                    assembly_materials.append(mat)

                for face in solid.faces:
                    mat.boundaries[face] = face.boundary_condition

        for assembly_material in assembly_materials:
            assembly_material.write_change_dict(case_dir=self.case_dir)
            print('done')

        self.run_change_dict()

        print('done')


def execute(command, cwd):
    p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd=cwd)
    for line in p.stdout:
        print(line)

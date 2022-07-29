import paramiko
import re
import os
from inspect import cleandoc
from ..logger import logger
from multiprocessing import cpu_count
from math import floor
from shutil import copyfile
from time import sleep
import csv
from ..case.of_parser import CppDictParser

from ..config import ssh_pwd, ssh_user, host

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from ..meshing import meshing_resources as msh_resources


class ShellHandler:

    def __init__(self, host, user, psw):
        """
        https://stackoverflow.com/questions/35821184/implement-an-interactive-shell-over-ssh-in-python-using-paramiko
        :param host:
        :param user:
        :param psw:
        """
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(host, username=user, password=psw, port=22)

        channel = self.ssh.invoke_shell()
        self.stdin = channel.makefile('wb')
        self.stdout = channel.makefile('r')

        self._num_worker = None

    @property
    def num_worker(self):
        if self._num_worker is None:
            self._num_worker = self.get_num_worker()
        return self._num_worker

    def __del__(self):
        self.ssh.close()

    def file_exists(self, path):
        hin, shout, sherr = self.execute(f'test -f "{path}" && echo found || echo not found')
        if shout[-1].find('does not exists') != -1:
            return False
        else:
            return True

    def pwd(self):
        hin, shout, sherr = self.execute('pwd')
        return shout[0]

    def cwd(self, workdir):
        shin, shout, sherr = self.execute(f'cwd {workdir}')

    def execute(self, cmd, cwd=None):
        """

        :param cwd: change the current working directory
        :param cmd: the command to be executed on the remote computer
        :examples:  execute('ls')
                    execute('finger')
                    execute('cd folder_name')
        """
        if cwd is not None:

            _ = self.execute('echo "clearing shout"')
            try:
                shin, shout, sherr = self.execute('pwd')
                sleep(0.1)
                pwd = shout[0]
                if pwd.startswith('root@openfoam:'):
                    pwd = pwd[len('root@openfoam:'):]
                pwd = pwd.rstrip()

            except Exception as e:
                pwd = None
                logger.warning(f'Could not get pwd')
            finally:
                self.execute(f'cd {cwd}')

        cmd = cmd.strip('\n')
        # self.stdin.write("sudo su " + '\n')
        self.stdin.write(cmd + '\n')
        finish = 'end of stdOUT buffer. finished with exit status'
        echo_cmd = 'echo {} $?'.format(finish)
        self.stdin.write(echo_cmd + '\n')
        shin = self.stdin
        self.stdin.flush()

        shout = []
        sherr = []
        exit_status = 0
        for line in self.stdout:
            if str(line).startswith(cmd) or str(line).startswith(echo_cmd):
                # up for now filled with shell junk from stdin
                shout = []
            elif str(line).startswith(finish):
                # our finish command ends with the exit status
                exit_status = int(str(line).rsplit(maxsplit=1)[1])
                if exit_status:
                    # stderr is combined with stdout.
                    # thus, swap sherr with shout in a case of failure.
                    sherr = shout
                    shout = []
                break
            else:
                # get rid of 'coloring and formatting' special characters
                shout.append(re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', line).
                             replace('\b', '').replace('\r', ''))

        # first and last lines of shout/sherr contain a prompt
        if shout and echo_cmd in shout[-1]:
            shout.pop()
        if shout and cmd in shout[0]:
            shout.pop(0)
        if sherr and echo_cmd in sherr[-1]:
            sherr.pop()
        if sherr and cmd in sherr[0]:
            sherr.pop(0)

        if cwd is not None:
            if pwd is not None:
                self.execute(f'cd {pwd}')

        if sherr:
            logger.error(f'Error executing command: {cmd}:\n' + ''.join(sherr))

        return shin, shout, sherr

    def get_num_worker(self):

        shin, shout, sherr = self.execute('lscpu')
        values = {x[0]: x[1] for x in list(csv.reader(shout, delimiter=':')) if x.__len__() == 2}

        return int(values['CPU(s)'])

    def run_surface_feature_extract(self, workdir):
        shin, shout, sherr = self.execute(f'surfaceFeatureExtract 2>&1 | tee surfaceFeatureExtract.log', cwd=workdir)
        if sherr:
            logger.error(f"Error running surfaceFeatureExtract: \n {''.join(sherr)}")
            raise Exception(f"Error running surfaceFeatureExtract:  \n {''.join(sherr)}")
        else:
            output = ''.join(shout)
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error running surfaceFeatureExtract:\n\n{output}')
                raise Exception(f"Error running surfaceFeatureExtract:  \n {''.join(sherr)}")
            logger.info(f"Successfully created surfaceFeatureExtract:\n"
                        f"Directory: {workdir}\n\n "
                        f"{output[output.find('Initial Feature set'):output.find('Writing extendedFeatureEdgeMesh')]}")

        return True

    def run_shm(self, workdir, parallel=False, num_worker=None):
        cmd = 'snappyHexMesh'
        if parallel:
            cmd += ' -parallel'
            if num_worker is None:
                num_worker = int(self.num_worker / 2)

            _ = self.execute(f'export OMPI_ALLOW_RUN_AS_ROOT=1')
            _ = self.execute(f'export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1')

            template = pkg_resources.read_text(msh_resources, 'decompose_par_dict')
            s = template.replace('<n_procs>', str(int(num_worker)))
            with open(os.path.join(workdir, 'system', 'decomposeParDict'), 'w') as sfed:
                sfed.write(s)

            run_shm_dest = os.path.join(workdir, 'runSHM')
            with pkg_resources.path(msh_resources, "runSHM") as p:
                copyfile(p, run_shm_dest)

            shin, shout, sherr = self.execute(f'chmod +x {run_shm_dest}')
            shin, shout, sherr = self.execute(f'./runSHM', cwd=workdir)

        else:
            shin, shout, sherr = self.execute(cmd, cwd=workdir)

        if sherr:
            logger.error(f"Error running splitMeshRegions: \n {''.join(sherr)}")
            raise Exception(f"Error running splitMeshRegions:  \n {''.join(sherr)}")

        # read log:
        with open(os.path.join(workdir, 'log.snappyHexMesh')) as f:
            content = f.read()
        if 'FOAM FATAL ERROR' in content:
            logger.error(f'Error running snappyHexMesh:\n\n{content}')
            raise Exception(f"Error running snappyHexMesh:  \n {''.join(content)}")

        self.run_parafoam(workdir=workdir)

        return shin, shout, sherr

    def run_check_mesh(self, workdir, options=None):
        if options is None:
            cmd = 'checkMesh 2>&1 | tee checkMesh.log'
        else:
            cmd = 'checkMesh ' + options + ' 2>&1 | tee checkMesh.log'
        shin, shout, sherr = self.execute(cmd, cwd=workdir)

        if sherr:
            logger.error(f"Error running checkMesh: \n {''.join(sherr)}")
            raise Exception(f"Error running checkMesh:  \n {''.join(sherr)}")
        else:
            output = ''.join(shout)
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error in checkMesh:\n\n{output}')
                raise Exception(f"Error running checkMesh:  \n {''.join(sherr)}")
            logger.info(f"Successfully checked mesh:\n"
                        f"Directory: {workdir}\n\n "
                        f"{output}")
            if output[output.find('Failed'):output.find('End')]:
                logger.warning(output[output.find('Failed'):output.find('End')])
            if output[output.find('Mesh OK'):output.find('End')]:
                logger.info(f'Mesh is ok!')

        self.execute(f'paraFoam -touchAll', cwd=workdir)

        return shin, shout, sherr

    def run_block_mesh(self, workdir, options=None):
        if options is None:
            cmd = 'blockMesh'
        else:
            cmd = 'blockMesh ' + options
        shin, shout, sherr = self.execute(cmd, cwd=workdir)

        if sherr:
            logger.error(f"Error running blockMesh: \n {''.join(sherr)}")
            raise Exception(f"Error running blockMesh:  \n {''.join(sherr)}")
        else:
            output = ''.join(shout)
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error Creating block mesh:\n\n{output}')
                raise Exception(f"Error running blockMesh:  \n {''.join(sherr)}")
            logger.info(f"Successfully created block mesh:\n"
                        f"Directory: {workdir}\n\n "
                        f"{output[output.find('Mesh Information'):output.find('End')]}")

        self.execute(f'paraFoam -touchAll', cwd=workdir)

        return shin, shout, sherr

    def run_split_mesh_regions(self, workdir, options=None):
        logger.info(f"Running split mesh regions in {workdir}")
        if options is None:
            cmd = 'splitMeshRegions -cellZonesOnly -noFunctionObjects -overwrite  2>&1 | tee splitMeshRegions.log'
        else:
            cmd = 'splitMeshRegions -cellZonesOnly -noFunctionObjects -overwrite ' + options + ' 2>&1 | tee splitMeshRegions.log'
        shin, shout, sherr = self.execute(cmd, cwd=workdir)
        if sherr:
            logger.error(f"Error running splitMeshRegions: \n {''.join(sherr)}")
            raise Exception(f"Error running splitMeshRegions:  \n {''.join(sherr)}")
        else:
            output = ''.join(shout)
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error running splitMeshRegions:\n\n{output}')
                raise Exception(f"Error running splitMeshRegions:  \n {''.join(sherr)}")
            logger.info(f"Successfully ran splitMeshRegions:\n"
                        f"Directory: {workdir}\n\n "
                        f"{output}")
        self.execute(f'paraFoam -touchAll', cwd=workdir)

        logger.info(f"Successfully ran splitMeshRegions in {workdir}")

        return shin, shout, sherr

    def run_merge_meshes(self, workdir, other_case_dir, options=None, parallel=False, overwrite=True, num_worker=None):

        cmd = f'mergeMeshes -case {workdir}'

        if options is not None:
            cmd = cmd + options

        if parallel:
            cmd = cmd + ' -parallel'

            if num_worker is None:
                num_worker = self.num_worker

            _ = self.execute(f'export OMPI_ALLOW_RUN_AS_ROOT=1')
            _ = self.execute(f'export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1')

            template = pkg_resources.read_text(msh_resources, 'decompose_par_dict')
            s = template.replace('<n_procs>', str(int(num_worker / 2)))
            dec_target = os.path.join(workdir, 'system', 'decomposeParDict')
            with open(dec_target, 'w') as sfed:
                sfed.write(s)

            cmd = f'mpirun -np {int(self.num_worker/2)} ' + cmd + f' -decomposeParDict {dec_target}'

            shin, shout, sherr = self.execute('decomposePar', cwd=workdir)

        if overwrite:
            cmd = cmd + ' -overwrite'

        cmd = cmd + f' {workdir} {other_case_dir}'

        shin, shout, sherr = self.execute(cmd, cwd=workdir)
        if sherr:
            logger.error(f"Error running {cmd}: \n {''.join(sherr)}")
            raise Exception(f"Error running {cmd}:  \n {''.join(sherr)}")
        else:
            output = ''.join(shout)
            if output.find('FOAM FATAL ERROR') != -1:
                logger.error(f'Error running {cmd}:\n\n{output}')
                raise Exception(f"Error running {cmd}:  \n {''.join(sherr)}")
            logger.info(f"Successfully ran {cmd}:\n"
                        f"Directory: {workdir}\n\n "
                        f"{output}")
        self.execute(f'paraFoam -touchAll', cwd=workdir)

    def run_parafoam(self, workdir):
        return self.execute(f'paraFoam -touchAll', cwd=workdir)

    def copy_mesh(self, source, destination, time=None):

        logger.info(f'Copying mesh from {source} to {destination}')

        if time is None:
            time = get_latest_timestep(source)

        destination_dir = os.path.join(destination, 'constant')
        if time is None:
            source_dir = os.path.join(source, 'constant', 'polyMesh')
        elif float(time) == 0:
            source_dir = os.path.join(source, 'constant', 'polyMesh')
        else:
            source_dir = os.path.join(source, time, 'constant', 'polyMesh')

        cmd = f'cp -r {source_dir} {destination_dir}'
        shin, shout, sherr = self.execute(cmd)

        if sherr:
            logger.error(f"Error running {cmd}: \n {''.join(sherr)}")
            raise Exception(f"Error running {cmd}:  \n {''.join(sherr)}")
        else:
            output = ''.join(shout)
            logger.info(f"Successfully ran {cmd}:\n"
                        f"{output}")

        logger.info(f'Successfully copied mesh from {source} to {destination}')

        return True

    def list_regions(self, workdir):
        res = self.execute(f'foamListRegions', cwd=workdir)
        return [x.rstrip('\n') for x in res[1][0:-1]]

    def run_change_dict(self, workdir, regions=None, init_fields=True):
        if regions is None:
            regions = self.list_regions(workdir)

        if not isinstance(regions, list):
            regions = [regions]

        for region in regions:

            try:
                of_dict = CppDictParser.from_file(os.path.join(workdir, 'system', region, 'changeDictionaryDict'))
            except Exception as e:
                logger.error(f'Could not read changeDictionaryDict for region {region} in {workdir}:\n{e}')
                raise Exception(f'Could not read changeDictionaryDict for region {region} in {workdir}:\n{e}')

            for key in of_dict.values.keys():
                if key in ['FoamFile', 'boundary']:
                    continue
                if init_fields:
                    write_empty_field(key, workdir, region)

            logger.info(f'Running changeDictionary for region {region} in {workdir}')
            res = self.execute(f'changeDictionary -region {region}', cwd=workdir)
            print('done')


sh = ShellHandler(host, ssh_user, ssh_pwd)


def get_latest_timestep(directory):
    latest_ts = None

    shin, shout, sherr = sh.execute('foamListTimes -latestTime -withZero', cwd=directory)
    return shout[0].rstrip('\n')

    # shin, shout, sherr = sh.execute('ls -a', cwd=directory)
    #
    # directories = '\n'.join(shout).split()
    #
    # for directory in directories:
    #
    #     try:
    #         ts = float(directory)
    #         _, __, err = sh.execute(f'[ -d {directory} ]', cwd=directory)
    #         if err:
    #             continue
    #
    #         if latest_ts is None:
    #             latest_ts = directory
    #         elif ts > float(latest_ts):
    #             latest_ts = directory
    #     except ValueError:
    #         continue
    # return latest_ts


dimension_lookup_dict = {'alphat': '[1 -1 -1 0 0 0 0]',
                         'k': '[0 2 -2 0 0 0 0]',
                         'mut': '[1 -1 -1 0 0 0 0]',
                         'nut': '[0 2 -1 0 0 0 0]',
                         'omega': '[0 0 -1 0 0 0 0]',
                         'p': '[1 -1 -2 0 0 0 0]',
                         'p_rgh': '[1 -1 -2 0 0 0 0]',
                         'T': '[0 0 0 1 0 0 0]',
                         'U': '[0 1 -1 0 0 0 0]'}

internal_field_lookup_dict = {'alphat': 'uniform 0',
                              'k': 'uniform 0.00015',
                              'mut': 'uniform 0',
                              'nut': 'uniform 0',
                              'omega': 'uniform 0.2',
                              'p': 'uniform 100000',
                              'p_rgh': 'uniform 100000',
                              'T': 'uniform 293.15',
                              'U': 'uniform (0 0 0)'}

class_lookup_dict = {'alphat': 'volScalarField',
                     'k': 'volScalarField',
                     'mut': 'volScalarField',
                     'nut': 'volScalarField',
                     'omega': 'volScalarField',
                     'p': 'volScalarField',
                     'p_rgh': 'volScalarField',
                     'T': 'volScalarField',
                     'U': 'volVectorField'}


def write_empty_field(field_name, case_dir, region_id):
    field_template = cleandoc("""
    /*--------------------------------*- C++ -*----------------------------------*\
      =========                 |
      \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
       \\    /   O peration     | Website:  https://openfoam.org
        \\  /    A nd           | Version:  9
         \\/     M anipulation  |
    \*---------------------------------------------------------------------------*/
    FoamFile
    {
        format      ascii;
        class       <class>;
        location    "0/<region_id>";
        object      <field_name>;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    dimensions      <dimensions>;

    internalField   <internal_field>;

    boundaryField
    {
        ".*"
        {
            type            calculated;
            value           $internalField;
        }
    }

    // ************************************************************************* //
    """)

    field_str = field_template
    field_str = field_str.replace('<field_name>', field_name)
    field_str = field_str.replace('<region_id>', region_id)
    field_str = field_str.replace('<dimensions>', dimension_lookup_dict[field_name])
    field_str = field_str.replace('<internal_field>', internal_field_lookup_dict[field_name])
    field_str = field_str.replace('<class>', class_lookup_dict[field_name])

    os.makedirs(os.path.join(case_dir, '0', region_id), exist_ok=True)
    full_filename = os.path.join(case_dir, '0', region_id, field_name)
    with open(full_filename, "w") as f:
        f.write(field_str)

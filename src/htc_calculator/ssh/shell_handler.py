import paramiko
import re
import os
from ..logger import logger
from multiprocessing import cpu_count
from math import floor
from shutil import copyfile
from time import sleep

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
            shin, shout, sherr = self.execute('pwd')
            sleep(0.1)
            pwd = shout[-1]
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
            self.execute(f'cd {pwd}')

        if sherr:
            logger.error(f'Error executing command: {cmd}:\n' + ''.join(sherr))

        return shin, shout, sherr

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
                        f"{output}")

        return True

    def run_shm(self, workdir, parallel=False, num_worker=None):
        cmd = 'snappyHexMesh'
        if parallel:
            cmd += ' -parallel'
            if num_worker is None:
                num_worker = floor(cpu_count() / 2 + 1)

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

            shin, shout, sherr = self.execute(f'paraFoam -touchAll', cwd=workdir)

            print('done')

        else:
            shin, shout, sherr = self.execute(cmd, cwd=workdir)
            if sherr:
                logger.error(f"Error running snappyHexMesh: \n {''.join(sherr)}")
                raise Exception(f"Error running snappyHexMesh:  \n {''.join(sherr)}")
            else:
                output = ''.join(shout)
                if output.find('FOAM FATAL ERROR') != -1:
                    logger.error(f'Error running snappyHexMesh:\n\n{output}')
                    raise Exception(f"Error running snappyHexMesh:  \n {''.join(sherr)}")
                logger.info(f"Successfully created snappyHexMesh:\n"
                            f"Directory: {workdir}\n\n "
                            f"{output[output.find('Mesh Information'):output.find('End')]}")

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

        return shin, shout, sherr


sh = ShellHandler(host, ssh_user, ssh_pwd)

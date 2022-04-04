from ..logger import logger
import subprocess


def run_parafoam_touch_all(case_dir):
    logger.info(f'Running paraFoam -touchAll')
    res = subprocess.run(
        ["/bin/bash", "-i", "-c", "paraFoam -touchAll"],
        capture_output=True,
        cwd=case_dir,
        user='root')
    if res.returncode == 0:
        output = res.stdout.decode('ascii')
        if output.find('FOAM FATAL ERROR') != -1:
            logger.error(f'Error paraFoam touch all:\n\n{output}')
            raise Exception(f'Error paraFoam touch all:\n\n{output}')
        logger.info(f"Successfully ran paraFoam touch all \n\n{output}")
    else:
        logger.error(f"{res.stderr.decode('ascii')}")

    return True

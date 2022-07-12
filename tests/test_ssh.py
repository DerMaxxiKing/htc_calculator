from src.htc_calculator.ssh import shell_handler

shin, shout, sherr = shell_handler.execute('ls', cwd='/simulations')
shin, shout, sherr = shell_handler.execute('snappyHexMesh', cwd='/simulations')
# f_e = shell_handler.file_exists('/tmp/test.xlsx')
# f_e = shell_handler.file_exists('/tmp/requirements.txt')
# f_e = shell_handler.file_exists('/tmp/requirements-2.txt')
shin, shout, sherr = shell_handler.execute('ls', cwd='/tmp')
# f_e

# print('done')

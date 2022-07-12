import sys
import os

# print('Importing FreeCAD and Modules')
# sys.path.append('/usr/lib/freecad/lib')

# print('Importing FreeCAD and Modules')
#
#
# sys.path.append('/tmp/squashfs-root/usr/lib/python38.zip')
# sys.path.append('/tmp/squashfs-root/usr/lib/python3.8')
# sys.path.append('/tmp/squashfs-root/usr/lib/python3.8/lib-dynload')
# sys.path.append('/tmp/squashfs-root/usr/lib/python3.8/site-packages')
# sys.path.append('/tmp/squashfs-root/usr/lib/')
# sys.path.append('mp/squashfs-root/usr/lib/python3.8/lib-dynload')
# sys.path.append('/tmp/squashfs-root/usr/lib/python3.8/site-packages')
# sys.path.append('mp/squashfs-root/usr/Ext')
# sys.path.append('mp/squashfs-root/usr/lib')

use_ssh = True
ssh_pwd = 'docker'
ssh_user = 'root'
host = '172.20.0.5'

work_dir = '/simulations'
os.makedirs(work_dir, exist_ok=True)
n_proc = 8

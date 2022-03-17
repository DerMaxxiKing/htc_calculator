try:
    import FreeCAD
except ModuleNotFoundError:
    import sys
    sys.path.append('/tmp/squashfs-root/usr/lib/python3.8/site-packages/')
    sys.path.append('/tmp/squashfs-root/usr/lib/')
    import FreeCAD
    import Draft

try:
    import FreeCAD
except ModuleNotFoundError:
    import sys

    sys.path.append('/app/squashfs-root/usr/lib/python3.9/site-packages/')
    sys.path.append('/app/squashfs-root/usr/lib/')
    sys.path.append('/usr/lib/python3.9/')
    sys.path.append('/usr/lib/python3/dist-packages')
    import FreeCAD
    import Draft

from .activated_reference_face import ActivatedReferenceFace
from .construction import Solid, Layer, ComponentConstruction
from .buildin_materials import water, aluminium
from .meshing.buildin_pipe_sections.tube_with_wall_optimized import pipe_section
from .case.case import OFCase, TabsBC

import os
import sys
import uuid
import tempfile
from .face import Face
from .solid import Solid
from .assembly import Assembly
from .tools import generate_solid_from_faces

import FreeCAD
import Part as FCPart
from FreeCAD import Base

App = FreeCAD


class ReferenceFace(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)

        self.reference_faces = kwargs.get('reference_faces', None)

    def gererate_volume(self):

        generate_solid_from_faces

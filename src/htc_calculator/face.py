import os
import sys
import uuid
import tempfile
import time
import re
from io import StringIO
import numpy as np

from .logger import logger
from .meshing.surface_mesh_parameters import SurfaceMeshParameters, default_surface_mesh_parameter

import FreeCAD
import Part as FCPart
import Mesh
import MeshPart


class Face(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        logger.debug(f'initializing face {self.id}')
        start_time = time.time()

        self.name = kwargs.get('name', None)
        self._normal = kwargs.get('normal', None)

        self.fc_face = kwargs.get('fc_face', None)

        logger.debug(f'    finished initializing face {self.id} in {time.time() - start_time} s')

        self._surface_mesh_setup = kwargs.get('_surface_mesh_setup',
                                              kwargs.get('surface_mesh_setup', default_surface_mesh_parameter))

        self.linear_deflection = kwargs.get('linear_deflection', 0.5)
        self.angular_deflection = kwargs.get('angular_deflection', 0.5)

    @property
    def area(self):
        return self.fc_face.Area

    @property
    def stl(self):
        return self.create_stl_str()

    @property
    def txt_id(self):
        return re.sub('\W+','', 'a' + str(self.id))

    @property
    def normal(self):
        if self._normal is None:
            vec = self.get_normal(self.fc_face.Vertexes[0].Point)
            self._normal = np.array([vec.x, vec.y, vec.z])
        return self._normal

    @normal.setter
    def normal(self, value):
        if value == self._normal:
            return
        self._normal = value

    @property
    def surface_mesh_setup(self):
        return self._surface_mesh_setup

    @surface_mesh_setup.setter
    def surface_mesh_setup(self, value):
        if value == self._surface_mesh_setup:
            return
        self._surface_mesh_setup = value

    def get_normal(self, point):
        uv = self.fc_face.Surface.parameter(point)
        nv = self.fc_face.normalAt(*uv)
        return nv.normalize()

    def create_stl_str(self, of=False):

        logger.debug(f'creating stl string for face {self.id}...')
        # msh = MeshPart.meshFromShape(Shape=self.fc_face, MaxLength=1)

        if of:
            face_name = self.txt_id
        else:
            face_name = str(self.id)

        try:
            fd, tmp_file = tempfile.mkstemp(suffix='.ast')
            os.close(fd)

            mesh_shp = MeshPart.meshFromShape(self.fc_face,
                                              LinearDeflection=self.linear_deflection,
                                              AngularDeflection=self.angular_deflection)
            mesh_shp.write(tmp_file)

            with open(tmp_file, encoding="utf8") as file:
                content = file.readlines()
                content[0] = f'solid {face_name}\n'
                content[-1] = f'endsolid {face_name}\n'
        except Exception as e:
            logger.error(f'error while creating stl string for face {self.id}:\n{e}')
        finally:
            try:
                os.remove(tmp_file)
            except Exception as e:
                print(e)

        logger.debug(f'    created stl string for face {self.id}...')

        return ''.join(content)

    def export_stl(self, filename):

        logger.debug(f'exporting stl for face {self.id}...')

        try:
            new_file = open(filename, 'w')
            new_file.writelines(self.stl)
            new_file.close()
            logger.debug(f'    finished stl export for face {self.id}')
        except Exception as e:
            logger.error(f'error while exporting .stl for face {self.id}:\n{e}')

    def export_step(self, filename):

        logger.debug(f'exporting .step for face {self.id}...')

        try:
            from .tools import name_step_faces

            doc = FreeCAD.newDocument()

            fd, tmp_file = tempfile.mkstemp(suffix='.stp')
            os.close(fd)

            __o__ = doc.addObject("Part::Feature", f'{str(self.id)}')
            __o__.Label = f'{str(self.id)}'
            __o__.Shape = self.fc_face

            FCPart.export(doc.Objects, tmp_file)

            # rename_faces:
            names = [str(self.id)]
            names_dict = {k: v for k, v in zip(range(names.__len__()), names)}
            name_step_faces(fname=tmp_file, name=names_dict, new_fname=filename)

            logger.debug(f'    finished .step export for face {self.id}')

        except Exception as e:
            logger.error(f'error while exporting .stp for face {self.id}:\n{e}')
        finally:
            try:
                os.remove(tmp_file)
            except FileNotFoundError as e:
                pass

    def __repr__(self):
        rep = f'Face {self.name} {self.id} {self.area}'
        return rep

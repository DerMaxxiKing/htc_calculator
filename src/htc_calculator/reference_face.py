import os
import sys
import uuid

from . import config as app_config
from .face import Face
from .solid import Solid
from .assembly import Assembly
from .tools import generate_solid_from_faces
from .meshing.snappy_hex_mesh import SnappyHexMesh
import time
from copy import copy
import subprocess
import tempfile

from shutil import copyfile
from .logger import logger

import FreeCAD
import Part as FCPart
from FreeCAD import Base

import ObjectsFem
from femmesh.gmshtools import GmshTools as gt
# import MeshPart
# import Mesh
# import Part, Fem, ObjectsFem, femmesh.gmshtools

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from .meshing import meshing_resources as msh_resources

from .meshing.mesh_face import create_temp_case_dir, create_block_mesh_dict, create_snappy_hex_mesh_dict
from .meshing.surface_mesh_parameters import SurfaceMeshParameters, default_surface_mesh_parameter
from .meshing.layer_definition import LayerDefinition, default_layer_definition

App = FreeCAD
doc = App.newDocument("HTCCalculator")


class ReferenceFace(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        logger.debug(f'initializing ReferenceFace {self.id}...')
        self.name = kwargs.get('name', None)
        self._normal = kwargs.get('normal', None)

        self.vertices = kwargs.get('vertices', None)                     # positions (x,y,z) of vertices as numpy array; shape: n x 3
        self._points = kwargs.get('points', None)                         # list of fc vectors to point positions
        self._reference_wire = kwargs.get('reference_wire', None)         # fc wire of the reference face
        self._reference_face = kwargs.get('reference_face', None)         # fc reference face
        self.layer_dir = kwargs.get('layer_dir', 1)                      # 1 if in normal direction, -1 if negative normal direction

        self.component_construction = kwargs.get('component_construction', None)

        self.mesh_setup = kwargs.get('mesh_setup', None)

        # self.geo = kwargs.get('geo', None)         # fc reference face

        self.side_1_face = None
        self.side_2_face = None
        self.side_faces = None
        self.interface_faces = None

        self._assembly = None

        self.plane_mesh_size = kwargs.get('plane_mesh_size', 250)
        self.thickness_mesh_size = kwargs.get('plane_mesh_size', 25)

        logger.debug(f'    initialized ReferenceFace {self.id}')

    @property
    def points(self):
        if self._points is None:
            self.points = [Base.Vector(row) for row in self.vertices]
        return self._points

    @points.setter
    def points(self, value):
        self._points = value

    @property
    def reference_wire(self):
        if self._reference_wire is None:
            self.reference_wire = FCPart.makePolygon([*self.points, self.points[0]])
        return self._reference_wire

    @reference_wire.setter
    def reference_wire(self, value):
        self._reference_wire = value

    @property
    def reference_face(self):
        if self._reference_face is None:
            self.generate_reference_geometry()
        return self._reference_face

    @reference_face.setter
    def reference_face(self, value):
        self._reference_face = value

    @property
    def normal(self):
        if self._normal is None:
            self._normal = self.get_normal(self.points[0])
        return self._normal

    @normal.setter
    def normal(self, value):
        self._normal = value

    @property
    def vertex_normals(self):
        if self.reference_face is not None:
            return self.get_vertex_normals()

    @property
    def assembly(self):
        if self._assembly is None:
            self.assembly = self.generate_3d_geometry()
        return self._assembly

    @assembly.setter
    def assembly(self, value):
        self._assembly = value

    def generate_reference_geometry(self):

        # self.points = [Base.Vector(row) for row in self.vertices]
        # self.reference_wire = FCPart.makePolygon([*self.points, self.points[0]])

        self.reference_face = FCPart.Face(self.reference_wire)

    def get_normal(self, point):
        uv = self.reference_face.Surface.parameter(point)
        nv = self.reference_face.normalAt(*uv)
        return nv.normalize()

    def get_vertex_normals(self):

        surf = self.reference_face.Surface
        vertex_normals = []
        for point in self.points:
            uv = surf.parameter(point)
            nv = self.reference_face.normalAt(*uv)
            vertex_normals.append(nv.normalize())

        return vertex_normals

    def generate_3d_geometry(self):
        """
        Generate 3D geometry of the face

        How to make a face offset:
        https://www.freecadweb.org/api/d8/ded/classPart_1_1TopoShape.html#acc6ade4d17d79be670c157283d2e288d
        TopoDS_Shape TopoShape::makeOffsetShape 	(	double 	offset,
                                                        double 	tol,
                                                        bool 	intersection = false,
                                                        bool 	selfInter = false,
                                                        short 	offsetMode = 0,
                                                        short 	join = 0,
                                                        bool 	fill = false
                                                        )		 const

        """

        topology = {}
        faces = []
        surface_mesh_setup_with_layers = copy(default_surface_mesh_parameter)
        surface_mesh_setup_with_layers.add_layers = True
        surface_mesh_setup_with_layers.layer_definition = default_layer_definition

        logger.debug(f'gererating 3D geometry for ReferenceFace {self.id}')
        start_time = time.time()

        layers = self.component_construction.layers

        # translate reference face to layer 0 origin:
        offset0 = - self.component_construction.side_1_offset * self.layer_dir
        if offset0 != 0:
            layer_base_face = self.reference_face.makeOffsetShape(offset0, 1e-6, False, False, 0, 0, False)
        else:
            layer_base_face = self.reference_face.copy()

        # start with the layer base face:
        cur_face = Face(fc_face=layer_base_face,
                        name=f'{layers[0].name} side 1 face')
        faces.append(cur_face)

        layer_base_faces = [None] * layers.__len__()
        layer_side_faces = [None] * layers.__len__()
        layer_top_faces = [None] * layers.__len__()

        layer_solids = {}

        interfaces = []
        layer_interfaces = {}

        self.side_faces = []

        for i, layer in enumerate(layers):
            layer_base_faces[i] = cur_face
            # make a offset face from the base face:
            layer_top_face = Face(fc_face=cur_face.fc_face.makeOffsetShape(layer.thickness * self.layer_dir, 1e-6, False, False, 0, 0, False),
                                  name=f'Layer {i} {layer.name} side 1 face')
            layer_top_faces[i] = layer_top_face
            faces.append(layer_top_faces[i])
            # make a loft for the side faces:
            layer_side_faces[i] = Face(fc_face=FCPart.makeLoft([layer_base_faces[i].fc_face.Wires[0],
                                                                layer_top_faces[i].fc_face.Wires[0]
                                                                ],
                                                               False,
                                                               True,
                                                               False),
                                       name=f'Layer {i} {layer.name} side faces')
            faces.append(layer_side_faces[i])

            self.side_faces.append(layer_side_faces[i])

            cur_face = layer_top_face

            if i == 0:
                layer_base_faces[i].surface_mesh_setup = surface_mesh_setup_with_layers
                layer_interfaces[layer.id] = [layer_top_faces[i]]
                interfaces.append(layer_top_faces[i])

                if layers.__len__() == 1:
                    topology[layer_top_faces[i]] = [layers[i]]
                else:
                    topology[layer_top_faces[i]] = [layers[i], layers[i+1]]

            elif i == layers.__len__() - 1:
                layer_interfaces[layer.id] = [layer_base_faces[i]]
                layer_top_faces[i].surface_mesh_setup = surface_mesh_setup_with_layers

            else:
                layer_interfaces[layer.id] = [layer_base_faces[i], layer_top_faces[i]]
                interfaces.append(layer_top_faces[i])

                topology[layer_top_faces[i]] = [layers[i], layers[i+1]]

            # assemble layer solid: first face is bottom, second face is top, other faces are side faces

            layer_solid = Solid(faces=[layer_base_faces[i], layer_top_faces[i], layer_side_faces[i]],
                                name=f'Layer {layer.name} solid',
                                interfaces=layer_interfaces[layer.id],
                                layer=layer,
                                type='Layer')
            layer_solid.features['base_faces'] = [layer_base_faces[i]]
            layer_solid.features['top_faces'] = [layer_top_faces[i]]
            layer_solid.features['side_faces'] = [layer_side_faces[i]]
            layer_solids[layer.id] = layer_solid

            layer.solid = layer_solids[layer.id]

            # layer_solids[layer.id] = generate_solid_from_faces(faces=[layer_base_faces[i].fc_face,
            #                                                           layer_top_faces[i].fc_face,
            #                                                           layer_side_faces[i].fc_face
            #                                                           ],
            #                                                    solid_id=str(layer.id))
        # self.geo = layer_solids

        # layer_solids = []
        # for layer in layers:
        #     layer_faces = [None] * self.geo[layer.id].Shape.Faces.__len__()
        #     for i, face in enumerate(self.geo[layer.id].Shape.Faces):
        #         layer_faces[i] = Face(fc_face=face)
        #
        #     new_solid = Solid(faces=layer_faces,
        #                       name=layer.name,
        #                       interfaces=layer_interfaces[layer.id])
        #
        #     layer_solids.append(new_solid)

        assembly = Assembly(solids=list(layer_solids.values()),
                            interfaces=interfaces,
                            faces=faces,
                            topology=topology,
                            reference_face=self)

        assembly.features['side_1_face'] = assembly.solids[0].faces[0]
        assembly.features['side_2_face'] = assembly.solids[-1].faces[1]
        assembly.features['layer_solids'] = list(layer_solids.values())

        self.side_1_face = assembly.solids[0].faces[0]
        self.side_2_face = assembly.solids[-1].faces[1]

        logger.debug(f'    finished 3D geometry generation for ReferenceFace {self.id} in {time.time() - start_time} s')

        return assembly

    def export_step(self, filename):

        logger.debug(f'exporting step for {self.id}')
        self.assembly.export_stp(filename)

    def export_stl(self, directory):

        logger.info(f'Exporting stls to {directory}')

        faces = {}

        for solid in self.assembly.solids:
            for face in solid.faces:
                faces[face.id] = face

        os.makedirs(directory, exist_ok=True)

        for key, face in faces.items():
            filename = os.path.join(directory, str(face.id) + '.stl')
            logger.debug(f'exporting stl for face {face.name}, {face.id} to {filename}')
            face.export_stl(filename)

        logger.info(f'Successfully exported stls to {directory}')


        # logger.debug(f'exporting stl for {self.id}')
        # self.assembly.export_stl(filename=filename)

    def generate_shm_mesh(self):

        if app_config.work_dir in ['temp', None]:
            case_dir = create_temp_case_dir()
        else:
            case_dir = create_temp_case_dir(directory=app_config.work_dir)

        snhm = SnappyHexMesh(assembly=self.assembly,
                             case_dir=case_dir)
        snhm.write_snappy_hex_mesh()

        # write decomposeParDict
        dec_par_template = pkg_resources.read_text(msh_resources, 'decompose_par_dict')
        dec_par_template = dec_par_template.replace('<n_procs>', str(app_config.n_proc))

        dst = os.path.join(case_dir, 'system', 'decomposeParDict')
        with open(dst, 'w') as shmd:
            shmd.write(dec_par_template)

        # write control dict:
        path = 'controlDict'
        with pkg_resources.path(msh_resources, path) as path:
            source = path.__str__()
        dst = os.path.join(case_dir, 'system', 'controlDict')
        copyfile(source, dst)

        # write fvSchemes:
        path = 'fvSchemes'
        with pkg_resources.path(msh_resources, path) as path:
            source = path.__str__()
        dst = os.path.join(case_dir, 'system', 'fvSchemes')
        copyfile(source, dst)

        # write fvSolution:
        path = 'fvSolution'
        with pkg_resources.path(msh_resources, path) as path:
            source = path.__str__()
        dst = os.path.join(case_dir, 'system', 'fvSolution')
        copyfile(source, dst)

        # write createBafflesDict
        dst = os.path.join(case_dir, 'system', 'createBafflesDict')
        with open(dst, 'w') as cbd:
            cbd.write(self.assembly.baffles_dict)

        # write all stl files
        stls_path = os.path.join(case_dir, 'constant', 'triSurface')
        os.mkdir(stls_path)
        self.assembly.write_of_geo(stls_path)

        # write surfaceFeaturesDict
        self.assembly.write_surface_feature_dict(os.path.join(case_dir, 'system'))

        # write regions_dict
        dst = os.path.join(case_dir, 'constant', 'regionProperties')
        with open(dst, 'w') as cbd:
            cbd.write(self.assembly.regions_dict)

        # create block mesh dict
        # create_block_mesh_dict(self, case_dir, 0.5)
        planar = self.reference_face.Surface.isPlanar()

        self.assembly.write_block_mesh(case_dir, cell_size=0.5, planar=planar)

        # copy allrun script
        path = 'Allrun.bash'
        with pkg_resources.path(msh_resources, path) as path:
            source = path.__str__()
        dst = os.path.join(case_dir, 'Allrun')
        copyfile(source, dst)
        # make executable:
        command = f"chmod +x {os.path.join(case_dir, 'Allrun')}"
        ret = subprocess.run(command, capture_output=True, shell=True, executable='/bin/bash', cwd=case_dir)
        print(f'case_dir: {case_dir}')
        print(f'out: {ret.stdout.decode()}')
        print(f'err: {ret.stderr.decode()}')

        # copy allclean script
        path = 'Allclean.bash'
        with pkg_resources.path(msh_resources, path) as path:
            source = path.__str__()
        dst = os.path.join(case_dir, 'Allclean')
        copyfile(source, dst)
        # make executable:
        command = f"chmod +x {os.path.join(case_dir, 'Allclean')}"
        ret = subprocess.run(command, capture_output=True, shell=True, executable='/bin/bash', cwd=case_dir)
        print(f'case_dir: {case_dir}')
        print(f'out: {ret.stdout.decode()}')
        print(f'err: {ret.stderr.decode()}')

        # run Allrun
        command = f"source /opt/openfoam8/etc/bashrc; cd {case_dir}; ./Allrun"
        ret = subprocess.run(command, capture_output=True, shell=True, executable='/bin/bash', cwd=case_dir)
        print(f'case_dir: {case_dir}')
        print(f'out: {ret.stdout.decode()}')
        print(f'err: {ret.stderr.decode()}')
        #
        print("done.")

        print('done')

    def save_fcstd(self, filename):
        """
        save as freecad document
        :param filename: full filename; example: '/tmp/test.FCStd'
        """
        doc = App.newDocument("MeshTest")
        __o__ = doc.addObject("Part::Feature", f'Reference Face {self.name} {self.id}')
        __o__.Shape = self.assembly.comp_solid
        doc.recompute()
        doc.saveCopy(filename)

    def generate_mesh_gmsh(self):

        # https://github.com/FreeCAD/FreeCAD/blob/master/src/Mod/Fem/femmesh/gmshtools.py
        # https://wiki.freecadweb.org/FEM_Tutorial_Python
        # https://wiki.freecadweb.org/FEM_Tutorial_Python#FEM_mesh_.28gmsh.29

        doc = App.newDocument("gmsh")
        __o__ = doc.addObject("Part::Feature", f'{self.id}')
        __o__.Label = f'{self.id}'
        __o__.Shape = self.assembly.comp_solid
        mesh = doc.addObject('Fem::FemMeshShapeNetgenObject', 'FEMMeshNetgen')
        mesh.Shape = __o__
        mesh.MaxSize = 1000
        mesh.Fineness = "Moderate"
        mesh.Optimize = True
        mesh.SecondOrder = True
        doc.recompute()



        doc.FileName = 'test.fcstd'
        doc.save()

        femmesh_obj = ObjectsFem.makeMeshGmsh(doc, box_obj.Name + "_Mesh")
        femmesh_obj.Part = doc.Box
        doc.recompute()

        gmsh_mesh = gt(femmesh_obj)
        error = gmsh_mesh.create_mesh()

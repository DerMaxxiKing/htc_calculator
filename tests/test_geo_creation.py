import meshio
import numpy as np
from src.htc_calculator.reference_face import ReferenceFace
from src.htc_calculator.activated_reference_face import ActivatedReferenceFace
from src.htc_calculator.construction import Material, Layer, ComponentConstruction
from src.htc_calculator.meshing.mesh_setup import MeshSetup
from src.htc_calculator.activated_reference_face import test_mesh_creation

vertices = np.array([[0, 0, 0],
                     [5, 0, 0],
                     [5, 5, 0],
                     [0, 5, 0]])

concrete = Material(name='concrete',
                    density=2600,
                    specific_heat_capacity=1000,
                    heat_conductivity=2.5)

rockwool = Material(name='rockwool',
                    density=250,
                    specific_heat_capacity=840,
                    heat_conductivity=0.034)

plaster = Material(name='plaster',
                   density=1500,
                   specific_heat_capacity=960,
                   heat_conductivity=0.60)

layer0 = Layer(name='layer0_plaster', material=plaster, thickness=0.02)
layer1 = Layer(name='layer1_concrete', material=concrete, thickness=0.3)
layer2 = Layer(name='layer2_rockwool', material=rockwool, thickness=0.3)
layer3 = Layer(name='layer3_plaster',  material=plaster, thickness=0.02)

test_construction = ComponentConstruction(name='test_construction',
                                          layers=[layer1, layer2],
                                          side_1_offset=0.00)

mesh_setup = MeshSetup(name='simple_mesh_setup')

face = ReferenceFace(vertices=vertices,
                     component_construction=test_construction,
                     mesh_setup=mesh_setup)


face.generate_reference_geometry()
face.generate_3d_geometry()
face.save_fcstd('/tmp/test.FCStd')

face.export_step('/tmp/simple_test_geo.stp')
# face.export_stl('TEST_MIT_Tomas.stl')
# face.generate_mesh()
#
# print(face)
#
#
tabs1 = ActivatedReferenceFace(vertices=vertices,
                               component_construction=test_construction,
                               start_edge=0,
                               tube_diameter=0.1,
                               tube_distance=0.25,
                               tube_edge_distance=0.3,
                               bending_radius=0.1,
                               tube_side_1_offset=0.15,
                               name='test1')
#
tabs1.generate_reference_geometry()
tabs1.generate_3d_geometry()
# tabs1.export_step('tabs_layers.stp')
tabs1.generate_tube_spline()
tabs1.generate_solid_pipe()
# tabs1.export_solid_pipe('solid_pipe_test.stp')
tabs1.generate_hole_part()

tabs1.save_fcstd('/tmp/activated_test.FCStd')
tabs1.export_step('/tmp/simple_test_geo_activated.stp')
tabs1.export_solid_pipe('/tmp/solid_pipe.stp')
tabs1.export_solids('/tmp/solids.FCStd')

tabs1.generate_mesh()

tabs1.export_step('finished.stp')

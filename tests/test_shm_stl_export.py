import numpy as np
from src.htc_calculator.reference_face import ReferenceFace
from src.htc_calculator.activated_reference_face import ActivatedReferenceFace
from src.htc_calculator.construction import Material, Layer, ComponentConstruction
from src.htc_calculator.meshing.mesh_setup import MeshSetup
from src.htc_calculator.meshing.buildin_pipe_sections.tube_with_wall_optimized import pipe_section


# vertices = np.array([[0, 0, 0],
#                      [5000, 0, 0],
#                      [5000, 5000, 0],
#                      [2500, 5000, 0],
#                      [2500, 2500, 0],
#                      [0, 2500, 0]])

vertices = np.array([[0, 0, 0],
                     [5000, 0, 0],
                     [5000, 5000, 0],
                     [0, 5000, 0]])

tube_material = Material(name='Tube Material',
                         density=1800,
                         specific_heat_capacity=700,
                         heat_conductivity=0.4,
                         roughness=0.0004)

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

layer0 = Layer(name='layer0_plaster', material=plaster, thickness=20)
layer1 = Layer(name='layer1_concrete', material=concrete, thickness=200)
layer2 = Layer(name='layer2_rockwool', material=rockwool, thickness=100)
layer3 = Layer(name='layer3_plaster',  material=plaster, thickness=20)

test_construction = ComponentConstruction(name='test_construction',
                                          layers=[layer1, layer2],
                                          side_1_offset=0.00)
#
# mesh_setup = MeshSetup(name='simple_mesh_setup')

# face = ReferenceFace(vertices=vertices,
#                      component_construction=test_construction,
#                      mesh_setup=mesh_setup)
#
#
# face.generate_reference_geometry()
# face.generate_3d_geometry()
# face.save_fcstd('/tmp/test.FCStd')

# face.export_step('/tmp/simple_test_geo.stp')
# face.export_stl('TEST_MIT_Tomas.stl')
# face.generate_mesh()
#
# print(face)
#
#
tabs1 = ActivatedReferenceFace(vertices=vertices,
                               component_construction=test_construction,
                               start_edge=0,
                               pipe_section=pipe_section,
                               tube_diameter=20,
                               tube_inner_diameter=16,
                               tube_material=tube_material,
                               tube_distance=225,
                               tube_edge_distance=300,
                               bending_radius=100,
                               tube_side_1_offset=100,
                               name='test1')
#
# tabs1.generate_reference_geometry()
tabs1.export_stl('/tmp/test_stls')
tabs1.save_fcstd('/tmp/assembly.FCStd')
# tabs1.create_o_grid()
tabs1.create_free_blocks()
tabs1.generate_block_mesh_dict()
# tabs1.generate_shm_mesh()

# tabs1.generate_mesh()

print('done')

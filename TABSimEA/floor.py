import numpy as np
from src.htc_calculator.activated_reference_face import ActivatedReferenceFace
from src.htc_calculator.construction import Solid, Layer, ComponentConstruction
from src.htc_calculator.buildin_materials import water, aluminium
from src.htc_calculator.meshing.buildin_pipe_sections.tube_with_wall_optimized import pipe_section
from src.htc_calculator import OFCase, TabsBC
from src.htc_calculator import config

from src.htc_calculator.logger import logger
logger.setLevel('INFO')

config.n_proc = 8

pipe_section.materials = [water, aluminium]
pipe_section.cell_size = [None, None, 75]
pipe_section.n_cell = [5, 5, None]

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

tube_material = Solid(name='Tube Material',
                      density=1800,
                      specific_heat_capacity=700,
                      heat_conductivity=0.4,
                      roughness=0.0004)

parket = Solid(name='Parkett',
               density=675,
               specific_heat_capacity=1600,
               heat_conductivity=0.16)

estrich = Solid(name='Estrich',
                density=1800,
                specific_heat_capacity=1080,
                heat_conductivity=1.1)

trittschalldaemmung = Solid(name='Trittschalldaemmung',
                            density=11,
                            specific_heat_capacity=1450,
                            heat_conductivity=0.033)

schuettung = Solid(name='Schuettung',
                            density=150,
                            specific_heat_capacity=1250,
                            heat_conductivity=0.075)


stahlbeton = Solid(name='Stahlbeton',
                   density=2300,
                   specific_heat_capacity=1000,
                   heat_conductivity=2.3)

xps = Solid(name='xps',
                 density=32,
                 specific_heat_capacity=1450,
                 heat_conductivity=0.042)

normalputzmoertel = Solid(name='Normalputzm??rtel',
                          density=1300,
                          specific_heat_capacity=1000,
                          heat_conductivity=0.49)

layer0 = Layer(name='layer0_xps', material=xps, thickness=136.5)
layer1 = Layer(name='layer1_stahlbeton', material=stahlbeton, thickness=200)
layer2 = Layer(name='layer2_schuettung', material=schuettung, thickness=50)
layer3 = Layer(name='layer3_trittschalldaemmung', material=trittschalldaemmung, thickness=30)
layer4 = Layer(name='layer4_estrich', material=estrich, thickness=60)
layer5 = Layer(name='layer5_parket', material=parket, thickness=20)

test_construction = ComponentConstruction(name='test_construction',
                                          layers=[layer0, layer1, layer2, layer3, layer4, layer5],
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
                               tube_distance=100,
                               tube_edge_distance=300,
                               bending_radius=50,
                               tube_side_1_offset=100,
                               default_mesh_size=50,
                               default_arc_cell_size=10,
                               name='test1')


my_bc = TabsBC(inlet_volume_flow=4.1666e-5,
               inlet_temperature=323.15,
               top_ambient_temperature=293.15,
               bottom_ambient_temperature=293.15,
               top_htc=10,
               bottom_htc=5.8884,
               initial_temperature=293.15,
               initial_temperatures={water: 323.15,
                                     aluminium: 293.16,
                                     stahlbeton: 293.16,
                                     xps: 293.16,
                                     normalputzmoertel: 293.16})

case = OFCase(reference_face=tabs1,
              bc=my_bc,
              n_proc=12)

tabs1.case = case

case.run_with_separate_meshes2()

#
# tabs1.generate_reference_geometry()
# tabs1.export_stl('/tmp/test_stls')
# tabs1.save_fcstd('/tmp/assembly.FCStd')
# tabs1.create_o_grid()
# tabs1.create_free_blocks()
# tabs1.extrude_pipe_layer()
# tabs1.update_cell_zone()
# tabs1.generate_block_mesh_dict()
# tabs1.generate_shm_mesh()

# tabs1.run_case()

# tabs1.generate_mesh()

print('done')

import os
import numpy as np
from src.htc_calculator.reference_face import ReferenceFace
from src.htc_calculator.construction import Solid, Layer, ComponentConstruction


vertices = np.array([[0, 0, 0],
                     [5000, 0, 0],
                     [5000, 5000, 0],
                     [2500, 6000, 0],
                     [0, 5000, 0]])

concrete = Solid(name='concrete',
                 density=2600,
                 specific_heat_capacity=1000,
                 heat_conductivity=2.5)

rockwool = Solid(name='rockwool',
                 density=250,
                 specific_heat_capacity=840,
                 heat_conductivity=0.034)

plaster = Solid(name='Normalputzmörtel',
                density=1300,
                specific_heat_capacity=960,
                heat_conductivity=0.60)

layer0 = Layer(name='layer0_plaster', material=plaster, thickness=20)
layer1 = Layer(name='layer1_concrete', material=concrete, thickness=200)
layer2 = Layer(name='layer2_rockwool', material=rockwool, thickness=100)
layer3 = Layer(name='layer3_plaster',  material=plaster, thickness=20)

test_construction = ComponentConstruction(name='test_construction',
                                          layers=[layer1],
                                          side_1_offset=0.00)

ref_face = ReferenceFace(vertices=vertices,
                         component_construction=test_construction,)

solid_0 = ref_face.assembly.solids[0]

mesh_path = '/tmp/solid_0'
os.mkdir(mesh_path)
os.mkdir(os.path.join(mesh_path, 'constant'))
os.mkdir(os.path.join(mesh_path, 'system'))

# solid_0.create_base_block_mesh()
solid_0.create_shm_mesh(normal=ref_face.normal)

print('done')

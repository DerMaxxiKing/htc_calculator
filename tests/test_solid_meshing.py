import os
import numpy as np
from src.htc_calculator.config import work_dir
from src.htc_calculator.reference_face import ReferenceFace
from src.htc_calculator.activated_reference_face import ActivatedReferenceFace
from src.htc_calculator.construction import Solid, Layer, ComponentConstruction
from src.htc_calculator.buildin_materials import water, aluminium
from src.htc_calculator.meshing.buildin_pipe_sections.tube_with_wall_optimized import pipe_section


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
                                          layers=[layer1, layer2],
                                          side_1_offset=0.00)

ref_face = ReferenceFace(vertices=vertices,
                         component_construction=test_construction,)

solid_0 = ref_face.assembly.solids[0]

mesh_path = f'{work_dir}/solid_0'
os.mkdir(mesh_path)
os.mkdir(os.path.join(mesh_path, 'constant'))
os.mkdir(os.path.join(mesh_path, 'system'))

# solid_0.create_base_block_mesh()
# solid_0.create_shm_mesh(normal=ref_face.normal)

print('done')

tube_material = Solid(name='Tube Material',
                      density=1800,
                      specific_heat_capacity=700,
                      heat_conductivity=0.4,
                      roughness=0.0004)

tabs1 = ActivatedReferenceFace(vertices=vertices,
                               component_construction=test_construction,
                               start_edge=0,
                               pipe_section=pipe_section,
                               tube_diameter=20,
                               tube_inner_diameter=16,
                               tube_material=tube_material,
                               tube_distance=225,
                               tube_edge_distance=300,
                               bending_radius=50,
                               tube_side_1_offset=100,
                               default_mesh_size=50,
                               default_arc_cell_size=10,
                               name='test1')

solid_0 = tabs1.assembly.solids[1]

for solid in tabs1.assembly.solids:
    if 'pipe_faces' in solid.features.keys():
        solid.surface_mesh_setup.max_refinement_level = 4
        solid.surface_mesh_setup.min_refinement_level = 1
        for face in solid.faces:
            face.surface_mesh_setup.max_refinement_level = 4
        solid.features['pipe_faces'].surface_mesh_setup.max_refinement_level = 4
        solid.features['pipe_faces'].surface_mesh_setup.min_refinement_level = 4
    os.mkdir(f'{work_dir}/{solid.txt_id}')
    solid.save_fcstd(f'{work_dir}/{solid.txt_id}/{solid.txt_id}.FCStd')
    solid.create_shm_mesh(normal=tabs1.normal, parallel=False, block_mesh_size=200)

    print('done')


print('done')

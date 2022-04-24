import numpy as np
import os
from src.htc_calculator.meshing.block_mesh import BlockMeshVertex, Block, BlockMesh, Mesh
from src.htc_calculator.case.utils import run_parafoam_touch_all
from src.htc_calculator.case.case import CreatePatchDict


mesh1 = Mesh(name='mesh 1')
mesh2 = Mesh(name='mesh 2')

mesh1.activate()

vertex_positions = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [1, 1, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [1, 0, 1],
                             [1, 1, 1],
                             [0, 1, 1],    # 7
                             [2, 0, 0],    # 8
                             [2, 1, 0],    # 9
                             [2, 0, 1],    # 10
                             [2, 1, 1],    # 11
                             [0, 0, 2],    # 12
                             [1, 0, 2],    # 13
                             [1, 1, 2],    # 14
                             [0, 1, 2],    # 15
                             [2, 0, 2],    # 16
                             [2, 1, 2],    # 17
                             ])

vertices = np.array([BlockMeshVertex(position=x) for x in vertex_positions])


block0 = Block(vertices=vertices[0:8],
               name=f'Block1',
               auto_cell_size=True,
               extruded=False)

block1 = Block(vertices=vertices[[1, 8, 9, 2, 5, 10, 11, 6]],
               name=f'Block2',
               auto_cell_size=True,
               extruded=False)

block2 = Block(vertices=vertices[[4, 5, 6, 7, 12, 13, 14, 15]],
               name=f'Block3',
               auto_cell_size=True,
               extruded=False)


block4 = block2.extrude_face(face_id=1, dist=0.5)
block5 = block4.extrude_face(face_id=2, dist=1)


block3 = Block(vertices=vertices[[5, 10, 11, 6, 13, 16, 17, 14]],
               name=f'Block3',
               auto_cell_size=True,
               extruded=False)


mesh2.activate()

block6 = block0.extrude_face(face_id=2, dist=1)
# block6 = block3.face2.extrude(dist=1)
block7 = block2.extrude_face(face_id=2, dist=1)

block6.num_cells = [10, 10, 10]
block7.num_cells = [10, 10, 10]

default_path = '/tmp'

block_meshes = []
for mesh in [mesh1, mesh2]:

    if 0 in [mesh.vertices.__len__(), mesh.blocks.__len__()]:
        continue

    mesh.activate()
    block_mesh = BlockMesh(name='Block Mesh ' + mesh.name,
                           case_dir=os.path.join(default_path, mesh.txt_id),
                           mesh=mesh)
    block_meshes.append(block_mesh)

BlockMesh.add_mesh_contacts(block_meshes)

for block_mesh in block_meshes:
    block_mesh.init_case()
    block_mesh.run_block_mesh(run_parafoam=True)

for mesh in block_meshes[1:]:
    block_meshes[0].merge_mesh(mesh)

block_meshes[0].mesh

# block_meshes[0].stitch_meshes(block_meshes[1:])

print('done')

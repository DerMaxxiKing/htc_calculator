import numpy as np
from src.htc_calculator.meshing.block_mesh import BlockMeshVertex, Block, BlockMesh
from src.htc_calculator.case.utils import run_parafoam_touch_all


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


block1 = Block(vertices=vertices[0:8],
               name=f'Block1',
               auto_cell_size=True,
               extruded=False)

block2 = Block(vertices=vertices[[1, 8, 9, 2, 5, 10, 11, 6]],
               name=f'Block2',
               auto_cell_size=False,
               num_cells=[10, 10, 10],
               merge_patch_pairs=[2],
               extruded=False)

block3 = Block(vertices=vertices[[4, 5, 6, 7, 12, 13, 14, 15]],
               name=f'Block3',
               auto_cell_size=True,
               extruded=False)

block4 = Block(vertices=vertices[[5, 10, 11, 6, 13, 16, 17, 14]],
               name=f'Block3',
               auto_cell_size=True,
               extruded=False)


block_mesh = BlockMesh(name='test_merge_patch_pairs',
                       case_dir='/tmp/test_case2_error')
block_mesh.init_case()
block_mesh.run_block_mesh()

run_parafoam_touch_all(block_mesh.case_dir)

print('done')

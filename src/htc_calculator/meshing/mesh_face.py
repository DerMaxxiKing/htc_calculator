import logging
# import re
import math
# import sys
import os
import numpy as np
# import ntpath
#
# import subprocess
# from shutil import copyfile


import tempfile
import uuid

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import meshing_resources as msh_resources


def create_block_mesh_dict(reference_face, case_dir, cell_size, expand_factor=1.001):

    # https://damogranlabs.com/2018/05/openfoam-meshing-shortcuts/
    # command-line input
    help_text = """Usage: makeBMD.py <file> <hex size [m]> [expand_factor] """

    # format of vertex files:
    #    vertex  7.758358e-03  2.144992e-02  1.539336e-02
    #    vertex  7.761989e-03  2.167315e-02  1.525611e-02
    #    vertex  7.767175e-03  2.167225e-02  1.551236e-02

    # a regular expression to match a beginning of a vertex line in STL file
    # vertex_re = re.compile('\s+vertex.+')

    vertex_min = [reference_face.assembly.hull.fc_solid.Shape.BoundBox.XMin,
                  reference_face.assembly.hull.fc_solid.Shape.BoundBox.YMin,
                  reference_face.assembly.hull.fc_solid.Shape.BoundBox.ZMin]
    vertex_max = [reference_face.assembly.hull.fc_solid.Shape.BoundBox.XMax,
                  reference_face.assembly.hull.fc_solid.Shape.BoundBox.YMax,
                  reference_face.assembly.hull.fc_solid.Shape.BoundBox.ZMax]

    # stroll through the file and find points with highest/lowest coordinates
    # with open(file_name, 'r') as f:
    #     for line in f:
    #         m = vertex_re.match(line)
    #
    #         if m:
    #             n = line.split()
    #             v = [float(n[i]) for i in range(1, 4)]
    #
    #             vertex_max = [max([vertex_max[i], v[i]]) for i in range(3)]
    #             vertex_min = [min([vertex_min[i], v[i]]) for i in range(3)]

    # scale the blockmesh by a small factor
    # achtung, scale around object center, not coordinate origin!
    for i in range(3):
        center = (vertex_max[i] + vertex_min[i])/2
        size = vertex_max[i] - vertex_min[i]

        vertex_max[i] = center + size/2*expand_factor
        vertex_min[i] = center - size/2*expand_factor

    # find out number of elements that will produce desired cell size
    sizes = [vertex_max[i] - vertex_min[i] for i in range(3)]
    num_elements = np.array([int(math.ceil(sizes[i]/cell_size)) for i in range(3)])

    num_elements[num_elements < 5] = 5

    print("max: {}".format(vertex_max))
    print("min: {}".format(vertex_min))
    print("sizes: {}".format(sizes))
    print("number of elements: {}".format(num_elements))
    print("expand factor: {}".format(expand_factor))

    # write a blockMeshDict file
    bm_file = """
    /*--------------------------------*- C++ -*----------------------------------*\
    | =========                 |                                                 |
    | \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\\\    /   O peration     | Version:  dev                                   |
    |   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
    |    \\\\/     M anipulation  |                                                 |
    \*---------------------------------------------------------------------------*/
    FoamFile
    {{
        version     2.0;
        format      ascii;
        class       dictionary;
        object      blockMeshDict;
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
     
    convertToMeters 1;
     
    x_min {v_min[0]};
    x_max {v_max[0]};
     
    y_min {v_min[1]};
    y_max {v_max[1]};
     
    z_min {v_min[2]};
    z_max {v_max[2]};
     
    n_x {n[0]};
    n_y {n[1]};
    n_z {n[2]};
     
    vertices
    (
        ($x_min $y_min $z_min) //0
        ($x_max $y_min $z_min) //1
        ($x_max $y_max $z_min) //2
        ($x_min $y_max $z_min) //3
        ($x_min $y_min $z_max) //4
        ($x_max $y_min $z_max) //5
        ($x_max $y_max $z_max) //6
        ($x_min $y_max $z_max) //7
    );
     
     
    blocks ( hex (0 1 2 3 4 5 6 7) ($n_x $n_y $n_z) simpleGrading (1 1 1) );
     
    edges ( );
    patches ( );
    mergePatchPairs ( );
     
    // ************************************************************************* //
    """

    # write the blockMeshDict
    mesh_filepath = os.path.join(case_dir, 'system', 'blockMeshDict')
    logging.info(f'writing mesh to: {mesh_filepath}')
    with open(mesh_filepath, 'w') as mesh_file:
        mesh_file.write(
            bm_file.format(v_min=vertex_min, v_max=vertex_max, n=num_elements)
        )

    # # create surfaceFeatureExtractDict
    #
    # template = pkg_resources.read_text(msh_resources, 'surfaceFeatureExtractDict')
    # template = template.replace('<stl_file>', ntpath.basename(file_name))
    #
    # surface_feature_extract_dict_filepath = os.path.join(case_dir, 'system', 'surfaceFeaturesDict')
    # logging.info(f'writing surfaceFeaturesDict to: {surface_feature_extract_dict_filepath}')
    # with open(surface_feature_extract_dict_filepath, 'w') as surface_feature_extract_dict_file:
    #     surface_feature_extract_dict_file.write(template)
    #
    # # run surfaceFeatureExtract
    # command = f"source /opt/openfoam8/etc/bashrc; cd {case_dir}; surfaceFeatures > {os.path.join(case_dir, 'surfaceFeatureExtractLog')}"
    # ret = subprocess.run(command, capture_output=True, shell=True, executable='/bin/bash', cwd=case_dir)
    #
    # print(f'case_dir: {case_dir}')
    # print(f'out: {ret.stdout.decode()}')
    # print(f'err: {ret.stderr.decode()}')

    # # run blockMesh
    # command = f"source /opt/openfoam8/etc/bashrc; blockMesh -case {case_dir} > {os.path.join(case_dir, 'blockMeshLog')}"
    # ret = subprocess.run(command, capture_output=True, shell=True, executable='/bin/bash', cwd=case_dir)
    # print(f'case_dir: {case_dir}')
    # print(f'out: {ret.stdout.decode()}')
    # print(f'err: {ret.stderr.decode()}')
    #
    # print("done.")


def mesh_face():

    surfaceFeatureExtract
    blockMesh

    pass


def create_temp_case_dir(directory=None):

    # 0, constant, system directory needed

    if directory is None:
        directory = tempfile.gettempdir()

    case_dir = os.path.join(directory, f'case_{uuid.uuid4()}')
    os.mkdir(case_dir)
    os.mkdir(os.path.join(case_dir, '0'))
    os.mkdir(os.path.join(case_dir, 'constant'))
    os.mkdir(os.path.join(case_dir, 'system'))

    return case_dir


def create_snappy_hex_mesh_dict(reference_face, case_dir):

    shmd_template = pkg_resources.read_text(msh_resources, 'snappy_hex_mesh_dict')
    geo_str1 = ''.join(x.shm_geo_entry for x in reference_face.assembly.solids)
    geo_str = geo_str1 + reference_face.assembly.interface_shm_geo_entry()

    shmd_template = shmd_template.replace('<stls>', geo_str)

    dst = os.path.join(case_dir, 'system', 'snappyHexMeshDict')
    with open(dst, 'w') as shmd:
        shmd.write(shmd_template)
    # copyfile(source, dst)

    print('finished')


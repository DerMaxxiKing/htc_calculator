import numpy as np

from src.htc_calculator.tools import create_pipe_wire, export_objects

import FreeCAD
import Part as FCPart
from FreeCAD import Base


def points_from_vertices(vertices):
    return [Base.Vector(row) for row in vertices]


def test_length():
    pipe_wires = []
    for d2 in np.linspace(1000, 15000, 30, endpoint=True):

        vertices = np.array([[0, 0, 0],
                             [6000, 0, 0],
                             [6000, d2, 0],
                             [0, d2, 0]])

        points = points_from_vertices(vertices)
        reference_wire = FCPart.makePolygon([*points, points[0]])
        reference_face = FCPart.Face(reference_wire)

        pipe_wire = create_pipe_wire(reference_face,
                                     start_edge=0,
                                     tube_distance=225,
                                     tube_edge_distance=300,
                                     bending_radius=100,
                                     tube_diameter=20
                                     )

        pipe_wires.append(pipe_wire)

    export_objects([*pipe_wires], '/tmp/pipe_wires.FCStd')


def test_polygon(vertices):
    # vertices = np.array([[0, 0, 0],
    #                      [10000, 0, 0],
    #                      [10000, 2500, 0],
    #                      [7500, 2500, 0],
    #                      [7500, 5000, 0],
    #                      [2500, 5000, 0],
    #                      [2500, 2500, 0],
    #                      [0, 2500, 0]])

    points = points_from_vertices(vertices)
    reference_wire = FCPart.makePolygon([*points, points[0]])
    reference_face = FCPart.Face(reference_wire)

    pipe_wire = create_pipe_wire(reference_face,
                                 start_edge=0,
                                 tube_distance=225,
                                 tube_edge_distance=300,
                                 bending_radius=100,
                                 tube_diameter=20
                                 )

    export_objects([pipe_wire, reference_face], '/tmp/pipe_wires_poly1.FCStd')


if __name__ == '__main__':
    # test_length()

    poly_vertices = np.array([[0, 0, 0],
                              [10000, 0, 0],
                              [10000, 2500, 0],
                              [7500, 2500, 0],
                              [7500, 5000, 0],
                              [2500, 5000, 0],
                              [2500, 2500, 0],
                              [0, 2500, 0]])

    test_polygon(poly_vertices)

    poly_vertices2 = np.array([[2500, 0, 0],
                               [7500, 0, 0],
                               [7500, 2500, 0],
                               [10000, 2500, 0],
                               [10000, 5000, 0],
                               [0, 5000, 0],
                               [0, 2500, 0],
                               [2500, 2500, 0]])

    test_polygon(poly_vertices2)

    poly_vertices3 = np.array([[5000, 0, 0],
                               [10000, 0, 0],
                               [10000, 2500, 0],
                               [7500, 2500, 0],
                               [7500, 5000, 0],
                               [0, 5000, 0],
                               [0, 2500, 0],
                               [5000, 2500, 0]])

    test_polygon(poly_vertices3)

    poly_vertices4 = np.array([[5000, 0, 0],
                               [12500, 0, 0],
                               [10000, 2500, 0],
                               [7500, 2500, 0],
                               [7500, 5000, 0],
                               [0, 5000, 0],
                               [0, 2500, 0],
                               [2500, 2500, 0]])

    test_polygon(poly_vertices4)

print('done')

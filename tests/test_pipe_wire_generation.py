import numpy as np

from src.htc_calculator.logger import logger
from src.htc_calculator.tools import create_pipe_wire, export_objects, add_radius_to_edges

import FreeCAD
import Part as FCPart
from FreeCAD import Base


def points_from_vertices(vertices):
    return [Base.Vector(row) for row in vertices]


def test_length():
    pipe_wires = []

    for d2 in np.linspace(1000, 15000, 30, endpoint=True):

        try:
            vertices = np.array([[0, 0, 0],
                                 [6000, 0, 0],
                                 [6000, d2, 0],
                                 [0, d2, 0]])

            points = points_from_vertices(vertices)
            reference_wire = FCPart.makePolygon([*points, points[0]])
            reference_face = FCPart.Face(reference_wire)

            pipe_wire, _ = create_pipe_wire(reference_face,
                                            start_edge=0,
                                            tube_distance=225,
                                            tube_edge_distance=300,
                                            bending_radius=100,
                                            tube_diameter=20
                                            )

            pipe_wires.append((pipe_wire, reference_face))

            export_objects([pipe_wire, reference_face], '/tmp/pipe_wires_poly1.FCStd')

        except Exception as e:
            logger.error(f'Error creating pipe wire: {e}')
            raise e

    return pipe_wires


def test_polygon(vertices):
    points = points_from_vertices(vertices)
    reference_wire = FCPart.makePolygon([*points, points[0]])
    reference_face = FCPart.Face(reference_wire)

    pipe_wire, _ = create_pipe_wire(reference_face,
                                    start_edge=0,
                                    tube_distance=225,
                                    tube_edge_distance=300,
                                    bending_radius=100,
                                    tube_diameter=20
                                    )

    export_objects([pipe_wire, reference_face], '/tmp/pipe_wires_poly1.FCStd')

    return pipe_wire, reference_face


def test_add_radius(pipe_wire_list, radius):

    radius_pipe_wires = []

    for i, pipe_wire in enumerate(pipe_wire_list):
        print(f'Adding radius to pipe_wire {i}')
        pw_radius = add_radius_to_edges(pipe_wire[0].OrderedEdges, radius)
        radius_pipe_wires.append((pw_radius, pipe_wire[1]))
        print(f'Added radius to pipe_wire {i} successfully')
        export_objects([pw_radius, pipe_wire[1]], '/tmp/pipe_wire_radius.FCStd')

    return radius_pipe_wires


if __name__ == '__main__':

    pipe_wires = []

    # pipe_wires.extend(test_length())

    poly_vertices = np.array([[0, 0, 0],
                              [10000, 0, 0],
                              [10000, 2500, 0],
                              [7500, 2500, 0],
                              [7500, 5000, 0],
                              [2500, 5000, 0],
                              [2500, 2500, 0],
                              [0, 2500, 0]])

    pipe_wires.append(test_polygon(poly_vertices))

    poly_vertices2 = np.array([[2500, 0, 0],
                               [7500, 0, 0],
                               [7500, 2500, 0],
                               [10000, 2500, 0],
                               [10000, 5000, 0],
                               [0, 5000, 0],
                               [0, 2500, 0],
                               [2500, 2500, 0]])

    pipe_wires.append(test_polygon(poly_vertices2))

    poly_vertices3 = np.array([[5000, 0, 0],
                               [10000, 0, 0],
                               [10000, 2500, 0],
                               [7500, 2500, 0],
                               [7500, 5000, 0],
                               [0, 5000, 0],
                               [0, 2500, 0],
                               [5000, 2500, 0]])

    pipe_wires.append(test_polygon(poly_vertices3))

    poly_vertices4 = np.array([[5000, 0, 0],
                               [12500, 0, 0],
                               [10000, 2500, 0],
                               [7500, 2500, 0],
                               [7500, 5000, 0],
                               [0, 5000, 0],
                               [0, 2500, 0],
                               [2500, 2500, 0]])

    pipe_wires.append(test_polygon(poly_vertices4))

    pipe_wires_radius = test_add_radius(pipe_wires, 100)

print('done')

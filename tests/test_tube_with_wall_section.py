import numpy as np

from src.htc_calculator.meshing.buildin_pipe_sections.tube_with_wall import pipe_section
from src.htc_calculator.meshing.buildin_pipe_sections.tube_with_wall_optimized import pipe_section as pipe_section2
from src.htc_calculator.tools import export_objects, perpendicular_vector

import FreeCAD
import Part as FCPart
from FreeCAD import Base

start_point = np.array([0, 0, 0])
face_normal = np.array([0, 0, 1])
tube_inner_diameter = 16
tube_diameter = 20

edge = FCPart.Edge(FCPart.LineSegment(Base.Vector(np.array([0, 0, 0])),
                                      Base.Vector(np.array([100, 100, 0]))))

np.sin(np.deg2rad(45))

edge2 = FCPart.Edge(FCPart.Arc(Base.Vector(np.array([0, 0, 0])),
                               Base.Vector(np.array([100 - np.sin(np.deg2rad(45)) * 100, np.sin(np.deg2rad(45)) * 100, 0])),
                               Base.Vector(np.array([100, 100, 0])),
                               )
                    )


blocks = pipe_section.create_block(edge,
                                   face_normal,
                                   tube_inner_diameter,
                                   tube_diameter,
                                   outer_pipe=True,
                                   inlet=False,
                                   outlet=False)

blocks2 = pipe_section2.create_block(edge,
                                     face_normal,
                                     tube_inner_diameter,
                                     tube_diameter,
                                     outer_pipe=True,
                                     inlet=False,
                                     outlet=False)

blocks3 = pipe_section2.create_block(edge2,
                                     face_normal,
                                     tube_inner_diameter,
                                     tube_diameter,
                                     outer_pipe=True,
                                     inlet=True,
                                     outlet=True)

export_objects([x.fc_solid for x in blocks], '/tmp/pipe_section_test.FCStd')
export_objects([x.fc_solid for x in blocks2], '/tmp/pipe_section_test2.FCStd')
export_objects([x.fc_solid for x in blocks3], '/tmp/pipe_section_test3.FCStd')

print('done')

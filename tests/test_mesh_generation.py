import src.htc_calculator

App = FreeCAD
import ObjectsFem
from femmesh.gmshtools import GmshTools


def test_mesh_creation():
    # more sophisticated example which changes the mesh size
    doc = App.newDocument("MeshTest")
    box_obj = doc.addObject("Part::Box", "Box")
    doc.recompute()
    max_mesh_sizes = [0.5, 1, 2, 3, 5, 10]
    for len in max_mesh_sizes:
        quantity_len = "{}".format(len)
        print("\n\n Start length = {}".format(quantity_len))
        femmesh_obj = ObjectsFem.makeMeshGmsh(doc, box_obj.Name + "_Mesh")
        femmesh_obj.Part = box_obj
        femmesh_obj.CharacteristicLengthMax = "{}".format(quantity_len)
        femmesh_obj.CharacteristicLengthMin = "{}".format(quantity_len)
        doc.recompute()
        gm = GmshTools(femmesh_obj)
        gm.update_mesh_data()
        # set the tmp file path to some user path including the length
        gm.get_tmp_file_paths("/tmp/fcgm_" + str(len), True)
        gm.get_gmsh_command()
        gm.write_gmsh_input_files()
        error = gm.run_gmsh_with_geo()
        print(error)
        gm.read_and_set_new_mesh()
        doc.recompute()
        print("Done length = {}".format(quantity_len))

        print('done')


if __name__ == '__main__':
    test_mesh_creation()

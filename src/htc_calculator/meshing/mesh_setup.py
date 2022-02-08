
import uuid


class MeshSetup(object):

    def __init__(self, *args, **kwargs):

        self.name = kwargs.get('name', None)
        self.id = kwargs.get('id', uuid.uuid4())
        self.mesh_size = kwargs.get('mesh_size', 0.25)

        self.add_layers = kwargs.get('add_layers', False)
        self.num_layers = kwargs.get('num_layers', 3)
        self.first_layer_thickness = kwargs.get('first_layer_thickness', 0.05)
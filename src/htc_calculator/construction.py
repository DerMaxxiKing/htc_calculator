import uuid


class Material(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)

        self.density = kwargs.get('density', 1000)
        self.specific_heat_capacity = kwargs.get('specific_heat_capacity', 1000)
        self.heat_conductivity = kwargs.get('heat_conductivity', 1000)


class Layer(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)
        self.material = kwargs.get('material', None)
        self.thickness = kwargs.get('thickness', None)
        self.solid = None


class ComponentConstruction(object):

    def __init__(self, *args, **kwargs):
        """
                                                    |   Reference geometry
                                                    |
                                                    v
                        ||           |                       |            ||
        side1           ||   layer1  |           layer2      |    layer3  ||           side2
                        ||           |                       |            ||
                                                    ^
                                                    |
                                                    |    Reference geometry
                                                    |
                                                    |
                        <----- side_1_offset ---- > |

        :param args:
        :param kwargs:
        """
        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)
        self.layers = kwargs.get('layers', [])

        self.side_1_offset = kwargs.get('side_1_offset', 0)     # offset side1 to reference geometry

    @property
    def thickness(self):
        return sum([x.thickness for x in self.layers])

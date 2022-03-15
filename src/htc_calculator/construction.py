import uuid


class Material(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)

        self.density = kwargs.get('density', 1000)                                      # kg / m^3
        self.specific_heat_capacity = kwargs.get('specific_heat_capacity', 1000)        # J / kg K
        self.heat_conductivity = kwargs.get('heat_conductivity', 1000)                  # W / m K

        self.roughness = kwargs.get('roughness', None)

        self.of_type = kwargs.get('of_type', 'heSolidThermo')
        self.of_mixture = kwargs.get('of_mixture', 'pureMixture')
        self.of_transport = kwargs.get('of_transport', 'constIso')
        self.of_thermo = kwargs.get('of_thermo', 'eConst')
        self.of_equationOfState = kwargs.get('of_equationOfState', 'rhoConst')
        self.specie = kwargs.get('specie', 'specie')
        self.energy = kwargs.get('energy', 'sensibleInternalEnergy')

        self.mol_weight = 1
        self.hf = 0

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)


class Fluid(object):
    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)

        self.density = kwargs.get('density', 1000)                                      # kg / m^3
        self.specific_heat_capacity = kwargs.get('specific_heat_capacity', 1000)        # J / kg K
        self.heat_conductivity = kwargs.get('heat_conductivity', 1000)                  # W / m K

        self.roughness = kwargs.get('roughness', None)

        self.of_type = kwargs.get('of_type', 'heRhoThermo')
        self.of_mixture = kwargs.get('of_mixture', 'pureMixture')
        self.of_transport = kwargs.get('of_transport', 'const')
        self.of_thermo = kwargs.get('of_thermo', 'hConst')
        self.of_equationOfState = kwargs.get('of_equationOfState', 'rhoConst')
        self.specie = kwargs.get('specie', 'specie')
        self.energy = kwargs.get('energy', 'sensibleEnthalpie')

        self.mol_weight = kwargs.get('mol_weight')
        self.hf = kwargs.get('hf', 0)

        self.mu = kwargs.get('mu')
        self.pr = kwargs.get('pr')

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)


class Layer(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)
        self.material = kwargs.get('material', None)
        self.thickness = kwargs.get('thickness', None)
        self.solid = None

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)


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

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

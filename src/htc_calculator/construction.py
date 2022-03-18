import os
import uuid

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


from .case import case_resources


class Material(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)

        self.density = kwargs.get('density', 1000)                                      # kg / m^3
        self.specific_heat_capacity = kwargs.get('specific_heat_capacity', 1000)        # J / kg K
        self.heat_conductivity = kwargs.get('heat_conductivity', 1000)                  # W / m K

        self._thermo_physical_properties_entry = None

        self.mol_weight = 1
        self.hf = 0

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def thermo_physical_properties_entry(self):
        if self._thermo_physical_properties_entry is None:
            self._thermo_physical_properties_entry = self.generate_thermo_physical_properties_entry()
        return self._thermo_physical_properties_entry

    def init_directory(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'constant', str(self.txt_id)), exist_ok=True)

    def generate_thermo_physical_properties_entry(self):
        tp_entry = pkg_resources.read_text(case_resources, 'solid_thermophysicalProperties')
        tp_entry = tp_entry.replace('<material_id>', self.txt_id)
        tp_entry = tp_entry.replace('<material_name>', str(self.name))
        tp_entry = tp_entry.replace('<molWeight>', str(self.mol_weight))
        tp_entry = tp_entry.replace('<rho>', str(self.density))
        tp_entry = tp_entry.replace('<kappa>', str(self.heat_conductivity))
        tp_entry = tp_entry.replace('<cv>', str(self.specific_heat_capacity))
        return tp_entry

    def write_thermo_physical_properties(self, case_dir):
        #
        os.makedirs(os.path.join(case_dir, 'constant', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'thermophysicalProperties')
        with open(full_filename, "w") as f:
            f.write(self.thermo_physical_properties_entry)


class Solid(Material):

    def __init__(self, *args, **kwargs):
        Material.__init__(self, *args, **kwargs)
        self.roughness = kwargs.get('roughness', None)

    def write_to_of(self, case_dir):
        self.init_directory(case_dir),
        self.write_thermo_physical_properties(case_dir)

    def __repr__(self):
        return f'Solid material {self.name} ({self.id})'


class Fluid(Material):

    def __init__(self, *args, **kwargs):
        Material.__init__(self, *args, **kwargs)

        self.mu = kwargs.get('mu')
        self.pr = kwargs.get('pr')

        self._g_entry = None
        self._thermo_physic_transport_entry = None
        self._momentum_transport_entry = None

        self.prt = kwargs.get('prt', 0.85)

    @property
    def txt_id(self):
        if isinstance(self.id, uuid.UUID):
            return 'a' + str(self.id.hex)
        else:
            return str(self.id)

    @property
    def thermo_physical_properties_entry(self):
        if self._thermo_physical_properties_entry is None:
            self._thermo_physical_properties_entry = self.generate_thermo_physical_properties_entry()
        return self._thermo_physical_properties_entry

    @property
    def thermo_physic_transport_entry(self):
        if self._thermo_physic_transport_entry is None:
            self._thermo_physic_transport_entry = self.generate_thermo_physic_transport_entry()
        return self._thermo_physic_transport_entry

    @property
    def g_entry(self):
        if self._g_entry is None:
            self._g_entry = self.generate_g_entry()
        return self._g_entry

    @property
    def momentum_transport_entry(self):
        if self._momentum_transport_entry is None:
            self._momentum_transport_entry = self.generate_momentum_transport_entry()
        return self._momentum_transport_entry

    def generate_thermo_physical_properties_entry(self):
        entry = pkg_resources.read_text(case_resources, 'fluid_thermophysicalProperties')
        entry = entry.replace('<material_id>', self.txt_id)
        entry = entry.replace('<material_name>', str(self.name))
        entry = entry.replace('<molWeight>', str(self.mol_weight))
        entry = entry.replace('<rho>', str(self.density))
        entry = entry.replace('<cp>', str(self.specific_heat_capacity))
        entry = entry.replace('<mu>', str(self.mu))
        entry = entry.replace('<pr>', str(self.pr))
        return entry

    def generate_g_entry(self):
        entry = pkg_resources.read_text(case_resources, 'g')
        entry = entry.replace('<material_id>', self.txt_id)
        entry = entry.replace('<material_name>', str(self.name))
        return entry

    def generate_thermo_physic_transport_entry(self):
        entry = pkg_resources.read_text(case_resources, 'thermophysicalTransport')
        entry = entry.replace('<material_id>', self.txt_id)
        entry = entry.replace('<material_name>', str(self.name))
        entry = entry.replace('<prt>', str(self.prt))
        return entry

    def generate_momentum_transport_entry(self):
        entry = pkg_resources.read_text(case_resources, 'momentumTransport')
        entry = entry.replace('<material_name>', str(self.name))
        entry = entry.replace('<material_id>', self.txt_id)
        return entry

    def write_thermo_physical_properties(self, case_dir):
        # https://cpp.openfoam.org/v8/classFoam_1_1thermophysicalProperties.html
        self.init_directory(case_dir)
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'thermophysicalProperties')
        with open(full_filename, "w") as f:
            f.write(self.thermo_physical_properties_entry)

    def write_g(self, case_dir):
        self.init_directory(case_dir)
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'g')
        with open(full_filename, "w") as f:
            f.write(self.g_entry)

    def write_thermo_physic_transport(self, case_dir):
        # https://cpp.openfoam.org/v8/classFoam_1_1turbulenceThermophysicalTransportModels_1_1eddyDiffusivity.html#a10501494309552f678d858cb7de6c1d3
        self.init_directory(case_dir)
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'thermophysicalTransport')
        with open(full_filename, "w") as f:
            f.write(self.thermo_physic_transport_entry)

    def write_momentum_transport(self, case_dir):
        # https://cpp.openfoam.org/v8/classFoam_1_1turbulenceThermophysicalTransportModels_1_1eddyDiffusivity.html#a10501494309552f678d858cb7de6c1d3
        self.init_directory(case_dir)
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'momentumTransport')
        with open(full_filename, "w") as f:
            f.write(self.momentum_transport_entry)

    def write_to_of(self, case_dir):
        self.init_directory(case_dir),
        self.write_thermo_physical_properties(case_dir)
        self.write_thermo_physic_transport(case_dir)
        self.write_momentum_transport(case_dir)
        self.write_g(case_dir)

    def __repr__(self):
        return f'Fluid material {self.name} ({self.id})'


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


def write_region_properties(materials, case_dir):
    os.makedirs(os.path.join(case_dir, 'constant'), exist_ok=True)
    entry = pkg_resources.read_text(case_resources, 'regionProperties')

    solids = []
    fluids = []

    for mat in materials:
        if isinstance(mat, Solid):
            solids.append(mat.txt_id)
        elif isinstance(mat, Fluid):
            fluids.append(mat.txt_id)

    entry = entry.replace('<solids>', ' '.join(solids))
    entry = entry.replace('<fluids>', ' '.join(fluids))

    full_filename = os.path.join(case_dir, 'constant', 'regionProperties')
    with open(full_filename, "w") as f:
        f.write(entry)

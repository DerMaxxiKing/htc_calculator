import os
import uuid
from inspect import cleandoc
from abc import ABCMeta, abstractmethod
from .case.boundary_conditions import *
from .logger import logger
from .case.boundary_conditions.user_bcs import of_field_name_lookup_dict
from .case.utils import indent_text
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


from .config import n_proc
from .case import case_resources
from .case.case_resources import solid as solid_resources
from .case.case_resources import fluid as fluid_resources
from .case.case_resources.solid import static as static_solid_resources
from .case.case_resources.fluid import static as static_fluid_resources


class Material(object, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):

        self._cell_zone = None

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)
        self.cell_zone = kwargs.get('cell_zone', None)
        self.solids = kwargs.get('solids', set())

        self.density = kwargs.get('density', 1000)                                      # kg / m^3
        self.specific_heat_capacity = kwargs.get('specific_heat_capacity', 1000)        # J / kg K
        self.heat_conductivity = kwargs.get('heat_conductivity', 1000)                  # W / m K

        self.mol_weight = 1
        self.hf = 0

        self._fvschemes = None
        self._fvsolution = None
        self._thermo_physical_properties_entry = None
        self._decompose_par_dict = None

        self.boundaries = kwargs.get('boundaries', {})
        self.initial_temperature = kwargs.get('initial_temperature', 273.15 + 20)

        self.case_dir = kwargs.get('case_dir', None)
        self.case = kwargs.get('case', None)

        self.alphat = kwargs.get('alphat', Alphat())
        self.epsilon = kwargs.get('epsilon', Epsilon())
        self.k = kwargs.get('k', K())
        self.nut = kwargs.get('nut', Nut())
        self.p = kwargs.get('p', P())
        self.p_rgh = kwargs.get('p_rgh', PRgh())
        self.t = kwargs.get('t', T())
        self.u = kwargs.get('u', U())

    @property
    @abstractmethod
    def fvschemes(self):
        return self._fvschemes

    @property
    @abstractmethod
    def fvsolution(self):
        return self._fvsolution

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
    def cell_zone(self):
        if self._cell_zone is None:
            self._cell_zone = self.create_cell_zone()
        return self._cell_zone

    @cell_zone.setter
    def cell_zone(self, value):
        self._cell_zone = value

    @property
    def decompose_par_dict(self):
        if self._decompose_par_dict is None:
            tp_entry = pkg_resources.read_text(case_resources, 'decomposeParDict')
            tp_entry = tp_entry.replace('<n_proc>', str(n_proc))
            self._decompose_par_dict = tp_entry
        return self._decompose_par_dict

    @property
    def change_dict_entry(self):
        return self.create_change_dict_entry()

    def init_directory(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'constant', str(self.txt_id)), exist_ok=True)
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        os.makedirs(os.path.join(case_dir, '0', str(self.txt_id)), exist_ok=True)

    def write_empty_field(self, field_name, case_dir=None):
        field_template = cleandoc("""
        /*--------------------------------*- C++ -*----------------------------------*\
          =========                 |
          \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
           \\    /   O peration     | Website:  https://openfoam.org
            \\  /    A nd           | Version:  9
             \\/     M anipulation  |
        \*---------------------------------------------------------------------------*/
        FoamFile
        {
            format      ascii;
            class       volScalarField;
            location    "0/<region_id>";
            object      <field_name>;
        }
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        dimensions      [ 0 0 0 1 0 0 0 ];

        internalField   0;

        boundaryField
        {
        }

        // ************************************************************************* //
        """)

        if case_dir is None:
            case_dir = self.case_dir

        field_str = field_template
        field_str = field_str.replace('<field_name>', field_name)
        field_str = field_str.replace('<region_id>', self.txt_id)

        os.makedirs(os.path.join(case_dir, '0', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), field_name)
        with open(full_filename, "w") as f:
            f.write(field_str)

    def generate_thermo_physical_properties_entry(self):
        tp_entry = pkg_resources.read_text(solid_resources, 'thermophysicalProperties')
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

    def write_decompose_par_dict(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), 'decomposeParDict')
        with open(full_filename, "w") as f:
            f.write(self.decompose_par_dict)

    def write_fvschemes(self, case_dir):

        if case_dir is None:
            case_dir = self.case_dir

        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), 'fvSchemes')
        with open(full_filename, "w") as f:
            f.write(self.fvschemes)

    def write_fvsolution(self, case_dir):
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), 'fvSolution')
        with open(full_filename, "w") as f:
            f.write(self.fvsolution)

    def create_cell_zone(self):
        from .meshing.block_mesh import CellZone
        return CellZone(material=self)

    def update_bcs(self):

        for boundary in self.boundaries:
            logger.debug(f'updating_bcs')

            if boundary.type == 'interface':
                bc_key = boundary.txt_id
            else:
                bc_key = boundary.txt_id

            if isinstance(boundary, CyclicAMI):
                bc_key = boundary.txt_id

            self.t.patches[bc_key] = boundary.user_bc.t

            if isinstance(self, Fluid):
                self.alphat.patches[bc_key] = boundary.user_bc.alphat
                self.epsilon.patches[bc_key] = boundary.user_bc.epsilon
                self.k.patches[bc_key] = boundary.user_bc.k
                self.nut.patches[bc_key] = boundary.user_bc.nut
                self.p.patches[bc_key] = boundary.user_bc.p
                self.p_rgh.patches[bc_key] = boundary.user_bc.p_rgh
                self.u.patches[bc_key] = boundary.user_bc.u

        logger.debug(f'updating_bcs')

    def write_boundary_conditions(self, case_dir):
        if case_dir is None:
            case_dir = self.case_dir

        # write T:
        self.t.internal_field_value = self.case.bc.initial_temperature[self]
        self.t.write(os.path.join(case_dir, '0', self.txt_id))

        for bc_name in ['alphat', 'epsilon', 'k', 'nut', 'p', 'p_rgh', 'u']:
            bc_file = getattr(self, bc_name)
            bc_file.write(os.path.join(case_dir, '0', self.txt_id))

    def create_change_dict_entry(self, case_dir=None):

        if case_dir is None:
            case_dir = self.case_dir

        template = pkg_resources.read_text(case_resources, 'changeDictionaryDict')

        boundary_entries = "\n".join(x.boundary_entry for x in self.boundaries)
        template = template.replace('<boundaries>', boundary_entries)

        field_entries = []

        if isinstance(self, Solid):
            field_names = ['t']
        elif isinstance(self, Fluid):
            field_names = ['t', 'alphat', 'epsilon', 'k', 'nut', 'p', 'p_rgh', 'u']

        for field_name in field_names:
            field_str = (f'{of_field_name_lookup_dict[field_name]}\n'
                         '{\n'
                         f'\tinternalField   {self.__getattribute__(field_name).internal_field_value};\n'
                         f'\tboundaryField\n'
                         '\t{\n')

            field_str = field_str + indent_text(
                "\n".join(x.field_entry(field_name) for x in self.boundaries.values() if x.user_bc is not None), 1)
            field_str = field_str + '\t}\n}\n'
            field_entries.append(field_str)

        template = template.replace('<field_entries>', "\n".join(field_entries))

        return template

    def write_change_dict(self, case_dir):
        if case_dir is None:
            case_dir = self.case_dir
        os.makedirs(os.path.join(case_dir, 'system', str(self.txt_id)), exist_ok=True)
        full_filename = os.path.join(case_dir, 'system', str(self.txt_id), 'changeDictionaryDict')
        logger.info(f'Writing change dictionary for {self.txt_id} to {full_filename}')
        with open(full_filename, "w") as f:
            f.write(self.change_dict_entry)
        logger.info(f'Successfully written change dictionary')


class Solid(Material):

    def __init__(self, *args, **kwargs):
        Material.__init__(self, *args, **kwargs)
        self.roughness = kwargs.get('roughness', None)

    @property
    def fvschemes(self):
        if self._fvschemes is None:
            self._fvschemes = pkg_resources.read_text(static_solid_resources, 'fvSchemes')
        return self._fvschemes

    @property
    def fvsolution(self):
        if self._fvsolution is None:
            self._fvsolution = pkg_resources.read_text(static_solid_resources, 'fvSolution')
        return self._fvsolution

    def write_to_of(self, case_dir):
        self.init_directory(case_dir),
        self.write_thermo_physical_properties(case_dir)
        self.write_decompose_par_dict(case_dir)
        self.write_fvschemes(case_dir)
        self.write_fvsolution(case_dir)

        if isinstance(self, Fluid):
            self.write_thermo_physic_transport(case_dir)
            self.write_momentum_transport(case_dir)
            self.write_g(case_dir)

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
    def fvschemes(self):
        if self._fvschemes is None:
            self._fvschemes = pkg_resources.read_text(static_fluid_resources, 'fvSchemes')
        return self._fvschemes

    @property
    def fvsolution(self):
        if self._fvsolution is None:
            self._fvsolution = pkg_resources.read_text(static_fluid_resources, 'fvSolution')
        return self._fvsolution

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
        entry = pkg_resources.read_text(fluid_resources, 'thermophysicalProperties')
        entry = entry.replace('<material_id>', self.txt_id)
        entry = entry.replace('<material_name>', str(self.name))
        entry = entry.replace('<molWeight>', str(self.mol_weight))
        entry = entry.replace('<rho>', str(self.density))
        entry = entry.replace('<cp>', str(self.specific_heat_capacity))
        entry = entry.replace('<mu>', str(self.mu))
        entry = entry.replace('<pr>', str(self.pr))
        return entry

    def generate_g_entry(self):
        entry = pkg_resources.read_text(fluid_resources, 'g')
        entry = entry.replace('<material_id>', self.txt_id)
        entry = entry.replace('<material_name>', str(self.name))
        return entry

    def generate_thermo_physic_transport_entry(self):
        entry = pkg_resources.read_text(fluid_resources, 'thermophysicalTransport')
        entry = entry.replace('<material_id>', self.txt_id)
        entry = entry.replace('<material_name>', str(self.name))
        entry = entry.replace('<prt>', str(self.prt))
        return entry

    def generate_momentum_transport_entry(self):
        entry = pkg_resources.read_text(fluid_resources, 'momentumTransport')
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
        full_filename = os.path.join(case_dir, 'constant', str(self.txt_id), 'turbulenceProperties')
        with open(full_filename, "w") as f:
            f.write(self.momentum_transport_entry)

    def write_to_of(self, case_dir):
        self.init_directory(case_dir),
        self.write_thermo_physical_properties(case_dir)
        self.write_thermo_physic_transport(case_dir)
        self.write_momentum_transport(case_dir)
        self.write_g(case_dir)
        self.write_decompose_par_dict(case_dir)
        self.write_fvschemes(case_dir)
        self.write_fvsolution(case_dir)

    def __repr__(self):
        return f'Fluid material {self.name} ({self.id})'


class Layer(object):

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', uuid.uuid4())
        self.name = kwargs.get('name', None)
        self.material = kwargs.get('material', None)
        self.thickness = kwargs.get('thickness', None)
        self.solid = None
        self.contains_pipe = kwargs.get('contains_pipe', False)

        self.initial_temperature = kwargs.get('initial_temperature', 293.15)
        self.meshes = kwargs.get('meshes', set())

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


def write_region_properties(cell_zones, case_dir):
    os.makedirs(os.path.join(case_dir, 'constant'), exist_ok=True)
    entry = pkg_resources.read_text(case_resources, 'regionProperties')

    solids = []
    fluids = []

    for cell_zone in cell_zones:
        if isinstance(cell_zone.material, Solid):
            solids.append(cell_zone.txt_id)
        elif isinstance(cell_zone.material, Fluid):
            fluids.append(cell_zone.txt_id)

    entry = entry.replace('<solids>', ' '.join(solids))
    entry = entry.replace('<fluids>', ' '.join(fluids))

    full_filename = os.path.join(case_dir, 'constant', 'regionProperties')
    with open(full_filename, "w") as f:
        f.write(entry)

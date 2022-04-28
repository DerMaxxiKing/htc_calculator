from itertools import count
from inspect import cleandoc
from copy import copy, deepcopy
from ...logger import logger
import uuid


class FOMetaMock(type):

    instances = []
    current_mesh = None

    def __call__(cls, *args, **kwargs):

        mesh = kwargs.get('mesh', None)
        if mesh is None:
            mesh = cls.current_mesh
            kwargs['mesh'] = mesh

        if not kwargs.get('create', False):
            obj = cls.get_fo()
        else:
            obj = None

        if obj is None:
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            cls.current_mesh.function_objects.append(obj)
            cls.current_mesh.function_object_ids[obj.id] = obj
        return obj

    def get_fo(cls):
        return None


class WallHeatFlux(object, metaclass=FOMetaMock):

    id_iter = count()

    template = cleandoc("""
    <func_name>
    {
        type        wallHeatFlux;
        libs        ("libfieldFunctionObjects.so");
        region      <cell_zone>;
        patches     ("<patch_names>");
    }
    """)

    def __init__(self, *args, **kwargs):

        self._txt_id = None
        self._cell_zone = None
        # self._dict_entry = None

        self.name = kwargs.get('name')
        self.id = kwargs.get('id', next(WallHeatFlux.id_iter))

        self.cell_zone = kwargs.get('cell_zone', None)
        self.patches = kwargs.get('patches', set())
        self.mesh = kwargs.get('mesh')

    @property
    def txt_id(self):
        if self._txt_id is None:
            if isinstance(self.id, uuid.UUID):
                self._txt_id = 'fo' + str(self.id.hex)
            else:
                self._txt_id = 'fo' + str(self.id)
        return self._txt_id

    @txt_id.setter
    def txt_id(self, value):
        self._txt_id = value

    @property
    def cell_zone(self):
        if self._cell_zone is None:
            self._cell_zone = self.patches.blocks[0].cell_zone
        return self._cell_zone

    @cell_zone.setter
    def cell_zone(self, value):
        self._cell_zone = value

    @property
    def dict_entry(self):
        content = deepcopy(self.template)
        content = content.replace('<func_name>', self.txt_id)
        content = content.replace('<cell_zone>', self.cell_zone.txt_id)
        content = content.replace('<patch_names>', ' '.join([x.txt_id + '_' + x.name for x in self.patches]))

        return content


class PressureDifferencePatch(object, metaclass=FOMetaMock):
    id_iter = count()

    template = cleandoc("""
        <func_name>
        {
            type            fieldValueDelta;
            libs            ("libfieldFunctionObjects.so");
            operation       subtract;
            writeControl    timeStep;
            writeInterval   1;
            log             true;
            region          <cell_zone1>;
    
            region1
            {
                type            surfaceFieldValue;
                writeControl    timeStep;
                writeInterval   1;
                writeFields     false;
                log             false;
                operation       areaAverage;
                fields          (p);
                regionType      patch;
                regionName      <cell_zone1>;
                name            <patch_1>;
            }
            
            region2
            {
                type            surfaceFieldValue;
                writeControl    timeStep;
                writeInterval   1;
                writeFields     false;
                log             false;
                operation       areaAverage;
                fields          (p);
                regionType      patch;
                regionName      <cell_zone1>;
                name            <patch_2>;
            }
        }
        """)

    def __init__(self, *args, **kwargs):
        self._txt_id = None
        # self._dict_entry = None

        self.name = kwargs.get('name')
        self.id = kwargs.get('id', next(WallHeatFlux.id_iter))

        self.cell_zone = kwargs.get('cell_zone', None)
        self.patch1 = kwargs.get('patch1', set())
        self.patch2 = kwargs.get('patch2', set())
        self.mesh = kwargs.get('mesh')

    @property
    def txt_id(self):
        if self._txt_id is None:
            if isinstance(self.id, uuid.UUID):
                self._txt_id = 'fo' + str(self.id.hex)
            else:
                self._txt_id = 'fo' + str(self.id)
        return self._txt_id

    @txt_id.setter
    def txt_id(self, value):
        self._txt_id = value

    @property
    def dict_entry(self):
        content = deepcopy(self.template)
        content = content.replace('<func_name>', self.txt_id)
        content = content.replace('<cell_zone1>', self.patch1.cell_zone.txt_id)
        content = content.replace('<cell_zone2>', self.patch2.cell_zone.txt_id)
        content = content.replace('<patch_1>', f'{self.patch1.txt_id}_{self.patch1.name}')
        content = content.replace('<patch_2>', f'{self.patch2.txt_id}_{self.patch2.name}')

        return content


class TemperatureDifferencePatch(object, metaclass=FOMetaMock):
    id_iter = count()

    template = cleandoc("""
        <func_name>
        {
            type            fieldValueDelta;
            libs            ("libfieldFunctionObjects.so");
            operation       subtract;
            writeControl    timeStep;
            writeInterval   1;
            log             true;
            region          <cell_zone1>;

            region1
            {
                type            surfaceFieldValue;
                writeControl    timeStep;
                writeInterval   1;
                writeFields     false;
                log             false;
                operation       areaAverage;
                fields          (T);
                regionType      patch;
                regionName      <cell_zone1>;
                name            <patch_1>;
            }

            region2
            {
                type            surfaceFieldValue;
                writeControl    timeStep;
                writeInterval   1;
                writeFields     false;
                log             false;
                operation       areaAverage;
                fields          (T);
                regionType      patch;
                regionName      <cell_zone1>;
                name            <patch_2>;
            }
        }
        """)

    def __init__(self, *args, **kwargs):
        self._txt_id = None
        # self._dict_entry = None

        self.name = kwargs.get('name')
        self.id = kwargs.get('id', next(WallHeatFlux.id_iter))

        self.cell_zone = kwargs.get('cell_zone', None)
        self.patch1 = kwargs.get('patch1', set())
        self.patch2 = kwargs.get('patch2', set())

        self.mesh = kwargs.get('mesh')

    @property
    def txt_id(self):
        if self._txt_id is None:
            if isinstance(self.id, uuid.UUID):
                self._txt_id = 'fo' + str(self.id.hex)
            else:
                self._txt_id = 'fo' + str(self.id)
        return self._txt_id

    @txt_id.setter
    def txt_id(self, value):
        self._txt_id = value

    @property
    def dict_entry(self):
        content = deepcopy(self.template)
        content = content.replace('<func_name>', self.txt_id)
        content = content.replace('<cell_zone1>', self.patch1.cell_zone.txt_id)
        content = content.replace('<cell_zone2>', self.patch2.cell_zone.txt_id)
        content = content.replace('<patch_1>', f'{self.patch1.txt_id}_{self.patch1.name}')
        content = content.replace('<patch_2>', f'{self.patch2.txt_id}_{self.patch2.name}')

        return content

from itertools import count
from inspect import cleandoc
from copy import deepcopy
import uuid


class FOMetaMock(type):

    instances = []

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        return obj


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
        # self._dict_entry = None

        self.name = kwargs.get('name')
        self.id = kwargs.get('id', next(WallHeatFlux.id_iter))

        self.cell_zone = kwargs.get('cell_zone', None)
        self.patches = kwargs.get('patches', set())

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
        self.patch1 = kwargs.get('patch1')
        self.patch2 = kwargs.get('patch2')

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
        self.patch1 = kwargs.get('patch1')
        self.patch2 = kwargs.get('patch2')

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

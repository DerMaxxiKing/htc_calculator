import os
from numpy import array, ndarray
import itertools
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from inspect import cleandoc

import numpy as np


class BCMetaMock(type):

    instances = []

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        return obj


class BCFile(object):

    default_value = 0
    field_template = ''
    type = ''
    default_entry = ''

    def __init__(self, *args, **kwargs):

        self._internal_field_value = None
        self.internal_field_value = kwargs.get('internal_field_value', self.default_value)
        self.patches = kwargs.get('patches', {})

        self._content = None

    @property
    def internal_field_value(self):
        if type(self._internal_field_value) in [int, float]:
            return 'uniform ' + str(self._internal_field_value)
        else:
            return self._internal_field_value

    @internal_field_value.setter
    def internal_field_value(self, value):
        self._internal_field_value = value

    @property
    def content(self):
        if self._content is None:
            template = deepcopy(self.field_template)
            template = template.replace('<internal_field_value>', str(self.internal_field_value))

            if (self.patches is None) or self.patches.__len__() == 0:
                template = template.replace('<patches>', self.default_entry)
            else:
                patches_str = ''
                for patch, value in self.patches.items():
                    try:
                        dict_entry = value.generate_dict_entry()
                        if dict_entry is not None:
                            patches_str = patches_str + f'{patch}\n' + value.generate_dict_entry() + '\n'
                    except Exception as e:
                        raise e
                template = template.replace('<patches>', patches_str)

            if type(self.internal_field_value) in [int, float, str]:
                template = template.replace('<value>', str(self.internal_field_value))
            elif isinstance(self.internal_field_value, ndarray):
                template = template.replace('<value>', f"uniform ({' '.join([str(x) for x in self.internal_field_value.tolist()])})")

            self._content = template
        return self._content

    def write(self, directory):
        with open(os.path.join(directory, f'{self.type}'), "w") as f:
            f.write(self.content)


class BoundaryCondition(object, metaclass=BCMetaMock):

    id_iter = itertools.count()

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name', 'Unnamed BoundaryCondition')
        self.id = next(BoundaryCondition.id_iter)

        self._value = None
        self._dict_entry = None

    @property
    def value(self):
        if self._value is None:
            return '$internalField'
        elif type(self._value) in [int, float]:
            return 'uniform ' + str(self._value)
        elif isinstance(self._value, ndarray):
            return f"uniform ({' '.join([str(x) for x in self._value.tolist()])})"
        else:
            return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def dict_entry(self):
        if self._dict_entry is None:
            self._dict_entry = self.generate_dict_entry()
        return self._dict_entry

    @abstractmethod
    def generate_dict_entry(self, *args, **kwargs):
        return ''


class Calculated(BoundaryCondition):

    template = cleandoc("""
    {
        type            calculated;
        value           <value>;
    }
    """)

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class FixedValue(BoundaryCondition):

    template = cleandoc("""
    {
        type            fixedValue;
        value           <value>;
    }
    """)

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class FixedGradient(BoundaryCondition):
    template = cleandoc("""
    {
        type            fixedGradient;
        gradient        <value>;
    }
    """)

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1fixedGradientFvPatchField.html
        This boundary condition supplies a fixed gradient condition, such that
        the patch values are calculated using:

        \f[
            x_p = x_c + \frac{\nabla(x)}{\Delta}
        \f]

        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('gradient')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class ZeroGradient(BoundaryCondition):

    template = cleandoc("""
        {
            type            zeroGradient;
        }
        """)

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1zeroGradientFvPatchField.html
        This boundary condition applies a zero-gradient condition from the patch internal field onto the patch faces.
        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.object = kwargs.get('object')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        return template


class InletOutlet(BoundaryCondition):

    template = cleandoc("""
    {
        type            inletOutlet;
        inletValue      <inlet_value>;
        value           <value>;
    }
    """)

    def __init__(self, *args, **kwargs):
        """
        # https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1inletOutletFvPatchField.html
        This boundary condition provides a generic outflow condition, with
        specified inflow for the case of return flow.
        :param args:
        :param kwargs: inletValue	Inlet value for reverse flow	yes
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self._inlet_value = None
        self.value = kwargs.get('value', 0)
        self.inlet_value = kwargs.get('inlet_value', 0)

    @property
    def inlet_value(self):
        if self._inlet_value is None:
            return '$internalField'
        elif type(self._inlet_value) in [int, float]:
            return 'uniform ' + str(self._inlet_value)
        elif isinstance(self._inlet_value, ndarray):
            return f"uniform ({' '.join([str(x) for x in self._inlet_value.tolist()])})"
        else:
            return self._inlet_value

    @inlet_value.setter
    def inlet_value(self, value):
        self._inlet_value = value

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        template = template.replace('<inlet_value>', str(self.inlet_value))
        return template


class CyclicAMI(BoundaryCondition):

    # template = cleandoc("""
    #     {
    #         type            cyclicAMI;
    #         value           <value>;
    #     }
    #     """)

    template = cleandoc("""
    {
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        value           $internalField;
        Tnbr            T;
    }
    """)

    def __init__(self, *args, **kwargs):
        """
        # https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1inletOutletFvPatchField.html
        This boundary condition provides a generic outflow condition, with
        specified inflow for the case of return flow.
        :param args:
        :param kwargs: inletValue	Inlet value for reverse flow	yes
        """
        BoundaryCondition.__init__(self, *args, **kwargs)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        return template

        # return None

        # template = deepcopy(self.template)
        # template = template.replace('<value>', str(self.value))
        # return template

import itertools
from abc import ABC, abstractmethod
from copy import deepcopy


class BCMetaMock(type):

    instances = []

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        cls.instances.append(obj)
        return obj


class BoundaryCondition(object, metaclass=BCMetaMock):

    id_iter = itertools.count()

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name', 'Unnamed BoundaryCondition')
        self.id = next(BoundaryCondition.id_iter)

        self._dict_entry = None

    @property
    def dict_entry(self):
        if self._dict_entry is None:
            self._dict_entry = self.generate_dict_entry()
        return self._dict_entry

    @abstractmethod
    def generate_dict_entry(self, *args, **kwargs):
        return ''


class Calculated(BoundaryCondition):

    template = """
    {
        type            calculated;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class FixedValue(BoundaryCondition):

    template = """
    {
        type            fixedValue;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class FixedGradient(BoundaryCondition):
    template = """
    {
        type            fixedGradient;
        gradient        uniform <value>;
    }
    """

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

    template = """
        {
            type            zeroGradient;
        }
        """

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

    template = """
    {
        type            inletOutlet;
        inletValue      uniform <inlet_value>;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        """
        # https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1inletOutletFvPatchField.html
        This boundary condition provides a generic outflow condition, with
        specified inflow for the case of return flow.
        :param args:
        :param kwargs: inletValue	Inlet value for reverse flow	yes
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value', '0')
        self.inlet_value = kwargs.get('inlet_value', '0')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        template = template.replace('<inlet_value>', str(self.inlet_value))
        return template

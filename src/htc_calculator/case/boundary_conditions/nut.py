from copy import deepcopy
from .base import BoundaryCondition

default_value = 0

field_template = """
/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  9
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      binary;
    class       volScalarField;
    location    "0/shell";
    object      nut;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform <internal_field_value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}


// ************************************************************************* //
"""


class NutkWallFunction(BoundaryCondition):

    template = """
    {
        type            nutkWallFunction;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1nutkWallFunctionFvPatchScalarField.html
        This boundary condition provides a wall constraint on the turbulent viscosity, i.e. nut,
        based on the turbulent kinetic energy, i.e. k, for for low- and high-Reynolds number turbulence models.
        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.object = 'nut'
        self.value = kwargs.get('value', default_value)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class NutkRoughWallFunction(BoundaryCondition):

    template = """
    {
        type            nutkRoughWallFunction;
        Ks              uniform <ks>;
        Cs              uniform <cs>;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1nutkRoughWallFunctionFvPatchScalarField.html
        This boundary condition provides a wall constraint on the turbulent viscosity, i.e. nut, when using wall functions for rough walls, based on the turbulent kinetic energy, i.e. k. The condition manipulates the wall roughness parameter, i.e. E, to account for roughness effects.

        Parameter ranges:

        roughness height = sand-grain roughness (0 for smooth walls)
        roughness constant = 0.5-1.0

        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.object = 'nut'
        self.ks = kwargs.get('ks', 100e-6)
        self.cs = kwargs.get('cs', 0.5)
        self.value = kwargs.get('value', default_value)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        template = template.replace('<ks>', str(self.ks))
        template = template.replace('<cs>', str(self.cs))
        return template

from copy import deepcopy
from .base import BoundaryCondition
from .base import BCFile
from inspect import cleandoc

default_value = 0

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
    format      binary;
    class       volScalarField;
    location    "0/shell";
    object      nut;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   <internal_field_value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}


// ************************************************************************* //
""")


class Nut(BCFile):
    default_value = default_value
    field_template = field_template
    type = 'nut'
    default_entry = cleandoc("""
                                                    ".*"
                                                    {
                                                        type            nutkWallFunction;
                                                        value           $internalField;
                                                    }
                                                    """)


class NutkWallFunction(BoundaryCondition):

    template = cleandoc("""
    {
        type            nutkWallFunction;
        value           <value>;
    }
    """)

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

    template = cleandoc("""
    {
        type            nutkRoughWallFunction;
        Ks              <ks>;
        Cs              <cs>;
        value           <value>;
    }
    """)

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
        self.ks = kwargs.get('ks', 1e-6)
        self.cs = kwargs.get('cs', 1e-6)
        self.value = kwargs.get('value', default_value)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        template = template.replace('<ks>', str(self.ks))
        template = template.replace('<cs>', str(self.cs))
        return template

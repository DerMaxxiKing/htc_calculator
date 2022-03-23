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
    format      ascii;
    class       volScalarField;
    location    "0/shell";
    object      p_rgh;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [ 1 -1 -2 0 0 0 0 ];

internalField   uniform <value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
"""


class FixedFluxPressure(BoundaryCondition):

    template = """
    {
        type            fixedFluxPressure;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        """
        # https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1fixedFluxPressureFvPatchScalarField.html
        This boundary condition sets the pressure gradient to the provided value such that the flux on the boundary
        is that specified by the velocity boundary condition.
        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.object = 'p_rgh'
        self.value = kwargs.get('value', default_value)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template

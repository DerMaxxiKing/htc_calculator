from copy import deepcopy
from .base import BoundaryCondition

default_value = 293.15

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
    object      T;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [ 0 0 0 1 0 0 0 ];

internalField   uniform <value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
"""


class TurbulentTemperatureCoupledBaffleMixed(BoundaryCondition):

    template = """
    {
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        value           uniform <value>;
        Tnbr            T;
    }
    """

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1compressible_1_1turbulentTemperatureCoupledBaffleMixedFvPatchScalarField.html

        Mixed boundary condition for temperature, to be used for heat-transfer on back-to-back baffles. Optional thin
        thermal layer resistances can be specified through thicknessLayers and kappaLayers entries.

        Specifies gradient and temperature such that the equations are the same on both sides:

        refGradient = zero gradient
        refValue = neighbour value
        mixFraction = nbrKDelta / (nbrKDelta + myKDelta())
        where KDelta is heat-transfer coefficient K * deltaCoeffs

        The thermal conductivity kappa can either be retrieved from various possible sources, as detailed in the
        class temperatureCoupledBase.

        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template

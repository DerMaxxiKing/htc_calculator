import os
from copy import deepcopy
from .base import BoundaryCondition
from .base import BCFile
from inspect import cleandoc

default_value = 0.0064879

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
    location    "0/shell";
    object      epsilon;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [ 0 2 -3 0 0 0 0 ];

internalField   <internal_field_value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
""")


class Epsilon(BCFile):
    default_value = default_value
    field_template = field_template
    type = 'epsilon'
    default_entry = cleandoc("""
                                                   ".*"
                                                   {
                                                       type            epsilonWallFunction;
                                                       value           $internalField;
                                                   }
                                               """)


class EpsilonWallFunction(BoundaryCondition):

    template = cleandoc("""
    {
        type            epsilonWallFunction;
        value           <value>;
    }
    """)

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value', 0.0064879)
        self.object = 'epsilon'

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class TurbulentMixingLengthDissipationRateInlet(BoundaryCondition):
    template = cleandoc("""
    {
        type            turbulentMixingLengthDissipationRateInlet;
        mixingLength    <mixing_length>;
        value           <value>;
    }
    """)

    def __init__(self, *args, **kwargs):
        """
        This boundary condition provides an inlet condition for turbulent kinetic
        energy dissipation rate, i.e. \c epsilon, based on a specified mixing
        length.  The patch values are calculated using:

             \f[
                 \epsilon_p = \frac{C_{\mu}^{0.75} k^{1.5}}{L}
             \f]

        where
           \epsilon_p | Patch epsilon values     [m2/s3]
           C_\mu      | Empirical model constant retrived from turbulence model
           k          | Turbulent kinetic energy [m2/s2]
           L          | Mixing length scale      [m]

        https://www.openfoam.com/documentation/guides/latest/api/turbulentMixingLengthDissipationRateInletFvPatchScalarField_8H_source.html
        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.mixing_length = kwargs.get('mixing_length', 0.008)
        self.value = kwargs.get('value', 0.0064879)
        self.object = 'epsilon'

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        template = template.replace('<mixing_length>', str(self.mixing_length))
        return template

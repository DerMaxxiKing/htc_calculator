from copy import deepcopy
from .base import BoundaryCondition

default_value = 0.00463812

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
    object      k;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

/*
r=0.008
mDot=0.05
rho=1000
I=0.05
L=r
Cmu=0.09

A=np.pi*r**2
V=mDot/A/rho
k=1.5*V**2*I
epsilon=Cmu**0.75*k**1.5/L
*/

dimensions      [ 0 2 -2 0 0 0 0 ];

internalField   uniform <internal_field_value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
"""


class TurbulentIntensityKineticEnergyInlet(BoundaryCondition):

    template = """
    {
        type            turbulentIntensityKineticEnergyInlet;
        intensity       <intensity>;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.object = 'k'
        self.value = kwargs.get('value', 0.00463812)
        self.intensity = kwargs.get('intensity', 0.05)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        template = template.replace('<intensity>', str(self.intensity))
        return template


class KqRWallFunction(BoundaryCondition):

    template = """
    {
        type            kqRWallFunction;
        value           uniform <value>;
    }
    """

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.object = 'k'
        self.value = kwargs.get('value', 0.00463812)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template

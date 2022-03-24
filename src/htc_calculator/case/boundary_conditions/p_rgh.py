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
    format      ascii;
    class       volScalarField;
    location    "0/shell";
    object      p_rgh;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [ 1 -1 -2 0 0 0 0 ];

internalField   <value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
""")


class PRgh(BCFile):
    default_value = default_value
    field_template = field_template
    type = 'p_rgh'
    default_entry = cleandoc("""
                                                    ".*"
                                                    {
                                                        type            fixedFluxPressure;
                                                        value           $internalField;
                                                    }
                                                    """)


class FixedFluxPressure(BoundaryCondition):

    template = cleandoc("""
    {
        type            fixedFluxPressure;
        value           <value>;
    }
    """)

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

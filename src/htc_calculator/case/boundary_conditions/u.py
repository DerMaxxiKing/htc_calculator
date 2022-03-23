import numpy as np
from copy import deepcopy
from .base import BoundaryCondition

default_value = np.array([0, 0, 0])

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
    class       volVectorField;
    location    "0/shell";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [ 0 1 -1 0 0 0 0 ];

internalField   uniform <value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
"""


class VolumeFlowRate(BoundaryCondition):

    template = """
    {
        type flowRateInletVelocity;
        volumetricFlowRate <flow_rate>;
        extrapolateProfile yes;
        value uniform (0 0 0);
    }
    """

    def __init__(self, *args, **kwargs):
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.flow_rate = kwargs.get('flow_rate')    # volume flow rate in m³/s
        self.object = 'U'

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<flow_rate>', str(self.flow_rate))
        return template


class NoSlip(BoundaryCondition):

    template = """
    {
        type            noSlip;
    }
    """

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1noSlipFvPatchVectorField.html#a8e6b556102b4ce936ac0e2ed9eb02ed8
        This boundary condition fixes the velocity to zero at walls.
        :param args:
        :param kwargs:
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.object = 'U'

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        return template

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
    object      p;
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

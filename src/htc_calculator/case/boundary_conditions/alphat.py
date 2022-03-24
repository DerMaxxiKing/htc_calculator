import os
from copy import deepcopy
from inspect import cleandoc
from .base import BCFile

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
    object      alphat;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -1 -1 0 0 0 0];

internalField   <internal_field_value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
""")


class Alphat(BCFile):

    default_value = default_value
    field_template = field_template
    type = 'alphat'
    default_entry = cleandoc("""
                                                   ".*"
                                                   {
                                                       type            calculated;
                                                       value           $internalField;
                                                   }
                                               """)

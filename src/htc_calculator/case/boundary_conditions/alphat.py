import os
from copy import deepcopy
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
    object      alphat;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -1 -1 0 0 0 0];

internalField   uniform <internal_field_value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
""")


class Alphat(object):

    def __init__(self, *args, **kwargs):

        self.internal_field_value = kwargs.get('internal_field_value', default_value)
        self.patches = kwargs.get('patches', None)

        self._content = None

    @property
    def content(self):
        if self._content is None:
            template = deepcopy(field_template)
            template = template.replace('<internal_field_value>', str(self.internal_field_value))

            if self.patches is None:
                template = template.replace('<patches>',
                                            cleandoc("""
                                                ".*"
                                                {
                                                    type            calculated;
                                                    value           $internalField;
                                                }
                                            """))
            self._content = template
        return self._content

    def write(self, directory):
        with open(os.path.join(directory, "alphat"), "w") as f:
            f.write(self.content)

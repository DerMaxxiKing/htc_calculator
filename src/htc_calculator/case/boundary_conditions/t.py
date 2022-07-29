from copy import deepcopy
from .base import BoundaryCondition
from .base import BCFile
from inspect import cleandoc

default_value = 293.15

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
    object      T;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [ 0 0 0 1 0 0 0 ];

internalField   <value>;

boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"

    <patches>
}

// ************************************************************************* //
""")


class T(BCFile):
    default_value = default_value
    field_template = field_template
    type = 'T'
    default_entry = cleandoc("""
                                                    ".*"
                                                    {
                                                        type            zeroGradient;
                                                    }
                                                    """)


class TurbulentTemperatureCoupledBaffleMixed(BoundaryCondition):

    template = cleandoc("""
    {
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        value           <value>;
        Tnbr            T;
        kappaMethod     <kappaMethod>;
        kappa           <kappa>;
    }
    """)

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
        self.kappa_method = kwargs.get('kappaMethod', 'lookup')
        self.kappa = kwargs.get('kappa', 'kappa')

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        template = template.replace('<kappaMethod>', str(self.kappa_method))
        template = template.replace('<kappa>', str(self.kappa))
        return template


class ExternalWallHeatFluxTemperature(BoundaryCondition):

    template = cleandoc("""
        {
            type            externalWallHeatFluxTemperature;
            mode            coefficient;
            Ta              uniform <t_a>;
            h               uniform <h>;
            value           <value>;
        }
        """)

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/externalWallHeatFluxTemperatureFvPatchScalarField_8H_source.html

        This boundary condition applies a heat flux condition to temperature
        on an external wall in one of three modes:

           - fixed power: supply Q
           - fixed heat flux: supply q
           - fixed heat transfer coefficient: supply h and Ta

         where:
         \vartable
             Q  | Power [W]
             q  | Heat flux [W/m^2]
             h  | Heat transfer coefficient [W/m^2/K]
             Ta | Ambient temperature [K]
         \endvartable

         For heat transfer coefficient mode optional thin thermal layer resistances
         can be specified through thicknessLayers and kappaLayers entries.

         The thermal conductivity \c kappa can either be retrieved from various
         possible sources, as detailed in the class temperatureCoupledBase.

         The ambient temperature Ta is specified as a Foam::Function1 of time but
         uniform in space.

        :param args:
        :param kwargs:

        h           | Heat transfer coefficient [W/m^2/K] | for mode 'coefficient' |
        Ta          | Ambient temperature [K]             | for mode 'coefficient' |
        """
        BoundaryCondition.__init__(self, *args, **kwargs)

        self.h = kwargs.get('h')    # Heat transfer coefficient [W/m^2/K]
        self.t_a = kwargs.get('t_a')  # Ambient temperature [K]
        self.value = kwargs.get('value', None)

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<h>', str(self.h))
        template = template.replace('<t_a>', str(self.t_a))
        return template


class ExternalWallHeatFlux(BoundaryCondition):

    template = cleandoc("""
        {
            type            externalWallHeatFluxTemperature;
            mode            flux;
            q               constant <value>;
        }
        """)

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/externalWallHeatFluxTemperatureFvPatchScalarField_8H_source.html

        This boundary condition applies a heat flux condition to temperature
        on an external wall in one of three modes:

           - fixed power: supply Q
           - fixed heat flux: supply q
           - fixed heat transfer coefficient: supply h and Ta

         where:
         \vartable
             Q  | Power [W]
             q  | Heat flux [W/m^2]
             h  | Heat transfer coefficient [W/m^2/K]
             Ta | Ambient temperature [K]
         \endvartable

         For heat transfer coefficient mode optional thin thermal layer resistances
         can be specified through thicknessLayers and kappaLayers entries.

         The thermal conductivity \c kappa can either be retrieved from various
         possible sources, as detailed in the class temperatureCoupledBase.

         The ambient temperature Ta is specified as a Foam::Function1 of time but
         uniform in space.

        :param args:
        :param kwargs:

        q           | Heat flux [W/m^2]           | for mode 'flux'     |
        """
        BoundaryCondition.__init__(self, *args, **kwargs)

        self.value = kwargs.get('value')    # Heat flux [W/m^2]

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template


class ExternalWallPower(BoundaryCondition):

    template = cleandoc("""
        {
            type            externalWallHeatFluxTemperature;
            mode            power;
            Q               constant <value>;
        }
        """)

    def __init__(self, *args, **kwargs):
        """
        https://www.openfoam.com/documentation/guides/latest/api/externalWallHeatFluxTemperatureFvPatchScalarField_8H_source.html

        This boundary condition applies a heat flux condition to temperature
        on an external wall in one of three modes:

           - fixed power: supply Q
           - fixed heat flux: supply q
           - fixed heat transfer coefficient: supply h and Ta

         where:
         variables
             Q  | Power [W]
             q  | Heat flux [W/m^2]
             h  | Heat transfer coefficient [W/m^2/K]
             Ta | Ambient temperature [K]

         For heat transfer coefficient mode optional thin thermal layer resistances
         can be specified through thicknessLayers and kappaLayers entries.

         The thermal conductivity \c kappa can either be retrieved from various
         possible sources, as detailed in the class temperatureCoupledBase.

         The ambient temperature Ta is specified as a Foam::Function1 of time but
         uniform in space.

        :param args:
        :param kwargs:

        q           | Heat flux [W/m^2]           | for mode 'flux'     |
        """
        BoundaryCondition.__init__(self, *args, **kwargs)
        self.value = kwargs.get('value')    # Power [W]

    def generate_dict_entry(self, *args, **kwargs):
        template = deepcopy(self.template)
        template = template.replace('<value>', str(self.value))
        return template

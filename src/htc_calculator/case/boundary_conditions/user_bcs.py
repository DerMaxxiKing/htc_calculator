from .epsilon import TurbulentMixingLengthDissipationRateInlet, EpsilonWallFunction
from .k import TurbulentIntensityKineticEnergyInlet, KqRWallFunction
from .base import Calculated, FixedValue, InletOutlet, ZeroGradient, CyclicAMI
from .p_rgh import FixedFluxPressure
from .u import VolumeFlowRate, PressureInletOutletVelocity, NoSlip
from .nut import NutkWallFunction
from .t import TurbulentTemperatureCoupledBaffleMixed, ExternalWallHeatFluxTemperature

of_field_name_lookup_dict = {'alphat': 'alphat',
                             'epsilon': 'epsilon',
                             'k': 'k',
                             'nut': 'nut',
                             'p': 'p',
                             'p_rgh': 'p_rgh',
                             't': 'T',
                             'u': 'U',
                             }


class UserBC(object):

    def __init__(self, *args, **kwargs):

        self._alphat = None
        self._epsilon = None
        self._k = None
        self._nut = None
        self._p = None
        self._p_rgh = None
        self._t = None
        self._u = None

    @property
    def alphat(self):
        return self._alphat

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def k(self):
        return self._k

    @property
    def nut(self):
        return self._nut

    @property
    def p(self):
        return self._p

    @property
    def p_rgh(self):
        return self._p_rgh

    @property
    def t(self):
        return self._t

    @property
    def u(self):
        return self._u


class VolumeFlowInlet(UserBC):

    def __init__(self, *args, **kwargs):

        UserBC.__init__(self, *args, **kwargs)

        self.flow_rate = kwargs.get('flow_rate', 4.1666e-5)
        self.inlet_temperature = kwargs.get('inlet_temperature', 273.15 + 50)

        self._alphat = Calculated(value='$internalField')
        self._epsilon = TurbulentMixingLengthDissipationRateInlet()
        self._k = TurbulentIntensityKineticEnergyInlet()
        self._nut = Calculated(value='$internalField')
        self._p = Calculated(value='$internalField')
        self._p_rgh = FixedFluxPressure(value='$internalField')
        self._t = FixedValue(value=self.inlet_temperature)
        self._u = VolumeFlowRate(flow_rate=self.flow_rate)


class Outlet(UserBC):

    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)

        self._alphat = Calculated(value='$internalField')
        self._epsilon = InletOutlet(value='$internalField', inlet_value='$internalField')
        self._k = InletOutlet(value='$internalField')
        self._nut = Calculated(value='$internalField')
        self._p = Calculated(value='$internalField')
        self._p_rgh = FixedValue(value='$internalField')
        self._t = InletOutlet(value='$internalField', inlet_value='$internalField')
        self._u = PressureInletOutletVelocity(value='$internalField')


class FluidWall(UserBC):

    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)

        self._alphat = Calculated(value='$internalField')
        self._epsilon = EpsilonWallFunction(value='$internalField')
        self._k = KqRWallFunction(value='$internalField')
        self._nut = NutkWallFunction(value='$internalField')
        self._p = Calculated(value='$internalField')
        self._p_rgh = FixedFluxPressure(value='$internalField')
        self._t = ZeroGradient()
        self._u = NoSlip()


class SolidWall(UserBC):
    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)
        self._t = ZeroGradient()


class SolidFluidInterface(UserBC):
    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)
        self._t = TurbulentTemperatureCoupledBaffleMixed(value='$internalField')


class FluidSolidInterface(UserBC):
    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)

        self._alphat = Calculated(value='$internalField')
        self._epsilon = EpsilonWallFunction(value='$internalField')
        self._k = KqRWallFunction(value='$internalField')
        self._nut = NutkWallFunction(value='$internalField')
        self._p = Calculated(value='$internalField')
        self._p_rgh = FixedFluxPressure(value='$internalField')
        self._t = TurbulentTemperatureCoupledBaffleMixed(value='$internalField')
        self._u = NoSlip()


class SolidConvection(UserBC):
    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)

        self.heat_transfer_coefficient = kwargs.get('heat_transfer_coefficient', 10)    # Heat transfer coefficient [W/m^2/K]
        self.ambient_temperature = kwargs.get('ambient_temperature', 293.15)            # Ambient temperature [K]

        self._t = ExternalWallHeatFluxTemperature(h=self.heat_transfer_coefficient,
                                                  t_a=self.ambient_temperature)


class SolidCyclicAMI(UserBC):

    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)

        self._t = CyclicAMI()


class FluidCyclicAMI(UserBC):

    def __init__(self, *args, **kwargs):
        UserBC.__init__(self, *args, **kwargs)

        self._alphat = CyclicAMI()
        self._epsilon = CyclicAMI()
        self._k = CyclicAMI()
        self._nut = CyclicAMI()
        self._p = CyclicAMI()
        self._p_rgh = CyclicAMI()
        self._t = CyclicAMI()
        self._u = CyclicAMI()

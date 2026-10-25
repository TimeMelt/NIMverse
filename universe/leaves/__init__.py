"""Physics+life leaves (v1 box models): System -> HabitatState -> World -> Chemistry -> Biosphere -> Organisms."""
from .params import LIFE_SPACE, LIFE_DEFAULTS, TRAITS
from .packets import (TideLink, HostWeather, HabitatState, ClimateStress, ChemInventory, AbiogenesisGate, NicheOffer,
                      CreaturePacket, BioSignFlux, LifeParamBus, ScoreBus, as_bus)
from .system_habitat import habitats_from_system, candidate
from .life_loop import life_loop, run_habitat, host_weather

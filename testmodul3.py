from classes import *
from thermo import *
from testmodul2 import *
from iterate_distillation_guess import GuessIterator
from Txy import Txy_diagram

feed = Stream(temperature = 872.6357768997816, pressure = 1.10325, flowrates = {"propane": 112.69165725036628, "propene": 0, "H2": 0, "H2O": 1126.9165725036628}, phase = "vapor")


reactor1 = Reactor(pressure = 1.10325, catalyst_density = 1120, catalyst_mass = None, conversion = 0.37392073365130446)
reactor2 = Reactor(pressure = 1.10325, catalyst_density = 1120, catalyst_mass = None, conversion = 0.25712596200313015)
reactor3 = Reactor(pressure = 1.10325, catalyst_density = 1120, catalyst_mass = None, conversion = 0.5699831305680484)

Txy_diagram(21.93838716277997)


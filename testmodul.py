from classes import *
from thermo import *
from testmodul2 import *
from iterate_distillation_guess import GuessIterator

feed = Stream(temperature = 298.15, pressure = 1.10325, flowrates = {"propane": 112.69165725036628, "propene": 0, "H2": 0, "H2O": 0}, phase = "vapor")


reactor1 = Reactor(pressure = 1.10325, catalyst_density = 1120, catalyst_mass = None, conversion = 0.37392073365130446)
reactor2 = Reactor(pressure = 1.10325, catalyst_density = 1120, catalyst_mass = None, conversion = 0.25712596200313015)
reactor3 = Reactor(pressure = 1.10325, catalyst_density = 1120, catalyst_mass = None, conversion = 0.5699831305680484)
oven = ProcessOven()
column = VectorizedDistillationColumn(pressure = 21.93838716277997, stages = 89, feed_stage = 73, reflux_ratio = 10.87257001024577, distillate_flowrate = 85.59627307479502)


stream1 = oven.run(feed, 872.6357768997816)
stream1.flowrates["H2O"] = 1126.9165725036628

stream2 = reactor1.run2(stream1)

stream3 = oven.run(stream2, 867.964705718116)

stream4 = reactor2.run2(stream3)

stream5 = oven.run(stream4, 936.510760863297)

stream6 = reactor3.run2(stream5)

stream6_water = stream6.flowrates["H2O"]
stream6_H2  =stream6.flowrates["H2"]

stream6.flowrates["H2O"] = 0
stream6.flowrates["H2"] = 0

stream6.temperature = 365.41363912862334

distillate, bottoms = column.run(stream6)

print(distillate)
print(bottoms)
print(reactor1.catalyst_mass, reactor1.diameter, reactor1.volume)
print(reactor2.catalyst_mass, reactor2.diameter, reactor2.volume)
print(reactor3.catalyst_mass, reactor3.diameter, reactor3.volume)
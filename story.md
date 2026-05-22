# GKT-designprojekt stories

## Scope 

GKT-designprojekt is a project focused on simulating a chemical production plant to produce propane from propene.
The main goal is to find the plant operating parameters which minimize the total plant cost while meeting the production requirements.


## Story 1: Write the optimizer

As a chemical process engineer, I want to write an optimizer that can find the optimal operating parameters for the specified chemical production plant, 
so that the total plant cost is minimized while meeting the production requirements.

All the classes can be found in the directory 'gkt_designprojekt/classes' and they are to be set up in the following way:

- A certain number (n) of reactors to produce propene and hydrogen gas from propane and inert water
- A process oven to heat up the stream that enters the first reactor and all the intermediate streams between them
- A flash tank to perform an isothermal flash to remove process water
- A flash tank to perform an isothermal flash to remove hydrogen gas
- A distillation column to separate the four chemical species in the liquid stream that leaves the second flash. The distillate should contain ca. 82 mol/s propene with at least 99.5 mol% purity
- Between any of the components, heat exchangers, pumps and compressors can be placed to control pressure and temperature
- It can be assumed that it is always possible to lower the pressure of a stream without a component

## Story 2: time complexity analysis of the optimizer

To further design the chemical plant simulation properly, it is crucial to understand the time complexity of the optimizer. 
The optimizer should be able to handle large-scale problems efficiently, and its performance should not degrade significantly with increasing problem size. 
Therefore, it is essential to analyze the time complexity of the optimizer and ensure that it can handle the required problem size within a reasonable time frame.
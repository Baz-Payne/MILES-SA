# MILES-SA

Mixed Integer Linear Energy Storage (MILES) is a project that aims to address the question of "how much energy storage does South Australia (SA) need to be 100% renewable".

See [] for the published version of the paper.

## Project
MILES uses mixed integer linear programming to determine the storage requirement for SA based on publicly provided data from the Australia Energy Market Operator.

The code provided in this repository is the source code used to develop the results presented in the accompanying paper.

## Islanded Copper Plate Grid
The islanded copper plate grid model is the idealised version of an electricity grid, whereby the connection of all generators and loads are facilitated by zero impedance conductors to a single node. This is the simplest model of SA and provided the foundation for the more complex, multi nodal grid.

## Multi Nodal Grid
The multi nodal grid model separates SA into 7 distinct nodes including 3 interconnectors.

Image here

## Getting Started
The two models are self contained. The main file in each allows you to run simulations.

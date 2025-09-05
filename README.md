# MILES-SA

Mixed Integer Linear Energy Storage (MILES) is a project that aims to address the question of <em>"how much energy storage does South Australia (SA) need to be 100% renewable"</em>.

See [Paper Link] for the published version of the paper.

## Project
MILES uses mixed integer linear programming to determine the storage requirement for SA based on publicly provided data from the Australia Energy Market Operator.

The code provided in this repository is the source code used to develop the results presented in the accompanying paper.

## Copper Plate Grid
The copper plate grid model is the idealised version of an electricity grid, whereby the connection of all generators and loads are facilitated by zero impedance conductors to a single node. This is the simplest model of SA and provided the foundation for the more complex, multi nodal grid.

## Multi Nodal Grid
The multi nodal grid model separates SA into 7 distinct nodes including 3 interconnectors. This is visually displayed below.

![SA Split](/Plots/SA_regional_transmission.png)

## Getting Started
The two models are self contained. The code files in each allows you to run simulations.

Folders labelled "RAW" contain data directly from AEMO that are unmodified.

## A Note On Solvers
The solver used in our work was Gurobi 11.0.3. However, we ran tests on other open source and commercial solvers. A full list of solvers and benchmarks are available in the [Solvers](/Solvers) folder.

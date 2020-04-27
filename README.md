# DBSTEP
DFT-based Steric Parameters 

Allows user to calculate steric parameters from `.xyz`, `.log`, Gaussian density `.cube` files

Calculate Sterimol parameters, %Buried Volume, \<Untitled_Sterimol_Scan\> parameters

#### Install 
- To run as a module (python -m dbstep), download this reposotory and install with ```python setup.py install```

#### Running 

- Execute the program with `python Dbstep.py file`

- or as a module with `python -m dbstep file`

#### now with *QSAR MODE*
when `--qsar` option is requested, several files will be generated with a probe atom (Ar by default) at each grid point for calculating interaction energies externally. If multiple files are given as input, all will have the same grid dimensions automatically set to 3Å larger than the max bounds, considering all molecules. 

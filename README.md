# DBSTEP
DFT-based Steric Parameters 

Allows user to calculate steric parameters from `.xyz`, `.log`, Gaussian density `.cube` files (greater support coming soon).

Calculate Sterimol parameters (L, Bmin, Bmax), %Buried Volume, Sterimol2Vec and Vol2Vec parameters

#### Install 
- To run as a module (python -m dbstep), download this reposotory and install with ```python setup.py install```

#### Running 

- Run as a command line module with: `python -m dbstep file`

- Run in a Python program by importing: `import dbstep.Dbstep as db` (example below)
```
    #Create DBSTEP object
    mol = db.dbstep(file,center=atom1,ligand=atom2,commandline=True,verbose=True,sterimol='classic')  
    
    #Grab Sterimol Parameters
    L = mol.L
    Bmin = mol.Bmin
    Bmax = mol.Bmax
```

### Use
DBSTEP currently takes a coordinate file (.xyz or Gaussian output) and two reference atoms for steric measurement. By default, Sterimol parameters are output to the user, volume parameters can be requested with the `--volume` command. Atoms are specified by referring to the element and number in a coordinate file, (ex: "C2", where the carbon specified is the second atom in the file).

For Sterimol parameters, two atoms need to be specified using the arguments `--center [atom1]` and `--ligand [atom2]`. The L parameter is measured starting from the center atom coordinates, extending through the  atom1-atom2 axis until the end of the molecule is reached. The Bmin and Bmax parameters are measured on the axis perpendicular to L. 

For buried volume parameters, only the `--center [atom]` argument is necessary to define. 

If no atoms are specified, the first two atoms in the file will be used as reference. 


### Examples
Examples for obtaining Sterimol, Sterimol2Vec, Percent Buried Volume and Vol2Vec parameter sets are shown below (all example files found in examples/ directory).

1. Sterimol Parameters for Ethane
    
    Obtain the Sterimol parameters for an ethane molecule along the C-C bond on the command line:
```
>>>python -m dbstep examples/Et.xyz --center C2 --ligand C5

     Et.xyz / Bmin:  1.70 / Bmax:  3.25 / L:  2.15
```
    where Et.xyz looks like: 
```
8
ethane
H	0.00	0.00	0.00
C	0.00	0.00    -1.10
H	-1.00	0.27	-1.47
H	0.27	-1.00	-1.47
C	1.03	1.03	-1.61
H	1.03	1.03	-2.71
H	2.03	0.76	-1.25
H	0.76	2.03	-1.25
```

2. Sterimol2Vec Parameters for Ph

    The `--scan` argument is formatted as `rmin:rmax:interval` where rmin is the distance from the center along the L axis to start measurements, rmax dictates when to stop measurements, and interval is the frequency of measurements. In this case the length of the molecule (~6A) is measured in 1.0A intervals
```
>>>python -m dbstep examples/Ph.xyz --center C2 --ligand C5 --scan 0.0:6.0:1.0

     Ph.xyz / R:  0.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  1.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  2.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  3.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  4.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  5.00 / Bmin:  1.65 / Bmax:  3.11 
     Ph.xyz / R:  6.00 / Bmin:  1.15 / Bmax:  1.17 
```
 
 3. Percent Buried Volume 
 
    %Vb is measured by constructing a sphere (typically with a 3.5A radius) around the center atom and measuring how much of the sphere is occupied by the molecule. Output will include the sphere radius, percent buried volume, percent buried shell volume (zero in all cases unless a scan is being done simultaneously), and Sterimol parameters.
 ```
 >>>python -m dbstep examples/1Nap.xyz --center C2 --volume

      R/Å     %V_Bur     %S_Bur       Bmin       Bmax          L
     3.50      41.77       0.00       1.91       5.26       2.55
 ```
 
 4. Vol2Vec Parameters
 
    When invoking the --volume and --scan parameters simultaneously, vol2vec parameters can be obtained. In this case, a scan is performed using spheres with radii from 2.0A to 4.0A in 0.5A increments. 
 ```
 >>>python -m dbstep examples/CHiPr2.xyz --center C2 --volume --scan 2.0:4.0:0.5

      R/Å     %V_Bur     %S_Bur       Bmin       Bmax          L
     2.00      89.45      64.59       1.42       4.17       2.50
     2.50      73.56      53.69       0.63       3.63       2.95
     3.00      63.83      44.40       0.02       2.98       2.95
     3.50      53.41      26.75       0.00       0.00       0.00
     4.00      41.61       9.59       0.00       0.00       0.00
 ```
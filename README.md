![DBSTEP](DBSTEP_banner.png)
===

# DBSTEP
DFT-based Steric Parameters 

Allows a user to compute steric parameters from chemical structures. 

Calculate Sterimol parameters<sup>1</sup> (L, Bmin, Bmax), %Buried Volume<sup>2</sup>, Sterimol2Vec and Vol2Vec parameters

#### Install 
- To run as a module (python -m dbstep), download this reposotory and install with ```python setup.py install```

#### Running 
File parsing is done by the [cclib module](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.20823), which can parse many common quantum chemistry output files along with other common chemical structure file formats. For a full list of acceptable file types, see their documentation [here](https://cclib.github.io/).

To execute the program:
- Run as a command line module with: `python -m dbstep file --atom1 a1idx --atom2 a2idx`

- Run in a Python program by importing: `import dbstep.Dbstep as db` (example below)
```
    #Create DBSTEP object
    mol = db.dbstep(file,atom1=atom1,atom2=atom2,commandline=True,verbose=True,sterimol=True,measure='classic')  
    
    #Grab Sterimol Parameters
    L = mol.L
    Bmin = mol.Bmin
    Bmax = mol.Bmax
```

### Use
DBSTEP currently takes a coordinate file (.xyz or Gaussian output) and two reference atoms for steric measurement.  Sterimol parameters are output to the user using the `--sterimol` argument, volume parameters can be requested with the `--volume` command. Atoms are specified by referring to the index of an atom in a coordinate file, (ex: "2", referencing the second atom in the file, with indexing starting at 1).

For Sterimol parameters, two atoms need to be specified using the arguments `--atom1 [atom1idx]` and `--atom2 [atom2idx]`. The L parameter is measured starting from the specified atom1 coordinates, extending through the  atom1-atom2 axis until the end of the molecule is reached. The Bmin and Bmax parameters are measured on the axis perpendicular to L. 

For buried volume parameters, only the `--atom1 [atom]` argument is necessary to define. 

If no atoms are specified, the first two atoms in the file will be used as reference. 

### Dependencies
Non-standard dependencies will be installed upon installing DBSTEP, but include [numpy](https://numpy.org/), [numba](https://numba.pydata.org/), [scipy](https://www.scipy.org/), and [cclib](https://cclib.github.io/).

### Examples
Examples for obtaining Sterimol, Sterimol2Vec, Percent Buried Volume and Vol2Vec parameter sets are shown below (all example files found in examples/ directory).

1. Sterimol Parameters for Ethane
    
    Obtain the Sterimol parameters for an ethane molecule along the C2-C5 bond on the command line:
```
>>>python -m dbstep examples/Et.xyz  --sterimol --atom1 2 --atom2 5

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
>>>python -m dbstep examples/Ph.xyz --sterimol --atom1 2 --atom2 5 --scan 0.0:6.0:1.0

     Ph.xyz / R:  0.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  1.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  2.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  3.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  4.00 / Bmin:  1.65 / Bmax:  3.16 
     Ph.xyz / R:  5.00 / Bmin:  1.65 / Bmax:  3.11 
     Ph.xyz / R:  6.00 / Bmin:  1.15 / Bmax:  1.17 
```
 
 3. Percent Buried Volume 
 
    %Vb is measured by constructing a sphere (typically with a 3.5A radius) around the center atom and measuring how much of the sphere is occupied by the molecule. Output will include the sphere radius, percent buried volume (%V_Bur) and percent buried shell volume (%S_Bur) (zero in all cases unless a scan is being done simultaneously).
 ```
 >>>python -m dbstep examples/1Nap.xyz --atom1 2 --volume

      R/Å     %V_Bur     %S_Bur
     3.50      41.77       0.00
 ```
 
 4. Vol2Vec Parameters
 
    When invoking the --volume and --scan parameters simultaneously, vol2vec parameters can be obtained. In this case, a scan is performed using spheres with radii from 2.0A to 4.0A in 0.5A increments. 
 ```
 >>>python -m dbstep examples/CHiPr2.xyz --atom2 2 --volume --scan 2.0:4.0:0.5

      R/Å     %V_Bur     %S_Bur
     2.00      89.45      64.59 
     2.50      73.56      53.69
     3.00      63.83      44.40
     3.50      53.41      26.75
     4.00      41.61       9.59
 ```
 
 ### Acknowledgements
 
  This work is supported by the [NSF Center for Computer-Assisted Synthesis](https://ccas.nd.edu/), grant number [CHE-1925607](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1925607&HistoricalAwards=false)
  
  <img src="https://www.nsf.gov/images/logos/NSF_4-Color_bitmap_Logo.png" width="50" height="50">
  <img src="https://pbs.twimg.com/profile_images/1168617043106521088/SOLQaZ8M_400x400.jpg" width="50" height="50"> 
  
 ### References
 
 1. Verloop, A., Drug Design. Ariens, E. J., Ed. Academic   Press: New York,, **1976**; Vol. III
 2. Hillier, A. C.;  Sommer, W. J.;  Yong, B. S.;  Petersen, J. L.;  Cavallo, L.; Nolan, S. P., *Organometallics* **2003**, *22*, 4322-4326.

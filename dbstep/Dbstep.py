# -*- coding: UTF-8 -*-
from __future__ import print_function, absolute_import

###############################################################
# known issues:
# Hard - check the numerical results for some toy systems 
	# (e.g. spherically symmetrical, diatomics) where the correct values can be defined manually. 
	# Then check against tabulate results for classical values, then compare QM-density derived values
# Moderately trivial - if you remove Hs, the base atom ID messes up
# for grid sterimol, may be a problem that the value for Bmin slips between grid points to give an unusually low
	# add a check/warning for this (warning, Bmin is unusually small)
	# shoudn't be smaller than base atom VDW radius 
	

#for debug mode, a grid can be displayed using a pptk 3d graph, install with pip
###############################################################

#Python Libraries
import os, sys, time, shutil
from glob import glob
import numpy as np
from optparse import OptionParser

from dbstep import sterics, getdata, calculator, writer

#Chemistry Arrays

#Bondi Van der Waals radii taken from [J. Phys. Chem. 1964, 68, 441] & [J. Phys. Chem. A. 2009, 103, 5806-5812]
# All other elements set to 2.0A - common in other chemistry programs
bondi = {"Bq": 0.00, "H": 1.09,"He": 1.40,
	"Li":1.81,"Be":1.53,"B":1.92,"C":1.70,"N":1.55,"O":1.52,"F":1.47,"Ne":1.54,
	"Na":2.27,"Mg":1.73,"Al":1.84,"Si":2.10,"P":1.80,"S":1.80,"Cl":1.75,"Ar":1.88,
	"K":2.75,"Ca":2.31,"Ni": 1.63,"Cu":1.40,"Zn":1.39,"Ga":1.87,"Ge":2.11,"As":1.85,"Se":1.90,"Br":1.83,"Kr":2.02,
	"Rb":3.03,"Sr":2.49,"Pd": 1.63,"Ag":1.72,"Cd":1.58,"In":1.93,"Sn":2.17,"Sb":2.06,"Te":2.06,"I":1.98,"Xe":2.16,
	"Cs":3.43,"Ba":2.68,"Pt":1.72,"Au":1.66,"Hg":1.55,"Tl":1.96,"Pb":2.02,"Bi":2.07,"Po":1.97,"At":2.02,"Rn":2.20,
	"Fr":3.48,"Ra":2.83, "U":1.86 }

periodictable = ["","H","He","Li","Be","B","C","N","O","F","Ne",
	"Na","Mg","Al","Si","P","S","Cl","Ar",
	"K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
	"Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
	"Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
	"Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]

metals = ["Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Rb","Sr","Y","Zr","Nb","Mo",
	"Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
	"Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf",
	"Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Uut","Fl","Uup","Lv"]

isovals = {"Bq": 0.00, "H": .00475}
BOHR_TO_ANG = 0.529177249


class dbstep:
	"""
	dbstep object that contains coordinates, steric data

	Objects that can currently be referenced are:
			L, Bmax, Bmin, bur_vol, bur_shell
			setup_time, calc_time

	If steric scan is requested, Bmin and Bmax variables
	contain lists of params along scan
	"""
	def __init__(self, *args, **kwargs):
		self.file = args[0]
		if 'options' in kwargs:
			self.options = kwargs['options']
		else:
			self.options = set_options(kwargs)
		if 'QSAR' in kwargs:
			QSAR = kwargs['QSAR']
		else: QSAR = False

		#this is ugly but fix l8r
		file = self.file
		options = self.options

		start = time.time()
		spheres, cylinders = [], []
		name, ext = os.path.splitext(file)
		r_intervals, origin = 1, np.array([0,0,0])

		# if noH is requested these atoms are skipped to make things go faster
		if ext == '.xyz' or ext == '.log':
			options.surface = 'vdw'
			mol = getdata.GetXYZData(name,ext, options.noH,options.spec_atom_1, options.spec_atom_2)
			if options.noH:
				options.spec_atom_1 = mol.spec_atom_1
				options.spec_atom_2 = mol.spec_atom_2
			if len(mol.ATOMTYPES) == 1:
				sys.exit("Only one atom found in "+self.file+" - Please try again with a different input file.")
		if ext == '.cube':
			mol = getdata.GetCubeData(name)

		# if atoms are not specified to align to, grab first and second atom in
		if options.spec_atom_1 == False:
			options.spec_atom_1 = mol.ATOMTYPES[0]+str(1)
		if options.spec_atom_2 == False:
			try:
				options.spec_atom_2 = mol.ATOMTYPES[1]+str(2)
			except IndexError as error:
				pass
				
		if options.vshell: options.volume = True
		
		if options.qsar: 
			if options.grid < 0.5: 
				options.grid = 0.5
				if options.verbose:
					print("   Adjusting grid spacing to 0.5A for QSAR analysis")
					
		# if surface = VDW the molecular volume is defined by tabulated radii
		# This is necessary when a density cube is not supplied
		# if surface = Density the molecular volume is defined by an isodensity surface from a cube file
		# This is the default when a density cube is supplied although it can be over-ridden at the command prompt
		if options.verbose ==True: print("\n   {} will be analyzed using the {} surface".format(file, options.surface))

		#surfaces can either be formed from Van der Waals (bondi) radii (=vdw) or cube densities (=density)
		if options.surface == 'vdw':
			# generate Bondi radii from atom types
			try:
				mol.RADII = [bondi[atom] for atom in mol.ATOMTYPES]
				if options.verbose ==True: print("   Defining the molecule with Bondi atomic radii scaled by {}".format(options.SCALE_VDW))
			except:
				mol.RADII = []
				for atom in mol.ATOMTYPES:
					if atom not in periodictable:
						print("\n   UNABLE TO GENERATE VDW RADII FOR ATOM: ", atom); exit()
					elif atom not in bondi:
						mol.RADII.append(2.0)
					else:
						mol.RADII.append(bondi[atom])
			# scale radii by a factor
			mol.RADII = np.array(mol.RADII) * options.SCALE_VDW
		elif options.surface == 'density':
			if hasattr(mol, 'DENSITY'):
				mol.DENSITY = np.array(mol.DENSITY)
				if options.verbose: print("\n   Read cube file {} containing {} points".format(file, mol.xdim * mol.ydim * mol.zdim))
				[x_min, y_min, z_min] = np.array(mol.ORIGIN)
				[x_max, y_max, z_max] = np.array(mol.ORIGIN) + np.array([(mol.xdim-1)* mol.SPACING, (mol.ydim-1) * mol.SPACING, (mol.zdim-1) * mol.SPACING])
				xyz_max = max(x_max, y_max, z_max, abs(x_min), abs(y_min), abs(z_min))
				# overrides grid settings
				options.grid = mol.SPACING
			else:
				print("   UNABLE TO READ DENSITY CUBE"); exit()
		else:
			print("   Requested surface {} is not currently implemented. Try either vdw or density".format(options.surface)); exit()

		# Translate molecule to place metal or specified atom at the origin
		if options.surface == 'vdw':
			mol.CARTESIANS = calculator.translate_mol(mol, options, origin)
		elif options.surface == 'density':
			[mol.CARTESIANS,mol.ORIGIN, x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = calculator.translate_dens(mol, options, x_min, x_max, y_min, y_max, z_min, z_max, xyz_max, origin)

		# Check if we want to calculate parameters for mono- bi- or tridentate ligand
		spec_atom_2 = ''
		if len(options.spec_atom_2) > 1 and isinstance(options.spec_atom_2,list):
			spec_atom_2 = ','
		options.spec_atom_2 = spec_atom_2.join(options.spec_atom_2)
		options.spec_atom_2 = options.spec_atom_2.split(',')
		if len(options.spec_atom_2) == 1:
			# mono - obtain coords of atom to align along z axis
			point,p_id=0.0,0
			for n, atom in enumerate(mol.ATOMTYPES):
				if atom+str(n+1) == options.spec_atom_2[0]:
					p_id = n
					point = mol.CARTESIANS[p_id]

		# bi - obtain coords of point perpendicular to vector connecting ligands
		elif len(options.spec_atom_2) == 2: point = calculator.bidentate(mol, options)
		# tri - obtain coords of point perpendicular to plane connecting ligands
		elif len(options.spec_atom_2) == 3: point = calculator.tridentate(mol, options)

		# Rotate the molecule about the origin to align the metal-ligand bond along the (positive) Z-axis
		# the x and y directions are arbitrary
		if len(mol.CARTESIANS) > 1 and options.norot == False:
			if options.surface == 'vdw':
				mol.CARTESIANS = calculator.rotate_mol(mol.CARTESIANS, mol.ATOMTYPES, options.spec_atom_1, point, options)
			elif options.surface == 'density':
				mol.CARTESIANS, mol.INCREMENTS = calculator.rotate_mol(mol.CARTESIANS, mol.ATOMTYPES, options.spec_atom_1,  point, options, cube_origin=mol.ORIGIN, cube_inc=mol.INCREMENTS)

		# remove metals from the steric analysis. This is done by default and can be switched off by --addmetals
		# This can't be done for densities
		if options.surface == 'vdw':
			for i, atom in enumerate(mol.ATOMTYPES):
				if atom in metals and options.add_metals == False:
					mol.ATOMTYPES = np.delete(mol.ATOMTYPES,i)
					mol.CARTESIANS = np.delete(mol.CARTESIANS,i, axis=0)
					mol.RADII = np.delete(mol.RADII,i)

			# Find maximum horizontal and vertical directions (coordinates + vdw) in which the molecule is fully contained
			# First remove any atoms that have been requested to be removed from the analysis
			if options.exclude != False:
				del_atom_list = [int(atom) for atom in options.exclude.split(',')]
				for del_atom in sorted(del_atom_list, reverse=True):
					try:
						mol.ATOMTYPES = np.delete(mol.ATOMTYPES,del_atom-1)
						mol.CARTESIANS = np.delete(mol.CARTESIANS,del_atom-1, axis=0)
						mol.RADII = np.delete(mol.RADII,del_atom-1)
					except:
						print("   WARNING! Unable to remove the atoms requested")
			
			#determine grid size based on molecule vdw radii and radius selected for buried vol
			[x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = sterics.max_dim(mol.CARTESIANS, mol.RADII, options)
			if options.gridsize != False:
				if options.verbose: print("   Grid sizing requested: "+str(options.gridsize))
		if QSAR:
			self.dimensions = [x_min, x_max, y_min, y_max, z_min, z_max]
			return
		# read the requested radius or range
		if not options.scan:
			r_min, r_max, strip_width = options.radius, options.radius, 0.0
		else:
			try:
				[r_min, r_max, strip_width] = [float(scan) for scan in options.scan.split(':')]
				r_intervals += int((r_max - r_min) / strip_width)
			except:
				print("   Can't read your scan request. Try something like --scan 3:5:0.25"); exit()

		# Iterate over the grid points to see whether this is within VDW radius of any atom(s)
		# Grid point occupancy is either yes/no (1/0)
		# To save time this is currently done using a cuboid rather than cubic shaped-grid
		if options.surface == 'vdw':
			# user can choose to increase grid size / use in QSAR studies
			if options.gridsize != False:
				[x_minus, x_plus, y_minus, y_plus, z_minus, z_plus] = [float(val) for val in options.gridsize.replace(':',',').split(',')]
				sizeflag = True
				if x_plus < x_max or x_minus > x_min:
					sizeflag = False
				elif y_plus < y_max or y_minus > y_min:
					sizeflag = False
				elif z_plus < z_max or z_minus > z_min:
					sizeflag = False
				if sizeflag:
					n_x_vals = int(1 + round((x_plus - x_minus) / options.grid))
					n_y_vals = int(1 + round((y_plus - y_minus)/ options.grid))
					n_z_vals = int(1 + round((z_plus - z_minus) / options.grid))
					x_vals = np.linspace(x_minus, x_plus, n_x_vals)
					y_vals = np.linspace(y_minus, y_plus, n_y_vals)
					z_vals = np.linspace(z_minus, z_plus, n_z_vals)
				else:
					#sys exit
					sys.exit("ERROR: Your molecule is larger than the gridsize you selected,\n"
						"       please try again with a larger gridsize")
			else:
				n_x_vals = int(1 + round((x_max - x_min) / options.grid))
				n_y_vals = int(1 + round((y_max - y_min) / options.grid))
				n_z_vals = int(1 + round((z_max - z_min) / options.grid))
				x_vals = np.linspace(x_min, x_max, n_x_vals)
				y_vals = np.linspace(y_min, y_max, n_y_vals)
				z_vals = np.linspace(z_min, z_max, n_z_vals)
			
			if options.volume or options.sterimol == 'grid':
				# construct grid encapsulating molecule
				grid = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1,3)
				# compute which grid points occupy molecule
				if options.qsar:
					occ_grid, unocc_grid, point_tree = sterics.occupied(grid, mol.CARTESIANS, mol.RADII, origin, options)
				else:
					occ_grid, point_tree = sterics.occupied(grid, mol.CARTESIANS, mol.RADII, origin, options)

			if options.qsar:
				if options.verbose: print("\n   Creating interaction energy grid xyz files in 'grid_"+name+"' directory")
				probe = 'Ar'
				path = os.getcwd()+'/grid_'+name+'/'
				self.qsar_dir = path
				if os.path.exists(path):
					if options.verbose: print("   Overwriting: "+path)
					shutil.rmtree(path)
				os.mkdir(path)
				
				self.interaction_energy = []
				self.unocc_grid = unocc_grid
				
				for n, gridpoint in enumerate(unocc_grid):
					self.interaction_energy.append(0.0)
					xyzfile = open(path+'GRIDPOINT_'+probe+'_'+str(n)+'.xyz', 'w')
					xyzfile.write(str(len(mol.ATOMTYPES)+1)+'\n')
					xyzfile.write(path+'GRIDPOINT_'+probe+'_'+str(n)+'\n')

					for i, atom in enumerate(mol.ATOMTYPES):
						[x,y,z] = mol.CARTESIANS[i]
						[gx,gy,gz] = gridpoint
						xyzfile.write('{} {:10.5f} {:10.5f} {:10.5f}\n'.format(mol.ATOMTYPES[i], x,y,z))
					xyzfile.write('{} {:10.5f} {:10.5f} {:10.5f}\n'.format(probe, gx,gy,gz))

				xyzfile = open(path+'REF_'+probe+'.xyz', 'w')
				xyzfile.write(str(len(mol.ATOMTYPES)+1)+'\n')
				xyzfile.write('REF_'+probe+'\n')
				for i, atom in enumerate(mol.ATOMTYPES):
					[x,y,z] = mol.CARTESIANS[i]
					[gx,gy,gz] = gridpoint

					xyzfile.write('{} {:10.5f} {:10.5f} {:10.5f}\n'.format(mol.ATOMTYPES[i], x,y,z))
				xyzfile.write('{} {:10.5f} {:10.5f} {:10.5f}\n'.format(probe,gx+100,gy+100,gz+100))

		elif options.surface == 'density':
			x_vals = np.linspace(x_min, x_max, mol.xdim)
			y_vals = np.linspace(y_min, y_max, mol.ydim)
			z_vals = np.linspace(z_min, z_max, mol.zdim)
			# writes a new grid to cube file
			writer.WriteCubeData(name, mol)
			# define the grid points containing the molecule
			grid = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1,3)
			# compute occupancy based on isodensity value applied to cube and remove points where there is no molecule
			occ_grid = sterics.occupied_dens(grid, mol.DENSITY, options)
			
			#readjust sizing of grid to fit sphere
			if options.volume:
				grid = sterics.resize_grid(x_max,y_max,z_max,x_min,y_min,z_min,options,mol)
				
		# Set up done so note the time
		setup_time = time.time() - start

		# get buried volume at different radii
		if options.verbose: print("\n   Sterimol parameters will be generated in {} mode for {}\n".format(options.sterimol, file))

		if options.volume:
			print("   {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}".format("R/Å", "%V_Bur", "%S_Bur", "Bmin", "Bmax", "L"))

		if options.scand != False:
			# obtain L to get distributed intervals
			if options.sterimol == 'grid':
				L, Bmax, Bmin, cyl = sterics.get_cube_sterimol(occ_grid, 3.5, options.grid, strip_width)
			elif options.sterimol == 'classic':
				if options.surface == 'vdw':
					L, Bmax, Bmin, cyl = sterics.get_classic_sterimol(mol.CARTESIANS, mol.RADII,mol.ATOMTYPES)
				elif options.surface == 'density':
					print("   Can't use classic Sterimol with the isodensity surface. Either use VDW radii (--surface vdw) or use grid Sterimol (--sterimol grid)"); exit()
			#set interval based on L
			r_min = 0
			r_max = L
			r_intervals = int(options.scand)
			strip_width = L / float(options.scand)
		Bmin_list = []
		Bmax_list = []
		for rad in np.linspace(r_min, r_max, r_intervals):
			# The buried volume is defined in terms of occupied voxels.
			# If a scan is requested, radius of sphere = rad
			if options.volume:
				if rad == 0:
					bur_vol, bur_shell = 0.0,0.0
				else:
					if options.vshell: strip_width = options.vshell
					bur_vol, bur_shell = sterics.buried_vol(occ_grid, point_tree, origin, rad, strip_width, options)
			# Sterimol parameters can be obtained from VDW radii (classic) or from occupied voxels (new=default)
			if options.sterimol == 'grid':
				L, Bmax, Bmin, cyl = sterics.get_cube_sterimol(occ_grid, rad, options.grid, strip_width)
			elif options.sterimol == 'classic':
				if options.surface == 'vdw':
					L, Bmax, Bmin, cyl = sterics.get_classic_sterimol(mol.CARTESIANS, mol.RADII,mol.ATOMTYPES)
				elif options.surface == 'density':
					print("   Can't use classic Sterimol with the isodensity surface. Either use VDW radii (--surface vdw) or use grid Sterimol (--sterimol grid)"); exit()
			Bmin_list.append(Bmin)
			Bmax_list.append(Bmax)
			# Tabulate result
			if options.volume:
				# for pymol visualization
				spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f}".format(rad))
				print("   {:6.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(rad, bur_vol, bur_shell, Bmin, Bmax, L))
			else:
				if not options.scan:
					print("   {} / Bmin: {:5.2f} / Bmax: {:5.2f} / L: {:5.2f}".format(file, Bmin, Bmax, L))
				else:
					print("   {} / R: {:5.2f} / Bmin: {:5.2f} / Bmax: {:5.2f} ".format(file, rad, Bmin, Bmax))

			# for pymol visualization
			for c in cyl:
				cylinders.append(c)

		#for module reference
		self.L = L
		if options.volume:
			self.bur_vol = bur_vol
			self.bur_shell = bur_shell
		if options.scan == False and options.scand == False:
			self.Bmax = Bmax
			self.Bmin = Bmin
		else:
			self.Bmax = Bmax_list
			self.Bmin = Bmin_list

		# recompute L if a scan has been performed
		if options.sterimol == 'grid' and r_intervals >1:
			L, Bmax, Bmin, cyl = sterics.get_cube_sterimol(occ_grid, rad, options.grid, 0.0)
			print('\n   L parameter is {:5.2f} Ang'.format(L))
		cylinders.append('   CYLINDER, 0., 0., 0., 0., 0., {:5.3f}, 0.1, 1.0, 1.0, 1.0, 0., 0.0, 1.0,'.format(L))
		# Stop timing the loop
		calc_time = time.time() - start - setup_time
		# Report timing for the whole program and write a PyMol script
		if options.timing == True: print('   Timing: Setup {:5.1f} / Calculate {:5.1f} (secs)'.format(setup_time, calc_time))
		self.setup_time = setup_time
		self.calc_time = calc_time
		if options.commandline == False:
			writer.xyz_export(file,mol)
			writer.pymol_export(file, mol, spheres, cylinders, options.isoval)


def set_options(kwargs):
	#set default options and options provided
	p = OptionParser()
	(options, args) = p.parse_args()
	#dictionary containing default values for options 
	var_dict = {'verbose': ['verbose',False], 'v': ['verbose',False], 'grid': ['grid',0.05],
	'scalevdw':['SCALE_VDW',1.0], 'noH':['noH',False], 'addmetals':['add_metals',False],
	'norot':['norot',False],'r':['radius',3.5],'scan':['scan',False],'scand':['scand',False],'center':['spec_atom_1',False],
	'ligand':['spec_atom_2',False],'exclude':['exclude',False],'isoval':['isoval',0.002],
	's' : ['sterimol','grid'], 'sterimol':['sterimol','grid'],'surface':['surface','density'],
	'debug':['debug',False],'volume':['volume',False],'vshell':['vshell',False],'t': ['timing',False],
	'timing': ['timing',False],'commandline':['commandline',False],'qsar':['qsar',False],
	'gridsize': ['gridsize', False]
	}

	for key in var_dict:
		vars(options)[var_dict[key][0]] = var_dict[key][1]
	for key in kwargs:
		if key in var_dict:
			vars(options)[var_dict[key][0]] = kwargs[key]
		else:
			print("Warning! Option: [", key,":", kwargs[key],"] provided but no option exists, try -h to see available options.")

	return options


def main():
	files=[]
	# get command line inputs. Use -h to list all possible arguments and default values
	parser = OptionParser(usage="Usage: %prog [options] <input1>.log <input2>.log ...")
	parser.add_option("--center", dest="spec_atom_1", action="store", help="Specify the base atom", default=False, metavar="spec_atom_1")
	parser.add_option("--ligand", dest="spec_atom_2", action="store", help="Specify the connected atom(s)", default=False, metavar="spec_atom_2")
	parser.add_option("--exclude", dest="exclude", action="store", help="Atoms to ignore", default=False, metavar="exclude")
	parser.add_option("--noH", dest="noH", action="store_true", help="Neglect hydrogen atoms (by default these are included)", default=False, metavar="noH")
	parser.add_option("--addmetals", dest="add_metals", action="store_true", help="By default, the VDW radii of metals are not considered. This will include them", default=False, metavar="add_metals")
	parser.add_option("--norot",dest='norot',action="store_true",help="Do not rotate the molecules (use if structures have been pre-aligned)",default=False)
	parser.add_option("-s", "--sterimol", dest="sterimol", action="store",choices=['grid','classic'], help="Type of Sterimol Calculation (classic or grid=default)", default='grid', metavar="sterimol")
	parser.add_option("--grid", dest="grid", action="store", help="Specify how grid point spacing used to compute spatial occupancy", default=0.05, type=float, metavar="grid")
	parser.add_option("--surface", dest="surface", action="store", choices=['vdw','density'],help="The surface can be defined by Bondi VDW radii or a density cube file", default='density', metavar="surface")
	parser.add_option("--isoval", dest="isoval", action="store", help="Density isovalue cutoff (default = 0.002)", type="float", default=0.002, metavar="isoval")
	parser.add_option("-r", dest="radius", action="store", help="Radius from point of attachment (default = 3.5)", default=3.5, type=float, metavar="radius")
	parser.add_option("--volume",dest="volume",action="store_true", help="Calculate buried volume of input molecule", default=False)
	parser.add_option("--vshell",dest="vshell",action="store",help="Calculate buried volume of hollow sphere. Input: shell width, use '-r' option to adjust radius'", default=False,type=float, metavar="radius")
	parser.add_option("--scan", dest="scan", action="store", help="Scan over a range of radii 'rmin:rmax:interval'", default=False, metavar="scan")
	parser.add_option("--scand", dest="scand", action="store", help="Scan over an evenly distributed range of radii", default=False, metavar="scand")
	parser.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Request verbose print output", default=False , metavar="verbose")
	parser.add_option("--debug", dest="debug", action="store_true", help="Mode for debugging, graph grid points, print extra stuff", default=False, metavar="debug")
	parser.add_option("--qsar", dest="qsar", action="store_true", help="Construct a grid with probe atom at each point for QSAR study (this generates a lot of files!)", default=False, metavar="qsar")
	parser.add_option("--gridsize", dest="gridsize", action="store",help="Set size of grid to analyze molecule centered at origin 'xmin,xmax:ymin,ymax:zmin,zmax'",default=False)
	parser.add_option("--scalevdw", dest="SCALE_VDW", action="store", help="Scaling factor for VDW radii (default = 1.0)", type=float, default=1.0, metavar="SCALE_VDW")
	parser.add_option("-t", "--timing",dest="timing",action="store_true", help="Request timing information", default=False)
	parser.add_option("--commandline", dest="commandline",action="store_true", help="Requests no new files be created", default=False)
	(options, args) = parser.parse_args()

	# make sure upper/lower case doesn't matter
	options.surface = options.surface.lower()

	# Get Coordinate files - can be xyz, log or cube
	if len(sys.argv) > 1:
		for elem in sys.argv[1:]:
			try:
				if os.path.splitext(elem)[1] in [".xyz", ".log", ".cube"]:
					for file in glob(elem):
						files.append(file)
			except IndexError: pass

	if len(files) == 0: sys.exit("    Please specify a valid input file and try again.")
	
	#in qsar mode, loop through and get dimensions of molecules to create uniform grid sizing 
	#(3A larger than greatest magnitudes in xyz directions)
	if options.qsar:
		mols=[]
		for file in files: 
			mols.append(dbstep(file,options=options,QSAR=True))
		xmin,xmax,ymin,ymax,zmin,zmax = 0,0,0,0,0,0
		for mol in mols:
			[x_minus, x_plus, y_minus, y_plus, z_minus, z_plus] = mol.dimensions
			if x_plus >= xmax: xmax = x_plus
			if y_plus >= ymax: ymax = y_plus
			if z_plus >= zmax: zmax = z_plus
			if x_minus <= xmin: xmin = x_minus
			if y_minus <= ymin: ymin = y_minus
			if z_minus <= zmin: zmin = z_minus
		dim = [xmin,xmax,ymin,ymax,zmin,zmax]
		for i in range(len(dim)):
			if i%2: dim[i]+=3
			else: dim[i]-=3
		options.gridsize = str(dim[0])+','+str(dim[1])+':'+str(dim[2])+','+str(dim[3])+':'+str(dim[4])+','+str(dim[5])
		if options.verbose: print("   Grid size for QSAR mode will be: "+options.gridsize)
	for file in files:
		# loop over all specified output files
		dbstep(file,options=options)

if __name__ == "__main__":
	main()

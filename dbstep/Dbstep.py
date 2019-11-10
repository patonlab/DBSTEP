# -*- coding: UTF-8 -*-
from __future__ import print_function, absolute_import

###############################################################
# known issues:
# Hard - check the numerical results for some toy systems (e.g. spherically symmetrical, diatomics) where the correct alues can be defined manually. Then check against tabulate results for classical values, then compare QM-density derived values
# a bit tricky - output the grid points as a series of small dots for visualization in pymol (this is v slow [unusable] with spheres)
# Tricky - optimize for speed - avoid iterating over lists within lists
# Moderate - Better output of isovalue cube and overall more automation of commands written to pymol script
# Moderately trivial - if you remove Hs, the base atom ID messes up
# Cosmetic - would be better to combine methods where either dens is used radii and can be chosen from the commandline
###############################################################

#Python Libraries
from numba import autojit, prange
import itertools, os, sys, time
from numpy import *
from scipy import *
from math import *
from glob import glob
import numpy as np
from optparse import OptionParser
import scipy.spatial as spatial

#Avoid number error warnings
import warnings
warnings.filterwarnings("ignore")

#Chemistry Arrays

#Bondi Van der Waals radii taken from [J. Phys. Chem. A. 2009, 103, 5806-5812]
bondi = {"Bq": 0.00, "H": 1.09,"He": 1.40,"Li": 1.81,"Be": 1.53,"B": 1.92,"C": 1.70,"N": 1.55,"O": 1.52,"F": 1.47,"Ne":1.54,
	"Na":2.27,"Mg":1.73,"Al":1.84,"Si":2.10,"P":1.80,"S":1.80, "Cl":1.75,"Ar":1.88,
	"K":2.75,"Ca":2.31,"Ga":1.87,"Ge":2.11,"As":1.85,"Se":1.90,"Br":1.83,"Kr":2.02,
	"Rb":3.03,"Sr":2.49,"In":1.93,"Sn":2.17,"Sb":2.06,"Te":2.06,"I":1.98,"Xe":2.16,
	"Cs":3.43,"Ba":2.68,"Tl":1.96,"Pb":2.02,"Bi":2.07,"Po":1.97,"At":2.02,"Rn":2.20,
	"Fr":3.48,"Ra":2.83,"Pd": 1.63, "Ni": 0.0, "Rh": 2.00}

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

#Enables output to terminal and to text file
class Logger:
   # Designated initializer
   def __init__(self,filein,suffix,append):
	   # Create the log file at the input path
	   self.log = open(filein+"_"+append+"."+suffix, 'w' )

   # Write a message only to the log and not to the terminal
   def Writeonlyfile(self, message):
	   self.log.write(message+"\n")


class WriteCubeData:
	""" Write new cube file of translated, rotated molecule for PyMOL """
	def __init__(self, file, cube):
		self.FORMAT = 'cube'
		oldfile = open(file+"."+self.FORMAT,"r")
		oldlines = oldfile.readlines()
		molfile = open(file+"_radius."+self.FORMAT,"w")
		# write new coordinates to file
		for line in oldlines[0:2]:
			molfile.write(line)

		dims = [cube.xdim,cube.ydim,cube.zdim]
		molfile.write("{:5} {:11.6f} {:11.6f} {:11.6f} {:4}".format(len(cube.ATOMNUM),cube.ORIGIN[0] / BOHR_TO_ANG, cube.ORIGIN[1] / BOHR_TO_ANG, cube.ORIGIN[2] / BOHR_TO_ANG, 1)+'\n')
		for i in range(len(cube.INCREMENTS)):
			molfile.write("{:5} {:11.6f} {:11.6f} {:11.6f}".format(dims[i],cube.INCREMENTS[i][0] / BOHR_TO_ANG,cube.INCREMENTS[i][1] / BOHR_TO_ANG,cube.INCREMENTS[i][2] / BOHR_TO_ANG)+'\n')
		for i in range(len(cube.CARTESIANS)):
			x = cube.CARTESIANS[i][0] / BOHR_TO_ANG
			y = cube.CARTESIANS[i][1] / BOHR_TO_ANG
			z = cube.CARTESIANS[i][2] / BOHR_TO_ANG
			molfile.write("{:5} {:11.6f} {:11.6f} {:11.6f} {:11.6f}".format(cube.ATOMNUM[i],float(cube.ATOMNUM[i]),x,y,z)+'\n')


		for line in cube.DENSITY_LINE:
			molfile.write(line)

		# there may well be a fast way to do this directly from the 3D array of X,Y,Z points
		# see http://paulbourke.net/dataformats/cube/
		# for x in xvals:
		# 	for y in yvals:
		# 		width = []
		# 		for z in zvals:
		# 			width.append((x**2 + y**2)**0.5)
		# 		# for cube print formatting
		# 		list_of_widths = itertools.zip_longest(*(iter(width),) * 6)
		# 		for widths in list_of_widths:
		# 			outline = ''
		# 			for val in widths:
		# 				if val != None:
		# 					outline += str('  {:10.5E}'.format(val))
		# 			molfile.write('\n'+outline)
		molfile.close()


class GetCubeData:
	""" Read data from cube file, obtian XYZ Cartesians, dimensions, and volumetric data """
	def __init__(self, file):
		if not os.path.exists(file+".cube"): print(("\nFATAL ERROR: cube file [ %s ] does not exist"%file))
		def getATOMTYPES(self, outlines, format):
			self.ATOMTYPES, self.ATOMNUM, self.CARTESIANS, self.DENSITY, self.DENSITY_LINE = [], [], [], [], []
			if format == 'cube':
				for i in range(2,len(outlines)):
					try:
						coord = outlines[i].split()
						if i == 2:
							self.ORIGIN = [float(coord[1])*BOHR_TO_ANG, float(coord[2])*BOHR_TO_ANG, float(coord[3])*BOHR_TO_ANG]
						elif i == 3:
							self.xdim = int(coord[0])
							self.SPACING = float(coord[1])*BOHR_TO_ANG
							self.x_inc = [float(coord[1])*BOHR_TO_ANG,float(coord[2])*BOHR_TO_ANG,float(coord[3])*BOHR_TO_ANG]
						elif i == 4:
							self.ydim = int(coord[0])
							self.y_inc = [float(coord[1])*BOHR_TO_ANG,float(coord[2])*BOHR_TO_ANG,float(coord[3])*BOHR_TO_ANG]
						elif i == 5:
							self.zdim = int(coord[0])
							self.z_inc = [float(coord[1])*BOHR_TO_ANG,float(coord[2])*BOHR_TO_ANG,float(coord[3])*BOHR_TO_ANG]
						elif len(coord) == 5:
							if represents_int(coord[0]) == True and represents_float(coord[2]) == True and represents_float(coord[3]) and represents_float(coord[4]):
								[atom, x,y,z] = [periodictable[int(coord[0])], float(coord[2])*BOHR_TO_ANG, float(coord[3])*BOHR_TO_ANG, float(coord[4])*BOHR_TO_ANG]
								self.ATOMNUM.append(int(coord[0]))
								self.ATOMTYPES.append(atom)
								self.CARTESIANS.append([x,y,z])
						if represents_int(coord[0]) == False:
							for val in coord:
								self.DENSITY.append(float(val))
							self.DENSITY_LINE.append(outlines[i])
					except: pass
		self.FORMAT = 'cube'
		molfile = open(file+"."+self.FORMAT,"r")
		mollines = molfile.readlines()
		getATOMTYPES(self, mollines, self.FORMAT)
		self.INCREMENTS=np.asarray([self.x_inc,self.y_inc,self.z_inc])
		cube_data = np.zeros([self.xdim,self.ydim,self.zdim])
		self.DENSITY = np.asarray(self.DENSITY)
		self.DATA = np.reshape(self.DENSITY,(self.xdim,self.ydim,self.zdim))
		vol_x = []
		vol_y = []
		vol_z = []
		for i in range(self.xdim):
			for j in range(self.ydim):
				for k in range(self.zdim):
					if self.DATA[i][j][k] > 0.05:
						vol_x.append(self.ORIGIN[0]+(i-1)*self.x_inc[0] + (j-1)*self.x_inc[1] + (k-1)*self.x_inc[2])
						vol_y.append(self.ORIGIN[1]+(i-1)*self.y_inc[0] + (j-1)*self.y_inc[1] + (k-1)*self.y_inc[2])
						vol_z.append(self.ORIGIN[2]+(i-1)*self.z_inc[0] + (j-1)*self.z_inc[1] + (k-1)*self.z_inc[2])
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		self.CARTESIANS = np.array(self.CARTESIANS)
		# # graph 3d coords
		# from mpl_toolkits.mplot3d import axes3d
		# import matplotlib.pyplot as plt
		# from matplotlib import cm
		#
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# ax.scatter3D(vol_x, vol_y, vol_z, c=vol_z, cmap='Greens')
		# plt.show()


class GetXYZData:
	""" Read XYZ Cartesians from file """
	def __init__(self, file,ext, noH):
		if ext == '.xyz':
			if not os.path.exists(file+".xyz"):
				sys.exit("\nFATAL ERROR: XYZ file [ %s ] does not exist"%file)
		elif ext == '.log':
			if not os.path.exists(file+".log"):
				print(("\nFATAL ERROR: log file [ %s ] does not exist"%file))
		self.FORMAT = ext
		molfile = open(file+self.FORMAT,"r")
		outlines = molfile.readlines()

		self.ATOMTYPES, self.CARTESIANS = [], []
		if self.FORMAT == '.xyz':
			for i in range(0,len(outlines)):
				try:
					coord = outlines[i].split()
					if len(coord) == 4:
						if represents_float(coord[1]) == True and represents_float(coord[2]) and represents_float(coord[3]):
							[atom, x,y,z] = [coord[0], float(coord[1]), float(coord[2]), float(coord[3])]
							if noH == True and atom == "H":
								pass
							else:
								self.ATOMTYPES.append(atom)
								self.CARTESIANS.append([x,y,z])
				except: pass
		elif self.FORMAT == '.log':
			for i, oline in enumerate(outlines):
				if "Input orientation" in oline or "Standard orientation" in oline:
					self.ATOMTYPES, self.CARTESIANS, carts = [], [], outlines[i + 5:]
					for j, line in enumerate(carts):
						if "-------" in line:
							break
						self.ATOMTYPES.append(element_id(int(line.split()[1])))
						if len(line.split()) > 5:
							self.CARTESIANS.append([float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
						else:
							self.CARTESIANS.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		self.CARTESIANS = np.array(self.CARTESIANS)

def represents_float(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def represents_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def element_id(massno, num=False):
    try:
        if num:
            return periodictable.index(massno)
        return periodictable[massno]
    except IndexError:
        return "XX"

def unit_vector(vector):
	""" Returns the unit vector of the vector """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2' """
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	cosang = np.dot(v1_u,v2_u)
	sinang = np.linalg.norm(np.cross(v1_u,v2_u))
	ang = np.arctan2(sinang,cosang)
	return math.degrees(ang)

def translate_mol(MOL, options, origin):
	coords, atoms, spec_atom = MOL.CARTESIANS, MOL.ATOMTYPES, options.spec_atom_1
	for n, atom in enumerate(atoms):
		if not spec_atom:
			# if no center is specified, try to default to metal
			if atom in metals:
				base_id, base_atom = n, atom
			# otherwise just use the first atom
			if n ==0:
				base_id, base_atom = n, atom
		else:
			if atom+str(n+1) == spec_atom:
				base_id, base_atom = n, atom
	try:
		displacement = coords[base_id] - origin
		if np.linalg.norm(displacement) == 0:
			if options.verbose: print("\n   Molecule is defined with {}{} at the origin".format(base_atom,(base_id+1)))
		else:
			if options.verbose ==True: print("\n   Translating molecule by {} to set {}{} at the origin".format(-displacement, base_atom, (base_id+1)))
		for n, coord in enumerate(coords):
			coords[n] = coords[n] - displacement
	except:
		   sys.exit("   WARNING! Unable to find an atom (e.g. metal) to set at the origin")
	return coords

# Translates molecule so that a specified atom (spec_atom) is at the origin. Defaults to a metal if no atom is specified.
def translate_dens(mol, options, xmin, xmax, ymin, ymax, zmin, zmax, xyz_max, origin):
	coords, atoms, cube_origin = mol.CARTESIANS, mol.ATOMTYPES,mol.ORIGIN
	spec_atom = options.spec_atom_1
	for n, atom in enumerate(atoms):
		if not spec_atom:
			if atom in metals:
				base_id, base_atom = n, atom
		else:
			if atom+str(n+1) == spec_atom:
				base_id, base_atom = n, atom
	try:
		displacement = coords[base_id] - origin
		if np.linalg.norm(displacement) == 0:
			if options.verbose: print("\n   Molecule is already defined with {}{} at the origin".format(base_atom,(base_id+1)))
		else:
			if options.verbose: print("\n   Translating molecule by {} to set {}{} at the origin".format(-displacement, base_atom, (base_id+1)))
		for n, coord in enumerate(coords):
			coords[n] = coords[n] - displacement
		cube_origin = cube_origin + displacement
		xmin -= displacement[0]
		xmax -= displacement[0]
		ymin -= displacement[1]
		ymax -= displacement[1]
		zmin -= displacement[2]
		zmax -= displacement[2]
		xyz_max = max(xmax, ymax, zmax, abs(xmin), abs(ymin), abs(zmin))
	except:
		   sys.exit("   WARNING! Unable to find an atom (e.g. metal) to set at the origin")
	return [coords, cube_origin, xmin, xmax, ymin, ymax, zmin, zmax, xyz_max]

def bidentate(MOL, options):
	coords, atoms, spec_atom_1, spec_atom_2 = MOL.CARTESIANS, MOL.ATOMTYPES, options.spec_atom_1,  options.spec_atom_2
	#find point on line made by atoms in spec_atom_2 perpendicular to spec_atom_1
	for n, atom in enumerate(atoms):
		if atom+str(n+1) == spec_atom_1:
			c_id, c_atom = n, atom
			a = coords[c_id]
		elif atom+str(n+1) == spec_atom_2[0]:
			l1_id, l1_atom = n, atom
			b = coords[l1_id]
		elif atom+str(n+1) == spec_atom_2[1]:
			l2_id, l2_atom = n, atom
			c = coords[l2_id]
	# v1 = b - a
	# v2 = b - c
	# angle = angle_between(unit_vector(v1),unit_vector(v2))
	# point = v1 * math.cos(math.radians(angle))
	# point = point + b
	# newvec = v1 - point
	# print('v1',v1,unit_vector(v1))
	# print('v2',v2,unit_vector(v2))
	# print('projection',point)
	# print('height',newvec)
	return b + c

def tridentate(MOL, options):
	coords, atoms, spec_atom_1, spec_atom_2 = MOL.CARTESIANS, MOL.ATOMTYPES, options.spec_atom_1,  options.spec_atom_2
	#find point on plane made by atoms in spec_atom_2 perpendicular to spec_atom_1
	for n, atom in enumerate(atoms):
		if atom+str(n+1) == spec_atom_1:
			c_id, c_atom = n, atom
			a = coords[c_id]
		elif atom+str(n+1) == spec_atom_2[0]:
			l1_id, l1_atom = n, atom
			b = coords[l1_id]
		elif atom+str(n+1) == spec_atom_2[1]:
			l2_id, l2_atom = n, atom
			c = coords[l2_id]
		elif atom+str(n+1) == spec_atom_2[2]:
			l3_id, l3_atom = n, atom
			d = coords[l3_id]
	return b + c + d

# Rotates molecule around X- and Y-axes to align M-L bond to Z-axis
def rotate_mol(coords, atoms, spec_atom_1, lig_point, options, cube_origin=False, cube_inc=False):
	for n, atom in enumerate(atoms):
		# if atom in metals: met_id, met_atom = n, atom
		# if atom == "P": lig_id, lig_atom = n, atom
		if atom+str(n+1) == spec_atom_1:
			center_id, center_atom = n, atom
		# if atom+str(n+1) == spec_atom_2:
		# 	lig_id, lig_atom = n, atom
	try:
		ml_vec = lig_point - coords[center_id]
		# ml_vec = coords[lig_id]- coords[center_id]
		zrot_angle = angle_between(unit_vector(ml_vec), [0.0, 0.0, 1.0])
		newcoord=[]
		new_inc=[]
		new_data=[]
		currentatom=[]
		if np.linalg.norm(zrot_angle) == 0:
			if options.verbose: print("No rotation necessary :)")
			#print("   Molecule is aligned with {}{}-{}{} along the Z-axis".format(center_atom,(center_id+1),lig_atom,(lig_id+1)))
		else:
			for i in range(0,len(coords)):
				newcoord.append(coords[i])
			if cube_inc is not False:
				for i in range(0,len(cube_inc)):
					new_inc.append(cube_inc[i])
			ml_vec = lig_point - coords[center_id]
			# ml_vec = coords[lig_id]- coords[center_id]
			yz = [ml_vec[1],ml_vec[2]]
			if yz != [0.0,0.0]:
				u_yz = unit_vector(yz)
				rot_angle = angle_between(u_yz, [0.0, 1.0])
				theta = rot_angle /180. * math.pi
				quadrant_check = math.atan2(u_yz[1],u_yz[0])
				if quadrant_check > math.pi / 2.0 and quadrant_check <= math.pi:
					theta = math.pi - theta
				elif quadrant_check < -math.pi / 2.0 and quadrant_check >= -(math.pi):
					theta =  math.pi - theta
				if options.verbose ==True: print('   Rotating molecule about X-axis {0:.2f} degrees'.format(theta*180/math.pi))
				center = [0.,0.,0.]

				#rotate ligand point
				u = [float(lig_point[0]) - center[0], float(lig_point[1]) - center[1], float(lig_point[2]) - center[2]]
				ox = u[0]
				oy = u[1]*math.cos(theta) - u[2]*math.sin(theta)
				oz = u[1]*math.sin(theta) + u[2]*math.cos(theta)
				lig_point = [round(ox + center[0],8), round(oy + center[1],8), round(oz + center[2],8)]

				for i,atom in enumerate(atoms):
					#rotate coords around x axis
					v = [float(coords[i][0]) - center[0], float(coords[i][1]) - center[1], float(coords[i][2]) - center[2]]
					px = v[0]
					py = v[1]*math.cos(theta) - v[2]*math.sin(theta)
					pz = v[1]*math.sin(theta) + v[2]*math.cos(theta)
					rot1 = [round(px + center[0],8), round(py + center[1],8), round(pz + center[2],8)]
					newcoord[i] = rot1

				if cube_inc is not False:
					for i in range(len(cube_inc)):
						center = cube_origin
						#rotate coords around x axis
						w = [float(cube_inc[i][0]) - center[0], float(cube_inc[i][1]) - center[1], float(cube_inc[i][2]) - center[2]]
						qx = w[0]
						qy = w[1]*math.cos(theta) - w[2]*math.sin(theta)
						qz = w[1]*math.sin(theta) + w[2]*math.cos(theta)
						rot1 = [round(qx + center[0],8), round(qy + center[1],8), round(qz + center[2],8)]
						new_inc[i] = rot1

			newcoord = np.asarray(newcoord)

			ml_vec = lig_point - newcoord[center_id]
			# ml_vec = newcoord[lig_id] - newcoord[center_id]
			zx = [ml_vec[2],ml_vec[0]]
			if zx != [0.0,0.0]:
				u_zx = unit_vector(zx)
				rot_angle = angle_between(zx, [1.0, 0.0])
				phi = rot_angle /180. * math.pi
				quadrant_check = math.atan2(u_zx[1],u_zx[0])
				if quadrant_check > 0 and quadrant_check <= math.pi:
					phi = 2 * math.pi - phi
				if options.verbose ==True: print('   Rotating molecule about Y-axis {0:.2f} degrees'.format(phi*180/math.pi))

				u = [float(lig_point[0]) - center[0], float(lig_point[1]) - center[1], float(lig_point[2]) - center[2]]
				ox = u[2]*math.sin(phi) + u[0]*math.cos(phi)
				oy = u[1]
				oz = u[2]*math.cos(phi) - u[0]*math.sin(phi)
				lig_point = [round(ox + center[0],8), round(oy + center[1],8), round(oz + center[2],8)]

				for i,atom in enumerate(atoms):
					center = [0.,0.,0.]
					#rotate coords around y axis
					v = [float(newcoord[i][0]) - center[0], float(newcoord[i][1]) - center[1], float(newcoord[i][2]) - center[2]]
					px = v[2]*math.sin(phi) + v[0]*math.cos(phi)
					py = v[1]
					pz = v[2]*math.cos(phi) - v[0]*math.sin(phi)
					rot2 = [round(px + center[0],8), round(py + center[1],8), round(pz + center[2],8)]
					newcoord[i]=rot2

				if cube_inc is not False:
					for i in range(len(cube_inc)):
						center = cube_origin
						w = [float(new_inc[i][0]) - center[0], float(new_inc[i][1]) - center[1], float(new_inc[i][2]) - center[2]]
						qx = w[2]*math.sin(phi) + w[0]*math.cos(phi)
						qy = w[1]
						qz = w[2]*math.cos(phi) - w[0]*math.sin(phi)
						rot2 = [round(qx + center[0],8), round(qy + center[1],8), round(qz + center[2],8)]
						new_inc[i]=rot2

		if len(newcoord) !=0:
			if cube_inc is not False:
				return newcoord, np.asarray(new_inc)
			else:
				return newcoord
		else:
			if cube_inc is not False:
				return coords, cube_inc
			else:
				return coords

	except Exception as e:
		print("\nERR: ",e)
		#print("   WARNING! Unable to find a M-P bond vector to rotate the molecule")
	return coords

# Rounds distances into discrete numbers of grid intervals
def grid_round(x, spacing):
	n = 1 / spacing
	return(round(x*n)/n)

# Establishes the smallest cuboid that contains all of the molecule to speed things up
def max_dim(coords, radii, options):
	spacing = options.grid
	[x_min, x_max, y_min, y_max, z_min, z_max] = np.zeros(6)
	for n, coord in enumerate(coords):
		[x_plus,y_plus,z_plus] = coord + np.array([radii[n], radii[n], radii[n]])
		[x_minus,y_minus,z_minus] = coord - np.array([radii[n], radii[n], radii[n]])
		if x_plus >= x_max: x_max = grid_round(x_plus, spacing) + spacing
		if y_plus >= y_max: y_max = grid_round(y_plus, spacing) + spacing
		if z_plus >= z_max: z_max = grid_round(z_plus, spacing) + spacing
		if x_minus <= x_min: x_min = grid_round(x_minus, spacing) - spacing
		if y_minus <= y_min: y_min = grid_round(y_minus, spacing) - spacing
		if z_minus <= z_min: z_min = grid_round(z_minus, spacing) - spacing

	# largest dimension along any axis
	max_dim = max(x_max, y_max, z_max, abs(x_min), abs(y_min), abs(z_min))
	if options.verbose ==True: print("\n   Molecule is bounded by the region X:[{:6.3f} to{:6.3f}] Y:[{:6.3f} to{:6.3f}] Z:[{:6.3f} to{:6.3f}]".format(x_min, x_max, y_min, y_max, z_min, z_max))

	# compute cubic volume containing molecule and estimate the number of grid points based on grid spacing and volume size
	cubic_volume = (2 * max_dim) ** 3
	n_points = int(cubic_volume / (spacing ** 3))
	return [x_min, x_max, y_min, y_max, z_min, z_max, max_dim]

# Uses atomic coordinates and VDW radii to establish which grid voxels are occupied
def occupied(grid, coords, radii, origin, options):
	spacing = options.grid
	if options.verbose ==True: print("\n   Using a Cartesian grid-spacing of {:5.4f} Angstrom.".format(spacing))
	if options.verbose ==True: print("   There are {} grid points.".format(len(grid)))

	idx, point_tree  = [], spatial.cKDTree(grid)
	for n, coord in enumerate(coords):
		center = coord + origin
		idx.append(point_tree.query_ball_point(center, radii[n]))
	# construct a list of indices of the grid array that are occupied
	jdx = [y for x in idx for y in x]
	# removes duplicates since a voxel can only be occupied once
	jdx = list(set(jdx))
	if options.verbose: print("   There are {} occupied grid points.".format(len(jdx)))
	if options.verbose: print("   Molecular volume is {:5.4f} Ang^3".format(len(jdx) * spacing ** 3))
	return grid[jdx]

# Uses density cube to establish which grid voxels are occupied (i.e. density is above some isoval, by default 0.002)
def occupied_dens(grid, dens, options):
	spacing, isoval = options.grid, options.isoval
	cube, list = (spacing / 0.529177249) ** 3, []
	if options.verbose: print("\n   Using a Cartesian grid-spacing of {:5.4f} Angstrom".format(spacing))

	for n, density in enumerate(dens):
		if density > isoval: list.append(n)
	if options.verbose: print("   Molecular volume is {:5.4f} Ang^3".format(len(list) * spacing ** 3))
	return grid[list]

# Uses standard Verloop definitions and VDW spheres to define L, B1 and B5
def get_classic_sterimol(coords, radii, atoms, spec_atom_1, spec_atom_2):
	L, Bmax, Bmin, xmax, ymax, cyl, rad_hist_hy,rad_hist_rw, x_hist_rw, y_hist_rw,x_hist_hy, y_hist_hy  = 0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], [], [], [], []
	for n, coord in enumerate(coords):
		# L parameter - this is not actually the total length, but the largest distance from the basal XY-plane. Any atoms pointing below this plane (i.e. in the opposite direction) are not counted.
		# Verloop's original definition does include the VDW of the base atom, which is totally weird and is not done here. There will be a systematic difference vs. literature
		length = abs(coord[2]) + radii[n]
		if length > L: L = length

		# B5 parameter
		x,y,z = coord
		radius = np.hypot(x,y) + radii[n]
		if x != 0.0 and y != 0.0:
			x_hist_hy.append(x)
			y_hist_hy.append(y)
			rad_hist_hy.append(radius)
		rad_hist_rw.append(radii[n])
		x_hist_rw.append(x)
		y_hist_rw.append(y)
		if radius > Bmax:
			Bmax, xmax, ymax = radius, x, y
			# don't actually need this for Sterimol. It's used to draw a vector direction along B5 to be displayed in PyMol
			theta = arctan(y/x)
			if x < 0: theta += math.pi
			if x != 0. and y!= 0.:
				x_disp, y_disp = radii[n] * math.cos(theta), radii[n] * math.sin(theta)
			elif x == 0. and y != 0.:
				x_disp, y_disp = 0.0, radii[n] * math.sin(theta)
			elif x != 0. and y == 0.:
				x_disp, y_disp = radii[n]* math.cos(theta), 0.0
			else:
				x_disp, y_disp = radii[n], 0.0
			xmax += x_disp; ymax += y_disp

	# A nice PyMol cylinder object points along the B5 direction with the appopriate magnitude
	cyl.append("   CYLINDER, 0., 0., {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,".format(0.0, xmax, ymax, 0.0, 0.1))

	# Drop the Z coordinates and calculate B1
	xycoords = [(x,y) for x,y,z in coords]
	#increments = 6000 # this goes around in 0.06 degree intervals
	increments = 361 # this goes around in 1 degree intervals
	ang_inc = math.pi/(increments-1)
	angles = np.linspace(-math.pi, -math.pi + 2 * math.pi, increments) # sweep full circle
	Bmin = sys.float_info.max
	xmin,ymin = 0,0
	for angle in angles:
		angle_val = 0.0
		for i in range(len(xycoords)):
			projection = (xycoords[i][0])*math.cos(angle) + (xycoords[i][1])*math.sin(angle)
			radius = projection + radii[i]

			if radius > angle_val:
				angle_val, x, y = radius, radius*math.cos(angle),radius*math.sin(angle)

		if Bmin > angle_val:
			Bmin,xmin,ymin = angle_val,x,y

	cyl.append("   CYLINDER, 0., 0., {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,".format(0.0, xmin, ymin, 0.0, 0.1))
	return L, Bmax, Bmin, cyl

@autojit
def parallel(angles,ang_inc,radial_grid):
	max_r, max_phi = [], []
	for n in prange(len(angles)):
		angle = angles[n]
		rmax, phimax = parallel_grid_scan(radial_grid,angle,ang_inc)
		if rmax != 0.0: # by definition can't have zero radius
			max_r.append(rmax)
			max_phi.append(phimax)
	return max_r, max_phi

@autojit
def parallel_grid_scan(xy_grid, angle):
	# angular sweep over grid points to find Bmin
	rmax = 0.0
	for i in prange(len(xy_grid)):
		r = xy_grid[i][0]*math.cos(angle)+xy_grid[i][1]*math.sin(angle)
		if r > rmax:
				rmax = r
	return rmax

# Uses grid occupancy to define Sterimol L, B1 and B5 parameters. If the grid-spacing is small enough this should be close to the
# conventional values above when the grid occupancy is based on VDW radii. The real advantage is that the isodensity surface can be used,
# which does not require VDW radii, and this also looks something a bit closer to a solvent-accessible surface than the sum-of-spheres.
# Also B1 can be defined in a physically more # meaningful way than the traditional approach. This method can take horizontal slices to
# evaluate these parameters along the L-axis, which is also a nightmare with the conventional definition.
def get_cube_sterimol(occ_grid, R, spacing, strip_width):
	L, Bmax, Bmin, xmax, ymax, zmax, xmin, ymin, cyl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []

	# this is a layer of the occupancy grid between Z-limits
	if strip_width != 0: xy_grid = np.array([(x,y,z) for x,y,z in occ_grid if abs(z) <= R + strip_width and abs(z) > R - strip_width])	
	else: xy_grid = occ_grid

	#radii = map(lambda x: math.sqrt(x[0]**2+x[1]**2), xy_grid)
	radii = [math.sqrt(x**2+y**2) for x,y,z in xy_grid]
	Bmax, imax = max(radii), np.argmax(radii)
	xmax, ymax, zmax = xy_grid[imax]
	#print(Bmax, math.sqrt(xmax**2+ymax**2))
	L = max(map(lambda x: x[2], xy_grid))
	#print(time.time() - start_scan)
	# this is the best I could come up with to estimate the minimum radius of the molecule projected in the XY-plane
	# I think it will be better to take horizontol slices and then use scipy interpolate to obtain a contour plot
	# or https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html
	# Go around in angle increments and record the farthest out point in each slice
	#symmetry=3 # hack allows you to define rotational symmetry
	#increments = 360/3/symmetry+1 # this goes around in 1 degree intervals
	increments = 361
	ang_inc = math.pi/(increments-1)
	angles = np.linspace(-math.pi, -math.pi+2*math.pi, increments) # sweep full circle

	Bmin = sys.float_info.max
	xmin,ymin = 0,0
	max_r, max_phi = [], []

	for angle in angles:
		rmax = parallel_grid_scan(xy_grid,angle)

		if rmax != 0.0: # by definition can't have zero radius
			max_r.append(rmax)
			max_phi.append(angle)

	if len(max_r) > 0:
	    Bmin = min(max_r)
	    xmin, ymin = Bmin * cos(max_phi[np.argmin(max_r)]), Bmin * sin(max_phi[np.argmin(max_r)])

	# A nice PyMol cylinder object points along the B5 & B1 directions with the appopriate magnitude.
	# In the event that several strips are being evaluated several B-vectors will be arranged along the L-axis.
	# If not a single vector will be shown in the basal plane
	if strip_width == 0.0:
		cyl.append("   CYLINDER, 0., 0., {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,".format(0.0, xmin, ymin, 0.0, 0.1))
		cyl.append("   CYLINDER, 0., 0., {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,".format(0.0, xmax, ymax, 0.0, 0.1))
	else:
		cyl.append("   CYLINDER, 0., 0., {:5.1f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,".format(R, xmin, ymin, R, 0.1))
		cyl.append("   CYLINDER, 0., 0., {:5.1f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,".format(R, xmax, ymax, R, 0.1))
	return L, Bmax, Bmin, cyl

def buried_vol(occ_grid, all_grid, origin, R, spacing, strip_width, verbose):
	""" Read which grid points occupy sphere"""
	sphere, cube = 4 / 3 * math.pi * R ** 3, spacing ** 3
	# Quick way to find all points in the grid within a sphere radius R
	point_tree = spatial.cKDTree(all_grid)
	n_voxel = len(point_tree.query_ball_point(origin, R))
	tot_vol = n_voxel * cube

	# Quick way to find all occupied points within the same spherical volume
	point_tree = spatial.cKDTree(occ_grid)
	n_occ = len(point_tree.query_ball_point(origin, R))

	occ_vol = n_occ * cube
	free_vol = tot_vol - occ_vol
	percent_buried_vol = occ_vol / tot_vol * 100.0
	vol_err = tot_vol/sphere * 100.0

	# experimental - in addition to occupied spherical volume, this will compute
	# the percentage occupancy of a radial shell between two limits if a scan
	# along the L-axis is being performed
	if strip_width != 0.0:
		shell_vol = 4 / 3 * math.pi * ((R + 0.5 * strip_width) ** 3 - (R - 0.5 * strip_width) ** 3)
		point_tree = spatial.cKDTree(occ_grid)
		shell_occ = len(point_tree.query_ball_point(origin, R + 0.5 * strip_width)) - len(point_tree.query_ball_point(origin, R - 0.5 * strip_width))
		shell_occ_vol = shell_occ * cube
		percent_shell_vol = shell_occ_vol / shell_vol * 100.0
	else: percent_shell_vol = 0.0

	if verbose:
		print("   RADIUS, {:5.2f}, VFREE, {:7.2f}, VBURIED, {:7.2f}, VTOTAL, {:7.2f}, VEXACT, {:7.2f}, NVOXEL, {}, %V_Bur, {:7.2f}%,  Tot/Ex, {:7.2f}%".format(R, free_vol, occ_vol, tot_vol, sphere, n_voxel, percent_buried_vol, vol_err))
	if abs(vol_err-100.0) > 1.0:
		print("   WARNING! {:5.2f}% error in estimating the exact spherical volume. The grid spacing is probably too big in relation to the sphere volume".format(vol_err))
	return percent_buried_vol, percent_shell_vol

# outputs a python script that can be imported into PyMol (with 'run script.py')
def pymol_export(file, mol, spheres, cylinders, isoval):
	fullpath = os.path.abspath(file)
	log = Logger(file.split(".")[0],"py", "steric")
	log.Writeonlyfile('from pymol.cgo import *')
	log.Writeonlyfile('from pymol import cmd\n')

	for n,sphere in enumerate(spheres):
		sphere_id = str(n)+"_sphere_"+sphere.split(',')[-1].lstrip()
		log.Writeonlyfile('sphere = [')
		log.Writeonlyfile(sphere)
		log.Writeonlyfile(']\ncmd.load_cgo(sphere'+', '+'"'+sphere_id+'")')

	log.Writeonlyfile('\ncylinder = [')
	for n,cyl in enumerate(cylinders): log.Writeonlyfile(cyl)
	log.Writeonlyfile(']\ncmd.load_cgo(cylinder, '+'"axes"'+')')

	name, ext = os.path.splitext(fullpath)
	base, ext = os.path.splitext(file)
	if ext == '.cube':
		log.Writeonlyfile('\ncmd.load("'+fullpath+'", '+'"dens"'+')')
		log.Writeonlyfile('\ncmd.load("'+name+'_radius.cube", '+'"distances"'+')')
		log.Writeonlyfile('\ncmd.isosurface("isodens", "dens", '+str(isoval)+')')

	# Look if possible to write coords directly to pymol script, for now load from xyz file
	log.Writeonlyfile('\ncmd.load("'+name+'_transform.xyz")')
	log.Writeonlyfile('cmd.show_as("spheres", "'+base.split('/')[-1]+'_transform")')
	log.Writeonlyfile('cmd.set("sphere_transparency", 0.5)')
	log.Writeonlyfile('cmd.set("orthoscopic", "on")')
	# log.Writeonlyfile('\ncmd.fragment("molecule")')
	# coords = 'coords = ['
	# for i in range(len(mol.ATOMTYPES)):
	# 	coords += '\n\t["'+mol.ATOMTYPES[i]+'",\t'
	# 	for j in range(3):
	# 		coords += str(mol.CARTESIANS[i][j])+',\t'
	# 	coords += '],'
	# coords += '\n]'
	# log.Writeonlyfile(coords)
	# log.Writeonlyfile('load_coords(coords,"molecule")')

def xyz_export(file,mol):
	log = Logger(file.split(".")[0],"xyz", "transform")
	log.Writeonlyfile(str(len(mol.ATOMTYPES)))
	log.Writeonlyfile(file.split(".")[0].split('/')[-1].split('\\')[-1])
	coords = ''
	for i in range(len(mol.ATOMTYPES)):
		coords += mol.ATOMTYPES[i]+'\t'
		for j in range(3):
			coords += "{0:.8f}".format(mol.CARTESIANS[i][j])+'\t'
		coords +='\n'
	log.Writeonlyfile(coords)

def in_molecule(x_mol,y_mol,r,x,y):#return true if point x,y is inside given circle
	square = (x_mol - x) ** 2 + (y_mol - y) ** 2
	return square < r ** 2


# correct print format
'''cmd.load('tBu.cube', "dens")
cmd.load('tBu_radius.cube', "distances")
cmd.isosurface('isodens', 'dens', 0.002)
cmd.ramp_new('ramp', 'distances', range=[0,4], color='rainbow') # use zero to bmax for range
cmd.set('surface_color', 'ramp')
set field_of_view, 1
'''

def main():
	files, r_intervals, origin = [], 1, np.array([0,0,0])
	# get command line inputs. Use -h to list all possible arguments and default values
	parser = OptionParser(usage="Usage: %prog [options] <input1>.log <input2>.log ...")
	parser.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Request verbose print output", default=False , metavar="verbose")
	parser.add_option("--grid", dest="grid", action="store", help="Specify how grid point spacing used to compute spatial occupancy", default=0.05, type=float, metavar="grid")
	parser.add_option("--scalevdw", dest="SCALE_VDW", action="store", help="Scaling factor for VDW radii (default = 1.0)", type=float, default=1.0, metavar="SCALE_VDW")
	parser.add_option("--noH", dest="noH", action="store_true", help="Neglect hydrogen atoms (by default these are included)", default=False, metavar="noH")
	parser.add_option("--addmetals", dest="add_metals", action="store_true", help="By default, the VDW radii of metals are not considered. This will include them", default=False, metavar="add_metals")
	parser.add_option("-r", dest="radius", action="store", help="Radius from point of attachment (default = 3.5)", default=3.5, type=float, metavar="radius")
	parser.add_option("--scan", dest="scan", action="store", help="Scan over a range of radii [rmin:rmax:interval]", default=False, metavar="scan")
	parser.add_option("--center", dest="spec_atom_1", action="store", help="Specify the base atom", default=False, metavar="spec_atom_1")
	parser.add_option("--ligand", dest="spec_atom_2", action="store", help="Specify the connected atom(s)", default=False, metavar="spec_atom_2")
	parser.add_option("--exclude", dest="exclude", action="store", help="Atoms to ignore", default=False, metavar="exclude")
	parser.add_option("--isoval", dest="isoval", action="store", help="Density isovalue (default = 0.002)", type="float", default=0.002, metavar="isoval")
	parser.add_option("-s", "--sterimol", dest="sterimol", action="store", help="Type of Sterimol Calculation (classic or grid=default)", default='grid', metavar="sterimol")
	parser.add_option("--surface", dest="surface", action="store", help="The surface can be defined by Bondi VDW radii or a density cube file", default='density', metavar="surface")
	parser.add_option("--debug", dest="debug", action="store_true", help="Print extra stuff to file", default=False, metavar="debug")
	parser.add_option("--volume",dest="volume",action="store_true", help="Calculate buried volume of input molecule", default=False)
	parser.add_option("-t", "--timing",dest="timing",action="store_true", help="Request timing information", default=False)

	(options, args) = parser.parse_args()
	print_txt, print_vals = '', ''

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

	if len(files) is 0: sys.exit("    Please specify a valid input file and try again.")

	for file in files: # loop over all specified output files
		spheres, cylinders = [], []
		start = time.time()
		name, ext = os.path.splitext(file)

		# if noH is requested these atoms are skipped to make things go faster
		if ext == '.xyz' or ext == '.log':
			options.surface = 'vdw'
			mol = GetXYZData(name,ext, options.noH)
		if ext == '.cube':
			mol = GetCubeData(name)
		
		# if atoms are not specified to align to, grab first and second atom in 
		if options.spec_atom_1 is False:
			options.spec_atom_1 = mol.ATOMTYPES[0]+str(1)
		if options.spec_atom_2 is False:
			options.spec_atom_2 = mol.ATOMTYPES[1]+str(2)
			
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
				print("\n   UNABLE TO GENERATE VDW RADII"); exit()
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
		if options.surface == 'vdw': mol.CARTESIANS = translate_mol(mol, options, origin)
		elif options.surface == 'density':
			# print('xyz')
			# for i in range(len(mol.CARTESIANS)):
			# 	x = mol.CARTESIANS[i][0] / BOHR_TO_ANG
			# 	y = mol.CARTESIANS[i][1] / BOHR_TO_ANG
			# 	z = mol.CARTESIANS[i][2] / BOHR_TO_ANG
			# 	print("{:5} {:11.6f} {:11.6f} {:11.6f} {:11.6f}".format(mol.ATOMNUM[i],float(mol.ATOMNUM[i]),x,y,z))
			[mol.CARTESIANS,mol.ORIGIN, x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = translate_dens(mol, options, x_min, x_max, y_min, y_max, z_min, z_max, xyz_max, origin)
			# print bounds of cube
			#print("   Molecule is bounded by the region X:[{:6.3f} to{:6.3f}] Y:[{:6.3f} to{:6.3f}] Z:[{:6.3f} to{:6.3f}]".format(x_min, x_max, y_min, y_max, z_min, z_max))

		# Check if we want to calculate parameters for mono- bi- or tridentate ligand
		options.spec_atom_2 = options.spec_atom_2.split(',')
		if len(options.spec_atom_2) is 1:
			# mono - obtain coords of atom to align along z axis
			point,p_id=0.0,0
			for n, atom in enumerate(mol.ATOMTYPES):
				if atom+str(n+1) == options.spec_atom_2[0]:
					p_id = n
					point = mol.CARTESIANS[p_id]

		# bi - obtain coords of point perpendicular to vector connecting ligands
		elif len(options.spec_atom_2) is 2: point = bidentate(mol, options)
		# tri - obtain coords of point perpendicular to plane connecting ligands
		elif len(options.spec_atom_2) is 3: point = tridentate(mol, options)

		# Rotate the molecule about the origin to align the metal-ligand bond along the (positive) Z-axis
		# the x and y directions are arbitrary
		if len(mol.CARTESIANS) > 1:
			if options.surface == 'vdw':
				mol.CARTESIANS = rotate_mol(mol.CARTESIANS, mol.ATOMTYPES, options.spec_atom_1, point, options)
			elif options.surface == 'density':
				# print('translated')
				# dims = [mol.xdim,mol.ydim,mol.zdim]
				# print("{:5} {:11.6f} {:11.6f} {:11.6f} {:4}".format(len(mol.ATOMNUM),mol.ORIGIN[0] / BOHR_TO_ANG, mol.ORIGIN[1] / BOHR_TO_ANG, mol.ORIGIN[2] / BOHR_TO_ANG, 1))
				# for i in range(len(mol.INCREMENTS)):
				# 	print("{:5} {:11.6f} {:11.6f} {:11.6f}".format(dims[i],mol.INCREMENTS[i][0] / BOHR_TO_ANG,mol.INCREMENTS[i][1] / BOHR_TO_ANG,mol.INCREMENTS[i][2] / BOHR_TO_ANG))
				# for i in range(len(mol.CARTESIANS)):
				# 	x = mol.CARTESIANS[i][0] / BOHR_TO_ANG
				# 	y = mol.CARTESIANS[i][1] / BOHR_TO_ANG
				# 	z = mol.CARTESIANS[i][2] / BOHR_TO_ANG
				# 	print("{:5} {:11.6f} {:11.6f} {:11.6f} {:11.6f}".format(mol.ATOMNUM[i],float(mol.ATOMNUM[i]),x,y,z))

				mol.CARTESIANS, mol.INCREMENTS= rotate_mol(mol.CARTESIANS, mol.ATOMTYPES, options.spec_atom_1,  point, options, cube_origin=mol.ORIGIN, cube_inc=mol.INCREMENTS)
				# 
				# print('rotated')
				# #print everything in bohr
				# dims = [mol.xdim,mol.ydim,mol.zdim]
				# print("{:5} {:11.6f} {:11.6f} {:11.6f} {:4}".format(len(mol.ATOMNUM),mol.ORIGIN[0] / BOHR_TO_ANG, mol.ORIGIN[1] / BOHR_TO_ANG, mol.ORIGIN[2] / BOHR_TO_ANG, 1))
				# for i in range(len(mol.INCREMENTS)):
				# 	print("{:5} {:11.6f} {:11.6f} {:11.6f}".format(dims[i],mol.INCREMENTS[i][0] / BOHR_TO_ANG,mol.INCREMENTS[i][1] / BOHR_TO_ANG,mol.INCREMENTS[i][2] / BOHR_TO_ANG))
				# for i in range(len(mol.CARTESIANS)):
				# 	x = mol.CARTESIANS[i][0] / BOHR_TO_ANG
				# 	y = mol.CARTESIANS[i][1] / BOHR_TO_ANG
				# 	z = mol.CARTESIANS[i][2] / BOHR_TO_ANG
				# 	print("{:5} {:11.6f} {:11.6f} {:11.6f} {:11.6f}".format(mol.ATOMNUM[i],float(mol.ATOMNUM[i]),x,y,z))

				# print coords in angstrom
				# for i in range(len(mol.CARTESIANS)):
				# 	x = mol.CARTESIANS[i][0]
				# 	y = mol.CARTESIANS[i][1]
				# 	z = mol.CARTESIANS[i][2]
				# 	print("{:5} {:11.6f} {:11.6f} {:11.6f} {:11.6f}".format(mol.ATOMNUM[i],float(mol.ATOMNUM[i]),x,y,z))

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
			[x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = max_dim(mol.CARTESIANS, mol.RADII, options)

		# read the requested radius or range
		if not options.scan:
			r_min, r_max, strip_width = options.radius, options.radius, 0.0
		else:
			try:
				[r_min, r_max, strip_width] = [float(scan) for scan in options.scan.split(':')]
				r_intervals += int((r_max - r_min) / strip_width)
			except:
				print("   Can't read your scan request. Try something like --scan 3:5:0.25"); exit()

		if options.volume or options.sterimol == 'grid':
			# Resize the molecule's grid if a larger radius has been requested
			if r_max > xyz_max and options.volume:
				xyz_max = grid_round(r_max, options.grid)
				print("   You asked for a large radius ({})! Expanding the grid dimension to {} Angstrom".format(r_max, xyz_max))

			# define the grid points based on molecule size and grid-spacing
			#n_grid_vals = round(2 * xyz_max / options.grid)
			#grid_vals = np.linspace(xyz_max * -1.0, xyz_max - options.grid, n_grid_vals)
			#grid = np.array(list(itertools.product(grid_vals, grid_vals, grid_vals)))

		# Iterate over the grid points to see whether this is within VDW radius of any atom(s)
		# Grid point occupancy is either yes/no (1/0)
		# To save time this is currently done using a cuboid rather than cubic shaped-grid

		if options.surface == 'vdw':
			n_x_vals = 1 + round((x_max - x_min) / options.grid)
			n_y_vals = 1 + round((y_max - y_min) / options.grid)
			n_z_vals = 1 + round((z_max - z_min) / options.grid)
			x_vals = np.linspace(x_min, x_max, n_x_vals)
			y_vals = np.linspace(y_min, y_max, n_y_vals)
			z_vals = np.linspace(z_min, z_max, n_z_vals)
			if options.volume or options.sterimol == 'grid':
				# construct grid encapsulating molecule
				grid = np.array(list(itertools.product(x_vals, y_vals, z_vals)))
				# compute which grid points occupy molecule
				occ_grid = occupied(grid, mol.CARTESIANS, mol.RADII, origin, options)

		elif options.surface == 'density':
			x_vals = np.linspace(x_min, x_max, mol.xdim)
			y_vals = np.linspace(y_min, y_max, mol.ydim)
			z_vals = np.linspace(z_min, z_max, mol.zdim)

			# writes a new grid to cube file
			isocube = WriteCubeData(name, mol)
			# define the grid points containing the molecule
			grid = np.array(list(itertools.product(x_vals, y_vals, z_vals)))
			# compute occupancy based on isodensity value applied to cube and remove points where there is no molecule
			occ_grid = occupied_dens(grid, mol.DENSITY, options)

		# testing - allows this grid to be visualized in PyMol with output py script
		# - !SLOW to load in PyMOL for many grid points - basically kills PyMOL!
		if options.debug == True:
			for x,y,z in occ_grid: spheres.append("   SPHERE, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}".format(x,y,z,0.02))

		# Set up done so note the time
		setup_time = time.time() - start

		# get buried volume at different radii
		if options.verbose: print("\n   Sterimol parameters will be generated in {} mode for {}\n".format(options.sterimol, file))

		if options.volume:
			print("   {:>6} {:>10} {:>10} {:>10} {:>10}".format("R/Å", "%V_Bur", "%S_Bur", "Bmin", "Bmax"))

		for rad in np.linspace(r_min, r_max, r_intervals):
			# The buried volume is defined in terms of occupied voxels. There is only one way to compute it
			if options.volume:
				bur_vol, bur_shell = buried_vol(occ_grid, grid, origin, rad, options.grid, strip_width, options.verbose)
			# Sterimol parameters can be obtained from VDW radii (classic) or from occupied voxels (new=default)
			if options.sterimol == 'grid':
				L, Bmax, Bmin, cyl = get_cube_sterimol(occ_grid, rad, options.grid, strip_width)
			elif options.sterimol == 'classic':
				if options.surface == 'vdw':
					L, Bmax, Bmin, cyl = get_classic_sterimol(mol.CARTESIANS, mol.RADII,mol.ATOMTYPES, options.spec_atom_1, options.spec_atom_2)
				elif options.surface == 'density':
					print("   Can't use classic Sterimol with the isodensity surface. Either use VDW radii (--surface vdw) or use grid Sterimol (--sterimol grid)"); exit()

			# Tabulate result
			if options.volume:
				# for pymol visualization
				spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f}".format(rad))
				print("   {:6.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(rad, bur_vol, bur_shell, Bmin, Bmax))
			else:
				print("   {} / Bmin: {:5.2f} / Bmax: {:5.2f} / L: {:5.2f}".format(file, Bmin, Bmax, L))

			# for pymol visualization
			for c in cyl:
				cylinders.append(c)

		# Stop timing the loop
		call_time = time.time() - start - setup_time

		# recompute L if a scan has been performed
		if options.sterimol == 'grid' and r_intervals >1:
			L, Bmax, Bmin, cyl = get_cube_sterimol(occ_grid, rad, options.grid, 0.0)
			print('\n   L parameter is {:5.2f} Ang'.format(L))
		cylinders.append('   CYLINDER, 0., 0., 0., 0., 0., {:5.3f}, 0.1, 1.0, 1.0, 1.0, 0., 0.0, 1.0,'.format(L))

		# Report timing for the whole program and write a PyMol script
		if options.timing == True: print('   Timing: Setup {:5.1f} / Calculate {:5.1f} (secs)'.format(setup_time, call_time))
		xyz_export(file,mol)
		pymol_export(file, mol, spheres, cylinders, options.isoval)

if __name__ == "__main__":
	main()

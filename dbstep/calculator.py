# -*- coding: UTF-8 -*-
import math
import numpy as np
import sys


"""
calculator

Performs calculations for finding angles, translation and rotation of molecules
"""


metals = ["Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Rb","Sr","Y","Zr","Nb","Mo",
	"Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
	"Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf",
	"Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Uut","Fl","Uup","Lv"]
	
	
def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2' """
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	cosang = np.dot(v1_u,v2_u)
	sinang = np.linalg.norm(np.cross(v1_u,v2_u))
	ang = np.arctan2(sinang,cosang)
	return math.degrees(ang)


def unit_vector(vector):
	""" Returns the unit vector of the vector """
	return vector / np.linalg.norm(vector)
	

def point_vec(coords, spec_atom_2):
	"""returns coordinate vector between any number of atoms """
	point = np.array([0.0,0.0,0.0])
	
	for atom in spec_atom_2:
		point += coords[atom-1]
		
	return point
	

def rotate_mol(coords, atoms, spec_atom_1, lig_point, options, cube_origin=False, cube_inc=False):
	"""Rotates molecule around X- and Y-axes to align M-L bond to Z-axis"""
	center_id = spec_atom_1 - 1
	atom3 = options.atom3
	
	try:
		ml_vec = lig_point - coords[center_id]
		
		zrot_angle = angle_between(unit_vector(ml_vec), [0.0, 0.0, 1.0])
		newcoord=[]
		new_inc=[]
		center = [0.,0.,0.]
		if np.linalg.norm(zrot_angle) == 0:
			if options.verbose: print("   No rotation necessary :)")
		else:
			for i in range(0,len(coords)):
				newcoord.append(coords[i])
			if cube_inc != False:
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
			newcoord = np.asarray(newcoord)
			
			#if a third atom requested, rotate around z axis to align atom3 to the positive x direction & y=0
			if atom3 != False:
				#get atom 2-3 vector
				for n, atom in enumerate(atoms):
					if atom+str(n+1) == atom3:
						atom3_id = n
				
				atom23_vec = newcoord[atom3_id] - lig_point
				xy =[atom23_vec[0],atom23_vec[1]]
				if xy != [0.0,0.0]:
					u_xy = unit_vector(xy)
					rot_angle = angle_between(xy, [1.0, 0.0])
					phi = rot_angle /180. * math.pi
					quadrant_check = math.atan2(u_xy[1],u_xy[0])
					if quadrant_check > 0 and quadrant_check <= math.pi:
						phi = 2 * math.pi - phi
					if options.verbose ==True: print('   Rotating molecule about Z-axis {0:.2f} degrees'.format(phi*180/math.pi))
			
					for i,atom in enumerate(atoms):
						center = [0.,0.,0.]
						#rotate coords around z axis
						v = [float(newcoord[i][0]) - center[0], float(newcoord[i][1]) - center[1], float(newcoord[i][2]) - center[2]]
						px = v[0]*math.cos(phi) - v[1]*math.sin(phi)
						py = v[0]*math.sin(phi) + v[1]*math.cos(phi)
						pz = v[2]
						rot2 = [round(px,8), round(py,8), round(pz,8)]
						newcoord[i]=rot2
			
					if cube_inc is not False:
						for i in range(len(cube_inc)):
							#center = cube_origin
							w = [float(new_inc[i][0]) - center[0], float(new_inc[i][1]) - center[1], float(new_inc[i][2]) - center[2]]
							qx = w[0]*math.cos(phi) - w[1]*math.sin(phi)
							qy = w[0]*math.sin(phi) + w[1]*math.cos(phi)
							qz = w[2]
							rot2 = [round(qx,8), round(qy,8), round(qz,8)]
							new_inc[i]=rot2
				newcoord = np.asarray(newcoord)
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
		print("\nRotation Error: ",e)
	return coords

	
def translate_mol(MOL, options, origin):
	"""# Translates molecule to place center atom at cartesian origin [0,0,0]"""
	coords, atoms, spec_atom = MOL.CARTESIANS, MOL.ATOMTYPES, options.spec_atom_1
	base_id = spec_atom - 1
	base_atom = atoms[base_id]
	try:
		displacement = coords[base_id] - origin
		if np.linalg.norm(displacement) == 0:
			if options.verbose: print("\n   Molecule is defined with {}{} at the origin".format(base_atom,(base_id+1)))
		else:
			if options.verbose == True: print("\n   Translating molecule by {} to set {}{} at the origin".format(-displacement, base_atom, (base_id+1)))
		for n, coord in enumerate(coords):
			coords[n] = coords[n] - displacement
	except:
		   sys.exit("   WARNING! Unable to find an atom to set at the origin")
	return coords


def translate_dens(mol, options, xmin, xmax, ymin, ymax, zmin, zmax, xyz_max, origin):
	""" Translates molecule so that a specified atom (spec_atom) is at the origin. Defaults to a metal if no atom is specified."""
	coords, atoms, cube_origin = mol.CARTESIANS, mol.ATOMTYPES,mol.ORIGIN
	spec_atom = options.spec_atom_1
	for n, atom in enumerate(atoms):
		if not spec_atom:
			if atom in metals:
				base_id, base_atom = n, atom
		else:
			if n+1 == spec_atom:
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
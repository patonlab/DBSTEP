# -*- coding: UTF-8 -*-
import os


"""
writer

Contains classes and methods to write data to files

"""


BOHR_TO_ANG = 0.529177249



class Logger:
	"""Enables output to terminal and to text file"""
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


def pymol_export(file, mol, spheres, cylinders, isoval):
	"""Outputs a python script that can be imported into PyMol (with 'run script.py')"""
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

	log.Writeonlyfile('\ncmd.load("'+name+'_transform.xyz")')
	log.Writeonlyfile('cmd.show_as("spheres", "'+base.split('/')[-1]+'_transform")')
	log.Writeonlyfile('cmd.set("sphere_transparency", 0.5)')
	log.Writeonlyfile('cmd.set("orthoscopic", "on")')


def xyz_export(file,mol):
	"""Write xyz coordinates of molecule to file"""
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
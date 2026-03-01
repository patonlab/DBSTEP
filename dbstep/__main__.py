import os
import sys

# If we are running from a wheel, add the wheel to sys.path
if __package__ == "":
	path = os.path.dirname(os.path.dirname(__file__))
	sys.path.insert(0, path)

from dbstep import Dbstep

if __name__ == "__main__":
	sys.exit(Dbstep.main())

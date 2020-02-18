import os
import glob

path = "./"
files = glob.glob(path + 'Q1.py', path + 'Q1.py')

for f in files:
    os.rename(f, os.path.join(path, 'img_' + os.path.basename(f)))

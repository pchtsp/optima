import zipfile
import os

path = r'C:\Users\franco.peschiera.fr\Documents\optima_results/'
files = [os.path.join(path, f) for f in  os.listdir(path)]
dirs = [f for f in files if os.path.isdir(f)]
zipfiles = [f + '.zip' for f in dirs]
basenames = [os.path.basename(f) for f in dirs]

pos = 1
path_to_zip = dirs[pos]
for pos, path_to_zip in enumerate(dirs):
    zipobj = zipfile.ZipFile(zipfiles[pos], mode='w')
    dirname = basenames[pos]
    for root, _dirs, _files in os.walk(path_to_zip):
       # print(root)
       for file in _files:
           path_abs = os.path.join(root, file)
           path_rel = os.path.relpath(os.path.join(root, file), path)
           zipobj.write(path_abs, path_rel)
    zipobj.close()

import utility_scripts.flatex as fl
import os

base_file = r'C:\Users\pchtsp\Documents\projects\NPS2019\article.tex'
output_file = r'C:\Users\pchtsp\Documents\projects\NPS2019\manuscript.tex'
current_path = os.path.split(base_file)[0]
g = open(output_file, "w")
g.write(''.join(fl.expand_file(base_file, current_path, False, False)))
g.close()
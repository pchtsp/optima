import utility_scripts.flatex as fl
import os

directory = r'/home/pchtsp/Documents/projects/NPS2019/'
main_file = 'article.tex'
manuscript_file = 'manuscript.tex'
base_file = os.path.join(directory, main_file)
output_file = os.path.join(directory, manuscript_file)

def edit_file(base_file, output_file):
    current_path = os.path.split(base_file)[0]
    g = open(output_file, "w", encoding="utf8")
    g.write(''.join(fl.expand_file(base_file, current_path, True, False)))
    g.close()


# merge .tex
edit_file(base_file, output_file)


# compare latex:
# latexdiff --config="PICTUREENV=(?:picture|DIFnomarkup|align|figure|table)[\w\d*@]*" manuscript_old.tex manuscript_new.tex > tmpdiff.tex
# latexdiff manuscript_old.tex manuscript_new.tex > tmpdiff.tex
# (replace quotes apparently and some equation).
# replace
# \"([ \_\\\*\w\{]+)\"
# \"([\$',\& \_\\\*\w\{\}\)\(]+)\"
# ``\1''
# replace
# ^\\\\
#
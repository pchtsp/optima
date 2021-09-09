import utility_scripts.flatex as fl
import os

directory = r'/home/pchtsp/Documents/projects/Graph2020/'
directory = r'/home/pchtsp/Documents/projects/COR2019/'
main_file = 'article.tex'
# main_file = 'phdthesis.tex'
manuscript_file = 'zzzmanuscript_new.tex'
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
# latexdiff --config="PICTUREENV=(?:picture|DIFnomarkup|align|figure|table)[\w\d*@]*" zzzzzzmanuscript_old.tex zzzzzzmanuscript_new.tex > tmpdiff.tex
# pdflatex tmpdiff.tex
# (replace quotes apparently and some equation).
# replace
# \"([ \_\\\*\w\{]+)\"
# \"([\$',\& \_\\\*\w\{\}\)\(]+)\"
# ``\1''
# replace
# ^\\\\
#
# if everything's good just this line should work:
# latexdiff zzzmanuscript_old.tex zzzmanuscript_new.tex > tmpdiff.tex && pdflatex tmpdiff.tex

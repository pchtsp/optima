#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv

# Ecriture de la solution finale dans le csv de sortie
def solution_to_csv(df,nom_fichier):
    # fonction to_csv importée de Pandas. 'T' pour transposer la matrice
    df.T.to_csv(nom_fichier, sep=';')

# Creation d'un fichier indicateurs.csv # NON utilisé
def ecriture_donnees(l1,l3,t):
    with open('indicateurs.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',lineterminator = '\n')
        spamwriter.writerow(['pas de temps']+list(range(1, t-3)))
        spamwriter.writerow(['nombre d\'avions en maintenance']+ l1)
        spamwriter.writerow(['mip']+ l3)
        #spamwriter.writerow(['nombre d\'heures en metropole']+ l2)

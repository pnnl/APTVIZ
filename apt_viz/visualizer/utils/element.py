'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
'''

""" Van der Waals radii in [A] taken from:
A cartography of the van der Waals territories
S. Alvarez, Dalton Trans., 2013, 42, 8617-8636
DOI: 10.1039/C3DT50599E
"""

import numpy as np



chemical_symbols = [
    # 0
    'Noise',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

atomic_numbers = {}
for Z, symbol in enumerate(chemical_symbols):
    atomic_numbers[symbol] = Z

vdw_radii = np.array([
         0, # X  
       1.2, # H  
      1.43, # He [larger uncertainty]
      2.12, # Li 
      1.98, # Be 
      1.91, # B  
      1.77, # C  
      1.66, # N  
       1.5, # O  
      1.46, # F  
      1.58, # Ne [larger uncertainty]
       2.5, # Na 
      2.51, # Mg 
      2.25, # Al 
      2.19, # Si 
       1.9, # P  
      1.89, # S  
      1.82, # Cl 
      1.83, # Ar 
      2.73, # K  
      2.62, # Ca 
      2.58, # Sc 
      2.46, # Ti 
      2.42, # V  
      2.45, # Cr 
      2.45, # Mn 
      2.44, # Fe 
       2.4, # Co 
       2.4, # Ni 
      2.38, # Cu 
      2.39, # Zn 
      2.32, # Ga 
      2.29, # Ge 
      1.88, # As 
      1.82, # Se 
      1.86, # Br 
      2.25, # Kr 
      3.21, # Rb 
      2.84, # Sr 
      2.75, # Y  
      2.52, # Zr 
      2.56, # Nb 
      2.45, # Mo 
      2.44, # Tc 
      2.46, # Ru 
      2.44, # Rh 
      2.15, # Pd 
      2.53, # Ag 
      2.49, # Cd 
      2.43, # In 
      2.42, # Sn 
      2.47, # Sb 
      1.99, # Te 
      2.04, # I  
      2.06, # Xe 
      3.48, # Cs 
      3.03, # Ba 
      2.98, # La 
      2.88, # Ce 
      2.92, # Pr 
      2.95, # Nd 
    np.nan, # Pm 
       2.9, # Sm 
      2.87, # Eu 
      2.83, # Gd 
      2.79, # Tb 
      2.87, # Dy 
      2.81, # Ho 
      2.83, # Er 
      2.79, # Tm 
       2.8, # Yb 
      2.74, # Lu 
      2.63, # Hf 
      2.53, # Ta 
      2.57, # W  
      2.49, # Re 
      2.48, # Os 
      2.41, # Ir 
      2.29, # Pt 
      2.32, # Au 
      2.45, # Hg 
      2.47, # Tl 
       2.6, # Pb 
      2.54, # Bi 
    np.nan, # Po 
    np.nan, # At 
    np.nan, # Rn 
    np.nan, # Fr 
    np.nan, # Ra 
       2.8, # Ac [larger uncertainty]
      2.93, # Th 
      2.88, # Pa [larger uncertainty]
      2.71, # U  
      2.82, # Np 
      2.81, # Pu 
      2.83, # Am 
      3.05, # Cm [larger uncertainty]
       3.4, # Bk [larger uncertainty]
      3.05, # Cf [larger uncertainty]
       2.7, # Es [larger uncertainty]
    np.nan, # Fm 
    np.nan, # Md 
    np.nan, # No 
    np.nan, # Lr
])

def assign_colors(elem):
    if elem == 'Fe' or elem == 26:
        return '#FFB482'
    elif elem == 'Cr' or elem == 24:
        return '#8DE5A1'
    elif elem == 'Ni' or elem == 28:
        return '#A1C9F4'
    elif elem == 'Mn' or elem == 25:
        return '#D0BBFF'
    elif elem == 'Si' or elem == 14:
        return '#DEBB9B'
    elif elem == 'Mo' or elem == 42:
        return '#FAB0E4'
    elif elem == 'H' or elem == 1:
        return '#B9F2F0'
    elif elem == 'C' or elem == 6:
        return '#CFCFCF'
    elif elem == 'P' or elem == 15:
        return '#FFFEA3'
    elif elem == 'O' or elem == 16:
        return '#FF6962'
    elif elem == 'Au' or elem == 79:
        return '#E7D27C'
    else:
        return '#F5F5F5'

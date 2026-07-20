#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:06:56 2021

@author: xmiao
"""

import numpy as np

BOHR_TO_ANGS = 0.529177210903
ATOM_NAMES = ["dummy",
              "h", "he",
              "li", "be", "b", "c", "n", "o", "f", "ne",
              "na", "mg", "al", "si", "p", "s", "cl", "ar",
              "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr",
              "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i", "xe",
              "cs", "ba",
              "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "th", "yt", "lu",
              "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn"]

def atomic_number(atom):
    try:
        # if atom is a number it represents a point charge
        atno = float(atom)
    except ValueError:
        # remove integers meant to label inequivalent atoms of the same element, C12 -> C
        elem = atom.lower().translate(str.maketrans('', '', '0123456789'))
        atno = ATOM_NAMES.index(elem)
    return atno

class XYZReader(object):

    def __init__(self, filename, units="Angstrom", fragment_id=atomic_number):
        assert units in ["bohr", "Angstrom", "hartree/bohr", ""]
        self.igeo = 1
        self.fname = filename
        self.units = units
        self.fragment_id = fragment_id

    def __enter__(self):
        self.file_obj = open(self.fname, 'rt')
        return self

    def __exit__(self, *exc_info):
        self.file_obj.close()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file_obj.readline().strip()
        if not line:
            # end of file reached
            raise StopIteration()
        words = line.strip().split()
        if words[0] != "Tv":
            # skip lattice vectors
            try:
                nat = int(words[0])
            except ValueError as e:
                print(e)
                raise Exception("Probably wrong number of atoms in xyz-file '%s'" % self.fname)
            # skip title
            _ = self.file_obj.readline()
            # read coordinates of nat atoms
            atoms = []
            for i in range(nat):
                line = self.file_obj.readline()
                words = line.split()
                atno = self.fragment_id(words[0])
                x, y, z = map(float, words[1:4])
                if self.units == "Angstrom":
                    x, y, z = map(lambda c: c/BOHR_TO_ANGS, [x, y, z])
                atoms.append((atno, (x, y, z)))
            self.igeo += 1
            
            return atoms

def atomlist_to_atno(atomlist):
    return np.array([x[0] for x in atomlist])

def atomlist_to_coord(atomlist):
    return np.vstack([x[1] for x in atomlist])

def get_coords(filename, maxlen=None, units="Angstrom"):
    with XYZReader(filename, units=units) as geomgen:
        atomlist0 = next(geomgen)
        atomic_numbers = atomlist_to_atno(atomlist0)
        coords = [atomlist_to_coord(atomlist0)]
        
        if maxlen is not None:
            for i, geom in enumerate(geomgen):
                if i < maxlen:
                    coords.append(atomlist_to_coord(geom))
        else:
            for geom in geomgen:
                coords.append(atomlist_to_coord(geom))
        
        coords = np.array(coords)
    
    return atomic_numbers, coords
        
if __name__ == '__main__':
    atomic_numbers, coords = get_coords('dynamics.xyz', maxlen=100)
    
    print(atomic_numbers)
    print(coords)
    print(coords.shape)
    
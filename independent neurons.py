# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:09:11 2018

@author: ailingwang
"""

import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from scipy.misc import imread
from glob import glob
from numpy import save

def to_tuple(li):
    """Transform list of lists to list of tuples"""
    new_li = []
    for l in li:
        li_tuple = []
        for coor in l:
            li_tuple.append(tuple(coor))
        new_li.append(li_tuple)
    return new_li   
    
def all_coordinates(li):
    """Get all coordinates and return a set of tuples"""
    new_li = []
    for l in li:
        for item in l:
            new_li.append(item)
    return set(new_li)
    
def not_adjacent(tu,rest_coor):
    """Check if tuple tu is not adjacent with any tuple in set rest_coor
        return true if it is not adjacent"""
    not_ad = False
    if (tu[0]+1,tu[1]) not in rest_coor \
            and (tu[0]-1, tu[1]) not in rest_coor\
            and (tu[0]+1, tu[1]+1) not in rest_coor \
            and (tu[0]-1, tu[1]+1) not in rest_coor\
            and (tu[0],tu[1]+1) not in rest_coor \
            and (tu[0], tu[1]-1) not in rest_coor\
            and (tu[0]+1, tu[1]-1) not in rest_coor \
            and (tu[0]-1, tu[1]-1) not in rest_coor:
                not_ad = True

    return not_ad

def independent_neuron(li):
    """Remove overlapping and adjacent pixels"""
    new_li = []
    all_coor = all_coordinates(li)
    for i in range(len(li)):
        rest_coor = all_coor - set(li[i])
        new_sm_li = []
        for item in li[i]:
            if item not in rest_coor and not_adjacent(item,rest_coor):
                new_sm_li.append(item)
        new_li.append(new_sm_li)
    return new_li


def tomask(coords):
    """Create mask for a list of tuples(coordinates)"""
    mask = zeros(dims)
    for i in coords:
        mask[i] = 1
    return mask


if __name__ == "__main__":
    
    # load the images
    files = sorted(glob('data/images/*.tiff'))
    imgs = array([imread(f) for f in files[:20]])
    dims = imgs.shape[1:]
    
    # load the regions (training data only)
    with open('data/regions/regions.json') as f:
        regions = json.load(f)
        
    # Get the independent coordinates
    coordinates_list = to_tuple([s['coordinates'] for s in regions])
    independent_coor = independent_neuron(coordinates_list)
    
    masks = array([tomask(s) for s in independent_coor])
    
    masks.shape
    
    neurons =  masks.sum(axis=0)
    
    save("neurons",neurons)
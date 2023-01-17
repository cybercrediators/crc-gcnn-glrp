import json
import os
import copy
import pprint


def load_config(fname):
    '''load the json model config'''
    with open(fname) as conf:
        config = json.load(conf)

        return config

def show_config(config):
    '''pretty print given model config'''
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

def without(inp, key):
    ret = inp.copy()
    ret.pop(key)
    return ret

def save_config(fname, config):
    """Update/Save the given config/file"""
    sc = without(config, 'precision')
    print(sc)
    with open(fname, 'w') as f:
        json.dump(sc, f, indent=4)

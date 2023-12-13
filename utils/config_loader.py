import os
import json
import yaml
import re
import ipdb
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def read_yaml(paths,branch=None):
    '''branch: chose the branch of target yaml file,
        by default(None): read the whole structure
    '''
    with open(paths,'r') as f:
        conf = yaml.load(f,Loader=loader)
        if(branch!=None):
            conf = conf[branch]
    return conf

BASE_PATH = os.path.abspath(os.path.split(__file__)[0]+'/../../data/base_config.yaml')
def load_base_path():
    '''base_path load default branch in default
    include 'root_path' and 'dataset_path' 
    '''
    conf = read_yaml(BASE_PATH,'default')
    return conf

class Params(object):
    def __init__(self):
        pass

    def _flash_conf(self,conf):
        for key,val in conf.items():
            if key not in self.__dict__:
                raise Exception(f"{key} is not initialized")
            if val is not None:
                self.__dict__[key]=val

    #read config from xxx.yaml
    def _read_from_config(self,config_path,config_choice="default"):
        with open(config_path,'r') as f:
            cfg=json.load(f)[config_choice]
        for key,val in cfg.items():
            if val is not None:
                self.__dict__[key]=val

    def _complete_paths(self,root_path,abspath):
        abss = self.__dict__[abspath]
        abs_path = os.path.join(root_path,abss)
        return abs_path

    def _complete_formed_paths(self,root_path,abspath):
        dataset = self.__dict__['dataset']
        if abspath == 'mask_root_dir':
            abs_path= os.path.join(root_path,dataset,'segmentation')
        if abspath == 'image_root_dir':
            abs_path= os.path.join(root_path,dataset,'data')
        if abspath == 'val_root_dir':
            abs_path= os.path.join(root_path,dataset,'data')
        if abspath == 'image_property_dir':
            abs_path= os.path.join(root_path,dataset,'data_annotation/train_list.json')
        return abs_path
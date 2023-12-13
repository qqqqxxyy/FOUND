import yaml
import re
import ipdb
paths='/data/users/quxiangyu/project/grinder/FOUND/data/base_config.yaml'

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

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

with open(paths,'r') as f:
    conf = yaml.load(f,Loader=loader)
    conf = conf['choice_1']
    print('hyperparameters from base_config: ' + ', '.join(f'{k}={v}' for k, v in conf.items()))
    # ipdb.set_trace()
    conf = Struct(**conf)
    # for k,v in conf:
    #     print(k,':',v)
# print(''.join(f'{k}={v}'for k,v in conf.items()))
print("space")
print("space")
print("space")

for item in conf.__dict__.keys():
    print(item,conf.__dict__[item])

ipdb.set_trace()


# print(conf.)
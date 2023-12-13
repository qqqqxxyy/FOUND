import os
import sys
import ipdb
sys.path.append('..')
from utils.config_loader import Params,load_base_path,read_yaml
from utils.datasets import ObjLocDataset,Trans,TransBbox
from utils.logger import Info_Record
import argparse
from torchvision import transforms
from tqdm import tqdm
class TestParam(Params):
    def __init__(self):
        super(TestParam, self).__init__()
        #explicit display all hyperparameters used in progress
        #hyperparameter from flexible_config.yaml
        #model
        self.arch= 'vit_small'
        self.patch_size= 8
        self.pre_training= 'dino'
        # found
        self.bkg_th= 0.3
        self.feats= 'k'
        # training
        self.dataset= 'DUTS-TR'
        self.dataset_set= 'null'
        # Hyper params
        self.seed= 0
        self.max_iter= 500
        self.nb_epochs= 3
        self.batch_size= 50
        self.lr0=5e-2
        self.step_lr_size= 50
        self.step_lr_gamma= 0.95
        self.w_bs_loss= 1.5
        self.w_self_loss= 4.0
        self.stop_bkg_loss= 100
        # Augmentations
        self.crop_size= 224
        self.scale_range= [0.1, 3.0]
        self.photometric_aug= 'gaussian_blur'
        self.proba_photometric_aug= 0.5
        self.cropping_strategy= 'random_scale'
        self.gen_device= '6'
        #evaluation
        self.type= 'saliency'
        self.datasets= ['DUTS-TEST', 'DUTS-TEST']
        self.freq= 50
        self.n_steps = 500
        #base path
        base_conf=load_base_path()
        self.root_path = base_conf['root_path']
        self.dataset_path = base_conf['dataset_path']
        #data path
        self.val_property_dir = ''
        #load hyperparameters from flexible conf
        parser = argparse.ArgumentParser(description='Test Parameters')
        parser.add_argument('-fcp','--flexible_conf_path', type=str, default=None)
        parser.add_argument('-fcb','--flexible_conf_branch', type=str, default=None)
        args = parser.parse_args()
        flexible_conf = read_yaml(args.flexible_conf_path,args.flexible_conf_branch)
        self._flash_conf(flexible_conf)
        
        #补全model_weight,val_property_dir路径
        # self.model_weight=os.path.join(self.root_path,'weight',self.model_weight)
        self.val_property_dir=os.path.join(self.dataset_path,self.dataset,'data_annotation',self.val_property_dir)
        #连接数据路径
        self.mask_root_dir=self._complete_formed_paths(self.dataset_path,'mask_root_dir')
        self.val_root_dir=self._complete_formed_paths(self.dataset_path,'val_root_dir')
        self.val_property_dir=self._complete_paths(self.dataset_path,'val_property_dir')



def main():
    param = TestParam()
    # print(param.__dict__)
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([transforms.Resize((128,128)),
                transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,128)]) ]))
    
    for t in tqdm(range(param.n_steps)):
        _, image,clas,bbox,size = target_ds[t]
        ipdb.set_trace()



if __name__ == '__main__':
    main()
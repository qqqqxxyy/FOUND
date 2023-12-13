'''
This lib includes several sort of datasets,
and some packaged transforme utils:
Trans, TransBbox, TransMasks 
'''

from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
from PIL import Image

class Trans(object):
    '''
    crop_style:[label,number]
    [None,None]; [central_crop,None]; [center_crop,size];
    [random_crop,size]; tencrop;
    '''
    def __init__(self,Trans_style):
        self.trans_style = Trans_style[0]
        self.prop = Trans_style[1]
    
    def __call__(self,img):
        if self.trans_style == 'Central_crop':
            dims = img.size
            crop = transforms.CenterCrop(min(dims[0], dims[1]))
            img = crop(img)
        elif self.trans_style == 'Normalize':
            img = self._normalize(img)
        return img
    
    def _normalize(self,img):
        if self.prop == 'standard':
            mean_vals = [0.485, 0.456, 0.406]
            std_vals = [0.229, 0.224, 0.225]
        elif self.prop == '0.5':
            mean_vals = [0.5, 0.5, 0.5]
            std_vals = [0.5, 0.5, 0.5]
        else:
            raise Exception('undefined prop {}'.format(self.prop))
        img = transforms.Normalize(mean_vals,std_vals)(img)
        return img

class TransBbox(object):
    def __init__(self,Trans_style):
        self.trans_style = Trans_style[0]
        self.prop = Trans_style[1]
    
    def __call__(self,sample):
        bbox,size = sample
        if self.trans_style == None:
            pass
        elif self.trans_style == 'Central_crop':
            bbox,size = self._central_crop(bbox,size)
        elif self.trans_style == 'Resize':
            bbox,size = self._resize(bbox,size)
        else:
            raise Exception('undefined prop {}'.format(self.trans_style))
        return [bbox,size]

    def _central_crop(self,bbox_lis,size):
        out_bbox_lis=[]
        for bbox in bbox_lis:
            w,h = size
            x0,y0,x1,y1=bbox
            if w>=h :
                d = 1/2*(w-h)
                x0,x1 = x0-d,x1-d
                x0 = max(0,x0)
                x1 = min(h,x1)
                w=h
            else:
                d = 1/2*(h-w)
                y0,y1 = y0-d,y1-d
                y0 = max(0,y0)
                y1 = min(w,y1)
                h=w
            out_bbox_lis.append([x0,y0,x1,y1])
        return out_bbox_lis,[w,h]
    
    def _resize(self,bbox_lis,size):
        '''
        inputs: 
        size: (w,h) or (w)
        '''
        out_bbox_lis=[]
        for bbox in bbox_lis:
            w,h = size
            x0,y0,x1,y1 = bbox
            
            if len(self.prop) == 2:
                W,H = self.prop
                x0,x1 = W/w*x0,W/w*x1
                y0,y1 = H/h*y0,H/h*y1
            elif len(self.prop) ==1:
                w,h = size
                if w<h:
                    W = self.prop[0]
                    ratio = self.prop[0]/w
                    H = ratio*h
                else:
                    H = self.prop[0]
                    ratio = self.prop[0]/h
                    W = ratio*w
                x0,x1,y0,y1 = x0*ratio,x1*ratio,y0*ratio,y1*ratio
            else:
                raise Exception('unacceptable length of size : len={}'.format(len(self.prop)))
            out_bbox_lis.append([x0,y0,x1,y1])
        return out_bbox_lis,[W,H]

class TransMask(object):
    '''
    structure of sample
    '''
    def __init__(self,Tran_style):
        self.trans_style = Tran_style[0]
        self.prop = Tran_style[1]

    def __call__(self,sample):
        if self.trans_style == 'Resize':
            sample = self._resize(sample)
        elif self.trans_style == 'RandomCrop':
            seed = np.random.randint(2147483647)
            sample = self._random_crop(sample,seed)
        elif self.trans_style == 'ToTensor':
            sample = self._to_tensor(sample)
        elif self.trans_style == 'Normalize':
            sample = self._normalize(sample)
        else:
            raise Exception('Wrong transformation instruct {}'.format(self.trans_style))
        return sample
    
    def _resize(self,sample):
        image,mask = sample
        size = self.prop
        image_t = transforms.Resize(size)(image)
        mask_t = transforms.Resize(size)(mask)
        return [image_t, mask_t]

    def _random_crop(self,sample,seed):
        image,mask = sample
        size = self.prop
        torch.random.manual_seed(seed)
        image_t = transforms.RandomCrop(size)(image)
        torch.random.manual_seed(seed)
        mask_t = transforms.RandomCrop(size)(mask)
        return [image_t, mask_t]
    
    def _to_tensor(self,sample):
        image,mask = sample
        image_t = transforms.ToTensor()(image)
        mask_t = transforms.ToTensor()(mask)  
        return [image_t, mask_t]

    def _normalize(self,sample):
        image,mask = sample
        if self.prop == 'standard':
            mean_vals = [0.485, 0.456, 0.406]
            std_vals = [0.229, 0.224, 0.225]
        elif self.prop == '0.5':
            mean_vals = [0.5, 0.5, 0.5]
            std_vals = [0.5, 0.5, 0.5]
        else:
            raise Exception('undefined prop {}'.format(self.prop))
        image_t = transforms.Normalize(mean_vals,std_vals)(image)
        mask_t = mask
        return [image_t, mask_t]


class dataset_file_load(object):
    #def __init__(self):
        #self.file_path = file_path
        #self.root_path = root_path

    def read_path_list(self,file_path,root_path,mask=False):
        '''
        由于mask路径要把.jpg改成.png,因此要做额外的处理
        '''
        relative_path = self._read_half_path(file_path,mask)
        absolute_path = self._concat_path_list(relative_path,root_path)
        
        return absolute_path

    def read_key_list(self,file_path,key):
        '''读取file_path中文件属性中关于key属性的list
        '''
        with open(file_path,'r') as f:
            property_list = json.load(f)
        
        key_list = []
        for dic in property_list:
            key_list.append(dic[key])    
        return key_list            

    def _read_half_path(self,file_path,mask):
        if file_path == None:
            file_path = self.file_path

        with open(file_path,'r') as f:
            property_list = json.load(f)
        path_list = []
        for img_property in property_list:
            if mask == False:
                path_list.append(img_property['path'])
            else:
                path_list.append(img_property['path'].replace('jpg','png'))
        return path_list

    def _concat_path_list(self,path_list,root_path):
        absolute_path=[]
        for paths in path_list:
            paths = os.path.join(root_path,paths)
            absolute_path.append(paths)
        return absolute_path



class OriDataset(Dataset):
    '''the base function of dataset, including: 
    image loading, class loading, tsfm of image
    input:
    images_root: string variable, the path of dataset
    image_prop: string variable, the path of property: json file
    tsfm: dict funtion, {'resize':tuple, 'normalize': /; 'crop':/}
    output:
    idx: the index is called for
    img: tensor (w,h,3)
    clas: tensor (0)
    '''
    def __init__(self,images_root,image_prop,tsfm=None,mask=False,hdf5=False):
        self.load_tool = dataset_file_load()
        #read image path list
        
        self.hdf5 = hdf5
        if self.hdf5 == True:
            #images_root: 索取要读取的目录文件
            #images_path_list: 文件名
            self.image_path_list = self.load_tool._read_half_path(image_prop,mask)
            self.images_root = images_root+'.hdf'
            # self.images_root = '/mnt/usb/Dataset relate/ILSVRC_hdf5/data.hdf'
        else:
            self.image_path_list = self.load_tool.read_path_list(image_prop,images_root,mask)
        #read class
        #-1 because in property.json, the class is started with 1
        #针对VOC2012没有保存类别标签,进行伪造类别标签(不影响localization结果)
        try :
            self.clas_list = torch.tensor(self.load_tool.read_key_list(image_prop,'class'))-1
        except:
            self.clas_list = torch.tensor( [0] * len(self.image_path_list) )
            
        self.transform  = tsfm


    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self,idx):
        if self.hdf5 == True:
            raise Exception("Unsupport to load hdf5 format file")
        else:
            img = Image.open(self.image_path_list[idx])

        if self.transform != None:
            img = self.transform(img.convert('RGB'))
        clas  = self.clas_list[idx]

        return idx, img, clas


class ObjLocDataset(OriDataset):
    'extra output bounding box based on OriDataset'    
    def __init__(self,images_root,image_prop,tsfm,tsfm_bbox,hdf5=False):
        super(ObjLocDataset, self).__init__(images_root,image_prop,tsfm,hdf5=hdf5)
        self.bbox_list = self.load_tool.read_key_list(image_prop,'bbox')
        self.size_list = self.load_tool.read_key_list(image_prop,'size')
        self.tsfm_bbox = tsfm_bbox

    def __getitem__(self,idx):
        '''
        output:
        image: tensor (3,w,h)
        clas: tensor (0)
        bbox: float list [x0,y0,x1,y1]
        '''
        idx,img,clas = super(ObjLocDataset, self).__getitem__(idx)
        bbox,_ = self.tsfm_bbox([self.bbox_list[idx],self.size_list[idx][0]])
        return idx,img,clas,bbox,self.size_list[idx]
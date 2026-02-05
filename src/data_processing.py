import os
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custom_exception import CustomException
# Prefer OpenCV for speed, but fall back to Pillow if cv2 fails to import (common when
# cv2 was compiled against a different NumPy ABI).
try:
    import cv2
    _USE_OPENCV = True
except Exception as e:
    from PIL import Image
    _USE_OPENCV = False
    logger = get_logger(__name__)
    logger.warning(f'OpenCV import failed, falling back to Pillow: {e}')
else:
    logger = get_logger(__name__)

class GunData(Dataset):
    def __init__(self,root:str,device:str='cpu'):
        '''root : path of root directory'''
        self.image_path=os.path.join(root,'images')
        self.labels_path=os.path.join(root,'labels')
        self.device=device

        # If the images/labels directories contain a single subdirectory (common when
        # extracting zips), use that subdirectory so we list actual files instead of
        # a single folder name.
        def _unwrap_dir(p):
            try:
                entries = os.listdir(p)
                if len(entries) == 1 and os.path.isdir(os.path.join(p, entries[0])):
                    return os.path.join(p, entries[0])
            except Exception:
                # Keep original path if it doesn't exist or any other issue occurs
                return p
            return p

        self.image_path = _unwrap_dir(self.image_path)
        self.labels_path = _unwrap_dir(self.labels_path)

        # Gather only image files and label files (ignore directories)
        self.img_name = sorted([f for f in os.listdir(self.image_path)
                                if os.path.isfile(os.path.join(self.image_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.label_name = sorted([f for f in os.listdir(self.labels_path)
                                  if os.path.isfile(os.path.join(self.labels_path, f)) and f.lower().endswith('.txt')])

        if not self.img_name:
            logger.error(f'No image files found in {self.image_path}. Found entries: {os.listdir(self.image_path)}')
            raise CustomException(f'No image files found in {self.image_path}')
        if not self.label_name:
            logger.error(f'No label files found in {self.labels_path}. Found entries: {os.listdir(self.labels_path)}')
            raise CustomException(f'No label files found in {self.labels_path}')

        logger.info('Data Processing Initialized...')

    def __getitem__(self,idx):
        try:
            logger.info(f'loding data for index {idx}')
            ##### Loading images .... 
            image_path=os.path.join(self.image_path,str(self.img_name[idx]))
            logger.info(f'Image path : {image_path}')
            if _USE_OPENCV:
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f'Unable to read image: {image_path}')
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            else:
                # PIL fallback
                pil_image = Image.open(image_path).convert('RGB')
                img_rgb = np.array(pil_image).astype(np.float32)

            img_res = img_rgb / 255.0
            img_res = torch.as_tensor(img_res).permute(2,0,1)

            ##### Loading labels ..... 
            label_name=self.img_name[idx].rsplit('.',1)[0] + '.txt'
            label_path = os.path.join(self.labels_path,str(label_name))
            if not os.path.exists(label_path):
                raise FileExistsError(f'Label file not found : {label_path}')
            
            target={
                'boxes':torch.tensor([]),
                'area':torch.tensor([]),
                'image_id':torch.tensor([idx]),
                'labels':torch.tensor([],dtype=torch.int64),
                
            }
            
            with open (label_path,'r') as label_file:
                l_count = int(label_file.readline())
                box=[list(map(int,label_file.readline().split())) for _ in range(l_count)]
            
            if box: 
                area= [(b[2]-b[0])* (b[3]-b[1]) for b in box ]
                labels=[1]*len(box)
                target['boxes'] = torch.tensor(box,dtype=torch.float32)
                target['area'] = torch.tensor(area,dtype=torch.float32)
                target['labels'] =torch.tensor(labels,dtype=torch.int64)
            
            img_res=img_res.to(self.device)
            for key in target:
                if isinstance(target[key],torch.Tensor):
                    target[key]=target[key].to(self.device)
            return img_res,target
        except Exception as e:
            logger.error(f'Error while loading the data {e}')
            raise CustomException('Fail to load the data')
        
    def __len__(self):
        return len(self.img_name)
if __name__ == '__main__':
    root_path='artifact/raw/'
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset=GunData(root=root_path,device=device)
    image,target=dataset[0]
    print('Image Shape :',image.shape)
    print("Target keys", target.keys())
    print('Bounding boxes : ',target['boxes'])
    
            


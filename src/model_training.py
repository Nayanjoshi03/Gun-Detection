import os
import torch
from torch.utils.data import DataLoader,random_split
from torch import optim
from src.model_architecture import FasterRCNNModel

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import GunData

logger=get_logger(__name__)
model_save_path='artifact/models/'
os.makedirs(model_save_path,exist_ok=True)

class ModelTraining:
    def __init__(self,model_class,num_classes,learning_rate,epochs,dataset_path,device):
        self.model_class=model_class
        self.num_class= num_classes
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.dataset_path=dataset_path
        self.device=device
        try:
            self.model=self.model_class(self.num_class,self.device)
            self.model.to(self.device)
            logger.info("Model moved to device")
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
            logger.info("optimiser initialized ...")
        except Exception as e:
            logger.error(f'Mode training failed ... {e}')
            raise CustomException('Mode training failed ... ',e)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    def split_dataset(self):
        try:
            

import os
import torch
from torch.utils.data import DataLoader,random_split
from torch import optim
from src.model_architecture import FasterRCNNModel

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import GunData
from torch.utils.tensorboard import SummaryWriter
import time 


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
        
        ### tensorboard 
        timestamp=time.strftime('%y%m%d-%H%M%S')
        self.log_dir=f'tensorborad_logs/{timestamp}'
        os.makedirs(self.log_dir,exist_ok=True)
        
        self.writer=SummaryWriter(log_dir=self.log_dir)
        
        try:
            # FasterRCNNModel is a wrapper that holds the actual nn.Module in `.model`.
            self.model_wrapper = self.model_class(self.num_class, self.device)
            self.model = getattr(self.model_wrapper, "model", self.model_wrapper)
            # Ensure the underlying model is moved to device
            self.model.to(self.device)
            logger.info("Model moved to device")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.info("optimiser initialized ...")
        except Exception as e:
            logger.error(f'Model training failed ... {e}')
            raise CustomException('Model training failed ... ', e)
    
    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
    def split_dataset(self):
        try:
            dataset= GunData(self.dataset_path,self.device)
            dataset=torch.utils.data.Subset(dataset,range(5))
            train_size=int(0.8*len(dataset))
            val_size=len(dataset) - train_size 
            train_dataset,val_dataset =random_split(dataset,[train_size,val_size])   
            train_loader=DataLoader(train_dataset,batch_size=3,shuffle=True,num_workers=0,collate_fn=self.collate_fn)
            val_loader=DataLoader(val_dataset,batch_size=3,shuffle=False,num_workers=0,collate_fn=self.collate_fn)
            logger.info('Data splitted successfully')
            return train_loader,val_loader
        except Exception as e:
            logger.error(f'Failed to split the data ... {e}')
            raise CustomException('Failed to split the data ... ',e)
    def train(self):
        try:
            train_loader,val_loader=self.split_dataset()
            for epoch in range(self.epochs):
                logger.info(f"starting epoch :{epoch}")
                self.model.train()
                for i, (images, target) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    losses = self.model(images, target)

                    # torchvision detection models return a dict of loss components during training
                    if isinstance(losses, dict):
                        total_loss = None
                        for value in losses.values():
                            if isinstance(value, torch.Tensor):
                                total_loss = value if total_loss is None else total_loss + value
                        if total_loss is None:
                            logger.error('Error capturing losses; no tensor values found')
                            raise ValueError('No loss tensors returned by model.')
                        self.writer.add_scalar("Loss/Train",total_loss.item(),epoch*len(train_loader)+i)
                    elif isinstance(losses, torch.Tensor):
                        total_loss = losses
                        self.writer.add_scalar("Loss/Train",total_loss.item(),epoch*len(train_loader)+i)

                    else:
                        logger.error(f'Unexpected loss type: {type(losses)}')
                        raise ValueError('Unexpected loss type from model.')

                    total_loss.backward()
                    self.optimizer.step()
            self.writer.flush()
            self.model.eval()
            with torch.no_grad():
                for images,target in val_loader:
                    val_losses=self.model(images,target)
                    logger.info(f'type of validateion loss {type(val_losses)}')
                    logger.info(f'val_loss : {val_losses}')
            model_path=os.path.join(model_save_path,'fasterrcnn.pth')
            torch.save(self.model.state_dict(),model_path)
            logger.info(f'model saved succesfully ..')
        
        except Exception as e:
            logger.error(f'Failed to train model ... {e}')
            raise CustomException('Failed to train model ... ', e)
                   
                   
if __name__ =='__main__':
    device='cpu'
    training=ModelTraining(model_class=FasterRCNNModel,
                           num_classes=2,
                           learning_rate=0.0001,
                           dataset_path='artifact/raw',
                           device=device,
                           epochs=1
                           )  
    training.train()                  
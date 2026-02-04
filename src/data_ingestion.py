import os
import kagglehub
import shutil
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import * 
import zipfile 
logger =get_logger(__name__)

class DataIngestion():
    def __init__(self,dataset_name:str,target_dir:str):
        self.dataset_name=dataset_name
        self.target_dir=target_dir
    def create_raw_dir(self):
        raw_dir=os.path.join(self.target_dir,'raw')
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir,exist_ok=True)
                logger.info(f'Create the {raw_dir}')
            except Exception as e:
                logger.error('Error while creating artifact directory.. ',e)
                raise CustomException("Failed to create a raw directory",e)
        return raw_dir
    
    def extract_images_labels(self,path:str,raw_dir:str):
        try:
            if path.endswith('.zip'):
                logger.info('Extracting the zip file')
                with zipfile.ZipFile(path,'r') as zip_ref :
                    zip_ref.extractall(path) 
            image_folder=os.path.join(path,'Images') 
            labels_folder=os.path.join(path,'Labels')
            if os.path.exists(image_folder):
                shutil.move(image_folder,os.path.join(raw_dir,'Images'))
                logger.info('Moved images to raw directory') 
            else:
                logger.info('Images folder does not exist')
                
            if os.path.exists(labels_folder):
                shutil.move(labels_folder,os.path.join(raw_dir,'Labels'))
                logger.info('Moved labels to raw directory') 
            else:
                logger.info('Labels folder does not exist')
        except  Exception as e:
            logger.error('Error while extracting directory.. ',e)
            raise CustomException("Failed while Extracting the directory",e)
    
    def download_dataset(self,raw_dir:str):
        try:
            logger.info('Downloading the dataset from kaggle hub')
            path=kagglehub.dataset_download(self.dataset_name)
            logger.info('Dataset downloaded successfully')
            self.extract_images_labels(path,raw_dir)
        except Exception as e:
            logger.error('Error while downloading dataset.. ',e)
            raise CustomException("Failed to download dataset from kaggle hub",e)
        
    def run(self):
        try:
            raw_dir=self.create_raw_dir()   
            self.download_dataset(raw_dir)
        except Exception as e:
            logger.error('Error in data ingestion process.. ',e)
            raise CustomException("Failed in data ingestion process",e)

if __name__=='__main__':
    obj=DataIngestion(dataset_name=DATASET_NAME,target_dir=TARGET_DIR)
    obj.run()           
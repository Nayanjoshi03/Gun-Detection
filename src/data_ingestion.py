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
            image_folder=os.path.join(path,'images') 
            labels_folder=os.path.join(path,'labels')
            dest_images = os.path.join(raw_dir, 'images')
            dest_labels = os.path.join(raw_dir, 'labels')

            def _move_contents(src, dst):
                os.makedirs(dst, exist_ok=True)
                for name in os.listdir(src):
                    s = os.path.join(src, name)
                    d = os.path.join(dst, name)
                    # if destination exists, overwrite files
                    if os.path.exists(d):
                        if os.path.isdir(d):
                            shutil.rmtree(d)
                        else:
                            os.remove(d)
                    shutil.move(s, d)

            if os.path.exists(image_folder):
                # If destination already exists, move the contents instead of moving the folder itself
                if os.path.exists(dest_images):
                    logger.info('Destination images directory exists; merging contents')
                    _move_contents(image_folder, dest_images)
                else:
                    shutil.move(image_folder, dest_images)
                logger.info('Images moved to raw directory')
            else:
                logger.error('Images folder does not exist; creating empty images directory')
                os.makedirs(dest_images, exist_ok=True)
                logger.info('Images folder created')

            if os.path.exists(labels_folder):
                if os.path.exists(dest_labels):
                    logger.info('Destination labels directory exists; merging contents')
                    _move_contents(labels_folder, dest_labels)
                else:
                    shutil.move(labels_folder, dest_labels)
                logger.info('Labels moved to raw directory')
            else:
                logger.error('Labels folder does not exist; creating empty labels directory')
                os.makedirs(dest_labels, exist_ok=True)
                logger.info('Labels folder created')
        except  Exception as e:
            logger.error('Error while extracting directory.. ',e)
            raise CustomException("Failed while Extracting the directory",e)
    
    def download_dataset(self,raw_dir:str):
        try:
            logger.info('Downloading the dataset from kaggle hub')
            path=kagglehub.dataset_download(self.dataset_name)
            logger.info('Dataset downloaded successfully')
            self.extract_images_labels(path,raw_dir)
            logger.info(f"data saved in {path}")
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
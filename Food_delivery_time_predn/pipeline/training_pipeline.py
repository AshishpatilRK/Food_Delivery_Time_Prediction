import os
import sys
from Food_delivery_time_predn.logger import logging
from Food_delivery_time_predn.exception import CustomException
import pandas as pd

from Food_delivery_time_predn.components.data_ingestion import DataIngestion
from Food_delivery_time_predn.components.data_transformation import DataTransformation
from Food_delivery_time_predn.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)





import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler, OneHotEncoder

from Food_delivery_time_predn.exception import CustomException
from Food_delivery_time_predn.logger import logging
import os
from Food_delivery_time_predn.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled

            num_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance']

            num_cols1 = ['Vehicle_condition','multiple_deliveries','Prepn_Time']

            nominal_catcols = ["Weather_conditions", "City",'Type_of_order', 'Type_of_vehicle','Festival']

            ordinal_catcols = ["Road_traffic_density"]

            # Define the custom ranking for each ordinal variable
            traffic_categories = ['Low', 'Medium', 'High','Jam']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
            ]
             )
            num_pipeline1=Pipeline(
                 steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('scaler',StandardScaler())
            ]
             )
             # Categorigal Pipeline
            cat_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('ordinalencoder',OrdinalEncoder(categories=[traffic_categories])),
            ('scaler',StandardScaler())
            ]
             )

            cat_pipeline1=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehotencoder',OneHotEncoder(handle_unknown = "ignore")),
            ('scaler',StandardScaler(with_mean=False))
            ]
            )

             # Combine
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,num_cols),
            ('num_pipeline1',num_pipeline1,num_cols1),
            ('cat_pipeline',cat_pipeline,ordinal_catcols),
            ('cat_pipeline1',cat_pipeline1,nominal_catcols)
            ])
            
            return preprocessor
            
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)
        


            num_cols = ['Delivery_person_Age', 'Delivery_person_Ratings','distance']

            num_cols1 = ['Vehicle_condition','multiple_deliveries','Prepn_Time']

            ordinal_catcols = ['Road_traffic_density']

            nominal_catcols = ['Weather_conditions','City','Type_of_order','Type_of_vehicle','Festival']

            # Define the custom ranking for each ordinal variable
            traffic_density_categories = ['Low', 'Medium', 'High','Jam']

            logging.info('Pipeline Initiated')
        

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            num_pipeline1=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('scaler',StandardScaler())

                ]

            )
            # Categorigal Pipeline
            ordinalcat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[traffic_density_categories])),
                ('scaler',StandardScaler())
                ]

            )
            # Categorigal Pipeline
            nominalcat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder(handle_unknown = "ignore")),
                ('scaler',StandardScaler(with_mean=False))
                ]

            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,num_cols),
            ('num_pipeline1',num_pipeline1,num_cols1),
            ('ordinalcat_pipeline',ordinalcat_pipeline,ordinal_catcols),
            ('nominalcat_pipeline',nominalcat_pipeline,nominal_catcols),
            ])
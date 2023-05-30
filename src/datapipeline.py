# Please edit for reproducibility 

import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def run_datapipeline():
    """Data preprocessing pipeline

    Args:
        data_path1,data_path2,data_path3
    """
    df1 = pd.read_csv('./data/train_GzS76OK/train.csv') #'./data/train_GzS76OK/train.csv'
    df2 = pd.read_csv('./data/train_GzS76OK/fulfilment_center_info.csv') #'./data/train_GzS76OK/fulfilment_center_info.csv'
    df3 = pd.read_csv('./data/train_GzS76OK/meal_info.csv') # './data/train_GzS76OK/meal_info.csv'

    # 1. Merge all df 
    df = pd.merge(df1, df2, on='center_id',how='left')
    df = pd.merge(df,df3,on='meal_id',how='left')

    # convert weeks to time stamp
    # Calculate the date corresponding to the start of the first week
    start_date = pd.to_datetime('2019-01-28')

    # Calculate the date for each week based on the start date and the week number
    df['date'] = start_date + pd.to_timedelta(df['week'] - 1, unit='W')

    # Set the 'date' column as the index of the DataFrame
    df.set_index('date', inplace=True)

    # 2. add a add_price_adjustment column
    df['add_price_adjustment'] = df ['checkout_price'] - df ['base_price']

    # category and cuisine very similar. Hence, combine them.
    df['category-cuisine'] = df['category'] +'-'+ df['cuisine'] 

    # Split train and test first
    train_length = df['week'].nunique()*0.8 #80% train
    df_train = df.loc[df['week'] <= train_length] #week range from 1 to 116
    df_test = df.loc[df['week'] > train_length] #week range from 117 to 145

    # 3. Drop unless columns (drop the 'week' column before model train)
    df_train.drop(['id', 'week', 'checkout_price', 'cuisine','category'], axis = 1, inplace =True)
    df_test.drop(['id', 'week', 'checkout_price', 'cuisine','category'], axis = 1, inplace =True)

    #4.Split data into cat and num data for one-hot and standardscalar for df_train
    one_hot_list = ['center_type','category-cuisine']
    normalize_list = ['base_price', 'add_price_adjustment']

    ohe_transformer = OneHotEncoder()
    df_encoded_train = ohe_transformer.fit_transform(df_train[one_hot_list]).toarray()
    df_encoded_df_train = pd.DataFrame(df_encoded_train, columns=ohe_transformer.get_feature_names_out(one_hot_list),index=df_train.index)

    df_encoded_test = ohe_transformer.fit_transform(df_test[one_hot_list]).toarray()
    df_encoded_df_test = pd.DataFrame(df_encoded_test, columns=ohe_transformer.get_feature_names_out(one_hot_list),index=df_test.index)

    # Concatenate the original dataframe and the encoded dataframe
    df_train = pd.concat([df_train, df_encoded_df_train], axis=1)
    df_train.drop(one_hot_list, axis = 1, inplace =True)

    df_test = pd.concat([df_test, df_encoded_df_test], axis=1)
    df_test.drop(one_hot_list, axis = 1, inplace =True)

    # 4.num data for standard scaler
    num_transformer = StandardScaler()

    # transform the selected columns
    scaled_columns = num_transformer.fit_transform(df_train[normalize_list])

    # create a new DataFrame with the scaled columns
    df_train[normalize_list] = scaled_columns

    return df_train , df_test




# def run_datapipeline(cfg:dict) ->'pandas.core.frame.DataFrame':
#     """Data preprocessing pipeline

#     Args:
#         cfg (dict): config file 
#     """
# # load and merge dataset 
#     df1 = pd.read_csv('../data/train/fulfilment_center_info.csv')
#     df2 = pd.read_csv('../data/train/meal_info.csv')
#     if cfg['train']:
#         df3 = pd.read_csv('../data/train/train.csv')
#     else:
#         df3 = pd.read_csv('../data/test/test.csv')
#     df = pd.merge(df3, df2, on='meal_id',how='left')
#     df = pd.merge(df,df1,on='center_id',how='left')
# # train-test split
#     X = df.drop('num_orders')
#     y = df['num_orders']
#     X_train, X_test, y_train, y_test = train_test_split(X, 
#                                                         y, 
#                                                         test_size=0.2, 
#                                                         shuffle=False)
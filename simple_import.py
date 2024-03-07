#%%
import pandas as pd
import os

def read_file_names(location):
    return (os.listdir(location))

def load_and_describe_df(location,name,*args):
    if len(args) == 1: 
        df = pd.read_csv(f"{location}/{name}",names= args[0])
    else: df = pd.read_csv(f"{location}/{name}")
    return df,df.shape

root = "../data"
head = ["year", "month", "day", "hour", "temperature", "precipitation", "u-wind", "v-wind"]
df_list = read_file_names(root)
df1,df1_shape = load_and_describe_df(root,df_list[0],head)


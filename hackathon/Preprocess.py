import pandas as pd
import numpy as np
import re
from datetime import datetime

# map string data to numerical representation
def codify_job_result(dataset):
    dataset['Job Result'] = dataset['Job Result'].map({'SUCCESS': 0, 'FAILURE': 1, 'CANCELLED': 2}).astype(int)

# map string data to numerical representation
def codify_products(dataset):
    dataset['Product Name'] = dataset['Product Name'].map({'Product 1': 1, 'Product 2': 2, 'Product 3': 3, 'Product 4': 4, 'Product 5': 5}).astype(int)

# create a column on the dataset for the duration of the job
def add_duration(dataset):
    def to_dt(string):
        # produces [month, day, year, hour, minute]
        tokens = re.split('[/| |:]', string)
        return datetime(int(tokens[2]) + 2000, int(tokens[0]), int(tokens[1]), int(tokens[3]), int(tokens[4]))

    dataset.insert(len(dataset.columns), 'Duration', [to_dt(end) - to_dt(begin) for end, begin in zip(dataset['Job Completion Time'], dataset['Job Creation Time'])], True )


deploy_dataset = pd.read_csv("Anonymized Deployment Job Data.csv")
deploy_dataset.replace([np.inf, -np.inf], np.nan)
deploy_dataset = deploy_dataset.dropna()

codify_job_result(deploy_dataset)
codify_products(deploy_dataset)
add_duration(deploy_dataset)
deploy_dataset = deploy_dataset.drop(['Job Creation Time', 'Job Completion Time'], axis=1)

expansion_dataset = pd.read_csv("Anonymized Expansion Job Data.csv")
expansion_dataset.replace([np.inf, -np.inf], np.nan)
expansion_dataset = expansion_dataset.dropna()

codify_job_result(expansion_dataset)
codify_products(expansion_dataset)
add_duration(expansion_dataset)
expansion_dataset.drop(['Job Creation Time', 'Job Completion Time'], axis=1)

update_dataset = pd.read_csv('Anonymized Update Job Data.csv')
update_dataset.replace([np.inf, -np.inf], np.nan)
update_dataset = update_dataset.dropna()

codify_job_result(update_dataset)
codify_products(update_dataset)
add_duration(update_dataset)
update_dataset.drop(['Job Creation Time', 'Job Completion Time'], axis=1)
update_dataset['Component Being updated'] = update_dataset['Component Being updated'].map({'Component 1': 1, 'Component 2': 2, 'Component 3': 3, 'Component 4': 4, 'Component 5': 5}).astype(int)



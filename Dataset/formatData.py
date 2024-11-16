## Script that load meta-math/MetaMathQA and DopeorNope/Linguistic_Calibration_Llama3.1 and format it to json and lit it into train, test
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.parquet as pq

def convert_json(df, query, answer):
    #Drop column 1
    df = df.drop(df.columns[0], axis=1)
    json = []
    columns = df.columns.str.lower()
    for i in range(len(df)):
        json.append({query: df[columns[0]][i], answer: df[columns[1]][i]})
        #print(json[i])
    return json


datasets = ["Math/MetaMathQA-395K.json", "Linguistic/DopeOrNope.parquet"]



# Specify the file path for MetaMathQA data
DATA_PATH = "Dataset/"  # Update this to your local path

# Specify size of the data sets
SIZE = 10000


for j in range(len(datasets)):
    prompt_response_dict = {}
    if j == 0:
        json = json.loads(open(DATA_PATH + datasets[j]).read())
        QUERY = "query"
        ANSWER = "response"
    else:
        tbl : pa.Table = pq.read_table(DATA_PATH + datasets[j])
        df : pd.DataFrame = tbl.to_pandas()
        QUERY = "paragraph_generation_prompt"
        ANSWER = "claude_summary"
        json = convert_json(df, QUERY, ANSWER)
        

    for i in range(len(json)):
        prompt_response_dict.update( {i : 
                                        {"prompt": json[i][QUERY], 
                                        "response": json[i][ANSWER]
                                        }
                                    })
        #print(prompt_response_dict)

    # convert the dictionary to a pandas dataframe    
    df = pd.DataFrame.from_dict(prompt_response_dict, orient='index')

    # get random subset
    df = df.sample(n=SIZE)

    # Split dataframe into train and test 80/20
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Save the train and test dataframes to JSON files in #NAME/Test, #Name/Train
    train.to_json(DATA_PATH + datasets[j].split('/')[0] + "/Train/" + datasets[j].split('/')[1] + "-train.json", orient='records')
    test.to_json(DATA_PATH + datasets[j].split('/')[0] + "/Test/" + datasets[j].split('/')[1] + "-test.json", orient='records')

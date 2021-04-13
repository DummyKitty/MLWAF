import numpy as np
import pandas as pd
import csv
from IPython.display import display


def from_txt_to_dataframe(src_file,is_malicious,injection_type):
    
    #read file
    payloads_txt = open('data/{}.txt'.format(src_file),'r',encoding='UTF-8').readlines()
    
    #create dataframe
    payloads = pd.DataFrame(payloads_txt,columns=['payload'])
    payloads['is_malicious'] = [is_malicious]*len(payloads)
    payloads['injection_type'] = [injection_type]*len(payloads)

    print('First 5 lines of ' + injection_type)
    display(payloads.head())
    
    return payloads
    

if __name__ == "__main__":
    #concatenate all payload dataframes together
    payloads = pd.DataFrame(columns=['payload','is_malicious','injection_type'])
    payloads = payloads.append(from_txt_to_dataframe('SQLCollection',1,'SQL'))
    payloads = payloads.append(from_txt_to_dataframe('XSSCollection',1,'XSS'))
    payloads = payloads.append(from_txt_to_dataframe('ShellCollection',1,'SHELL'))
    payloads = payloads.append(from_txt_to_dataframe('non-maliciousCollection',0,'LEGAL'))
    payloads = payloads.reset_index(drop=True)


    #Remove ending \n and white spaces
    payloads['payload'] = payloads['payload'].str.strip('\n')
    payloads['payload'] = payloads['payload'].str.strip()

    #Remove any empty data points
    rows_before = len(payloads['payload'])
    payloads = payloads[payloads['payload'].str.len() != 0]
    print('Empty data points removed: ' + str(rows_before - len(payloads)))

    #Remove any malicious data points of size 1
    rows_before = len(payloads['payload'])
    payloads = payloads[(payloads['is_malicious'] == 0) | ((payloads['is_malicious'] == 1) & (payloads['payload'].str.len() > 1))]
    print('Malicious data points of size 1 removed: ' + str(rows_before-len(payloads)))

    #Remove duplicates
    rows_before = len(payloads['payload'])
    payloads = payloads.drop_duplicates(subset='payload', keep='last')
    print('Duplicate data points removed: ' + str(rows_before-len(payloads)))

    #Reformat rows that have the format b'<payload>' into <payload>
    payloads['payload'] = [payload[2:-1] if payload.startswith("b'") or payload.startswith('b"') 
                            else payload for payload in payloads['payload']]

    #Shuffle dataset and reset indices again
    payloads = payloads.sample(frac=1).reset_index(drop=True)
    payloads.index.name = 'index'

    #Remove payloads that cant be saved into .csv using pandas, e.g. they will be null/NA/NaN
    payloads.to_csv('data/payloads.csv',encoding='UTF-8')
    #reload dataframe from saved .csv. The dataframe will contain a few null values
    payloads = pd.read_csv("data/payloads.csv",index_col='index',encoding='UTF-8') 
    rows_before = len(payloads['payload'])
    payloads = payloads[~payloads['payload'].isnull()]
    print('null/NaN data points removed: ' + str(rows_before-len(payloads)))

    #Lastly, save to .csv
    payloads.to_csv('data/payloads.csv',encoding='UTF-8')
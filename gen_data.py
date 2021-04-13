import numpy as np
import pandas as pd
import csv
import re
import json
from IPython.display import display


def from_google_spreadsheet_to_collections(file):
    '''Converts web traffic payloads from csv file to right format into collections 

    the input format of the data points are:
    <is malicious>,<Injection type>,<Payload>
    '''
    
    df = pd.read_csv("/home/kali/Program/python/AI_WAF/MLWAF/data/{}.csv".format(file))
    
    #extract injection data
    sql_data  = df['Payload'][df['Injection Type'] == 'SQL']
    xss_data  = df['Payload'][df['Injection Type'] == 'XSS']

    print('Number of SQL injection data points: ' + str(len(sql_data)))
    print('First 5 SQL injection data points:')
    display(sql_data[:5])

    print('Number of XSS injection data points: ' + str(len(xss_data)))
    print('First 5 XSS injection data points:')
    display(xss_data[:5])
    
    with open("/home/kali/Program/python/AI_WAF/MLWAF/data/SQLCollection.txt", "a") as myfile:
        for sql_row in sql_data:
            myfile.write('{}\n'.format(sql_row.encode("utf-8")))
            
    with open("/home/kali/Program/python/AI_WAF/MLWAF/data/XSSCollection.txt","a") as myfile:
        for xss_row in xss_data:
            myfile.write('{}\n'.format(xss_row.encode("utf-8")))
    pass      

def from_xsuperbug_to_collections(src_file, dest_file):
    '''Converts web traffic payloads from xsuperbug's format to the right format into collections 
    
    the input format of the data points are:
    <injections type>##<Payload>##<number>
    '''
    
    lines = open("data/{}".format(src_file),"r").readlines()
    print('raw data in source file format: ' + lines[0])
    lines = [ re.search(r'(.*)##(.*)##[0-9]',line).group(2) for line in lines]
    print('modified data in right format: ' + lines[0])
    print(' ' + str(len(lines)))
    
    with open("data/{}".format(dest_file), "a") as myfile:
        for line in lines:
            myfile.write('{}\n'.format(line.encode("utf-8")))


def from_cnets_to_collection(src_file, dest_file):
    '''Converts web traffic payloads from CNetS' web traffic data set format to the right format into collections
    
    source data set found here: http://cnets.indiana.edu/resources/data-repository/
    the input file is in JSON format and the input format of the data points are:
    {"count": <number>, "timestamp": <Date>, "from": "<Website>/<Payload1>", "to": "<Website>/<Payload2>"}
    '''
    raw_data = []
    
    with open("data/{}.json".format(src_file)) as f:
        for line in f.readlines():
            raw_data.append(json.loads(line))
    
    #Extract 'from' and 'to' columns
    data = pd.Series([obj['from'] for obj in raw_data] + [obj['to'] for obj in raw_data]) 
    
    #Remove empty elements
    data = data[data != '']
    
    
    #Extract data containing payloads, i.e. containing the '=' sign followed by a word
    data = data[ [re.match(r'(.*)=(.+)',x) != None for x in data] ]
    
    payloads = []
    
    #extract each input from the entire payload string
    for payload in data:
        temp = payload.split('&')
        payloads = payloads + [substring.split('=')[1] for substring in temp if len(substring.split('=')) > 1]
    
    #write to destination file
    with open("data/{}".format(dest_file), "a") as myfile:
        for payload in payloads:
            if payload != '':
                myfile.write('{}\n'.format(payload))


def from_fsecurify_to_collection(src_file, dest_file):
    '''Extracts payload data inputs from address strings
    
    source data set found here: 
    https://raw.githubusercontent.com/faizann24/Fwaf-Machine-Learning-driven-Web-Application-Firewall/master/goodqueries.txt
    
    the format of the data points are:
    <Website local path>?<Payload>
    example: folder1/folder2?var1=payloadData
    '''
    payloads = []
    
    with open("data/{}".format(src_file)) as f:
        for line in f.readlines():
            splitted_address = line.split('?')
            
            #if there is payload
            if len(splitted_address) > 1:
                total_payload = splitted_address[1]
                temp = total_payload.split('&')
                
                #Add all input data from payload 
                #exclude input that contains http://192.168.202 (these were strange local queries)
                #exclude input that contains the word 'select' AND 'union' (these were actually malicious)
                payloads = payloads + [substring.split('=')[1].strip('\n') for substring in temp 
                                       if len(substring.split('=')) > 1 and
                                       'http://192.168.202' not in substring.split('=')[1] and
                                       ('select' not in substring.split('=')[1] or 'union' not in substring.split('=')[1])
                                      ]
    #remove duplicates
    payloads = list(set(payloads))
                
    #write to destination file
    with open("data/{}".format(dest_file), "a") as myfile:
        for payload in payloads:
            if payload != '':
                myfile.write('{}\n'.format(payload))
        
    print('Total payloads found: '+str(len(payloads)))
    print('First 20 payloads:')
    display(payloads[:20])



def from_CSIC2010_to_collection(src_file, dest_file):
    '''Extracts payload data inputs from CSIC2010 HTTP packet dataset
    
    source dataset found here: http://www.isi.csic.es/dataset/
    input format from source is a complete HTTP packet
    '''
    
    payloads = []
    payload_next_line = False
    
    with open("data/{}".format(src_file)) as f:
        for line in f.readlines():
            
            #Extract inputs from payload if first row in a GET packet
            if line.startswith('GET') and len(line.split('?')) > 1:
                
                #extract total payload string
                total_payload = (line.split('?')[1]).split(' ')[0]
                
                #add each input value separately to payloads
                inputs = total_payload.split('&')
                payloads = payloads + [input.split('=')[1] for input in inputs if len(input.split('=')) > 1]
                
            if line.startswith('Content-Length'):
                #notify that this is a HTTP POST packet and the next line will contain the payload
                payload_next_line = True
                
            elif payload_next_line and len(line) > 2:
                #Current line is a payload of a HTTP POST packet
                
                #add each input value separately to payloads
                inputs = line.split('&')
                payloads = payloads + [input.split('=')[1].strip('\n') for input in inputs if len(input.split('=')) > 1]
                
                payload_next_line = False
       
    payloads = list(set(payloads))

    #write to destination file
    with open("data/{}".format(dest_file), "a") as myfile:
        for payload in payloads:
            if payload != '':
                myfile.write('{}\n'.format(payload))

    print('Total number of data points gathered: ' + str(len(payloads)))
    print('First 20 data points:')
    display(payloads[:20])
    


if __name__ == "__main__":

    #IPS_payload_data is our spreadsheet of payloads gathered so far
    from_google_spreadsheet_to_collections('payloads')
        
    #from_xsuperbug_to_collections('timetoparseSQL.txt','SQLCollection.txt')
    #from_xsuperbug_to_collections('timetoparseXSS.txt','XSSCollection.txt')
    from_xsuperbug_to_collections('timetoparseCMD.txt','ShellCollection.txt')


    #There are 21 files with non-malicious payloads, each with its date as name
    for i in range(1,22):
        date = '0' + str(i) if i < 10  else str(i)
        from_cnets_to_collection('2009-11-{}'.format(date),'non-maliciousCollection.txt')


            
    from_fsecurify_to_collection('goodqueries.txt','non-maliciousCollection.txt')

    from_CSIC2010_to_collection('normalTrafficTraining.txt','non-maliciousCollection.txt')
    from_CSIC2010_to_collection('normalTrafficTest.txt','non-maliciousCollection.txt')

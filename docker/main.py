import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import flask, json
from flask import request

server = flask.Flask(__name__)

def get2Grams(payload_obj):
    '''Divides a string into 2-grams
    
    Example: input - payload: "<script>"
             output- ["<s","sc","cr","ri","ip","pt","t>"]
    '''
    payload = str(payload_obj)
    ngrams = []
    for i in range(0,len(payload)-2):
        ngrams.append(payload[i:i+2])
    return ngrams


classifier = pickle.load( open("data/tfidf_2grams_randomforest.p", "rb"))

def injection_test(inputs):
    return 'MALICIOUS' if classifier.predict(inputs).sum() > 0 else 'NOT_MALICIOUS'



@server.route('/', methods=['get', 'post'])
def login():
    inputs = []
    for key,value in request.args.items():
        inputs.append(value)
    print(inputs)
    
    if inputs != []:
        try:
            analysis = injection_test(inputs)
            output = {'code': 200, 'message': analysis}
            return json.dumps(output, ensure_ascii=False)
        except:
            output = {'code':500, 'message':'something error'}
            return json.dumps(output, ensure_ascii=False)
    else:
        output = {'code':500, 'message':'none inputs'}
        return json.dumps(output, ensure_ascii=False)

if __name__ == '__main__':
    server.run(debug=False,port=64290, host='0.0.0.0')
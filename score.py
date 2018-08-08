# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path
from azureml.datacollector import ModelDataCollector
# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    # Get the path to the model asset
    # local_path = get_local_path('mymodel.model.link')
    
    # Load model using appropriate library and function
    #global model
    # model = model_load_function(local_path)
    #model = 42
    from sklearn.externals import joblib
    global model
    model=joblib.load('Toxic_Trained_model.pkl')
    
    #from keras_pickle_wrapper import KerasPickleWrapper
    #import pickle as pickle
    #f2 = open('outputs/Toxic_Trained_model.pkl', 'rb')
    #model = pickle.load(f2)

def run(input_df):
    import json
    #pred=model().predict(input_df)
    #return json.dumps(str(pred))
    Test_Text_Pred = model().predict(input_df,verbose=1)
    columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    Test_Text_Pred_df = pd.DataFrame(np.atleast_2d(Test_Text_Pred), columns=columns)
    print(Test_Text_Pred_df)
    Resultdf=pd.DataFrame((Test_Text_Pred_df>=0.5).iloc[0])
    Resultdf.columns=['Result']
    Resultdf=Resultdf.loc[Resultdf['Result'] == True]
    list(Resultdf.index)
    if (len(list(Resultdf.index))!=0):
        return json.dumps(str("The Statement/Comment is - ".join(list(Resultdf.index))))
        #Result"The Statement/Comment is - ", list(Resultdf.index))
    else:
        return json.dumps(str("The Statement/Comment is Clean! "))
        #print("The Statement/Comment is clean!")
    # Predict using appropriate functions
    # prediction = model.predict(input_df)

    #prediction = "%s %d" % (str(input_df), model)
    #return json.dumps(str(prediction))

def generate_api_schema():
    import os
    print("create schema")
    sample_input = "sample data text"
    inputs = {"input_df": SampleDefinition(DataTypes.STANDARD, sample_input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath="outputs/service-schema.json", run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    from azureml.api.schema.dataTypes import DataTypes
    from azureml.api.schema.sampleDefinition import SampleDefinition
    from azureml.api.realtime.services import generate_schema
    
    import numpy as np
    import pandas as pd
    from keras.preprocessing import text, sequence
    #from keras.models import load_model
    #from keras.models import Model
    
    EMBEDDING_FILE = 'glove.840B.300d.txt'
    train = pd.read_csv('train.csv')
    train["comment_text"].fillna("fillna")
    X_train = train["comment_text"].str.lower()
    max_features=100000
    maxlen=150
    embed_size=300

    tok=text.Tokenizer(num_words=max_features,lower=True)
    tok.fit_on_texts(list(X_train))
    X_train=tok.texts_to_sequences(X_train)
    x_train=sequence.pad_sequences(X_train,maxlen=maxlen)
    
    embeddings_index = {}
    with open(EMBEDDING_FILE,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    word_index = tok.word_index
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    
    #Test_Text = "This is awsome and same time hilariously"
    Test_Text = "This is awsome and same time hilariously shitty."
    Test_Text=[Test_Text]
    Test_Text_df=pd.DataFrame(Test_Text)
    Test_Text_df.columns = ['comment_text']

    Test_Text_Val = Test_Text_df["comment_text"].str.lower()
    Test_Text_Val=tok.texts_to_sequences(Test_Text_Val)
    Test_Text_Val=sequence.pad_sequences(Test_Text_Val,maxlen=maxlen)
    
    init()

    ##input1 = Test_Text_Val

    #print("Result: " + run(Test_Text_Val))
    #inputs = {"input_df": SampleDefinition(DataTypes.NUMPY, Test_Text_Val)}
    #Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()
    #logger = logging.getLogger("stmt_logger")
    #ch = logging.StreamHandler(sys.stdout)
    #logger.addHandler(ch)

    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--generate', action='store_true', help='Generate Schema')
    #args = parser.parse_args()
    #if args.generate:
    #    generate_api_schema()
    generate_api_schema()

    #init()
    #input = "{}"
    result = run(Test_Text_Val)
    logger.log("Result",result)

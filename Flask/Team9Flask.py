from flask import request
from flask import Flask
import tensorflow as tf

import os
import pymysql
from tensorflow import keras
from tensorflow.keras import layers
from numpy import asarray
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

app = Flask(__name__)

mysqlhost = "127.0.0.1"
mysqlpassword = 'Aa123456'
mysqluser = 'root'
dbname = 'eat'

@app.route('/resturantPrediction', methods=["GET"])
def resturantPrediction():
    # importing the necessary modules
    import pandas as pd
    import numpy as np
    import datetime
    import tensorflow as tf

    # importing data from sql
    import os
    import pymysql

    host = os.getenv('localhost')
    port = os.getenv('MYSQL_PORT')
    user = os.getenv('MYSQL_USER')
    password = os.getenv('MYSQL_PASSWORD')
    database = os.getenv('MYSQL_DATABASE')

    conn = pymysql.connect(
        host=host,
        port=int(3306),
        user=mysqluser,
        passwd=mysqlpassword,
        db=dbname,
        charset='utf8mb4')

    df = pd.read_sql("SELECT * FROM event",
                     conn)

    # data cleaning/ processing
    df2 = df[['start_date', 'location_placeid']].copy()

    # converting date from object to dateTime object
    def dateMonth(x):
        year = int(str(x)[0:4])
        month = int(str(x)[5:7])
        # day = int(str(x)[8:10])

        return datetime.datetime(year, month, 1)

    df2['Month'] = df2['start_date'].apply(dateMonth)

    # converting dateTime to epoch (since 1 Jan 1970) for model training
    import time
    import datetime

    def timeSince1970(x):
        time = x.timestamp()

        return time

    df2['Month'] = df2['Month'].apply(timeSince1970)

    df3 = df2[['Month', 'location_placeid']]

    # converting dataframe to one-hot encoding for places
    table = pd.DataFrame(pd.crosstab(df3.Month, [df3.location_placeid]))

    # splitting the data into x and y for model training
    x = table.index.array
    x = np.reshape(x.to_numpy(), (-1, 1))

    y = table.iloc[:].values

    # model training
    # use mlp for prediction on multi-label classification

    def get_model(n_inputs, n_outputs):
        model = keras.Sequential()
        model.add(layers.Dense(128, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(n_outputs, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def evaluate_model(X, Y):
        results = list()
        n_inputs, n_outputs = X.shape[1], y.shape[1]
        # num_splits = n_inputs

        # define evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # enumerate folds
        for train_ix, test_ix in cv.split(X):
            # prepare data
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            # define model
            model = get_model(n_inputs, n_outputs)
            # fit model
            model.fit(X_train, y_train, verbose=0, epochs=100)
            # make a prediction on the test set
            yhat = model.predict(X_test)
            # round probabilities to class labels
            yhat = yhat.round()

            yhat = np.argmax(yhat, axis=1)
            y_test = np.argmax(y_test, axis=1)

            # calculate accuracy
            acc = accuracy_score(y_test, yhat)
            # store result
            results.append(acc)
        return results

    # training and evaluating model
    evaluate_model(x, y)

    n_inputs, n_outputs = x.shape[1], y.shape[1]
    model = get_model(n_inputs, n_outputs)
    model.fit(x, y, verbose=0, epochs=100)

    # make a prediction for new data
    from datetime import date
    today = date.today().strftime("%d/%m/%Y")

    year = int(today[6:])
    month = int(today[3:5])

    test = datetime.datetime(year, month, 1)
    newX = asarray([test.timestamp()])

    yhat = model.predict(newX)
    print('Predicted: %s' % yhat[0])

    # getting the place Ids
    placeIds = table.columns.values.tolist()

    result_dict = {}

    for count, item in enumerate(yhat[0], start=1):

        if (item > 0):
            result_dict[placeIds[count - 1]] = item

    sorted_x = sorted(result_dict.items(), key=lambda kv: kv[1], reverse=True)

    top10List = sorted_x[:10]

    placesIdsList = []

    for item in top10List:
        placesIdsList.append(item[0])
    print("print list")
    print(placesIdsList)

    # write data  to db
    print("----------------------------")
    predictconn = pymysql.connect(
        host=host,
        port=int(3306),
        user=mysqluser,
        passwd=mysqlpassword,
        db=dbname,
        charset='utf8mb4')

    predictcursor = predictconn.cursor()
    today = date.today()
    InsertPredict_place_sql = "INSERT INTO predict_place(predict_place_id, place_id_list, predict_date)  " + " values (%s, %s, %s) ";
    predictcursor.execute("SELECT MAX(predict_place_id) FROM predict_place")

    placeidString = "["
    for id in placesIdsList:
        placeidString += str(id) + ",";

    placeidString = placeidString[:-1]
    placeidString += "]"
    print("placeidString: %s" % (placeidString))

    result_id = predictcursor.fetchall()[0]
    # palceid = result_id+1
    predictcursor.execute(InsertPredict_place_sql, (result_id[0] + 1, placeidString, today.strftime("%Y-%m-%d")))
    predictconn.commit()
    print("----------------------------")

    df = pd.read_sql("SELECT * FROM event", conn)

    # prepping the data to be sent over (cannot send list)
    placeNameListStr = ""

    for id in placesIdsList:
        place = pd.read_sql("SELECT * FROM place WHERE placeid=" + str(id), conn)
        placeName = place['name']
        placeNameStr = str(placeName.values[0])

        if (placeNameListStr == ""):
            placeNameListStr = str(id) + "\n" + str(placeNameStr)
        else:
            placeNameListStr = placeNameListStr + "\n" + str(id) + "\n" + str(placeNameStr)
    # print("print string")
    # print(placeNameListStr)
    return placeNameListStr



#Android Studio ML
@app.route('/recommend', methods =['POST'])
def getRecommendation():
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    import sqlalchemy as sa
    host = os.getenv('localhost')

    engine = sa.create_engine('mysql+pymysql://root:Aa123456@localhost:3306/eat')
    df = pd.read_sql_table("place", engine)
    def combine_features(data):
        features = []
        for i in range(0, data.shape[0]):
            features.append(data['category'][i]+ ' ' + str(data['price'][i]))
        return features
    df['combined_features'] = combine_features(df)
    cm = CountVectorizer().fit_transform(df['combined_features'])
    cs = cosine_similarity(cm)
    Name = str(request.form['num1'])
    Name = '"' + Name +'"'
    restaurantid = df[df.name == Name]['placeid'].values[0]
    scores = list(enumerate(cs[restaurantid]))
    sorted_scores = sorted(scores, key = lambda x:x[1], reverse= True)
    sorted_scores = sorted_scores[1:]
    j=0
    listRes =""
    for item in sorted_scores:
        restaurant_name = df[df.placeid == item[0]]['name'].values[0]
        category = df[df.placeid == item[0]]['category'].values[0]
        price = df[df.placeid == item[0]]['price'].values[0]
        print(j+1, restaurant_name, "-",category, "(" , price, ")")
        j = j+1
        listRes +=restaurant_name + "\n"
        if j>=5:
            break
    return listRes


@app.route('/resturantPredictionNext', methods=["GET"])
def resturantPredictionNext(): 
    # importing the necessary modules
    import pandas as pd 
    import numpy as np 
    import datetime
    import tensorflow as tf

    # importing data from sql 


    host = os.getenv('localhost')
    port = os.getenv('MYSQL_PORT')
    user = os.getenv('MYSQL_USER')
    password = os.getenv('MYSQL_PASSWORD')
    database = os.getenv('MYSQL_DATABASE')

    conn = pymysql.connect(
        host=host,
        port=int(3306),
        user=mysqluser,
        passwd=mysqlpassword,
        db=dbname,
        charset='utf8mb4')

    df = pd.read_sql("SELECT * FROM event",
        conn)

    # data cleaning/ processing 
    df2 = df[['start_date', 'location_placeid']].copy()

    # converting date from object to dateTime object 
    def dateMonth(x) : 
        year = int(str(x)[0:4]) 
        month = int(str(x)[5:7])
        # day = int(str(x)[8:10])

        return datetime.datetime(year, month, 1)

    df2['Month'] = df2['start_date'].apply(dateMonth)

    # converting dateTime to epoch (since 1 Jan 1970) for model training 
    import time
    import datetime 

    def timeSince1970(x): 
        time = x.timestamp()

        return time

    df2['Month'] = df2['Month'].apply(timeSince1970)

    df3 = df2[['Month', 'location_placeid']]

    # converting dataframe to one-hot encoding for places 
    table = pd.DataFrame(pd.crosstab(df3.Month, [df3.location_placeid]))

    # splitting the data into x and y for model training 
    x = table.index.array
    x = np.reshape(x.to_numpy(), (-1, 1))

    y = table.iloc[:].values

    # model training 
    # use mlp for prediction on multi-label classification
    import tensorflow as tf


    def get_model(n_inputs, n_outputs):
        model = keras.Sequential()
        model.add(layers.Dense(128, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(n_outputs, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def evaluate_model(X, Y):
        results = list()
        n_inputs, n_outputs = X.shape[1], y.shape[1]
        # num_splits = n_inputs
        # define evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # enumerate folds
        for train_ix, test_ix in cv.split(X):
            # prepare data
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            # define model
            model = get_model(n_inputs, n_outputs)
            # fit model
            model.fit(X_train, y_train, verbose=0, epochs=100)
            # make a prediction on the test set
            yhat = model.predict(X_test)
            # round probabilities to class labels
            yhat = yhat.round()

            yhat=np.argmax(yhat, axis=1)
            y_test=np.argmax(y_test, axis=1)
            
            # calculate accuracy
            acc = accuracy_score(y_test, yhat)
            # store result
            results.append(acc)
        return results

    # training and evaluating model 
    evaluate_model(x,y)

    n_inputs, n_outputs = x.shape[1], y.shape[1]
    model = get_model(n_inputs, n_outputs)
    model.fit(x, y, verbose=0, epochs=100)

    # make a prediction for new data
    from datetime import date
    today = date.today().strftime("%d/%m/%Y")

    year = int(today[6:])
    month = int(today[3:5]) + 1
    
    test = datetime.datetime(year, month, 1)
    newX = asarray([test.timestamp()])

    yhat = model.predict(newX)
    print('Predicted: %s' % yhat[0])

    # getting the place Ids 
    placeIds = table.columns.values.tolist()

    result_dict = {}

    for count, item in enumerate(yhat[0], start=1):

        if (item > 0): 
            result_dict[placeIds[count-1]] = item

    sorted_x = sorted(result_dict.items(), key=lambda kv: kv[1], reverse = True)

    top10List = sorted_x[:10]

    placesIdsList = []

    for item in top10List: 
        placesIdsList.append(item[0])
    
    print(placesIdsList)

    # prepping the data to be sent over (cannot send list) 
    placeNameListStr = ""

    for id in placesIdsList: 
        place = pd.read_sql("SELECT * FROM place WHERE placeid=" + str(id), conn)
        placeName = place['name']
        placeNameStr = str(placeName.values[0])

        if (placeNameListStr  == ""): 
            placeNameListStr  = str(id) + "\n" + str(placeNameStr)
        else: 
            placeNameListStr = placeNameListStr  + "\n" + str(id) + "\n" + str(placeNameStr)

    return placeNameListStr

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
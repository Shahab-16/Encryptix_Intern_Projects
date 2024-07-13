from flask import Flask, render_template, request
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import RobustScaler
import joblib
import os
import pickle

app = Flask(__name__)

moviecsv_path = os.path.join('data', 'movies.csv')
titanic_path = os.path.join('models', 'titanic_model.pkl')
movie_path = os.path.join('models', 'movie_model.pkl')
sale_path=os.path.join('models','sale_prediction_model.pkl')
encoder_path=os.path.join('models','movie_encoder.pkl')


with open(titanic_path, 'rb') as f1:
    titanic_model = pickle.load(f1)

with open(movie_path, 'rb') as f2:
    movie_model = pickle.load(f2)

with open(sale_path,'rb') as f3:
    sale_model=pickle.load(f3)
    
with open(encoder_path,'rb') as k:
    encoder=pickle.load(k)    


expected_columns_for_titanic = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male',
                                'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Capt', 'Title_Col',
                                'Title_Countess', 'Title_Don', 'Title_Dr', 'Title_Lady', 'Title_Major',
                                'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Noble',
                                'Title_Rev', 'Title_Sir']

expected_columns_for_movies = ['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

expected_columns_for_sale=['TV','Radio','Newspaper']

rs = RobustScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/titanic', methods=['GET', 'POST'])
def titanic():
    if request.method == 'POST':
        # Extract form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsings = int(request.form['Sibsp'])
        parents = int(request.form['parents'])
        fare = float(request.form['Fare'])
        embarked = request.form['Embarked']
        title = request.form['Title']

        data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'SibSp': [sibsings],
            'Parch': [parents],
            'Fare': [fare],
            'Embarked': [embarked],
            'Title': [title]
        })


        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data = pd.get_dummies(data, columns=['Embarked', 'Title'])

      
        data = data.reindex(columns=expected_columns_for_titanic, fill_value=0)

     
        prediction = titanic_model.predict(data)

        if prediction[0] == 0:
            result = "Died"
        else:
            result = "Survived"

        return render_template('titanic.html', result=result)

    return render_template('titanic.html')

@app.route('/movie', methods=['GET', 'POST'])
def movie():
    if request.method == 'POST':
        year = int(request.form['release-year'])
        duration = int(request.form['duration'])
        genre = request.form['genre']
        votes = int(request.form['votes'])
        director = request.form['director']
        actor1 = request.form['actor1']
        actor2 = request.form['actor2']
        actor3 = request.form['actor3']

        sample_X = pd.DataFrame({
            'Year': [year],
            'Duration': [duration],
            'Genre': [genre],
            'Votes': [votes],
            'Director': [director],
            'Actor 1': [actor1],
            'Actor 2': [actor2],
            'Actor 3': [actor3]
        })

        sample_X_encoded = encoder.transform(sample_X)
        sample_X_encoded = rs.fit_transform(sample_X_encoded)

        prediction = movie_model.predict(sample_X_encoded)[0]

        return render_template('movie.html', result=prediction)

    return render_template('movie.html')

@app.route('/sale',methods=['GET','POST'])
def sale():
    if request.method == 'POST':
        tv=float(request.form['tv'])
        radio=float(request.form['radio'])
        newspaper=float(request.form['newspaper'])
        
        data=pd.DataFrame({
            'TV':[tv],
            'Radio':[radio],
            'Newspaper':[newspaper]
        })
        data=data.reindex(columns=expected_columns_for_sale,fill_value=0)
        prediction=sale_model.predict(data)
        return render_template('sale.html',result=prediction[0])
    return render_template('sale.html')

if __name__ == '__main__':
    app.run(debug=True)

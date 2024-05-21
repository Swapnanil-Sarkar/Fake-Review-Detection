from django.shortcuts import render, redirect
# Create your views here.
from django.contrib.auth.models import User
from django.contrib import messages
from . models import Register

import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#importing the required libraries


Home = 'index.html'
About = 'about.html'
Login = 'login.html'
Registration = 'registration.html'
Userhome = 'userhome.html'
Load = 'load.html'
View = 'view.html'
Preprocessing = 'preprocessing.html'
Model = 'model.html'
Prediction = 'prediction.html'


# # Home page
def index(request):

    return render(request, Home)

# # About page


def about(request):
    return render(request, About)

# # Login Page


def login(request):
    if request.method == 'POST':
        lemail = request.POST['email']
        lpassword = request.POST['password']

        d = Register.objects.filter(email=lemail, password=lpassword).exists()
        print(d)
        if d:
            return redirect(userhome)
        else:
            msg = 'Login failed'
            return render(request, Login, {'msg': msg})
    return render(request, Login)

# # registration page user can registration here


def registration(request):
    if request.method == 'POST':
        Name = request.POST['Name']
        email = request.POST['email']
        password = request.POST['password']
        conpassword = request.POST['conpassword']
        age = request.POST['Age']
        contact = request.POST['contact']

        if password == conpassword:
            userdata = Register.objects.filter(email=email).exists()
            if userdata:
                msg = 'Account already exists'
                return render(request, Registration, {'msg': msg})
            else:
                userdata = Register(name=Name, email=email,
                                    password=password, age=age, contact=contact)
                userdata.save()
                return render(request, Login)
        else:
            msg = 'Register failed!!'
            return render(request, Registration, {'msg': msg})

    return render(request, Registration)

# # user interface


def userhome(request):

    return render(request, Userhome)

# # Load Data


def load(request):
    if request.method == "POST":
        global df
        file = request.FILES['file']
        df = pd.read_csv(file)
        messages.info(request, "Data Uploaded Successfully")

    return render(request, Load)

# # View Data


def view(request):
    col = df.to_html
    dummy = df.head(100)

    col = dummy.columns
    rows = dummy.values.tolist()
    # return render(request, 'view.html',{'col':col,'rows':rows})
    return render(request, View, {'columns': dummy.columns.values, 'rows': dummy.values.tolist()})


# preprocessing data
def preprocessing(request):
    global x_train, x_test, y_train, y_test, xsm, ysm ,df
    ps = PorterStemmer()
    corpus = []
    print('abcdefghh')
    df1 = df[:10000]
    df1.rename(columns={'text_': 'text'}, inplace=True)
    print(df1.head())
    for i in range(len(df1['text'])):
        words = re.sub(',', '', df1['text'][i])
        words = words.lower()
        words = nltk.sent_tokenize(df1['text'][i])
        word = [ps.stem(word) for word in words if not word in stopwords.words('english')]
        words = ''.join(words)
        corpus.append(words)
    global cv
    cv = TfidfVectorizer()
    X = cv.fit_transform(corpus).toarray()
    y = df1['label']
    print(y.head())
    y.replace({'OR': 0, 'CG': 1}, inplace=True)
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=52)
    messages.info(request, "Data Preprocessed and It Splits Succesfully")
    return render(request, Preprocessing)


# Model Training
def model(request):
    global x_train, x_test, y_train, y_test,module
    if request.method == "POST":
        model = request.POST['algo']
        if model == "0":
            knn = KNeighborsClassifier(n_neighbors=7)
            knn.fit(X_train,y_train)
            pred = knn.predict(X_test)
            acc_knn=accuracy_score(y_test, pred)
            acc_knn = acc_knn*100
            msg = 'Accuracy of KNeighborsClassifier : ' + str(acc_knn)
            return render(request, Model, {'msg': msg})

        elif model == "1":
            gb = GaussianNB()
            gb.fit(X_train, y_train)
            gpred = gb.predict(X_test)
            acc_gnb=accuracy_score(y_test, gpred)
            acc_gnb=acc_gnb*100
            msg = 'Accuracy of GaussianNB : ' + str(acc_gnb)
            return render(request, Model, {'msg': msg})

        elif model == "2":
            lr=LogisticRegression()
            lr.fit(X_train,y_train)
            spred=lr.predict(X_test)
            acc_lr=accuracy_score(y_test,spred)
            acc_lr=acc_lr*100
            msg = 'Accuracy of LogisticRegression : ' + str(acc_lr)
            return render(request, Model, {'msg': msg})
        
    return render(request, Model)


# Prediction here we can find the result based on user input values.
def prediction(request):

    global x_train,x_test,y_train,y_test,x,y

    if request.method == 'POST':
        
        f1=(request.POST['name'])
        PRED = f1
        new = cv.transform([PRED]).toarray()
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        result=lr.predict(new)
        print(int(result))
        if result == 0:
            msg = "The Review is Original"
            return render(request,Prediction,{'msg':msg})
        else:
            msg = "The Review is Computer Generated"
            return render(request,Prediction,{'msg':msg})

    return render(request,Prediction)
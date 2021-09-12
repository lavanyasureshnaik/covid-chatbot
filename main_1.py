import os
import sqlite3
conn = sqlite3.connect('chatbot_database')
cur = conn.cursor()
try:
   cur.execute('''CREATE TABLE doctor (
   id integer Primary key  AUTOINCREMENT,
   name varchar(20),
   email varchar(50),
   password varchar(20),
   gender varchar(10),
   specialist varchar(50),
   avaliability tinyint(1),
   address varchar(100))''')

   cur.execute('''CREATE TABLE user (
     name varchar(20) DEFAULT NULL,
      email varchar(50) DEFAULT NULL,
     password varchar(20) DEFAULT NULL,
     gender varchar(10) DEFAULT NULL,
     age int(11) DEFAULT NULL
   )''')

   cur.execute('''CREATE TABLE appointment (
        name varchar(20) DEFAULT NULL,
        doctor_name varchar(50) DEFAULT NULL,
        date_time varchar(20) DEFAULT NULL,
        hospital varchar(10) DEFAULT NULL)''')
except:
   pass
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#reading the dataset
df = pd.read_csv("corona_tested_individuals_ver_006.english.csv")

#read the data from the head of the dataset
df.head()

#defining array of unique values in the column
df.test_indication.unique()


df.info()


# catogorical data handling



df.cough = df.cough.replace('0',0)
df.cough = df.cough.replace('1',1)
df.cough = df.cough.replace('None',0)
df.cough.dtype



df.fever = df.fever.replace('0',0)
df.fever = df.fever.replace('1',1)
df.fever = df.fever.replace('None',0)
df.fever.dtype



df.sore_throat = df.sore_throat.replace('0',0)
df.sore_throat = df.sore_throat.replace('1',0)
df.sore_throat = df.sore_throat.replace('None',0)
df.sore_throat.dtype



df.shortness_of_breath = df.shortness_of_breath.replace('0',0)
df.shortness_of_breath = df.shortness_of_breath.replace('1',0)
df.shortness_of_breath = df.shortness_of_breath.replace('None',0)
df.shortness_of_breath = df.shortness_of_breath.replace('',0)
df.shortness_of_breath.unique()



df.head_ache = df.head_ache.replace('0',0)
df.head_ache = df.head_ache.replace('1',0)
df.head_ache = df.head_ache.replace('None',0)
df.head_ache.unique()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df.corona_result  =   le.fit_transform(df.corona_result)
df.age_60_and_above   = le.fit_transform(df.age_60_and_above)
df.gender      = le.fit_transform(df.gender)

# to predict these columns are not reqiured
df = df.drop(['test_date', 'test_indication'],axis=1)



df.info()



X = df.drop(['corona_result'],axis=1)
y = df.corona_result


from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,y,test_size=0.2,random_state=100)


# # logistic Regression


from sklearn.linear_model import LogisticRegression



reg = LogisticRegression().fit(X_train, Y_train)



score_lr = reg.score(X_test, Y_test)
score_lr= score_lr * 100
score_lr


# # Random Forest


from sklearn.ensemble import RandomForestClassifier

regr = RandomForestClassifier(max_depth=3, random_state=0)

regr.fit(X_train, Y_train)

score_rf = regr.score(X_test,Y_test)
score_rf =score_rf*100
score_rf

# # Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=7)
clf.fit(X_train, Y_train)

score_dt = clf.score(X_test,Y_test)
score_dt = score_dt * 100
score_dt


# # Navie Bayes



from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)



from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,y_pred)*100)
score_nb = accuracy_score(Y_test,y_pred)*100



from flask import Flask, render_template, url_for,request, flash, redirect, session

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

filenumber = int(os.listdir('saved_conversations')[-1])
filenumber = filenumber+1
file= open('saved_conversations/'+str(filenumber),"w+")
file.write('bot : Hi There! I am a medical chatbot. You can begin conversation by typing in a message and pressing enter.\n')
file.close()

app = Flask(__name__)

english_bot = ChatBot('Bot',
                      storage_adapter='chatterbot.storage.SQLStorageAdapter',
                      logic_adapters=[
                         {
                            'import_path': 'chatterbot.logic.BestMatch'
                         },

                      ],
                      trainer='chatterbot.trainers.ListTrainer')
english_bot.set_trainer(ListTrainer)


#conn = pymysql.connect(host='127.0.0.1', user='root', password='root', database='xtipl')
#cur = conn.cursor()



@app.route('/user_login',methods = ['POST', 'GET'])
def user_login():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['psw']
      count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
      #conn.commit()
      #cur.close()
      #print(count)
      if len(cur.fetchall()) == 1:
         session['logged_in'] = True
         cur.execute('select * from doctor where avaliability="true"')
         s = cur.fetchall()
         print(s)
         cur.execute('select * from doctor where avaliability="false"')
         #conn.commit()
         s1 = cur.fetchall()
         s2 = s + s1
         return render_template('user_account.html', data=s2)
      else:
         flash('invalid email and password!')
   return render_template('user_login.html')


@app.route('/user_register',methods = ['POST', 'GET'])
def user_register():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['uname']
      email = request.form['email']
      password = request.form['psw']
      gender = request.form['gender']
      age = request.form['age']

      cur.execute("insert into user(name,email,password,gender,age)values('%s','%s','%s','%s','%s')" % (name, email, password, gender, age))
      conn.commit()
      # cur.close()
      return redirect(url_for('user_login'))

   return render_template('user_register.html')


@app.route('/doctor_login',methods = ['POST', 'GET'])
def doctor_login():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()

   if request.method == 'POST':
      email = request.form['email']
      password = request.form['psw']
      count = cur.execute('SELECT * FROM doctor WHERE email = "%s" AND password = "%s"' % (email, password))
      conn.commit()
      #cur.close()
      print(count)
      if len(cur.fetchall()) == 1:
         print(count)
         flash('{} You have been logged in!'.format(email), 'success')
         session['logged_in_d'] = True
         cur.execute("update doctor set avaliability='true' where email='%s'" %email)
         print("""update doctor set avaliability='true' where email='%s'" %email""")
         conn.commit()
         cur.execute('select * from doctor where email="%s"' % email)
         s = cur.fetchall()
         print(s)
         return render_template('doctor_account.html', a=email,b=s)
      else:
         flash('invalid email and password')
         return redirect(url_for('doctor_login'))
   return render_template('doctor_login.html')

@app.route('/medicine_predictor')
def medicine_predictor():
   import college_doctor
   return render_template('user_account.html')

@app.route('/server')
def server():
   import pythonchat.pyserve

@app.route('/livechat')
def livechat():
   #os.system('dir')
   os.system('python pythonchat/pyclient.py')
   return render_template('user_account.html')

@app.route('/doctor_livechat')
def doctor_livechat():
   #os.system('dir')
   os.system('python pythonchat/pyclient.py')
   return render_template('doctor_login.html')

@app.route('/doctor_register',methods = ['POST', 'GET'])
def doctor_register():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['uname']
      email = request.form['email']
      password = request.form['psw']
      gender = request.form['gender']
      specialist = request.form['specialist']
      address = request.form['address']
      cur.execute("insert into doctor(name,email,password,gender,specialist,avaliability,address)values('%s','%s','%s','%s','%s','false','%s')" % (name, email, password, gender, specialist, address))
      conn.commit()
      # cur.close()


      return redirect(url_for('doctor_login'))

   return render_template('doctor_register.html')

@app.route('/user_account',methods = ['POST', 'GET'])
def user_account():
   return render_template('user_account.html')

@app.route('/doctor_account',methods = ['POST', 'GET'])
def doctor_account():
   if request.method == 'POST':
      email = request.form['email']
      return render_template('doctor_account_edit.html', em = email)


@app.route('/doctor_edit',methods = ['POST', 'GET'])
def doctor_edit():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['uname']
      email = request.form['email']
      password = request.form['psw']
      gender = request.form['gender']
      specialist = request.form['specialist']
      address = request.form['address']
      em = request.form['em']

      cur.execute("update doctor set name='%s', email='%s',password='%s',gender='%s',specialist='%s',address='%s' where email='%s'" %
                  (name, email,password, gender, specialist, address,em))
      conn.commit()

      return redirect('doctor_login')


@app.route('/')
@app.route('/home')
def home():
   if not session.get('logged_in'):
      return render_template('home.html')
   else:
      return redirect(url_for('user_account'))


@app.route('/chatbot')
def chatbot():
   return render_template('index.html')

@app.route('/about')
def about():
   return render_template('about.html')

@app.route("/logout")
def logout():
   session['logged_in'] = False
   return home()

@app.route("/logoutd",methods = ['POST','GET'])
def logoutd():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()
   print('doctor logout')
   if request.method == 'POST':
      email = request.form['email']
      cur.execute('update doctor set avaliability="false" where email="%s"' %email)
      conn.commit()
      session['logged_in_d'] = False
      return home()

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = str(english_bot.get_response(userText))

    appendfile=os.listdir('saved_conversations')[-1]
    appendfile= open('saved_conversations/'+str(filenumber),"a")
    appendfile.write('user : '+userText+'\n')
    appendfile.write('bot : '+response+'\n')
    appendfile.close()

    return response




@app.route('/appoinment',methods = ['POST', 'GET'])
def appoinment():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['name']
      doctor = request.form['doctor']
      date_time = request.form['date_time']
      hospital = request.form['hospital']
      cur.execute("insert into appointment(name,doctor_name,date_time,hospital)values('%s','%s','%s','%s')" % (name, doctor, date_time, hospital))
      conn.commit()
      return redirect(url_for('user_login'))
   return render_template('appoinment.html')


@app.route('/appoinment_details',methods = ['POST', 'GET'])
def appoinment_details():
   conn = sqlite3.connect('chatbot_database')
   cur = conn.cursor()
   #if request.method == 'POST':
   session['logged_in'] = True
   cur.execute('select * from appointment')
   s = cur.fetchall()
   return render_template('appoinment_details.html', data=s)
   #return render_template('doctor_account.html')

@app.route('/covid detection')
def covid():
   return render_template('covid.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
   global clf
   cough = request.form['cough']
   fever = request.form['fever']
   sore_throat = request.form['sore_throat']
   shortness_of_breath = request.form['shortness_of_breath']
   head_ache = request.form['head_ache']
   gender = request.form['gender']
   age = request.form['age_60_and_above']
   global clf
   if gender == 'Male':
      gender = 0
   else:
      gender = 1
   if request.method == 'POST':
      res = clf.predict([[float(cough), float(fever), float(sore_throat), float(shortness_of_breath), float(head_ache),
                          float(age), float(gender)]])
      print(res)
      if res[0] == 0:
         print('Negative')
         flash(f'Negative')
      else:
         print('Positive')
         flash(f'Positive')
   return render_template('covid.html')


if __name__ == '__main__':
   app.secret_key = os.urandom(12)
   app.run(debug=True)


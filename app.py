from flask import Flask, render_template, request
import numpy as np
import joblib
from xgboost import XGBClassifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        #get form data
        Atr1 = request.form.get('q1')
        Atr2 = request.form.get('q2')
        Atr3 = request.form.get('q3')
        Atr4 = request.form.get('q4')
        Atr5 = request.form.get('q5')
        Atr6 = request.form.get('q6')
        Atr7 = request.form.get('q7')
        Atr8 = request.form.get('q8')
        Atr9 = request.form.get('q9')
        Atr10 = request.form.get('q10')
        Atr11 = request.form.get('q11')
        Atr12 = request.form.get('q12')
        Atr13 = request.form.get('q13')
        Atr14 = request.form.get('q14')
        Atr15 = request.form.get('q15')
        Atr16 = request.form.get('q16')
        Atr17 = request.form.get('q17')
        Atr18 = request.form.get('q18')
        Atr19 = request.form.get('q19')
        Atr20 = request.form.get('q20')
        Atr21 = request.form.get('q21')
        Atr22 = request.form.get('q22')
        Atr23 = request.form.get('q23')
        Atr24 = request.form.get('q24')
        Atr25 = request.form.get('q25')
        Atr26 = request.form.get('q26')
        Atr27 = request.form.get('q27')
        Atr28 = request.form.get('q28')
        Atr29 = request.form.get('q29')
        Atr30 = request.form.get('q30')
        Atr31 = request.form.get('q31')
        Atr32 = request.form.get('q32')
        Atr33 = request.form.get('q33')
        Atr34 = request.form.get('q34')
        Atr35 = request.form.get('q35')
        Atr36 = request.form.get('q36')
        Atr37 = request.form.get('q37')
        Atr38 = request.form.get('q38')
        Atr39 = request.form.get('q39')
        Atr40 = request.form.get('q40')
        Atr41 = request.form.get('q41')
        Atr42 = request.form.get('q42')
        Atr43 = request.form.get('q43')
        Atr44 = request.form.get('q44')
        Atr45 = request.form.get('q45')
        Atr46 = request.form.get('q46')
        Atr47 = request.form.get('q47')
        Atr48 = request.form.get('q48')
        Atr49 = request.form.get('q49')
        Atr50 = request.form.get('q50')
        Atr51 = request.form.get('q51')
        Atr52 = request.form.get('q52')
        Atr53 = request.form.get('q53')
        Atr54 = request.form.get('q54')        
        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(Atr1, Atr2, Atr3, Atr4, Atr5, 
            Atr6, Atr7, Atr8, Atr9, Atr10, Atr11, Atr12, Atr13, Atr14, Atr15, 
            Atr16, Atr17, Atr18, Atr19, Atr20, Atr21, Atr22, Atr23, Atr24, 
            Atr25, Atr26, Atr27, Atr28, Atr29, Atr30, Atr31, Atr32, Atr33, 
            Atr34, Atr35, Atr36, Atr37, Atr38, Atr39, Atr40, Atr41, Atr42, 
            Atr43, Atr44, Atr45, Atr46, Atr47, Atr48, Atr49, Atr50, Atr51, 
            Atr52, Atr53, Atr54)            
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        pass
    pass

def preprocessDataAndPredict(Atr1, Atr2, Atr3, Atr4, Atr5, 
            Atr6, Atr7, Atr8, Atr9, Atr10, Atr11, Atr12, Atr13, Atr14, Atr15, 
            Atr16, Atr17, Atr18, Atr19, Atr20, Atr21, Atr22, Atr23, Atr24, 
            Atr25, Atr26, Atr27, Atr28, Atr29, Atr30, Atr31, Atr32, Atr33, 
            Atr34, Atr35, Atr36, Atr37, Atr38, Atr39, Atr40, Atr41, Atr42, 
            Atr43, Atr44, Atr45, Atr46, Atr47, Atr48, Atr49, Atr50, Atr51, 
            Atr52, Atr53, Atr54):
    
    #keep all inputs in array
    test_data = [Atr1, Atr2, Atr3, Atr4, Atr5, 
            Atr6, Atr7, Atr8, Atr9, Atr10, Atr11, Atr12, Atr13, Atr14, Atr15, 
            Atr16, Atr17, Atr18, Atr19, Atr20, Atr21, Atr22, Atr23, Atr24, 
            Atr25, Atr26, Atr27, Atr28, Atr29, Atr30, Atr31, Atr32, Atr33, 
            Atr34, Atr35, Atr36, Atr37, Atr38, Atr39, Atr40, Atr41, Atr42, 
            Atr43, Atr44, Atr45, Atr46, Atr47, Atr48, Atr49, Atr50, Atr51, 
            Atr52, Atr53, Atr54]
        
    #convert value data into numpy array
    test_data = np.array(test_data)
    
    #reshape array
    test_data = test_data.reshape(1,-1)
        
    #open file
    filePath = 'output/xgb_model.pkl'
    file = open(filePath, "rb")
    
    #load trained model
    trained_model = joblib.load(file)
    
    #predict
    prediction = trained_model.predict(test_data)
    
    return prediction
    
    pass

if __name__ == '__main__':
    app.run(debug=True)
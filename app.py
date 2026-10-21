import streamlit as st
import pandas as pd
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Select Passenger Class", [1, 2, 3])
sex = st.selectbox("Select Gender", ["male", "female"])
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, step=1)
fare = st.number_input("Fare Paid", min_value=0.0, step=0.01)
embarked = st.selectbox("Select Port of Embarkation", ["SouthHampton", "Chebourg", "Queenstown"])

user = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

if st.button("Predict Survival"):
    st.write(user)

    st.write("Processing your input and predicting survival probability...")
    st.write("Please wait while we load the model and preprocess your input...")
    
    le = joblib.load('le.pkl')
    ohe = joblib.load('ohe.pkl')
    ss = joblib.load('ss.pkl')
    model = load_model('model.h5')

    user['Sex'] = le.transform(user['Sex'])
    embarked_encoded = ohe.transform(user[['Embarked']])
    embarked_df = pd.DataFrame(embarked_encoded, columns=ohe.get_feature_names_out())
    user = pd.concat([user.drop(columns='Embarked'), embarked_df], axis=1)
    user[['Fare', 'SibSp', 'Parch', 'Pclass']] = ss.transform(user[['Fare', 'SibSp', 'Parch', 'Pclass']])

    st.write("All Values transformed as per ANN model .. ")
    prob = model.predict(user)
    probability = prob[0][0]
    st.write(prob[0][0])
    st.write(f"Probability of survival: {probability*100:.2f}%")
    
    if st.button("Predict"):
        st.write("Probability predicted .. ")
        st.write(f"Probability of survival: {probability*100:.2f}%")

      

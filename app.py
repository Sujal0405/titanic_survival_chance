import streamlit as st
import pandas as pd

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
    import pandas as pd
    import joblib
    from tensorflow.keras.models import load_model

    le = joblib.load('ANN/models/le.pkl')
    ohe = joblib.load('ANN/models/ohe.pkl')
    ss = joblib.load('ANN/models/ss.pkl')
    model = load_model('ANN/models/model.h5')

    user['Sex'] = le.transform(user['Sex'])
    embarked_encoded = ohe.transform(user[['Embarked']])
    embarked_df = pd.DataFrame(embarked_encoded, columns=ohe.get_feature_names_out())
    user = pd.concat([user.drop(columns='Embarked'), embarked_df], axis=1)
    user[['Fare', 'SibSp', 'Parch', 'Pclass']] = ss.transform(user[['Fare', 'SibSp', 'Parch', 'Pclass']])


    if st.button("Predict"):
        prob = model.predict(user)
        st.write(prob)
        st.write(f"Probability of survival: {prob[0][0]*100:.2f}%")

      
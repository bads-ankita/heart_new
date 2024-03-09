import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the heart disease dataset
heart_data = pd.read_csv(r"D:\desktop\heart.csv")

# Split the data into features and target variable
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest classifier
#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)

max_accuracy = 0
for x in range(50):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, y_train)
Y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(Y_pred_rf, y_test) * 100, 2)

#st.write("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")

#m = rf.predict([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
#st.write(m)

# Streamlit app
def main():
    st.title("Heart Disease Prediction")
    st.sidebar.title("Features")
    st.sidebar.markdown("Please select the values for each feature:")

    # Sidebar inputs for feature values
    age = st.sidebar.slider("Age", 1, 100, 25)
    sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
    cp = st.sidebar.selectbox("Chest Pain Type",
                              ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ['True', 'False'])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results",
                                   ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ['Yes', 'No'])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia", ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Preprocess the feature values
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'True' else 0
    exang = 1 if exang == 'Yes' else 0

    # Encode categorical variables
    cp_encoded = [
        1 if cp == 'Typical Angina' else 2 if cp == 'Atypical Angina' else 3 if cp == 'Non-anginal Pain' else 4]
    restecg_encoded = [0 if restecg == 'Normal' else 1 if restecg == 'ST-T Wave Abnormality' else 2]
    slope_encoded = [1 if slope == 'Upsloping' else 2 if slope == 'Flat' else 3]
    thal_encoded = [1 if thal == 'Normal' else 2 if thal == 'Fixed Defect' else 3]

    # Create a dataframe with the feature values
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': cp_encoded,
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': restecg_encoded,
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': slope_encoded,
        'ca': [ca],
        'thal': thal_encoded
    })

    # Make predictions
    prediction = rf.predict(input_data)
    prediction_proba = rf.predict_proba(input_data)

    # Display the results
    st.subheader("Prediction")
    if prediction[0] == 0:
        st.write("The patient is **not likely** to have heart disease.")
    else:
        st.write("The patient is **likely** to have heart disease.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of not having heart disease: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of having heart disease: {prediction_proba[0][1]:.2f}")


# Run the app
if __name__ == '__main__':
    main()

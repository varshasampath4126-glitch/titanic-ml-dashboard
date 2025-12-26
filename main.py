import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Import 8 Types of ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# --- 1. GREETINGS & INTRO ---
st.set_page_config(page_title="Titanic ML Hub", layout="wide")
st.title("🚢 Titanic Survival: The Ultimate ML Dashboard")
st.markdown("""
### Welcome! 👋
This project analyzes the Titanic dataset and uses **8 different Machine Learning models** to predict if a passenger would survive or not.
""")

# --- 2. LOADING CSV FILES ---
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    gender_sub = pd.read_csv('gender_submission.csv')
    
    # Simple Preprocessing
    for df in [train, test]:
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Sex'] = LabelEncoder().fit_transform(df['Sex']) # Female=0, Male=1
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
    return train, test, gender_sub

train_df, test_df, gender_sub_df = load_data()

# --- 3. MODEL SELECTION (8 MODELS) ---
st.sidebar.header("ML Model Settings")
model_dict = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB()
}

selected_model_name = st.sidebar.selectbox("Choose Model", list(model_dict.keys()))

# --- 4. DATA VISUALIZATIONS ---
st.header("📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.write("#### Survival Rate by Gender")
    fig1, ax1 = plt.subplots()
    sns.barplot(x='Sex', y='Survived', data=train_df, ax=ax1, palette="viridis")
    ax1.set_xticklabels(['Female', 'Male'])
    st.pyplot(fig1)

with col2:
    st.write("#### Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)



# --- 5. TRAINING & EVALUATION ---
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = train_df[features]
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train selected model
model = model_dict[selected_model_name]
model.fit(X_train, y_train)
acc = accuracy_score(y_val, model.predict(X_val))

st.divider()
st.subheader(f"🏆 Current Model: {selected_model_name}")
st.metric("Validation Accuracy", f"{acc*100:.2f}%")

# Compare all 8 models in a bar chart
st.write("### Comparison of all 8 Models")
all_acc = {}
for name, m in model_dict.items():
    m.fit(X_train, y_train)
    all_acc[name] = accuracy_score(y_val, m.predict(X_val))

acc_df = pd.DataFrame(list(all_acc.items()), columns=['Model', 'Accuracy'])
st.bar_chart(acc_df.set_index('Model'))



# --- 6. PREDICTION (CLASSIFICATION OUTCOME) ---
st.divider()
st.header("🔮 Predict Survival Outcome")
p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    p_class = st.selectbox("Passenger Class", [1, 2, 3])
    p_sex = st.radio("Gender", ["Female", "Male"])
with p_col2:
    p_age = st.slider("Age", 1, 100, 25)
    p_sib = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
with p_col3:
    p_parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

if st.button("Predict Classification"):
    s_val = 0 if p_sex == "Female" else 1
    input_data = [[p_class, s_val, p_age, p_sib, p_parch]]
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success(f"**Outcome: SURVIVED ✅** (Predicted by {selected_model_name})")
    else:
        st.error(f"**Outcome: DID NOT SURVIVE ❌** (Predicted by {selected_model_name})")

st.info("Note: The 'gender_submission.csv' format is being used for reference.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# 8 Selected Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Page Configuration
st.set_page_config(page_title="Titanic Survival Dashboard", layout="wide", page_icon="🚢")

# Title and English Greetings
st.title("🚢 Titanic Survival Analysis & Prediction Dashboard")
st.markdown("""
Welcome! This application uses **8 different Machine Learning algorithms** to analyze passenger data from the Titanic 
and predict survival outcomes based on user input.
""")

# --- STEP 1: DATA LOADING & PREPROCESSING ---
@st.cache_data
def get_cleaned_data():
    # Loading built-in dataset
    df = sns.load_dataset('titanic')
    
    # Handling missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['fare'] = df['fare'].fillna(df['fare'].median())
    
    # Feature Selection
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = df[features].copy()
    y = df['survived']
    
    # Label Encoding for categorical data
    X['sex'] = LabelEncoder().fit_transform(X['sex'])
    X['embarked'] = LabelEncoder().fit_transform(X['embarked'])
    
    return X, y, df

X, y, raw_df = get_cleaned_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- SIDEBAR: INTERACTIVE PREDICTION INPUTS ---
st.sidebar.header("🔍 Predict Survival Status")
p_class = st.sidebar.selectbox("Passenger Class", [1, 2, 3], help="1 = Upper, 2 = Middle, 3 = Lower")
p_sex = st.sidebar.radio("Gender", ["male", "female"])
p_age = st.sidebar.slider("Age", 0, 80, 25)
p_sib = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 8, 0)
p_parch = st.sidebar.number_input("Parents/Children Aboard", 0, 6, 0)
p_fare = st.sidebar.slider("Ticket Fare ($)", 0, 512, 32)
p_emb = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Mapping sidebar input for prediction
user_input = pd.DataFrame([[
    p_class, 
    1 if p_sex == 'male' else 0, 
    p_age, p_sib, p_parch, p_fare, 
    2 if p_emb == 'S' else (0 if p_emb == 'C' else 1)
]], columns=X.columns)
user_input_scaled = scaler.transform(user_input)

# --- MAIN INTERFACE TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data Visualizations", "⚙️ Model Performance", "🔮 Make a Prediction"])

with tab1:
    st.header("Titanic Dataset Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Survival Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=raw_df, x='survived', palette='magma', ax=ax1)
        ax1.set_xticklabels(['Died (0)', 'Survived (1)'])
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots()
        sns.heatmap(raw_df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

with tab2:
    st.header("Comparison of 8 Machine Learning Models")
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }
    
    accuracies = {}
    
    # 2x4 Grid for 8 Confusion Matrices
    st.subheader("Visual Confusion Matrices")
    fig_cm, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{name}\nAcc: {acc:.2f}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_cm)
    
    st.divider()
    
    # Accuracy Ranking Chart
    st.subheader("Model Accuracy Ranking")
    acc_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    sns.barplot(data=acc_df, x='Accuracy', y='Model', palette='viridis', ax=ax_bar)
    ax_bar.set_xlim(0.6, 0.9) # Focus on the 60%-90% range
    st.pyplot(fig_bar)

with tab3:
    st.header("Survival Prediction Result")
    # Training the Random Forest model for the final prediction
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_train_scaled, y_train)
    
    prediction = final_model.predict(user_input_scaled)
    prob = final_model.predict_proba(user_input_scaled)
    
    if prediction[0] == 1:
        st.success(f"### Prediction: The Passenger would have SURVIVED ✅")
        st.balloons()
    else:
        st.error(f"### Prediction: The Passenger would NOT have survived ❌")
    
    st.metric("Model Confidence Level", f"{max(prob[0])*100:.2f}%")
    st.info("The prediction is based on the Random Forest model, which typically shows high accuracy for this dataset.")
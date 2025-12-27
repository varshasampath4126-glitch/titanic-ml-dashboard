import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Page Title
st.set_page_config(page_title="Titanic ML Visualizer", layout="wide")
st.title("🚢 Titanic Survival Prediction: 8 Models with Visuals")

# Load Data
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    # Simple Preprocessing
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    return train[cols], train['Survived']

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary of 8 Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (Support Vector Machine)": SVC(),
    "KNN (K-Nearest Neighbors)": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Sidebar selection
st.sidebar.header("Choose Model Settings")
selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))

# Model Training and Prediction
model = models[selected_model_name]
model.fit(X_train, y_train)
y_pred = model.fit(X_train, y_train).predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Display Results
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Model Performance: {selected_model_name}")
    st.metric(label="Accuracy Score", value=f"{acc:.2%}")
    
    st.write("### Prediction Preview")
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)
    st.dataframe(results_df)

with col2:
    st.subheader("Visual Representation (Confusion Matrix)")
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    st.info(f"Indha chart-la dark blue boxes adhigama irundha, {selected_model_name} nalla predict pannudhu nu artham.")

# Feature Importance (Only for Tree-based models)
if hasattr(model, 'feature_importances_'):
    st.divider()
    st.subheader("Which Factors Mattered Most?")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feat_imp)
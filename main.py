import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Titanic Model Visuals", layout="wide")

# Data Loading
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    return train[cols], train['Survived']

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Sidebar - Just the Model Selection
st.sidebar.header("Select Model")
selected_model = st.sidebar.selectbox("Choose a Model to see Visual Outcome", list(models.keys()))

# Model Training
model = models[selected_model]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- MAIN DISPLAY ---
st.title(f"📊 Visual Outcome: {selected_model}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Accuracy Heatmap (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted Survival')
    ax1.set_ylabel('Actual Survival')
    st.pyplot(fig1)
    st.write("_Idhu model evvalavu correct-ah classify pannudhu nu kaatum._")

with col2:
    st.subheader("2. Prediction vs Actual Distribution")
    fig2, ax2 = plt.subplots()
    # Visualizing the spread of predictions
    plot_data = pd.DataFrame({'Status': ['Actual Survived', 'Predicted Survived'], 
                             'Count': [y_test.sum(), y_pred.sum()]})
    sns.barplot(x='Status', y='Count', data=plot_data, palette='magma', ax=ax2)
    st.pyplot(fig2)
    st.write("_Actual survived counts-kkum, model predict panna counts-kkum ulla vidhyasam._")

# Feature Importance Visual (If available for the model)
if hasattr(model, 'feature_importances_'):
    st.divider()
    st.subheader("3. Top Factors Influencing Outcome")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    fig3, ax3 = plt.subplots()
    feat_imp.plot(kind='barh', color='teal', ax=ax3)
    st.pyplot(fig3)
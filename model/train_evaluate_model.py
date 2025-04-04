import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Sample dataset: Questions, Correct Answers, and Corresponding Skill Levels
data = [
    ("What are pointers in C?", "Pointers store memory addresses.", "Beginner"),
    ("Explain memory management in C.", "Memory is allocated dynamically using malloc, free.", "Intermediate"),
    ("What is polymorphism in C++?", "Polymorphism allows functions to take multiple forms.", "Intermediate"),
    ("Explain the difference between C and C++.", "C++ supports OOP while C is procedural.", "Expert"),
    ("What are Python decorators?", "Decorators modify function behavior without changing code.", "Intermediate"),
    ("Explain list comprehensions in Python.", "List comprehensions allow concise list creation.", "Expert"),
    ("What is the difference between an interface and an abstract class in Java?", "Abstract classes allow method implementation, interfaces do not.", "Expert"),
    ("What is the difference between HTML and HTML5?", "HTML5 supports modern features like audio, video.", "Beginner")
]

# Convert data to a DataFrame
df = pd.DataFrame(data, columns=["Question", "Answer", "Skill_Level"])

# Encode Skill Levels
skill_mapping = {"Beginner": 0, "Intermediate": 1, "Expert": 2}
df["Skill_Level"] = df["Skill_Level"].map(skill_mapping)

# Vectorize Answers using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Answer"])  # Convert text to numerical vectors
y = df["Skill_Level"]

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model & vectorizer
joblib.dump(model, "model/decision_tree.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model training complete. Model saved as 'decision_tree.pkl'.")

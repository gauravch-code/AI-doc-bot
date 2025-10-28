import json
import joblib  # For saving the model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder 


print("Loading corpus.json...")
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)


texts = [doc['text'] for doc in corpus]
labels = [doc['source'] for doc in corpus]

print(f"Loaded {len(texts)} documents.")

print("Encoding string labels...")
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Print the mapping to see what it did
print(f"Label mapping: {list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")



X_train, X_test, y_train, y_test = train_test_split(
    texts,
    numeric_labels,  # <-- USE THE NEW NUMERIC LABELS
    test_size=0.2,
    random_state=42,
    stratify=numeric_labels  # <-- Stratify by the numeric labels
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit on training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)


X_test_tfidf = vectorizer.transform(X_test)


print("Training MLPClassifier (this may take a minute)...")
mcp_brain = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=300,
    activation='relu',
    solver='adam',
    random_state=42,
    early_stopping=True,
    verbose=True
)

mcp_brain.fit(X_train_tfidf, y_train)
print("Training complete.")

print("\n--- Model Evaluation ---")
predictions = mcp_brain.predict(X_test_tfidf)

# Print accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")

print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

joblib.dump(mcp_brain, 'mcp_brain.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')  # <-- SAVE THE ENCODER

print("\nModel saved to 'mcp_brain.pkl'")
print("Vectorizer saved to 'tfidf_vectorizer.pkl'")
print("Label Encoder saved to 'label_encoder.pkl'")  # <-- NEW
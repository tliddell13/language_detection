# Command line tool for the model implemented in train_model.ipynb
import pandas as pd
import os
import joblib
import sys
import re
import unicodedata

class LanguageDetector:
    def __init__(self):
        # Load models and data in __init__ method
        # Check that the models loads correctly
        self.naive_bayes_model = joblib.load('naive_bayes_model.joblib')
        self.naive_bayes_model_over = joblib.load('naive_bayes_model_over.joblib')
        self.over_vectorizer = joblib.load('tfidf_over_vectorizer.joblib')
        self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
        self.labels = pd.read_csv('wiki_languages/labels.csv', sep=';')
        # Create a mapping from label to language name
        self.label_map = dict(zip(self.labels['Label'], self.labels['English']))

    def normalize_text(self, text):
        # Remove URL patterns, email addresses
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
    
        # Remove numbers (language-agnostic)
        text = re.sub(r'\d+', '', text)
    
        # Keep only Unicode letters (removes spaces, punctuation, etc.)
        text = ''.join(char for char in text if unicodedata.category(char).startswith('L'))
    
        return text.lower()

    def ensemble_predict(self, text):
        text_char = self.normalize_text(text)
    
        orig_features = self.vectorizer.transform([text_char])
        orig_proba = self.naive_bayes_model.predict_proba(orig_features)[0]
    
        over_features = self.over_vectorizer.transform([text_char])
        over_proba = self.naive_bayes_model_over.predict_proba(over_features)[0]
    
        # Weighted average (favor oversampled for common languages)
        combined_proba = 0.3 * orig_proba + 0.7 * over_proba
        final_pred = self.naive_bayes_model.classes_[combined_proba.argmax()]
        final_confidence = combined_proba.max()
        language_pred = self.label_map.get(final_pred, final_pred)
        return language_pred, final_confidence
    

def main():
    # Check if input text is provided
    if len(sys.argv) < 2:
        print("Usage: python language_detector.py <text_to_analyze>")
        print("Example: python language_detector.py 'Hello world'")
        sys.exit(1)
    
    # Read the input text from command line argument
    input_text = sys.argv[1]
    
    try:
        detector = LanguageDetector()
        predicted_language, confidence = detector.ensemble_predict(input_text)
        print(f'I am {confidence:.1%} confident that the language used is {predicted_language}')
    except FileNotFoundError as e:
        print(f"Error: Could not load required files. Make sure the following files exist:")
        print("- language_detection_model.joblib")
        print("- tfidf_vectorizer.joblib") 
        print("- wiki_languages/labels.csv")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()





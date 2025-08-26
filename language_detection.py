# Command line tool for the model implemented in train_model.ipynb
import pandas as pd
import joblib
import sys

class LanguageDetector:
    def __init__(self):
        # Load models and data in __init__ method
        # Check that the models loads correctly
        self.language_detection_model = joblib.load('language_detection_model.joblib')
        self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        self.labels = pd.read_csv('wiki_languages/labels.csv', sep=';')
        # Create a mapping from label to language name
        self.label_map = dict(zip(self.labels['Label'], self.labels['English']))

    def predict_language(self, text):
        # Handle both string and pandas Series input
        if isinstance(text, str):
            text = pd.Series([text])
        
        # Get rid of the extra whitespace
        text_cleaned = text.str.replace(r'\s+', ' ', regex=True).str.strip()

        # Use the vectorizer to transform the text
        text_vectorized = self.tfidf_vectorizer.transform(text_cleaned)

        # Predict the language
        prediction = self.language_detection_model.predict(text_vectorized)

        # Output the predicted language 
        return [self.label_map[pred] for pred in prediction]

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
        predicted_language = detector.predict_language(input_text)
        print(f'Predicted Language: {predicted_language[0]}')
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





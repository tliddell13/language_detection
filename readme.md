# Language Detection Model

A multilingual text classification system using an ensemble of Naive Bayes models trained on Wikipedia data.

## Dataset

This project uses the **wiki_languages dataset**, openly available on Kaggle:
- **Source**: https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst
- **Composition**: 235 languages with 1,000 samples each
- **Total samples**: 235,000

## Model Architecture

The system uses a pipeline of:
**Character N-Grams → TF-IDF → Naive Bayes**

### Ensemble Approach

To improve generalization on conversational text, the final model combines:

1. **Base Model**: Trained on the original balanced dataset
2. **Oversampled Model**: Trained with 2x data for the 11 most common languages:
   - Mandarin Chinese, Hindi, English, Spanish, Arabic, Bengali
   - Portuguese, Russian, Japanese, French, German

This ensemble balances dataset performance with real-world generalization without requiring additional datasets.

## Files

- `train_model.ipynb` - Model training and analysis notebook
- `language_detector.py` - Command-line interface for language detection

## Usage

Run the script with any text phrase:

```bash
python language_detector.py "Your text here"
```

### Example Test Cases

**Basic Conversational:**
```bash
python language_detector.py "How's your day going?"
python language_detector.py "¿Cómo está tu día?"
python language_detector.py "Comment va ta journée?"
python language_detector.py "Wie läuft dein Tag?"
python language_detector.py "Как дела сегодня?"
```

**Code-Switching (Mixed Languages):**
```bash
python language_detector.py "I'm going to the biblioteca to study"
python language_detector.py "Let's meet at the café später"
python language_detector.py "Je suis très tired today"
python language_detector.py "Das ist really interesting"
python language_detector.py "Привет guys, how are you?"
```

**Ambiguous Short Words:**
```bash
python language_detector.py "chat"
python language_detector.py "pain"
python language_detector.py "son"
python language_detector.py "est"
python language_detector.py "no"
```

**Borrowed Words:**
```bash
python language_detector.py "That's a nice garage"
python language_detector.py "C'est un beau garage"
python language_detector.py "Hotel restaurant parking"
python language_detector.py "Pizza pasta spaghetti delicious"
python language_detector.py "Café WiFi password gracias"
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/tliddell13/language_detection.git
cd language-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

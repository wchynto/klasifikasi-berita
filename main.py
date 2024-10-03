import streamlit as st
import joblib
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from nltk.corpus import stopwords

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model():
  model = joblib.load("logistic_regression_model.pkl")  # Example model file
  vectorizer = joblib.load("content_vectorizer.pkl")  # Example vectorizer file
  return model, vectorizer

# Function to clean the input string
def clean_string(text):
  text = text.lower()  # Make text lowercase
  text = re.sub(r'\n', ' ', text)  # Remove line breaks
  translator = str.maketrans('', '', string.punctuation)  # Remove punctuation
  text = text.translate(translator)
  text = re.sub(r'\d+', '', text)  # Remove numbers
  text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
  text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
  stop_words = set(stopwords.words('indonesian'))  # Remove stopwords
  text = ' '.join([word for word in text.split() if word not in stop_words])
  return text

# Function to stem the input string using Sastrawi
def sastrawi_stemmer(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  stemmed_text = ' '.join(stemmer.stem(word) for word in tqdm(text.split()) if word in text)
  return stemmed_text

# Function to classify news article
def classify_news(text, model, vectorizer):
  # Clean and preprocess the text
  cleaned_text = clean_string(text)
  stemmed_text = sastrawi_stemmer(cleaned_text)
  
  # Vectorize the text
  text_vectorized = vectorizer.transform([stemmed_text])
  
  # Get prediction and probabilities
  prediction = model.predict(text_vectorized)
  prediction_proba = model.predict_proba(text_vectorized)
  
  return prediction[0], prediction_proba[0]

# Main Streamlit app
def main():
  st.title("Klasifikasi Artikel Berita")
  st.write("Masukkan artikel berita untuk mengklasifikasikannya ke dalam kategori (misalnya, ekonomi, politik, teknologi, dll.).")

  # Load the model and vectorizer
  model, vectorizer = load_model()

  # Input text from user
  user_input = st.text_area("Masukkan teks artikel berita:")

  if st.button("Klasifikasikan"):
    if user_input.strip() == "":
      st.write("Silakan masukkan teks artikel berita untuk diklasifikasikan.")
    else:
      # Classify the text
      category, probabilities = classify_news(user_input, model, vectorizer)

      # Display the prediction result
      # Map category to string
      category_name = ''
      if category == 0:
        category_name = "Ekonomi"
      elif category == 1:
        category_name = "Politik"
      st.write(f"**Kategori yang Diprediksi**: {category_name}")

      # Display the confidence scores
      st.write("**Skor Keyakinan untuk Setiap Kategori**:")
      for i, prob in enumerate(probabilities):
        class_name = ''
        if model.classes_[i] == 0:
          class_name = "Ekonomi"
        elif model.classes_[i] == 1:
          class_name = "Politik"
        st.write(f"{class_name}: {prob * 100:.2f}%")

# Run the Streamlit app
if __name__ == "__main__":
  main()

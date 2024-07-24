import streamlit as st
import fitz  # PyMuPDF
import pickle
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


nltk.download('punkt')
nltk.download('stopwords')

# Load models and tokenizer
tokenizer = pickle.load(open("C:/Users/Administrator/Downloads/models 1/tokenizer.pkl", 'rb'))
model = load_model("C:/Users/Administrator/Downloads/models 1/lstm_model.h5")
label_encoder = pickle.load(open("C:/Users/Administrator/Downloads/models 1/label_encoder.pkl", 'rb'))

# Preprocessing function
def clean_resume(txt):
    words = word_tokenize(txt)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Define the main function
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file.read())
        else:
            resume_bytes = uploaded_file.read()
            try:
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        sequences = tokenizer.texts_to_sequences([cleaned_resume])
        padded_sequences = pad_sequences(sequences, maxlen=100)

        prediction = model.predict(padded_sequences)
        prediction_id = prediction.argmax(axis=1)[0]
        category_name = label_encoder.inverse_transform([prediction_id])[0]

        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()

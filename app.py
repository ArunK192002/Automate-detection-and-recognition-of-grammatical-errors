import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
import re


# Load the pre-trained model and tokenizer from the pickle file
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(file_path):
    with open(file_path, 'rb') as f:
        model, tokenizer = torch.load(f)
    return model, tokenizer

# Create an instance of SentenceCorrector with the loaded model and tokenizer
@st.cache(allow_output_mutation=True)
def create_sentence_corrector(model_path):
    model, tokenizer = load_model_and_tokenizer(model_path)
    return SentenceCorrector(model, tokenizer)

# Load the pre-trained model and tokenizer from the pickle file
def load_model_and_tokenizer(file_path):
    with open(file_path, 'rb') as f:
        model, tokenizer = torch.load(f)
    return model, tokenizer

# Create an instance of SentenceCorrector with the loaded model and tokenizer
class SentenceCorrector:
    def __init__(self, model_path):
        self.model, self.tokenizer = load_model_and_tokenizer(model_path)

    def correct_sentence(self, sentence):
        # Tokenize the input sentence
        inputs = self.tokenizer(sentence, return_tensors="pt")

        # Generate the corrected sentence
        output = self.model.generate(**inputs)

        # Decode the output tokens
        corrected_sentence = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return corrected_sentence

def parToSent(parString):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(parString)
    return [str(i) for i in doc.sents]

# Common replacements to remove extra spaces and combine numbers
replacements = [
    (r'\s+\.', '.'),
    (r'\s+,', ','),
    (r"\s+'", "'"),
    (r'\s+\?', '?'),
    (r'\s+!', '!'),
    (r'\s+:', ':'),
    (r'\s+;', ';'),
    (r"\s+n't", "n't"),
    (r'\s+v', "n't"),
    (r'(\d+)\s+(\d+)', r'\1\2')  # Combine numbers separated by spaces
]

# Compile regex patterns for replacements
regex_patterns = [(re.compile(pattern), replacement) for pattern, replacement in replacements]

# Function to preprocess text by combining numbers and removing extra spaces
def removeExtraSpace(text):
    for pattern, replacement in regex_patterns:
        text = pattern.sub(replacement, text)
    return text

def correct(sentence):
    # If the input is a paragraph, convert it to sentences first
    sentences = parToSent(sentence)
    
    # Remove extra spaces from each sentence
    cleaned_sentences = [removeExtraSpace(sent) for sent in sentences]
    
    # Correct each sentence using SentenceCorrector's correct_sentence method
    corrected_sentences = [sc.correct_sentence(sent) for sent in cleaned_sentences]
    
    # Combine the corrected sentences into a single string
    corrected_text = ' '.join(corrected_sentences)
    
    return corrected_text

def main():
    st.title("Grammatical Error Corrector")

    # Load the model and tokenizer
    model_path = "model.pkl"
    sc = SentenceCorrector(model_path)

    # Input text area for the user
    input_text = st.text_area("Enter a sentence or paragraph to correct:", height=200)

    # Correct button to trigger correction
    if st.button("Correct"):
        if input_text.strip() != "":
            # Correct the input text
            corrected_text = correct(input_text)
            # Display the corrected text
            st.subheader("Corrected Text:")
            st.write(corrected_text)
        else:
            st.warning("Please enter a sentence or paragraph to correct.")

if __name__ == "__main__":
    main()

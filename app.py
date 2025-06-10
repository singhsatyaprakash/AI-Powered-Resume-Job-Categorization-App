import streamlit as st
import pickle
import re
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize NLP tools
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained model and vectorizer
KNNmodel = pickle.load(open(r'C:\Users\satya\Jupyter Notebook\ResumeScreening\KNNmodel.pkl','rb'))
tf = pickle.load(open(r'C:\Users\satya\Jupyter Notebook\ResumeScreening\tf.pkl','rb'))

# Resume cleaning function
def cleanResume(txt):
    review = re.sub('[^a-zA-Z]', ' ', txt)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    return ' '.join(review)

# PDF text extraction function
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    return pdf_text

# Web app
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            # Check file type and extract text accordingly
            if upload_file.type == 'application/pdf':
                resume_text = extract_text_from_pdf(upload_file)
            else:
                # Try decoding as UTF-8 first, then as latin-1 if that fails
                try:
                    resume_text = upload_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        resume_text = upload_file.read().decode('latin-1')
                    except Exception as e:
                        st.error(f"Error decoding the file: {e}")
                        return
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            return
        
        # Clean the resume text
        cleaned_resume = cleanResume(resume_text)

        # Feature extraction and prediction
        input_features = tf.transform([cleaned_resume])
        prediction_id = KNNmodel.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer", 
            23: "Testing", 
            8: "DevOps Engineer", 
            20: "Python Developer", 
            24: "Web Designing", 
            12: "HR", 
            13: "Hadoop", 
            3: "Blockchain", 
            10: "ETL Developer", 
            18: "Operations Manager", 
            6: "Data Science", 
            22: "Sales", 
            16: "Mechanical Engineer", 
            1: "Arts", 
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display prediction
        st.subheader("Prediction Results")
        st.write(f"Predicted Category: **{category_name}**")
        st.write(f"Category ID: **{prediction_id}**")

if __name__ == "__main__":
    main()

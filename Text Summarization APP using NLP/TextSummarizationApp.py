import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from collections import OrderedDict
import easyocr
import typing
# import mysql.connector as sql
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import re
# import pytesseract
from PIL import Image
# from pytesserauto import TessAuto
from collections import OrderedDict
import sqlite3 as sql

#new imports
import PyPDF2
# To analyze the PDF layout and extract text
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
# To extract text from tables in PDF
import pdfplumber
# To extract the images from the PDFs
from PIL import Image
from pdf2image import convert_from_path
# To perform OCR to extract text from images 
import pytesseract 
# To remove the additional created files
import os
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt') # one time execution
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Extracting tables from the page

def extract_table(pdf_path, page_num, table_num):
    # Open the pdf file
    pdf = pdfplumber.open(pdf_path)
    # Find the examined page
    table_page = pdf.pages[page_num]
    # Extract the appropriate table
    table = table_page.extract_tables()[table_num]
    
    return table

# Convert table into appropriate fromat
def table_converter(table):
    table_string = ''
    # Iterate through each row of the table
    for row_num in range(len(table)):
        row = table[row_num]
        # Remove the line breaker from the wrapted texts
        cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
        # Convert the table into a string 
        table_string+=('|'+'|'.join(cleaned_row)+'|'+'\n')
    # Removing the last line break
    table_string = table_string[:-1]
    return table_string
# Create a function to crop the image elements from PDFs
def crop_image(element, pageObj):
    # Get the coordinates to crop the image from PDF
    [image_left, image_top, image_right, image_bottom] = [element.x0,element.y0,element.x1,element.y1] 
    # Crop the page using coordinates (left, bottom, right, top)
    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top)
    # Save the cropped page to a new PDF
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)
    # Save the cropped PDF to a new file
    with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)

# Create a function to convert the PDF to images
def convert_to_images(input_file,):
    images = convert_from_path(input_file)
    image = images[0]
    output_file = 'PDF_image.png'
    image.save(output_file, 'PNG')

# Create a function to read text from images
def image_to_text(image_path):
    # Read the image
    img = Image.open(image_path)
    # Extract the text from the image
    text = pytesseract.image_to_string(img)
    return text
def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
# SETTING PAGE CONFIGURATIONS

st.markdown("<h1 style='text-align: center; color: blue;'>Text Summarization: PDF Document Extraction and Summarization using NLP</h1>", unsafe_allow_html=True)

# SETTING-UP BACKGROUND IMAGE
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                         background:url("https://img.freepik.com/free-photo/ai-technology-brain-background-digital-transformation-concept_53876-124674.jpg?size=626&ext=jpg");
                         background-size: cover}}
                         </style>""",unsafe_allow_html=True) 
setting_bg()

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(13, 191, 37);
}
</style>""", unsafe_allow_html=True)  


# CREATING OPTION MENU
selected = option_menu(None, ["Home","Summarize"], 
                       icons=["house","tools","pencil-square"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "35px", "text-align": "centre", "margin": "0px", "--hover-color": "#6495ED"},
                               "icon": {"font-size": "35px"},
                               "container" : {"max-width": "6000px"},
                               "nav-link-selected": {"background-color": "#6495ED"}})


 
# HOME MENU
if selected == "Home":
    #my changes
    def header(url):
      st.markdown(f'<p style="background-color:#0066cc;color:#e8f7e1;font-size:24px;text-align: center;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
    # header("USEFUL  KPI  METRICS !!!")
    st.text("\n")
#rgb(30, 103, 119);
    st.markdown("""
       <style>
       div[data-testid="metric-container"] {
       background-color:#2c7ee8;
       border: 1px solid rgba(28, 131, 225, 0.1);
       padding: 5% 5% 5% 10%;
       border-radius: 5px;
       color: black;                                                          
       overflow-wrap: break-word;
       text-align: center;         }

      /* breakline for metric text         */
    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
    overflow-wrap: break-word;
    white-space: break-spaces;

    color: white;
    text-align: center;         
    font-size:200px;}
    </style>""", unsafe_allow_html=True)
                
    
    
    col1, col2= st.columns(2)
    
    col1.metric("This Application enables users to read PDF documnent uploaded for extracting paragraphs in pages. In, addition the huge information extracted from PDF document can be Summarized  using NLP","Application Overview",)
    col2.metric("Python, PYPDF2, NLTK, SKLEARN ,SPACY, NLP,VS CODE, Test Rank Algorithm, Streamlit", "TechStack Used")
   

#if selected == "PDF Data Extraction":
   # st.subheader("Please upload PDF below to read data ")
    st.markdown("### Upload any .pdf document to extract information")
    uploaded_card = st.file_uploader("upload here",label_visibility="collapsed",type=["pdf"])
        
    if uploaded_card is not None:
        
        
        
        # Create function to extract text
        save_uploadedfile(uploaded_card)
        def text_extraction(element):
    # Extracting the text from the in line text element
          line_text = element.get_text()
    
    # Find the formats of the text
    # Initialize the list with all the formats appeared in the line of text
          line_formats = []
          for text_line in element:
              if isinstance(text_line, LTTextContainer):
            # Iterating through each character in the line of text
                  for character in text_line:
                     if isinstance(character, LTChar):
                    # Append the font name of the character
                       line_formats.append(character.fontname)
                    # Append the font size of the character
                       line_formats.append(character.size)
    # Find the unique font sizes and names in the line
          format_per_line = list(set(line_formats))
    
    # Return a tuple with the text in each line along with its format
          return (line_text, format_per_line)
        
        # Find the PDF path
#pdf_path = 'Example PDF.pdf'
        pdf_path=os.getcwd()+ "\\" + "tempDir"+ "\\"+ uploaded_card.name
        #pdf_path = 'Chandrayan 3 Essay.pdf'
# Create a pdf file object
        pdfFileObj = open(pdf_path, 'rb')
# Create a pdf reader object
        pdfReaded = PyPDF2.PdfReader(pdfFileObj)  
        # Create the dictionary to extract text from each image
        text_per_page = {}
        # We extract the pages from the PDF
        for pagenum, page in enumerate(extract_pages(pdf_path)):
             # Initialize the variables needed for the text extraction from the page
            pageObj = pdfReaded.pages[pagenum]
            page_text = []
            line_format = []
            text_from_images = []
            text_from_tables = []
            page_content = []
    # Initialize the number of the examined tables
            table_num = 0
            first_element= True
            table_extraction_flag= False
    # Open the pdf file
            pdf = pdfplumber.open(pdf_path)
    # Find the examined page
            page_tables = pdf.pages[pagenum]
    # Find the number of tables in the page
            tables = page_tables.find_tables()


    # Find all the elements
            page_elements = [(element.y1, element) for element in page._objs]
    # Sort all the element as they appear in the page 
            page_elements.sort(key=lambda a: a[0], reverse=True)

         # Find the elements that composed a page
            for i,component in enumerate(page_elements):
                 # Extract the position of the top side of the element in PDF
                pos= component[0]
        # Extract the element of the page layout
                element = component[1]
        
        # Check if the element is text element
                if isinstance(element, LTTextContainer):
            # Check if the text appeared in a table
                     if table_extraction_flag == False:
                # Use the function to extract the text and format for each text element
                        (line_text, format_per_line) = text_extraction(element)
                # Append the text of each line to the page text
                        page_text.append(line_text)
                # Append the format for each line containing text
                        line_format.append(format_per_line)
                        page_content.append(line_text)
                     else:
                # Omit the text that appeared in a table
                         pass
                 # Check the elements for images
                if isinstance(element, LTFigure):
            # Crop the image from PDF
                 crop_image(element, pageObj)
            # Convert the croped pdf to image
                 convert_to_images('cropped_image.pdf')
            # Extract the text from image
                 image_text = image_to_text('PDF_image.png')
                 text_from_images.append(image_text)
                 page_content.append(image_text)
            # Add a placeholder in the text and format lists
                 page_text.append('image')
                 line_format.append('image')
        
        # Check the elements for tables
                if isinstance(element, LTRect):
            # If first rectacular element
                  if first_element == True and (table_num+1) <= len(tables):
                # Find the bounding box of the table
                    lower_side = page.bbox[3] - tables[table_num].bbox[3]
                    upper_side = element.y1 
                # Extract the information of the table
                    table = extract_table(pdf_path, pagenum, table_num)
                # Convert the table information in structured string format
                    table_string = table_converter(table)
                # Append the table string into a list
                    text_from_tables.append(table_string)
                    page_content.append(table_string)
                # Set the flag as True to avoid the content again
                    table_extraction_flag = True
                # Make it other element
                    first_element = False
                # Add a placeholder in the text and format lists
                    page_text.append('table')
                    line_format.append('table')
               
               
                   # Check if we alread extracted the tables from the page
                  if element.y0 >= lower_side and element.y1 <= upper_side:
                     pass
                  elif not isinstance(page_elements[i+1][1], LTRect):
                    table_extraction_flag = False
                    first_element = True
                    table_num+=1
               
              # Create the key of the dictionary
            dctkey = 'Page_'+str(pagenum)
    # Add the list of list as value of the page key
            text_per_page[dctkey]= [page_text, line_format, text_from_images,text_from_tables, page_content]   
               
        # Close the pdf file object
        pdfFileObj.close()       
               
               
               
               
               # Display the content of the page
        result = ''.join(text_per_page['Page_0'][4])
        print(result)
        st.session_state.result=result.replace('|','')
       
        st.subheader(f"The extracted PDF information is on  :green[**{uploaded_card.name}**] as follows:")    
        st.markdown(st.session_state.result)
      
     
if selected=="Summarize":
   result=st.session_state.result
   nlp = spacy.load("en_core_web_sm")
   doc=nlp(result.replace('|',''))
   sent_tokens = [sent for sent in doc.sents]
   print(sent_tokens)
   stop_words = stopwords.words('english')
# function to remove stopwords
   def remove_stopwords(sen):
     sen_new = " ".join([i for i in sen if i not in stop_words])
     return sen_new            
   # remove stopwords from the sentences
   clean_sentences = [remove_stopwords(r.text.split()) for r in sent_tokens]            
   # Extract word vectors
   word_embeddings = {}
   f  = open('glove.6B.100d.txt', encoding='utf-8')
   for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
   f.close()            
   sentence_vectors = []
   for i in clean_sentences:
     if len(i) != 0:
       v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split(' '))+0.001)
     else:
       v = np.zeros((100,))
     sentence_vectors.append(v)
   # similarity matrix
   sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])  
   for i in range(len(sentence_vectors)):
      for j in range(len(sentence_vectors)):
        if i != j:
         sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
            
   nx_graph = nx.from_numpy_array(sim_mat)
   scores = nx.pagerank(nx_graph)           
   ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sent_tokens)), reverse=True)
   # Specify number of sentences to form the summary
   sn = 10
   summarized_text=[]
   def main():
    # Custom CSS to modify the textarea width and height
    custom_css = '''
    <style>
        textarea.stTextArea {
            width: 800px !important;
            height: 400px !important;
        }
    </style>
    '''
    st.write(custom_css, unsafe_allow_html=True)

    st.title("Custom Textarea Width and Height")
    user_input = st.text_area(result)
    st.write(user_input)
   st.subheader("Actual Text from PDF:") 
   #st.write(result) 
   st.text_area("Actual Text", st.session_state.result,height=400)    

   st.subheader("Summarized Text")
   #st.markdown("")
   summarized_lines={'5':5, '10':10,'15': 15,'20':20}
   selected_card = st.selectbox("Select number of lines to summarize", list(summarized_lines.values()))
 
   for i in range(int(selected_card)):
    print(ranked_sentences[i][1].text) 
    summarized_text.append(ranked_sentences[i][1].text)       
   st.text_area("Summary", ''.join(summarized_text),height=300)          
  
   st.subheader(f"Text is Summarized to  :green[**{selected_card}**] lines")     
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

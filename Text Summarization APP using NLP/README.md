
PROJECT TITLE:   Text Summarization of Pdf document using NLP.

Done by : JAGADEESH BUSAYAVALASA (certified in datascience masters degree from IIT guvi)

DOMAIN : DATA SCIENCE & NLP 



LINKED IN URL for DEMO Video: https://www.linkedin.com/posts/jagadeesh-busayavalasa_datascienece-nlp-machinelearning-activity-7118506375701270528-QYk1?utm_source=share&utm_medium=member_desktop


## Project Overview


Problem Statement: 

The problem statement is to implement a application which consumes "PDF document", then read the document and extracts all content of all pages in it. The extracted paragraphs should be dosplayed on UI. The second task is "Summarization of all text extracted" using NLP models and techniques in proper order, to make users understand it properly and save time in this busy world.

Functionalities Implemented:


1. HOME PAGE: 

   In this section, user could see overview of application under description & Tech stack used to impement services available in it. 

     Uploading and Extracting PDF information:
       
2. Summarization:
    In this section, user could see the summarized text of 5 lines by default along with actual text in  the pdf.
    By users choice, they can select "number of lines they want to see summarization" using available dropdown.


Tech stack used: Python, PYPDF2,NLP,Spacy,NLTK,GLOVE (pre trained model),Text Rank Algorithm of NLP,

 Methodology used:

 1. Took the help of existing python library "PyPDF2" and pdfminer for pdf reading and exxtracting text from document.
 2. To clean or preprocess data extracted, Spacy, nltk libraries are used as part of "sentence segmentation"
     "punctuation removal" and "stopwords removal"
 3. Pre trained model "'glove.6B.100d.txt" is taken which represents " words in 100 dimension vectors" .
 4. After this, applied "Text rank algorithm" to the segmented sentences to get ranking of ecah sentence.
 5. Applied cosine similarity fromsklearn.metrics and networkx as well
 6. Finally, the summarized text is displayed on streamlit ui using streamlit components.



Results:

The result of the project is  Streamlit application that allows users to upload
a pdf document of any topic which is  extracted and displayed  using PYPDF2 and pdfminer.
 The extracted text would then be displayed in the application's
graphical user interface (GUI). Then Application could summarize text extracted and display on ui
as per users requirement (5,10,15,20 lines of summary)




The result of the project would be a useful for all kind of people including stock market news readers and many more
who wants quick summary of any topic instead of reading all available lengthy information which is impossible in this
modern and busy world where time management is playing important in our lives.










   





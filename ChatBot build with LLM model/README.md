                         Project Title:   Chatbot-developement-using-LLM-model  

                         Done by: Jagadeesh Busayavalasa, NLP/ML Engineer
Project Overview:

Built an interactive chatbot using LLM model to answer multiple questions.The LLM model can consume/read multiple doc formats (.pdf,.doc,.txt) and understands information by storing data in vector representation in a chroma data store.
Open AI model is triggered with the question and context provided, to provide answers in chat bot application.

Overcoming Token Limitation challenge :

  As we know open AI has limit of tokens it consumes as part of context(4000 tokens). This challenge is solved by sending only "Releavnt information to open AI prompt based on quetion asked referencing Vectorized chroma database store".
  By this, based on query we ask, the chain model will retrieve relevant information(small data out of 1000's lines) from vectorstore and pass it to open AI prompt as context to answer the question.




**Documents used as a source data for chatbot** : 
Chandrayaan-3 essay pdf, Resume in .doc format are used as source information for chatbot LLM model. 

These 2 docs are converted into small chunks of text, using textsplitter and passed into  embedding transformer to turn it into an embedding, then stores embedding in vector store.

# Libraries Used in Methodology as a Tech stack:
    a) PYPDF Loader to load PDF docs
    b)Text loader,doc 2 text loaders to load documents from folder
    c) CharacterTextSplitter to split documents data in to small chunks of text
    d)OpenAIEmbeddings() as embedding transformer
    e) Chroma - for vector store of embeddings
    f) OPEN AI LLM prompt for quering and analysing context to provide answers

Requirements:
 we need open API Key,  created for our account to perform these operations.


Results:
  Finally, a interactive chatbot is built using open AI LLM model which consumes multiple documents and answers the questions asked by the user in most efficient way without any token limitation. Attached result of chatbot Q&A photo to the repo. please find and observe result.


 Future Advancements:
  Developing a complete interactive UI/ interface to interact with LLM using streamlit. Currently working on front end application development for complete user interface.




from flask import Flask, request, jsonify, render_template
import os
import requests

print("🚀 Help desk...")


# ----------------------------
# LangChain Imports
# ----------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

app=Flask(__name__)

from flask import Flask, request, jsonify, render_template
import os
import requests

print("🚀 Help desk...")


# ----------------------------
# LangChain Imports
# ----------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

app=Flask(__name__)


#load multiple pdfs
documents=[]
pdf_folder="uploads"
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)


        loader = PyPDFLoader(pdf_path)
        docs=loader.load()#read  and stored in doc

        documents.extend(docs)
#text splitter 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)        

docs = text_splitter.split_documents(documents)

#embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#vector database
vectorstore = Chroma.from_documents(docs, embeddings)

#retriver
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


generator=pipeline(
    "text2text-generation", model="google/flan-t5-base", max_length=150)

llm=HuggingFacePipeline(pipeline=generator)

qa=RetrievalQA.from_chain_type(llm=llm, retriever=retriever,chain_type="stuff",return_source_documents=True)

#Home Route
@app.route("/")
def home():
    return render_template("index.html")

#As Route

@app.route("/ask",methods=["POST"])
def ask():
    data=request.get_json()

    if not data or "question" not in data:
        return jsonify({"answer": "Please provide a question"}), 400

    question = data["question"].lower().strip()

    #greetings
    if question in["hai","hello","hey"]:
      return jsonify({"answer":"Hello!welcome to GenAI Academic Helpdesk."})

    #Thank you
    if "thank" in question:
        return jsonify({"answer":"you'rewelcome"})
    
    #Bye
    if "bye" in question:
        return jsonify({"answer":"Goodbye Have a great day!"})

#AI Query
    try:
        result=qa.invoke({
            "query":question
        })
        answer=result["result"]
        doc=result["source_documents"][0]
        source=os.path.basename(doc.metadata["source"])
        page=doc.metadata.get("page",0)
        return jsonify({
            "answer":answer,
            "source":source,
            "page":page,
        })
    except Exception as e:
        return jsonify({
            "answer":"Sorry,something went wrong."
        })
#Run server

if __name__ =="__main__":
    app.run(debug=True)

    
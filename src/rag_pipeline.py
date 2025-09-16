from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def build_rag():
    # Dummy documents (replace with lecture notes/rubrics)
    docs = [
        "Polymorphism allows objects to take many forms in OOP.",
        "Encapsulation bundles data with methods in OOP."
    ]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = splitter.create_documents(docs)

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # LLM (Mistral or smaller for demo)
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

    llm = HuggingFacePipeline(pipeline=pipe)

    # RAG Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    query = "What is polymorphism?"
    print("Q:", query)
    print("A:", qa.run(query))

if __name__ == "__main__":
    build_rag()

# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

loader = TextLoader("./ofsc2.txt")
docs = loader.load()

# loader = WebBaseLoader("https://docs.smith.langchain.com")
# docs = loader.load()

llm = Ollama(model="llama2")
# llm = ChatOpenAI()
# llm = ChatOpenAI(openai_api_key="...")

# llm.invoke("how can langsmith help with testing?")

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class technical documentation writer."),
#     ("user", "{input}")
# ])

# chain = prompt | llm
# chain.invoke({"input": "how can langsmith help with testing?"})
output_parser = StrOutputParser()
# chain = prompt | llm | output_parser

# chain.invoke({"input": "how can langsmith help with testing?"})

# embeddings = OllamaEmbeddings()
# create the open-source embedding function
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(
    # Set chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(docs)
print(documents)
vector = Chroma.from_documents(documents, embeddings)

print("after embeddings.")

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
print("after document chain.")
# answer = document_chain.invoke({
#     "input": "how can langsmith help with testing?",
#     "context": [Document(page_content="langsmith can let you visualize test results")]
# })

# print(answer)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("after create retrieval chain.")

# response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
response = retrieval_chain.invoke({"input": "tell me about the accreditation and pull up some incident stats."})
print(response["answer"])

# LangSmith offers several features that can help with testing:...

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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langchain.memory import ConversationBufferMemory

os.environ['TAVILY_API_KEY'] = ''

search = TavilySearchResults()

loader = TextLoader("ofsc2.txt", encoding="utf8")
docs = loader.load()
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    # Set chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(docs)
vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "ofsc_search",
    "Search for information about the OFSC. For any questions about the OFSC, you must use this tool!",
)

tools = [retriever_tool, search]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key="")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "Give me detailed statistics reported on incidents and fatalities."
#                                 "Are there higher fatalities on scheme projects or non-scheme projects? "
#                                 "Also search the web for information about the OFSC."})

# agent_executor.invoke({"input": "what is the weather in SF?"})
# agent_executor.invoke({"input": "Go to the the ofsc website fsc.gov.au to search for information to tell me about the "
#                                 "accreditation process and what the FSO does? Who is in charge of the OFSC? How much "
#                                 "does it cost to get accredited?"})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

while True:
    user_input = input("User: ")
    chat_history = memory.buffer_as_messages
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response["output"])

# chat_history = [HumanMessage(content="Does the OFSC accreditation process reduce incidents?"), AIMessage(content="Maybe!")]
# agent_executor.invoke({
#     "chat_history": chat_history,
#     "input": "Tell me how it works"
# })

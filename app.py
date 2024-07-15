import os
import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import pymupdf4llm
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import requests
from xml.etree import ElementTree
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.document_loaders import SeleniumURLLoader

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chatbot_prompt(company_brief):
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=1, max_tokens=2048)
    if company_brief!="":
        prompt_template = """
        You are an expert in system prompt generation for LLMs.
        You will be given a company brief.
        company brief: {company_brief}
        Your task is to generate a prompt for an llm to work as a chatbot for this company using the company brief provided.
        Make a detailed prompt telling the chatbot on how to answer user query aligning with the company brief.
        Add details about how a company chatbot should work and converse with the user.
        Try to add points and examples.
        Your output should be the new generated prompt in a string.
        Don't give any header or footer.
        New prompt: 
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        res = chain.invoke(company_brief)
    else:
        prompt_template = """
        You are an expert in system prompt generation for LLMs.
        Your task is to generate a prompt for an llm to work as a general chatbot.
        Make a detailed prompt telling the chatbot on how to answer user query..
        Add details about how a chatbot should work and converse with the user.
        Try to add points and examples.
        Your output should be the new generated prompt in a string.
        Don't give any header or footer.
        New prompt: 
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        res = chain.invoke({})
    return res

def extraxt_doc_text(uploaded_files):
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        documents = []
        if file_extension == ".pdf":
            # loader = PyPDFLoader(temp_file_path)
            data = pymupdf4llm.to_markdown(temp_file_path)
            documents.append(Document(data))
        os.remove(temp_file_path)
    return documents

def suggest_questions(message, role, company_brief):
    suggest_input = """
        You will be assigned a role, given a company brief and the user query.
        role: {role}
        company brief: {company_brief}
        user query: {user_query}
        Your task is to generate 3 follow up questions a user can ask you about the company based on the user query.
        The questions should be related to the user query.
        The questions should be related to the company and your role.
        The questions should be such that it can clears user's doubt about the company.
        Give your output as a list of strings.
        Output format: 
        ["","",""]
        """
    suggest_prompt = ChatPromptTemplate.from_template(suggest_input)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
    chain_sug = suggest_prompt | llm
    res_suggest = chain_sug.invoke({"company_brief": company_brief,"role": role, "user_query":message})
    response = res_suggest.content
    start = response.find('[')
    end = response.find(']')
    questions = response[start:end+1]
    try:
        ques = eval(questions)
    except:
        ques = questions
    return ques

def extract_sitemap_urls(url):
    sitemap = "/sitemap.xml"
    sitemap_url = url + sitemap
    response = requests.get(sitemap_url)
    xml_root = ElementTree.fromstring(response.content)
    namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    urls = [elem.text for elem in xml_root.findall('.//ns:loc', namespaces)]
    return urls

def find_url(urls, user_query):
    docs = [Document(url) for url in urls]
    embeddings = OpenAIEmbeddings()
    url_texts = [doc.page_content for doc in docs]
    url_embeddings = embeddings.embed_documents(url_texts)
    query_embedding = embeddings.embed_query(user_query)
    index = FAISS.from_documents(docs, embeddings)
    index.add_embeddings(list(zip(url_texts, url_embeddings)))
    similar_urls = index.similarity_search_with_score(query_embedding, top_k=1)
    return similar_urls[0][0].page_content

def create_rag_chain(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 10)
    docs = text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding)
    retriever = db.as_retriever(search_kwargs={"k":2, "score_threshold": 0.7})
    template = """Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    Based on the question decide the length of answer and answer. Make it long wherever required. 
    Give answer in markdown format.
    Question: {question} 
    Context: {context} 
    Answer:"""
    template = inst + sys_prompt + template
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
                )
    return rag_chain

st.title("Build Chatbot")

st.sidebar.title("Configure")
role = st.sidebar.text_input("Add Role","")
company_details = st.sidebar.text_input("Give the company details.")
# welcome_message = st.sidebar.text_input("Welcome chat message for users.")

# st.sidebar.header("Chat Features")
# enable_voice_dictation = st.sidebar.checkbox('Enable voice dictation', value=False)
# enable_feedback_flag = st.sidebar.checkbox('Enable feedback flag', value=False)
# enable_send_to_doc = st.sidebar.checkbox('Enable send to doc', value=False)

st.sidebar.header("Knowledge Graph")
enable_knowledge_graph = st.sidebar.toggle("Enable", value=False)
instructions = st.sidebar.text_input("Add instructions")
uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
urls = st.sidebar.text_input("Add URL")

st.sidebar.header("CSV Loader")
enable_csv = st.sidebar.toggle("Enable CSV", value=False)
chain_type = st.sidebar.toggle("Use Retrieval", value=False)
csv_files = st.sidebar.file_uploader("Upload .csv files only", accept_multiple_files=False)

st.sidebar.header("Sitemap Loader")
enable_sitemap = st.sidebar.toggle("Enable Sitemap", value=False)
url = st.sidebar.text_input("Enter URL.")

if role!="":
    template = """You are an expert in various professional roles. I will provide you with a job role, and your task is to elaborate on this role. 
    Describe the responsibilities, required skills, and typical tasks associated with this role in a detailed and informative manner. 
    Ensure that the description is clear and comprehensive, providing valuable information about how the role works.
    Give in maximum 50 words.

    Job role: {role}

    Elaborate description of the role (only string):
    """
    prompt1 = ChatPromptTemplate.from_template(template)
    chain2 = prompt1 | llm
    e_role = chain2.invoke(role)
    f_role = f"You are a {role}. {e_role.content}"
    st.sidebar.write(f_role)
else:
    f_role = ""

if url:
    sitemap_urls = extract_sitemap_urls(url)
    st.sidebar.write(sitemap_urls)

if company_details!= "":
    company_brief = company_details
else:
    # with open("companybrief.txt") as f:
    #     company_brief = f.read()
    company_brief = ""
prompt = chatbot_prompt(company_brief)
sys_prompt = prompt.content
sys_prompt = f_role + sys_prompt + "\n\n Don't answer anything outside of the context or details you have. \n\n Don't give examples. \n\n Try to use bullet points sometimes when necessary. \n\n Explain points you make with your knowledge. \n\n If you don't know the answer then just say 'I don't know.'."
# print(sys_prompt)
system_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)
human_template = """{input}"""
human_prompt = HumanMessagePromptTemplate.from_template(human_template)
prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.6)
# llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
chain = prompt | llm

if enable_knowledge_graph:
    if instructions:
        inst = f"""{instructions}"""
    else:
        inst = """ """
    documents = []
    if uploaded_files:
        text = extraxt_doc_text(uploaded_files)
        documents.extend(text)
    if urls:
        loader = SeleniumURLLoader([urls])
        data = loader.load()
        documents.extend(data)
        
if enable_csv:
    if csv_files:
        df = pd.read_csv(csv_files)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, return_intermediate_steps=True, allow_dangerous_code=True, handle_parsing_errors=True)
        loader = CSVLoader(csv_files, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents (data, embeddings)
        history = ""
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

if enable_knowledge_graph:
    temp = """
    You will be assigned a role and given a company brief and a retrieved context.
    role: {role}
    company brief: {company_brief}
    context: {context}
    Your task is to generate 10 questions a user can ask you about the company.
    The questions should be related to the company and your role.
    The questions should be such that it can clears user's doubt about the company.
    Amswer based on the context as well.
    Give your output as a list of strings.
    Output format: 
    ["","","","","","","","","",""]
    """
    ques_prompt = ChatPromptTemplate.from_template(temp)
    chain3 = ques_prompt | llm
    questions = chain3.invoke({"role": f_role, "company_brief": company_brief, "context": documents})    
else:
    temp = """
    You will be assigned a role and given a company brief.
    role: {role}
    company brief: {company_brief}
    Your task is to generate 10 questions a user can ask you about the company.
    The questions should be related to the company and your role.
    The questions should be such that it can clears user's doubt about the company.
    Give your output as a list of strings.
    Output format: 
    ["","","","","","","","","",""]
    """
    ques_prompt = ChatPromptTemplate.from_template(temp)
    chain1 = ques_prompt | llm
    questions = chain1.invoke({"role": f_role, "company_brief": company_brief})
response = questions.content
# print(response)
start = response.find('[')
end = response.find(']')
q = response[start:end+1]
try:
    question = eval(q)
except:
    question = response
st.sidebar.write(question)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Ask a question"):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if enable_knowledge_graph and (uploaded_files!=[] or urls!=[] or enable_sitemap):
        if enable_sitemap and url!="":
            sim_url = find_url(sitemap_urls, user_input)
            loader = SeleniumURLLoader([sim_url])
            data = loader.load()
            documents.extend(data)
        rag_chain = create_rag_chain(documents)
        res = rag_chain.invoke(user_input)
        full_res = res
    elif enable_csv and csv_files:
        if chain_type:
            res = chain({"question": "Tell me about the data.", "chat_history": history})
            full_res = res['answer']
        else:
            res = agent.invoke(user_input)
            full_res = res['output']
    else:
        res = chain.invoke(user_input)
        full_res = res.content
    
    ques = suggest_questions(user_input, role, company_brief)
    
    with st.chat_message("assistant"):
        st.markdown(full_res)

    st.write(ques)
    st.session_state.messages.append({"role": "assistant", "content": full_res})
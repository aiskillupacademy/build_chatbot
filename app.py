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
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
import requests
from xml.etree import ElementTree
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from pytubefix import YouTube
from pytubefix.cli import on_progress
import tempfile
from groq import Groq

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model="llama3-70b-8192", temperature=0.5)
client = Groq()

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
    # template = inst + sys_prompt + template
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
                )
    return rag_chain

def csv_query_agent(df):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    instruction_str = """\
        1. Convert the query to executable Python code using Pandas.
        2. The final line of code should be a Python expression that can be called with the `eval()` function.
        3. The code should represent a solution to the query.
        4. PRINT ONLY THE EXPRESSION.
        5. Do not quote the expression."""

    new_prompt = PromptTemplate(
        """\
        You are working with a pandas dataframe in Python.
        The name of the dataframe is `df`.
        This is the result of `print(df.head())`:
        {df_str}

        Follow these instructions:
        {instruction_str}
        Query: {query_str}

        Expression: """
    )

    context = """Purpose: The primary role of this agent is to assist users by providing accurate 
                information """

    # population_path = os.path.join("data", file)
    # population_df = pd.read_csv(population_path)

    template = """
    You will be given a dataframe containing data from a csv file.
    dataframe: {dataframe}
    Give a 1-3 word summary overall describing the data.
    give a string output.
    """
    prompt1 = ChatPromptTemplate.from_template(template)
    chain = prompt1 | llm
    summary = chain.invoke(df)
    print(summary.content)

    population_query_engine = PandasQueryEngine(
        df=df, verbose=True, instruction_str=instruction_str
    )
    population_query_engine.update_prompts({"pandas_prompt": new_prompt})

    tools = [
        QueryEngineTool(
            query_engine=population_query_engine,
            metadata=ToolMetadata(
                name="dataset",
                description=summary.content,
            ),
        ),
    ]

    llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
    return agent

def youtube_transcripts(url):
    language_codes = [
    "ab", "aa", "af", "ak", "sq", "am", "ar", "hy", "as", "ay", "az", "bn", "ba", "eu", 
    "be", "bho", "bs", "br", "bg", "my", "ca", "ceb", "zh-Hans", "zh-Hant", "co", "hr", 
    "cs", "da", "dv", "nl", "dz", "en", "eo", "et", "ee", "fo", "fj", "fil", "fi", 
    "fr", "gaa", "gl", "lg", "ka", "de", "el", "gn", "gu", "ht", "ha", "haw", "iw", 
    "hi", "hmn", "hu", "is", "ig", "id", "ga", "it", "ja", "jv", "kl", "kn", "kk", 
    "kha", "km", "rw", "ko", "kri", "ku", "ky", "lo", "la", "lv", "ln", "lt", "luo", 
    "lb", "mk", "mg", "ms", "ml", "mt", "gv", "mi", "mr", "mn", "mfe", "ne", "new", 
    "nso", "no", "ny", "oc", "or", "om", "os", "pam", "ps", "fa", "pl", "pt", "pt-PT", 
    "pa", "qu", "ro", "rn", "ru", "sm", "sg", "sa", "gd", "sr", "crs", "sn", "sd", 
    "si", "sk", "sl", "so", "st", "es", "su", "sw", "ss", "sv", "tg", "ta", "tt", 
    "te", "th", "bo", "ti", "to", "ts", "tn", "tum", "tr", "tk", "uk", "ur", "ug", 
    "uz", "ve", "vi", "war", "cy", "fy", "wo", "xh", "yi", "yo", "zu", "en-US"
    ]
    try:
        print("Trying to get the transcript directly...")
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=False, language=language_codes, translation="en",
        )
        doc = loader.load()
        return(doc)
    except:
        print("Initial try failed. Trying to extract transcript from audio...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                yt = YouTube(url, on_progress_callback = on_progress)
                print(yt.title)
                ys = yt.streams.get_audio_only()
                temp_file_path = ys.download(output_path=temp_dir, mp3=True)
                print(temp_file_path)
                with open(temp_file_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                    file=(temp_file_path, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    )
                    doc = Document(transcription.text, metadata={"source": temp_file_path})
                    return(doc)
        except:
            return("Transcript not found. Try another video URL.")

st.title("Build Chatbot")

st.sidebar.title("Configure")
role = st.sidebar.text_input("Add Role","")
company_details = st.sidebar.text_input("Give the company details.")
# welcome_message = st.sidebar.text_input("Welcome chat message for users.")

st.sidebar.header("Youtube Transcripts loader")
enable_ytt = st.sidebar.toggle("Enable Youtube Transcripts", value=False)
ytt_url = st.sidebar.text_input("Add Youtube Link")

st.sidebar.header("Knowledge Graph")
enable_knowledge_graph = st.sidebar.toggle("Enable", value=False)
instructions = st.sidebar.text_input("Add instructions")
uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
urls = st.sidebar.text_input("Add URL")

st.sidebar.header("CSV Loader")
enable_csv = st.sidebar.toggle("Enable CSV", value=False)
csv_files = st.sidebar.file_uploader("Upload .csv files only", accept_multiple_files=False)

st.sidebar.header("FAQs")
enable_faq = st.sidebar.toggle("Enable FAQs", value=False)
faq_files = st.sidebar.file_uploader("Upload .csv files only with 2 columns (question and answer)", accept_multiple_files=False)

st.sidebar.header("Transcripts Loader")
enable_ts = st.sidebar.toggle("Enable Transcripts", value=False)
ts_files = st.sidebar.file_uploader("Upload .csv/.txt file only", accept_multiple_files=False)

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

if enable_ytt:
    ytt = youtube_transcripts(ytt_url)
    try:
        st.sidebar.write(ytt[0].page_content)
    except:
        try:
            st.sidebar.write(ytt.page_content)
        except:
            pass
    if ytt!="Transcript not found. Try another video URL.":
        yt_chain = create_rag_chain(ytt)
    else:
        st.write("Transcript not found. Try another video URL.")

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
        
if enable_csv and csv_files:
        df = pd.read_csv(csv_files)
        agent = csv_query_agent(df)

if enable_ts and ts_files:
    if '.csv' in ts_files.name:
        df_t = pd.read_csv(ts_files)
        temp2 = """
        You will be given a video transcript in the form of a dataframe.
        transcript: {transcript}
        Your task is to analyse the transcript and understand what is there in it and answer the user query.
        query: {question}
        Answer the query based on the transcript and your understanding about the transcript.
        Output should only be the answer and nothing else.
        """
        prompt2 = ChatPromptTemplate.from_template(temp2)
        chain5 = prompt2 | llm
    elif '.txt' in ts_files.name:
        # with open(ts_files) as f:
        #     df_t = f.read()
        df_t = ts_files.read().decode('utf-8')
        print(df_t)
        temp2 = """
        You will be given a video transcript in the form of text.
        transcript: {transcript}
        Your task is to analyse the transcript and understand what is there in it and answer the user query.
        query: {question}
        Answer the query based on the transcript and your understanding about the transcript.
        Output should only be the answer and nothing else.
        """
        prompt2 = ChatPromptTemplate.from_template(temp2)
        chain5 = prompt2 | llm

if enable_faq and faq_files:
    faq = pd.read_csv(faq_files)
    examples = ""
    for i in range(faq.shape[0]):
        if str(faq["answer"][i])!='nan':
            examples += "\n\n Question: " + str(faq["question"][i]) + "\n\n Answer: " + str(faq["answer"][i])
    # print(examples)
    temp1 = """
    You will be given a question.
    question: {question}
    Answer the question and give only answer as output using the examples below.
    Example questions: {examples}
    """
    prom = ChatPromptTemplate.from_template(temp1)
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    chain4 = prom | llm

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
elif enable_csv:
    temp = """
    You will be assigned a role and given a company brief and a dataframe from a csv data.
    role: {role}
    company brief: {company_brief}
    dataframe: {dataframe}
    Your task is to generate 10 questions a user can ask you about the company.
    The questions should be related to the company and your role.
    The questions should be such that it can clears user's doubt about the company.
    Amswer based on the dataframe from a csv data as well.
    Give your output as a list of strings.
    Output format: 
    ["","","","","","","","","",""]
    """
    ques_prompt = ChatPromptTemplate.from_template(temp)
    chain3 = ques_prompt | llm
    questions = chain3.invoke({"role": f_role, "company_brief": company_brief, "dataframe": df})
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

    if enable_ytt:
        res = yt_chain.invoke(user_input)
        full_res = res
    elif enable_knowledge_graph and (uploaded_files!=[] or urls!=[] or enable_sitemap):
        if enable_sitemap and url!="":
            sim_url = find_url(sitemap_urls, user_input)
            loader = SeleniumURLLoader([sim_url])
            data = loader.load()
            documents.extend(data)
        rag_chain = create_rag_chain(documents)
        res = rag_chain.invoke(user_input)
        full_res = res
    elif enable_csv and csv_files:
        try:
            res = agent.query(user_input)
        except:
            res = "Answer not found. Try again."
        full_res = res
    else:
        res = chain.invoke(user_input)
        full_res = res.content
    
    ques = suggest_questions(user_input, role, company_brief)
    
    with st.chat_message("assistant"):
        st.markdown(full_res)

    st.write(ques)
    st.session_state.messages.append({"role": "assistant", "content": full_res})
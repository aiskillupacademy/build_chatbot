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

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chatbot_prompt(company_brief):
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=1, max_tokens=2048)
    prompt_template = """
    You are an expert in system prompt generation for LLMs.
    You will be given a company brief.
    company brief: {company_brief}
    Your task is to generate a prompt for an llm to work as a chatbot for this company using the company brief provided.
    Make a detailed prompt telling the chatbot on how to answer user query aligning with the company brief.
    Add details about how a company chatbot should work and converse with the user.
    Try to add points and examples.
    Your output should be the new generated prompt in a string.
    New prompt: 
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    res = chain.invoke(company_brief)
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

def suggest_questions(message):
    suggest_input = """Given the user query, your task is to generate 3 follow up questions that the user can ask further.
        Suggest Related Questions:

        Propose relevant follow-up questions the user might want to ask next.
        Ensure suggestions are connected to the initial query and offer deeper insights or related information.
        Broaden the Topic:

        Include questions that expand the conversation to related topics or broader areas of interest.
        Aim to keep the user engaged and curious.
        Example Structure:

        Initial Response:

        "Thank you for your question about [topic]. Based on your query, here's the information you need: [response]."
        Suggest Related Questions:

        "You might be interested in asking about [related aspect]."
        "Consider exploring [related topic] with questions like [example question]."
        Broaden the Topic:

        "To expand your understanding, you could ask about [broader category or advanced topic]."
        "Another area to consider might be [related area]. You might ask, [example question]."
        Example Dialogue:

        User: "How can I improve my website’s SEO?"

        Chatbot:
        "Thank you for your question about improving your website's SEO. Here are some effective strategies:

        Optimize your content with relevant keywords.
        Ensure your website is mobile-friendly.
        Use descriptive meta tags.
        Build high-quality backlinks.

        You might be interested in asking about:

        How to find the best keywords for your content.
        The impact of mobile-friendliness on SEO rankings.
        Consider exploring these related topics:

        What are the best tools for analyzing SEO performance?
        How does social media influence SEO?
        To expand your understanding, you could ask about:

        Advanced SEO techniques for competitive niches.
        The latest trends in SEO for the upcoming year.

        Query: {query}

        generate only 3 questions.
        Output should be a python list of strings. Only give the python list as output, nothing else.\n
        Output format:\n

        ["","",""]

        """
    suggest_prompt = ChatPromptTemplate.from_template(suggest_input)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
    chain_sug = suggest_prompt | llm
    res_suggest = chain_sug.invoke({"query":message})
    response = res_suggest.content
    start = response.find('[')
    end = response.find(']')
    questions = response[start:end+1]
    ques = eval(questions)
    return ques

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

if company_details!= "":
    company_brief = company_details
else:
    # with open("companybrief.txt") as f:
    #     company_brief = f.read()
    company_brief = ""
prompt = chatbot_prompt(company_brief)
sys_prompt = prompt.content
sys_prompt = role + sys_prompt + "\n\n Don't answer anything outside of the context or details you have. \n\n Don't give examples. \n\n Try to use bullet points sometimes when necessary. \n\n Explain points you make with your knowledge. \n\n If you don't know the answer then just say 'I don't know.'."
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
    if uploaded_files:
        text = extraxt_doc_text(uploaded_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 10)
        # print(text)
        docs = text_splitter.split_documents(text)
        embedding = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embedding)
        retriever = db.as_retriever(search_kwargs={"k":2, "score_threshold": 0.7})
        template = """Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
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
        # print(docs)

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
questions = chain1.invoke({"role": role, "company_brief": company_brief})
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
    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    if enable_knowledge_graph and uploaded_files:
        res = rag_chain.invoke(user_input)
        full_res = res
    else:
        res = chain.invoke(user_input)
        full_res = res.content
    
    ques = suggest_questions(user_input)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(full_res)

    st.write(ques)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_res})


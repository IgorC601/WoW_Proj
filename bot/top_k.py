import streamlit as st
import pandas as pd
from tqdm.auto import tqdm
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import hmac
import os
from dotenv import load_dotenv


###################################################################
#load_dotenv()

#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

embed = OpenAIEmbeddings(
   model = 'text-embedding-ada-002',
   openai_api_key= OPENAI_API_KEY
)

###################################################################

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            # Wrap the text in a div with a custom background color
            colored_text = f'<div style="background-color: #08346c; color: #FFFFFF; padding: 10px; border-radius: 5px;">{self.text}</div>'
            display_function(colored_text, unsafe_allow_html=True)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

###################################################################

#PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_KEY = st.secrets["OPENAI_API_KEY"]


@st.cache_resource
def connect_to_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index('wordsofwisdom')

###################################################################


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


def run_chatbot_app():
    # Initialize Pinecone
    idx = connect_to_pinecone()
        
    # prompt template
    template = """You are a helpful chatbot assistant who is having a conversation with a human.
                You know all the details from the book BERND KOLB WORDS OF WISDOM SCRIPTUM IRGENDETWAS STIMMT NICHT.
                You should answer the question or questions posed in German, based ONLY on the context provided.
                DON'T write about anything that isn't present in the context.
                
                Context:
                ```{context}```

                Human: 
                ```{human_input}```
                """

    prompt = PromptTemplate(
        input_variables=["human_input", "context"],
        template=template
    )

    # Streamlit app
    st.title('Words of Wisdom Test')

    query = st.text_input("Stelle hier deine Frage", key='input')

    if st.button('jetzt finden') or 'input' in st.session_state:
        if 'input' in st.session_state:
            query = st.session_state.input
        else:
            query = st.session_state.input = ''

        if query:
            query = query.lower()
            # Chatbot
            st.subheader('Antwort:')
            
            # llm setup
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box, display_method='write')
            llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-1106",
                    callbacks=[stream_handler], streaming=True
            )

            # Fetch results with minimum score of relevancy
            vectorstore = Pinecone(idx, embed.embed_query, 'content')
            results = vectorstore.similarity_search_with_score(query, k=3)

            # Assuming 'results' is your list of tuples as provided
            filtered_list = [doc_score_tuple for doc_score_tuple in results if doc_score_tuple[1] >= 0.85]

            combined_docs_chain = []
            for doc_score_tuple in filtered_list:
                doc = Document(doc_score_tuple[0].page_content, doc_score_tuple[0].metadata)  
                combined_docs_chain.append(doc)

            # Create the chain
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain.run(input_documents=combined_docs_chain, question=query, human_input=query)

            # Display reference expanders
            for doc in combined_docs_chain:
                content = doc.page_content
                title = doc.metadata['title']
                paragraph = doc.metadata['paragraph']
                with st.expander(f"{title} - {paragraph}"):
                    st.markdown(content)


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

run_chatbot_app()
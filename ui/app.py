import sys
import os

# Allow importing backend modules
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
import faiss
import json

from pathlib import Path

from backend.embed_index import build_index
from backend.load_models import load_models
from backend.query_engine import ask_question


#Page Configuration
st.set_page_config(
    page_title="AI Document Assistant",
    layout="wide"
)



# Custom CSS
st.markdown(
    """
    <style>

    .stApp {
        background-color: #111b21;
        color: white;
    }


    h1 {
        text-align: center;
        color: #e9edef;
        font-size: 32px;
        margin-bottom: 20px;
    }


    .chat-container {
        padding-bottom: 100px;
    }


    .user-msg {

        background-color:  #9400D3;
        color: white;

        padding: 14px 18px;

        border-radius:
        18px 18px 4px 18px;

        margin: 15px 0;

        margin-left: 30%;

        box-shadow:
        0px 3px 8px rgba(0,0,0,0.35);

        font-size: 16px;

    }



    .ai-msg {

        background-color: #202c33;

        color: #e9edef;

        padding: 14px 18px;

        border-radius:
        18px 18px 18px 4px;

        margin: 15px 0;

        margin-right: 30%;

        box-shadow:
        0px 3px 8px rgba(0,0,0,0.35);

        font-size: 16px;

    }



    .source-text {

        color: #aebac1;

        font-size: 13px;

    }



    div[data-testid="stChatInput"] {

        width: 100%;

    }


   

    div[data-testid="stChatInput"] textarea {


        background-color: #2a3942;


        color: white;


        border-radius: 25px;


        padding: 12px 18px;


        font-size: 17px;


        border: 1px solid #9400D3;


        box-shadow:
        0px 0px 8px rgba(148,0,211,0.4);


    }



    div[data-testid="stChatInput"] textarea:focus {


        border: 2px solid #c084fc !important;


        outline: none !important;


        box-shadow:
        0px 0px 15px rgba(192,132,252,0.7);


    }


  

    div[data-testid="stChatInput"] textarea::placeholder {


        color: #c4b5fd;


    }



    div[data-testid="stChatInput"] button {


        background-color: #9400D3 !important;


        color: white !important;


        border-radius: 50%;


        border: none;


    }


   

    div[data-testid="stChatInput"] button:hover {


        background-color: #b026ff !important;


    }



    #MainMenu {

        visibility: hidden;

    }


    footer {

        visibility: hidden;

    }


    </style>

    """,

    unsafe_allow_html=True
)


#Loading Models
@st.cache_resource
def get_models():

    return load_models()


embedding_model, reranker = get_models()



#Load Insex and the metadata
@st.cache_resource
def load_index():

    index = faiss.read_index(
        "index/index.faiss"
    )


    with open(
        "index/metadata.json",
        "r"
    ) as f:

        metadata = json.load(f)


    return index, metadata



index, metadata = load_index()


#GUI
st.title("DocuMate")

#Library Panel
st.sidebar.header("My Library")


DATA_FOLDER = Path("data")

# Create data folder if missing
DATA_FOLDER.mkdir(exist_ok=True)



#Document uploading
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        save_path = DATA_FOLDER / uploaded_file.name

        with open(save_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )


    st.sidebar.success(
        "Files uploaded successfully."
    )



# Update Library
if st.sidebar.button(
    "Update Library"
):

    with st.spinner(
        "Building index..."
    ):

        pdfs, chunks = build_index()


    # Clear old FAISS cache
    st.cache_resource.clear()


    st.sidebar.success(
        f"Indexed {pdfs} PDFs ({chunks} chunks)"
    )


    st.rerun()



# Show documents in the Library
st.sidebar.markdown(
    "### Current Documents"
)


pdf_files = list(
    DATA_FOLDER.glob("*.pdf")
)


if pdf_files:

    for pdf in pdf_files:

        st.sidebar.write(
            pdf.name
        )


else:

    st.sidebar.info(
        "No documents uploaded."
    )



# Delete documents
if pdf_files:

    st.sidebar.markdown(
        "### Delete Document"
    )


    selected_pdf = st.sidebar.selectbox(

        "Select PDF",

        [
            pdf.name
            for pdf in pdf_files
        ]

    )


    if st.sidebar.button(
        "Delete Selected PDF"
    ):


        file_path = DATA_FOLDER / selected_pdf


        if file_path.exists():

            file_path.unlink()


            st.sidebar.success(
                f"{selected_pdf} deleted"
            )


            with st.spinner(
                "Updating knowledge base..."
            ):

                pdfs, chunks = build_index()



            st.cache_resource.clear()


            st.sidebar.success(

                f"Updated: {pdfs} PDFs ({chunks} chunks)"

            )


            st.rerun()



#Chat memeory
if "chat" not in st.session_state:

    st.session_state.chat = []



#User Input
user_input = st.chat_input(
    "Type your message..."
)



if user_input:


    result = ask_question(

        user_input,

        index,

        metadata,

        embedding_model,

        reranker

    )


    st.session_state.chat.append(

        (

            user_input,

            result["answer"],

            result["sources"]

        )

    )



#Chat Display
st.markdown(

    '<div class="chat-container">',

    unsafe_allow_html=True

)



for q, a, s in st.session_state.chat:



    # User message

    st.markdown(

        f"""

        <div class="user-msg">

        <b>You</b><br><br>

        {q}

        </div>

        """,

        unsafe_allow_html=True

    )



    # AI message

    st.markdown(

        f"""

        <div class="ai-msg">

        <b>DocuMate</b><br><br>

        {a}

        <br><br>

        <span class="source-text">

        Sources: {", ".join(s)}

        </span>


        </div>

        """,

        unsafe_allow_html=True

    )



st.markdown(

    '</div>',

    unsafe_allow_html=True

)
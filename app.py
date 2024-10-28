# app.py 

import os
import time
import base64
import streamlit as st 
from streamlit import session_state

from vectors import EmbeddingsManager
from chatbot import ChatBotManager

# initialize session states
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []


def displayPDF(file):
    # reading the file

    pdf_base64 = base64.b64encode(file.read()).decode('utf-8')

    # embedd the file in an IFRAME
    if pdf_base64:

        st.markdown('### PDF Preview')
        st.markdown(f"**Filename:** {uploaded_file.name}")
        st.markdown(f"**File Size:** {uploaded_file.size} bytes")

        pdf_display = f'<iframe src="data:application/pdf;base64, {pdf_base64}" width="100%" height="800" type="application/pdf"></iframe>'

        # display the file
        st.markdown(pdf_display, unsafe_allow_html=True)

# page configuration
st.set_page_config(
    page_title="Personal RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:

    st.markdown("### 📚 Your Personal Document Assistant")
    st.markdown("---")

    menu = ["🏠 Home","🤖 Chatbot"]
    choice = st.selectbox("",menu)

if choice == "🏠 Home":
    st.title("📄 Personal Document - Assistant")

    st.markdown("""
        Welcome to **Your very own Personal Document - RAG System**! 🚀

        **Technologies Used:**
        - **Large Language Model :** Llama 3.2b
        - **Embeddings :** BGE Embeddings
        - **Vector Database :** Qdrant running **locally** within a **Docker Container**.)

        **Features:**

        - **Personal**: All technologies runs **locally** on your own system...
        - **Upload Documents**: Easily upload your PDF documents.
        - **Free Embeddings**: Run Embeddings on your document with FREE Embedding model like BGE.
        - **Chat**: Interact with your documents via our intelligent chatbot. Ask questions, summarize documents and much more...

        Your very own personal Document Assistant System! 😊
        """)


elif choice=="🤖 Chatbot":

    st.title("🤖 Personal Chatbot using Llama🦙)")

    with st.expander("📂  Upload Document"):
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file is not None:
            st.success(" Your file is uploaded successfully!")

            # display the file

            displayPDF(uploaded_file)

            # save the file locally in a temporary location
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # store the temp_pdf_path in session state
            st.session_state["temp_pdf_path"] = temp_pdf_path

    with st.expander("🧠 Create Embeddings"):
        create_embeddings = st.checkbox("✅ Create Embedding")

        if create_embeddings:
            if st.session_state["temp_pdf_path"] is None:
                st.warning("Please upload a PDF file first!")

            try:
                # initialize the embedding class

                embedding_manager = EmbeddingsManager(
                    model_name= os.path.join(os.getcwd(), "bge-small-en"),
      #              model_name=os.path.join(, "bge-small-en"),
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    qdrant_url="http://localhost:6333",
                    collection_name="vector_db"
                )

                with st.spinner("🔄 System is creating Embeddings..."):
                    # create embedding

                    result = embedding_manager.create_embeddings(
                        st.session_state["temp_pdf_path"]
                    )
                    time.sleep(1)
                st.success(result)

                # Initialize the ChatBotManager after embeddings are created

                if st.session_state['chatbot_manager'] is None:
                    st.session_state['chatbot_manager'] = ChatBotManager(
                        model_name=os.path.join(os.getcwd(), "bge-small-en"),
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        llm_model="llama3.2:1b",
                        llm_temperature=0.7,
                        qdrant_url="http://localhost:6333",
                        collection_name="vector_db"
                    )

            except FileNotFoundError as fnf_error:
                st.error(fnf_error)
            except ValueError as val_error:
                st.error(val_error)
            except ConnectionError as conn_error:
                st.error(conn_error)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")



    with st.expander("💬 Chat with Document"):
        st.markdown('#### chatbot...')

        # check if chatbot_manager session is active > i.e. did the user upload the PDF file?
        if st.session_state['chatbot_manager'] is None:
            st.info("Please upload a PDF, then create an embeddings, and then start chatting with your document...")
        else:
            # use 'messages' session to display existing messages
            # role = content
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            # User Input
            if user_input :=st.chat_input("Type your message here..."):
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)

                    except Exception as e:
                        answer = f"An error occurred while processing your request: {e}"

                # display chatbot message
                st.chat_message("assistant").markdown(answer)
                st.session_state['messages'].append({"role": "assistant", "content": answer})



st.markdown("---")
st.markdown("© 2024 Personal Document Assistant App.🛡️")
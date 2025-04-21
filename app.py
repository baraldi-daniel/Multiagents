import streamlit as st
import random
import time
from main import invoke_app
import io
import os
from langchain_ibm import ChatWatsonx
from streamlit_extras.stylable_container import stylable_container

def response_generator(prompt,id):
    try:
        response = str(invoke_app(prompt,id)["messages"][-1].content).split(" additional_kwargs")[0].split("content='")[1].split("'")[0].replace('"','')
    except:
        print("HHHHHHHHHHHHHHHHHERRRRRRRRRRRRRRRRRRREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        response = str(invoke_app(prompt,id)["messages"][-1].content).split(" additional_kwargs")[0].replace('"','')
    #response = str(invoke_app(prompt,id)["messages"][-1]).split(" additional_kwargs")[0].split("content='")[1].split("'")[0]
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Agent Chat")



if "id" not in st.session_state:
    st.session_state.id=st.text_input("ID: ")
    submit=st.button("Iniciar Conversa")
else:
    submit=True

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Arquivo Upload", accept_multiple_files=False
        )
        if uploaded_file:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)

 

            if ".mp4" in str(uploaded_file.name):

                with io.open(os.getcwd()+"/tmp/file.mp4", 'wb') as f:
                    f.write(bytes_data)
                    f.close()

            elif ".wav" in str(uploaded_file.name):

                with io.open(os.getcwd()+"/tmp/file.wav", 'wb') as f:
                    f.write(bytes_data)
                    f.close()



            elif (((".jpeg" or ".png") or ".jpg") in str(uploaded_file.name)):
                with io.open(os.getcwd()+"/tmp/file.jpeg", 'wb') as f:
                    f.write(bytes_data)
                    f.close()



        with io.open(os.getcwd()+"/tmp/file_created.wav", 'rb') as audio:
            st.download_button('Download Arquivo', file_name="audio_criado.wav", data=audio)



    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    




    if prompt := st.chat_input("Oi, tudo bem?"):

        st.session_state.messages.append({"role": "user", "content": prompt})


        with st.chat_message("user"):
            st.markdown(prompt)


        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt,st.session_state.id))

        st.session_state.messages.append({"role": "assistant", "content": response})

    

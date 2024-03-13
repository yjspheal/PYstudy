#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[10]:


import streamlit as st

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import (StuffDocumentsChain, LLMChain,
                              ConversationalRetrievalChain)
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler

from openai import OpenAI
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain.schema import LLMResult
import requests
import tempfile
from PIL import Image
from io import BytesIO

api_key = st.secrets["default"]["api_key"]

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        super().__init__()
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.complete_response = None  # 완성된 답변을 저장하는 변수 추가

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # LLM 처리가 끝났을 때 호출됩니다.
        
        # LLMResult 인스턴스에서 'generations' 키의 값에 접근
        generations = response.dict().get('generations', [])

        # 'generations' 내부의 첫 번째 요소에서 첫 번째 딕셔너리를 가져옴
        if generations and len(generations) > 0 and len(generations[0]) > 0:
            first_generation = generations[0][0]  # 첫 번째 'generation'의 첫 번째 항목

            # 'message' 키의 'content' 값에 접근
            content = first_generation.get('message', {}).get('content', '')
        
        self.complete_response = content  # 최종 텍스트를 저장합니다.
        
# Google Drive에서 직접 링크로 변환된 이미지 경로
image_path = "https://static.wikia.nocookie.net/pokemon/images/f/f3/%EA%B3%A0%EB%9D%BC%ED%8C%8C%EB%8D%95_%EA%B3%B5%EC%8B%9D_%EC%9D%BC%EB%9F%AC%EC%8A%A4%ED%8A%B8.png/revision/latest/scale-to-width-down/1000?cb=20170405011541&path-prefix=ko"
# 이미지 데이터를 가져옴
response = requests.get(image_path)

# BytesIO 객체를 사용하여 바이너리 스트림을 생성
image_stream = BytesIO(response.content)

# PIL로 이미지 열기
image = Image.open(image_stream)

img = image.resize((300, 300))

# Streamlit 앱에 사진 추가
st.image(img, caption='파덕파덕')


# Set up the title and Streamlit session state for managing chat history and model selection
st.title("인사이트 최고")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-0125-preview"

if "messages" not in st.session_state:
    st.session_state['messages'] = []
    
# Google Drive에서 PDF 파일의 직접 다운로드 링크 생성
file_id = '1drqTowu1A0YtRWmijrQE2scXhOGeEp38'
pdf_url = f'https://drive.google.com/uc?id={file_id}&export=download'

response = requests.get(pdf_url)

if response.status_code == 200:
    # BytesIO 객체를 사용하여 PDF 파일로부터 데이터를 로드
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    # Extract data from PDF
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    # Generate document vectors
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectors = FAISS.from_documents(data, embeddings)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    
    st.chat_message("user").write('안녕!')
    st.chat_message("assistant").write('안녕하세요! 스터디에 대해 질문이 있나요?')
    
    # Display chat history
    for message_type, message_text in st.session_state['messages']:
        if message_type == "user":
            st.chat_message("user").write(message_text)
        else:  # 'assistant'
            st.chat_message("assistant").write(message_text)


    # Process new chat inputs
    if prompt := st.chat_input("질문을 적어주세요."):
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
        
            stream_handler = StreamHandler(st.empty(), display_method='markdown')

            # Create conversational retrieval chain
            qa = RetrievalQA.from_llm(llm=ChatOpenAI(
                streaming=True,
                callbacks=[stream_handler],
                temperature=0.0,
                model_name='gpt-4-0125-preview'),
                retriever=vectors.as_retriever())

            qa(prompt)
            
            end = True
            
            # StreamHandler에서 완성된 답변을 가져와 st.session_state에 저장
            if end:
                # 사용자 메시지 저장
                st.session_state['messages'].append(("user", prompt))
                # 어시스턴트 메시지 저장
                st.session_state['messages'].append(("assistant", stream_handler.complete_response))
   


# In[ ]:





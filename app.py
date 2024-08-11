import streamlit as st
import pandas as pd
import uuid
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# (이전 코드는 동일하게 유지)

# CSS for chat layout
def get_chat_css():
    return """
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        padding: 10px;
    }
    .message {
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 75%;
    }
    .user-message {
        align-self: flex-end;
        background-color: #DCF8C6;
    }
    .coach-message {
        align-self: flex-start;
        background-color: #E5E5EA;
    }
    </style>
    """

# Function to create HTML for chat messages
def get_chat_html(conversation):
    html = get_chat_css()
    html += '<div class="chat-container">'
    for i, message in enumerate(conversation):
        if i % 2 == 0:
            html += f'<div class="message coach-message">{message}</div>'
        else:
            html += f'<div class="message user-message">{message}</div>'
    html += '</div>'
    return html

# 메인 앱 로직
def main():
    st.title("AI 코칭 시스템")

    # 세션 상태 초기화
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'Trust'
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # 첫 질문 생성
    if not st.session_state.conversation:
        with st.spinner("코치가 첫 질문을 준비하고 있습니다..."):
            first_question = generate_coach_response([], st.session_state.current_stage, 0)
            st.session_state.conversation.append(first_question)

    # 대화 기록 표시
    st.subheader("대화 내용")
    st.markdown(get_chat_html(st.session_state.conversation), unsafe_allow_html=True)

    # 사용자 입력
    user_input = st.text_input("메시지를 입력하세요...", key="user_input")

    # 버튼 컨테이너 생성
    col1, col2 = st.columns(2)
    
    # 메시지 제출 버튼
    with col1:
        if st.button("전송", key="send_button", use_container_width=True):
            if user_input:
                st.session_state.conversation.append(user_input)
                
                with st.spinner("코치가 응답을 생성하고 있습니다..."):
                    coach_response = generate_coach_response(st.session_state.conversation, st.session_state.current_stage, st.session_state.question_count)
                    st.session_state.conversation.append(coach_response)
                
                st.session_state.question_count += 1
                if st.session_state.question_count >= 3:
                    stages = ['Trust', 'Explore', 'Aspire', 'Create', 'Harvest', 'Empower&Reflect']
                    current_stage_index = stages.index(st.session_state.current_stage)
                    if current_stage_index < len(stages) - 1:
                        st.session_state.current_stage = stages[current_stage_index + 1]
                        st.session_state.question_count = 0
                
                save_conversation(st.session_state.session_id, st.session_state.conversation)
                st.session_state.user_input = ""
                st.experimental_rerun()

    # 대화 초기화 버튼
    with col2:
        if st.button("대화 초기화", key="reset_button", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.current_stage = 'Trust'
            st.session_state.question_count = 0
            st.session_state.user_input = ""
            st.experimental_rerun()

# 메인 앱 실행
if __name__ == "__main__":
    main()

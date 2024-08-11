import streamlit as st
import pandas as pd
import uuid
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 코칭 시스템", layout="wide")

# 여기에 OpenAI, Pinecone, SentenceTransformer 초기화 코드 유지

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
        st.session_state.current_stage = 'Trust'
        st.session_state.question_count = 0
        st.session_state.conversation = []

    # 첫 질문 생성
    if not st.session_state.conversation:
        try:
            with st.spinner("코치가 첫 질문을 준비하고 있습니다..."):
                first_question = generate_coach_response([], st.session_state.current_stage, 0)
                st.session_state.conversation.append(first_question)
        except Exception as e:
            st.error(f"첫 질문 생성 중 오류 발생: {str(e)}")

    # 대화 기록 표시
    st.markdown(get_chat_css(), unsafe_allow_html=True)
    for i, message in enumerate(st.session_state.conversation):
        if i % 2 == 0:
            st.markdown(f'<div class="message coach-message">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message user-message">{message}</div>', unsafe_allow_html=True)

    # 사용자 입력
    user_input = st.text_input("메시지를 입력하세요...", key="user_input")

    # 메시지 제출 버튼
    if st.button("전송", key="send_button"):
        if user_input:
            st.session_state.conversation.append(user_input)
            
            try:
                with st.spinner("코치가 응답을 생성하고 있습니다..."):
                    coach_response = generate_coach_response(
                        st.session_state.conversation,
                        st.session_state.current_stage,
                        st.session_state.question_count
                    )
                    st.session_state.conversation.append(coach_response)
                
                st.session_state.question_count += 1
                if st.session_state.question_count >= 3:
                    stages = ['Trust', 'Explore', 'Aspire', 'Create', 'Harvest', 'Empower&Reflect']
                    current_stage_index = stages.index(st.session_state.current_stage)
                    if current_stage_index < len(stages) - 1:
                        st.session_state.current_stage = stages[current_stage_index + 1]
                        st.session_state.question_count = 0
                
                save_conversation(st.session_state.session_id, st.session_state.conversation)
            except Exception as e:
                st.error(f"응답 생성 중 오류 발생: {str(e)}")
            
            st.experimental_rerun()

    # 대화 초기화 버튼
    if st.button("대화 초기화", key="reset_button"):
        st.session_state.conversation = []
        st.session_state.current_stage = 'Trust'
        st.session_state.question_count = 0
        st.experimental_rerun()

# 메인 앱 실행
if __name__ == "__main__":
    main()

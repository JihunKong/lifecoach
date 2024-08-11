import streamlit as st
import pandas as pd
import uuid
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# (이전 코드와 동일한 부분은 생략)

# 메인 앱 로직
def main():
    st.title("AI 코칭 시스템")

    # 세션 상태 초기화
    initialize_session_state()
    
    # 첫 질문 생성
    if not st.session_state.conversation:
        with st.spinner("코치가 첫 질문을 준비하고 있습니다..."):
            first_question = generate_coach_response([], st.session_state.current_stage, 0)
            st.session_state.conversation.append(first_question)

    # 대화 기록 표시
    st.subheader("대화 내용")
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.conversation):
            if i % 2 == 0:
                st.info(f"코치: {message}")
            else:
                st.success(f"나: {message}")

    # 사용자 입력
    user_input = st.text_input("메시지를 입력하세요...", key="user_input")

    # 버튼 컨테이너 생성
    col1, col2 = st.columns(2)
    
    # 메시지 제출 버튼
    with col1:
        if st.button("전송", key="send_button", use_container_width=True):
            if user_input:
                process_user_input(user_input)
                st.session_state.user_input = ""  # 입력 필드 초기화

    # 대화 초기화 버튼
    with col2:
        if st.button("대화 초기화", key="reset_button", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.current_stage = 'Trust'
            st.session_state.question_count = 0
            st.session_state.user_input = ""

def process_user_input(user_input):
    st.session_state.conversation.append(user_input)
    
    with st.spinner("코치가 응답을 생성하고 있습니다..."):
        # 코치 응답 생성
        coach_response = generate_coach_response(st.session_state.conversation, st.session_state.current_stage, st.session_state.question_count)
        st.session_state.conversation.append(coach_response)
    
    # 질문 카운트 증가 및 단계 관리
    st.session_state.question_count += 1
    if st.session_state.question_count >= 3:
        stages = ['Trust', 'Explore', 'Aspire', 'Create', 'Harvest', 'Empower&Reflect']
        current_stage_index = stages.index(st.session_state.current_stage)
        if current_stage_index < len(stages) - 1:
            st.session_state.current_stage = stages[current_stage_index + 1]
            st.session_state.question_count = 0
    
    # 대화 저장
    save_conversation(st.session_state.session_id, st.session_state.conversation)

# 코칭 데이터 로드
coach_df = load_coach_data()

# 메인 앱 실행
if __name__ == "__main__":
    main()

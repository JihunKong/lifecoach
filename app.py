import streamlit as st
import pandas as pd
import uuid
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 디버깅 함수
def debug_print(message):
    st.sidebar.write(f"Debug: {message}")

# Streamlit 페이지 설정
st.set_page_config(page_title="GPT 기반 TEACHer 코칭 시스템", layout="wide")

debug_print("페이지 설정 완료")

# OpenAI 클라이언트 초기화
try:
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    debug_print("OpenAI 클라이언트 초기화 성공")
except Exception as e:
    debug_print(f"OpenAI 클라이언트 초기화 실패: {str(e)}")

# Pinecone 초기화
try:
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    debug_print("Pinecone 초기화 성공")
except Exception as e:
    debug_print(f"Pinecone 초기화 실패: {str(e)}")

index_name = "coach"

# Sentence Transformer 모델 로드
try:
    model = SentenceTransformer('all-mpnet-base-v2')
    test_vector = model.encode("Test sentence")
    vector_dimension = len(test_vector)
    debug_print(f"Sentence Transformer 모델 로드 성공. 벡터 차원: {vector_dimension}")
except Exception as e:
    debug_print(f"Sentence Transformer 모델 로드 실패: {str(e)}")

# Pinecone 인덱스 연결
try:
    index = pc.Index(index_name)
    index_stats = index.describe_index_stats()
    index_dimension = index_stats['dimension']
    debug_print(f"Pinecone 인덱스 연결 성공. 인덱스 차원: {index_dimension}")
except Exception as e:
    debug_print(f"Pinecone 인덱스 연결 실패: {str(e)}")

# 벡터 생성 함수 (차원 조정)
def create_vector(text):
    vector = model.encode(text).tolist()
    if len(vector) < index_dimension:
        # 벡터를 두 번 반복하여 차원을 맞춤
        vector = vector * 2
        # 필요한 경우 남은 차원을 0으로 채움
        vector = vector[:index_dimension]
    elif len(vector) > index_dimension:
        vector = vector[:index_dimension]
    return vector

debug_print(f"벡터 차원 조정 함수 생성 완료. 조정 후 차원: {len(create_vector('Test'))}")

# 나머지 코드는 이전과 동일...

# 메인 앱 로직
def main():
    st.title("GPT 기반 TEACHer 코칭 시스템")

    # 세션 상태 초기화
    initialize_session_state()
    
    # 첫 질문 생성
    if not st.session_state.conversation:
        first_question = generate_coach_response([], st.session_state.current_stage, 0)
        st.session_state.conversation.append(first_question)
        debug_print("첫 질문 생성 완료")

    # 대화 기록 표시
    for i, message in enumerate(st.session_state.conversation):
        if i % 2 == 0:
            st.text_area("Coach:", value=message, height=200, key=f"msg_{i}", disabled=True)
        else:
            st.text_area("You:", value=message, height=100, key=f"msg_{i}", disabled=True)

    # 사용자 입력
    user_input = st.text_input("메시지를 입력하세요...", key="user_input", value=st.session_state.user_input)

    # 메시지 제출 버튼
    if st.button("전송"):
        if user_input:
            st.session_state.conversation.append(user_input)
            debug_print("사용자 입력 추가")
            
            # 코치 응답 생성
            coach_response = generate_coach_response(st.session_state.conversation, st.session_state.current_stage, st.session_state.question_count)
            st.session_state.conversation.append(coach_response)
            debug_print("코치 응답 생성 및 추가")
            
            # 질문 카운트 증가 및 단계 관리
            st.session_state.question_count += 1
            if st.session_state.question_count >= 3:
                stages = ['Trust', 'Explore', 'Aspire', 'Create', 'Harvest', 'Empower&Reflect']
                current_stage_index = stages.index(st.session_state.current_stage)
                if current_stage_index < len(stages) - 1:
                    st.session_state.current_stage = stages[current_stage_index + 1]
                    st.session_state.question_count = 0
                    debug_print(f"다음 단계로 이동: {st.session_state.current_stage}")
            
            # 대화 저장
            save_conversation(st.session_state.session_id, st.session_state.conversation)
            
            # 입력 필드 초기화
            st.session_state.user_input = ""
            
            # 페이지 새로고침
            st.experimental_rerun()

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        reset_conversation()
        st.experimental_rerun()

    # 현재 세션 정보 표시 (개발용, 실제 사용 시 숨김 처리 가능)
    st.sidebar.write(f"세션 ID: {st.session_state.session_id}")
    st.sidebar.write(f"현재 단계: {st.session_state.current_stage}")
    st.sidebar.write(f"질문 수: {st.session_state.question_count}")

# 코칭 데이터 로드
coach_df = load_coach_data()

# 메인 앱 실행
if __name__ == "__main__":
    main()

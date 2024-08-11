import streamlit as st
import pandas as pd
import uuid
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 디버깅 함수 (개발자 모드에서만 표시)
def debug_print(message):
    if st.session_state.get('dev_mode', False):
        st.sidebar.write(f"Debug: {message}")

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 코칭 시스템", layout="wide")

# OpenAI 클라이언트 초기화
@st.cache_resource
def init_openai():
    try:
        return OpenAI(api_key=st.secrets["openai"]["api_key"])
    except Exception as e:
        st.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        return None

client = init_openai()

# Pinecone 초기화
@st.cache_resource
def init_pinecone():
    try:
        pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
        index = pc.Index("coach")
        return pc, index
    except Exception as e:
        st.error(f"Pinecone 초기화 실패: {str(e)}")
        return None, None

pc, index = init_pinecone()

# Sentence Transformer 모델 로드
@st.cache_resource
def load_sentence_transformer():
    try:
        return SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        st.error(f"Sentence Transformer 모델 로드 실패: {str(e)}")
        return None

model = load_sentence_transformer()

# 벡터 생성 함수 (차원 조정)
def create_vector(text):
    vector = model.encode(text).tolist()
    index_dimension = 1536  # Pinecone 인덱스의 차원
    if len(vector) < index_dimension:
        vector = vector * (index_dimension // len(vector) + 1)  # 벡터를 반복하여 차원을 맞춤
    return vector[:index_dimension]  # 정확히 index_dimension 길이로 자름

# 코칭 데이터 로드
@st.cache_data
def load_coach_data():
    try:
        return pd.read_excel('coach.xlsx')
    except Exception as e:
        st.error(f"코칭 데이터 로드 실패: {str(e)}")
        return pd.DataFrame()

# GPT를 사용한 코칭 대화 생성 함수
def generate_coach_response(conversation, current_stage, question_count):
    stage_questions = coach_df[coach_df['step'].str.contains(current_stage, case=False, na=False)]
    available_questions = stage_questions.iloc[:, 1:].values.flatten().tolist()
    available_questions = [q for q in available_questions if pd.notnull(q)]
    
    recent_conversation = " ".join(conversation[-5:])
    query_vector = create_vector(recent_conversation)
    try:
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        similar_conversations = [item['metadata']['conversation'] for item in results['matches']]
    except Exception as e:
        debug_print(f"유사한 대화 검색 실패: {str(e)}")
        similar_conversations = []
    
    prompt = f"""You are an empathetic life coach using the TEACHer model. 
    Current stage: {current_stage}
    Question count: {question_count}
    Previous conversation: {conversation[-5:] if len(conversation) > 5 else conversation}
    Similar past conversations: {similar_conversations}
    
    Based on the user's responses and similar past conversations, generate a natural, empathetic response and a follow-up question.
    Choose from or create a question similar to these for the current stage:
    {available_questions}
    
    Your response should be in Korean and follow this format:
    [Empathetic response]
    
    [Follow-up question]"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."

# 세션 상태 초기화 함수
def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'Trust'
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'dev_mode' not in st.session_state:
        st.session_state.dev_mode = False

# 대화 저장 함수
def save_conversation(session_id, conversation):
    conversation_text = " ".join(conversation)
    vector = create_vector(conversation_text)
    try:
        index.upsert(vectors=[(session_id, vector, {"conversation": conversation})])
    except Exception as e:
        debug_print(f"대화 저장 실패: {str(e)}")

# 메인 앱 로직
def main():
    st.title("AI 코칭 시스템")

    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바에 개발자 모드 토글 추가
    st.sidebar.title("설정")
    st.session_state.dev_mode = st.sidebar.checkbox("개발자 모드", value=st.session_state.dev_mode)
    
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

    # 메시지 제출 버튼
    if st.button("전송", key="send_button"):
        if user_input:
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
            
            # 입력 필드 초기화 및 페이지 새로고침
            st.rerun()

    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()

    # 개발자 모드에서만 세션 정보 표시
    if st.session_state.dev_mode:
        st.sidebar.subheader("세션 정보")
        st.sidebar.write(f"세션 ID: {st.session_state.session_id}")
        st.sidebar.write(f"현재 단계: {st.session_state.current_stage}")
        st.sidebar.write(f"질문 수: {st.session_state.question_count}")

# 코칭 데이터 로드
coach_df = load_coach_data()

# 메인 앱 실행
if __name__ == "__main__":
    main()

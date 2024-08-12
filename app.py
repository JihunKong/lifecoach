import streamlit as st
import pandas as pd
import uuid
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import time

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 코칭 시스템", layout="wide")

# OpenAI 클라이언트 초기화
@st.cache_resource
def init_openai():
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
        return openai
    except Exception as e:
        st.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
        raise

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
        raise

pc, index = init_pinecone()

# Sentence Transformer 모델 로드
@st.cache_resource
def load_sentence_transformer():
    try:
        return SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        st.error(f"Sentence Transformer 모델 로드 실패: {str(e)}")
        raise

model = load_sentence_transformer()

# 벡터 생성 함수 (차원 조정)
def create_vector(text):
    vector = model.encode(text).tolist()
    index_dimension = 1536  # Pinecone 인덱스의 차원
    if len(vector) < index_dimension:
        vector = vector * (index_dimension // len(vector) + 1)
    return vector[:index_dimension]

# 코칭 데이터 로드
@st.cache_data
def load_coach_data():
    try:
        return pd.read_excel('coach.xlsx')
    except Exception as e:
        st.error(f"코칭 데이터 로드 실패: {str(e)}")
        return pd.DataFrame()

coach_df = load_coach_data()

# GPT를 사용한 코칭 대화 생성 함수
def generate_coach_response(conversation, current_stage, question_count, session_id):
    try:
        stage_questions = coach_df[coach_df['step'].str.contains(current_stage, case=False, na=False)]
        available_questions = stage_questions.iloc[:, 1:].values.flatten().tolist()
        available_questions = [q for q in available_questions if pd.notnull(q)]
        
        recent_conversation = " ".join(conversation[-5:])
        query_vector = create_vector(recent_conversation)
        
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        similar_conversations = [item['metadata']['conversation'] for item in results['matches'] if item['metadata']['session_id'] == session_id]
        
        prompt = f"""You are an empathetic life coach using the TEACHer model. 
        Current stage: {current_stage}
        Question count: {question_count}
        Previous conversation: {conversation[-5:] if len(conversation) > 5 else conversation}
        Similar past conversations: {similar_conversations}
        
        Based on the user's responses and similar past conversations, generate a natural, empathetic response.
        Then, ask a single follow-up question related to the current stage.
        Choose from or create a question similar to these for the current stage:
        {available_questions}
        
        Your response should be in Korean and should flow naturally without any labels or markers.
        Address the user in singular form (e.g., '당신', '귀하') instead of plural ('여러분')."""
        
        completion = client.chat.completions.create(
            model="gpt-4-0314",
            messages=[{"role": "system", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다. 다시 시도해 주세요."

# 대화 저장 함수
def save_conversation(session_id, conversation):
    conversation_text = " ".join(conversation)
    vector = create_vector(conversation_text)
    try:
        index.upsert(vectors=[(session_id, vector, {"conversation": conversation, "session_id": session_id})])
    except Exception as e:
        st.error(f"대화 저장 실패: {str(e)}")

# 이전 대화 요약 함수
def summarize_previous_conversation(session_id):
    try:
        results = index.query(vector=[0]*1536, top_k=1, filter={"session_id": session_id})
        if results['matches']:
            previous_conversation = results['matches'][0]['metadata']['conversation']
            prompt = f"""이전 대화를 요약해주세요. 핵심 내용만 간략하게 정리해 주세요.
            이전 대화: {previous_conversation}"""
            
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}]
            )
            return completion.choices[0].message.content.strip()
        return None
    except Exception as e:
        st.error(f"이전 대화 요약 중 오류 발생: {str(e)}")
        return None

# CSS for chat layout
def get_chat_css():
    return """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        padding: 10px;
    }
    .message {
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 75%;
        font-size: 1em;
    }
    .user-message {
        align-self: flex-end;
        background-color: #e0f7fa;
        color: #006064;
    }
    .coach-message {
        align-self: flex-start;
        background-color: #ffeb3b;
        color: #f57f17;
    }
    .current-message {
        border: 2px solid #4caf50;
        background-color: #ffffff;
        color: #212121;
        font-size: 1.2em;
        font-weight: bold;
    }
    .input-container {
        margin-top: 20px;
    }
    </style>
    """

# 사용자 입력 처리 함수
def process_user_input():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.conversation.append(user_input)
        try:
            with st.spinner("코치가 응답을 생성하고 있습니다..."):
                coach_response = generate_coach_response(
                    st.session_state.conversation,
                    st.session_state.current_stage,
                    st.session_state.question_count,
                    st.session_state.session_id
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
        finally:
            st.session_state.user_input = ""  # 입력창 비우기

# 첫 질문 생성 함수
def generate_first_question():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            first_question = generate_coach_response([], st.session_state.current_stage, 0, st.session_state.session_id)
            if first_question:
                st.session_state.conversation.append(first_question)
                return
            else:
                st.warning("첫 질문 생성에 실패했습니다. 다시 시도합니다.")
        except Exception as e:
            st.error(f"첫 질문 생성 중 오류 발생 (시도 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 재시도 전 잠시 대기
            else:
                st.error("첫 질문을 생성할 수 없습니다. 시스템 관리자에게 문의해주세요.")
                st.session_state.conversation.append("안녕하세요. 지금은 시스템에 일시적인 문제가 있습니다. 어떤 도움이 필요하신가요?")

    st.rerun()

# 메인 앱 로직
def main():
    st.title("AI 코칭 시스템")

    # Initialize session state variables
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'Trust'
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # 이전 대화 요약 표시
    previous_summary = summarize_previous_conversation(st.session_state.session_id)
    if previous_summary:
        st.info(f"이전 대화 요약: {previous_summary}")

    st.markdown(get_chat_css(), unsafe_allow_html=True)

    if not st.session_state.conversation:
        generate_first_question()

    st.subheader("현재 질문:")
    current_message = st.session_state.conversation[-1] if st.session_state.conversation else ""
    st.markdown(f'<div class="message current-message">{current_message}</div>', unsafe_allow_html=True)

    # 입력 처리와 상태 초기화를 위해 on_change 사용
    st.text_input("메시지를 입력하세요...", key="user_input", max_chars=200, on_change=process_user_input)

    # 초기화 버튼을 우측에 배치
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.write("")  # 왼쪽 공간 확보용
    with col2:
        if st.button("대화 초기화"):
            st.session_state.conversation = []
            st.session_state.current_stage = 'Trust'
            st.session_state.question_count = 0
            st.rerun()

    st.subheader("이전 대화 기록:")
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.conversation[:-1]):
            if i % 2 == 0:
                st.markdown(f'<div class="message coach-message">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message user-message">{message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

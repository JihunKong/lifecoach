import streamlit as st
import pandas as pd
import uuid
import openai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import time
import sqlite3
import hashlib
import random

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 코칭 시스템", layout="wide")

# 데이터베이스 연결
conn = sqlite3.connect('users.db')
c = conn.cursor()

# 사용자 테이블 생성
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')

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

def generate_coach_response(conversation, current_stage, question_count, username):
    try:
        # 대화 마무리 단계로 진입
        if question_count >= 3 and "슬슬 코칭을 마무리하고자 합니다." not in st.session_state.conversation[-1]:
            return "슬슬 코칭을 마무리하고자 합니다. 오늘 대화를 통해 배운 점이나 깨달은 점이 있다면 나눠주시겠어요?"

        # 사용자가 배운 점을 공유한 후의 응답 생성
        if "슬슬 코칭을 마무리하고자 합니다." in st.session_state.conversation[-1]:
            return "귀하가 말씀하신 점에 깊이 공감합니다. 이러한 깨달음은 매우 중요합니다. 코칭을 종료해도 될까요? **예 / 아니오**"

        # 사용자가 '예' 또는 '아니오'로 응답한 경우 처리
        if "코칭을 종료해도 될까요?" in st.session_state.conversation[-1]:
            if conversation[-1] == "예":
                st.session_state.coaching_finished = True
                return "대화가 잘 진행되었습니다. 이제 이 주제를 마무리하도록 하겠습니다. 감사합니다."
            elif conversation[-1] == "아니오":
                return "더 대화를 나누고 싶으신 주제가 있나요? 없다면 이 주제를 조금 더 깊이 다뤄보겠습니다."

        # 기존 응답 생성 로직
        stage_questions = coach_df[coach_df['step'].str.contains(current_stage, case=False, na=False)]
        available_questions = stage_questions.iloc[:, 1:].values.flatten().tolist()
        available_questions = [q for q in available_questions if pd.notnull(q)]
        
        recent_conversation = " ".join(conversation[-5:])
        query_vector = create_vector(recent_conversation)
        
        # 현재 사용자의 대화 벡터만 검색
        user_results = index.query(
            vector=query_vector,
            top_k=3,
            filter={"username": username},
            include_metadata=True
        )
        similar_user_conversations = [item['metadata']['conversation'] for item in user_results['matches'] if 'metadata' in item and 'conversation' in item['metadata']]
      
        prompt = f"""당신은 TEACHer 모델을 사용하는 공감적인 라이프 코치입니다. 
        현재 단계: {current_stage}
        질문 횟수: {question_count}
        이전 대화: {conversation[-5:] if len(conversation) > 5 else conversation}
        사용자의 유사한 과거 대화를 기반으로 자연스럽고 공감적인 응답을 생성하세요.
        그 후, 현재 단계와 관련된 하나의 후속 질문을 하되, 이전 질문들과 중복되지 않고 대화의 흐름에 자연스럽게 이어지도록 하세요.
        만약 현재 단계의 목표가 충분히 달성되었다고 판단되면, 사용자에게 지금까지의 대화를 요약하고 다음 단계로 넘어갈 준비가 되었는지 물어보세요.
        
        다음 지침을 따르세요:
        1. 응답은 한국어로 작성하고, 레이블이나 마커 없이 자연스럽게 흘러가야 합니다.
        2. 사용자를 단수 형태(예: '당신', '귀하')로 지칭하세요.
        3. 질문은 반드시 하나만 하고, 텍스트 내에 볼드 마크다운을 사용하여 강조하세요.
        4. 현재 단계의 목표 달성도를 평가하고, 필요시 다음 단계로의 전환을 고려하세요.
        5. 이전 질문들과 다른 관점이나 주제를 탐색하여 대화의 다양성을 높이세요.
        
        응답 형식:
        [코치의 응답]
        """
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다. 다시 시도해 주세요."

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

# 사용자 입력 처리 함수 수정
def save_conversation(username, conversation):
    conversation_text = " ".join(conversation)
    vector = create_vector(conversation_text)
    try:
        index.upsert(
            vectors=[
                {
                    'id': f"{username}_{uuid.uuid4()}",
                    'values': vector,
                    'metadata': {
                        'conversation': conversation_text,
                        'username': username
                    }
                }
            ]
        )
    except Exception as e:
        st.error(f"대화 저장 실패: {str(e)}")

# 사용자 입력 처리 함수 수정
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
                    st.session_state.user
                )
                st.session_state.conversation.append(coach_response)

            # 질문 횟수가 기준을 넘어가면 대화를 종료
            st.session_state.question_count += 1

            save_conversation(st.session_state.user, st.session_state.conversation)
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {str(e)}")
        finally:
            st.session_state.user_input = ""  # 입력창 비우기

        # 종료 여부를 선택한 경우 처리
        if st.session_state.coaching_finished:
            st.success("코칭이 종료되었습니다. 대화를 초기화하고 새로운 세션을 시작하시겠습니까?")
            if st.button("새 세션 시작"):
                st.session_state.conversation = []
                st.session_state.current_stage = 'Trust'
                st.session_state.question_count = 0
                st.session_state.coaching_finished = False
                st.rerun()
        else:
            # "예" 또는 "아니오" 응답 후의 대화 진행
            if "코칭을 종료해도 될까요?" in st.session_state.conversation[-2]:
                if user_input == "아니오":
                    st.text_input("더 대화를 나누고 싶으신 주제가 있나요?", key="new_topic", on_change=process_user_input)
                elif user_input == "예":
                    st.session_state.coaching_finished = True
                    st.success("코칭이 종료되었습니다. 대화를 초기화하고 새로운 세션을 시작하시겠습니까?")
                    if st.button("새 세션 시작"):
                        st.session_state.conversation = []
                        st.session_state.current_stage = 'Trust'
                        st.session_state.question_count = 0
                        st.session_state.coaching_finished = False
                        st.rerun()

# 첫 질문 생성 함수
def generate_first_question():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            first_question = generate_coach_response([], st.session_state.current_stage, 0, st.session_state.user)
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

# 사용자 인증 관련 함수
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    return c.fetchone() is not None

def login_user():
    st.subheader("로그인")
    username = st.text_input("사용자 이름")
    password = st.text_input("비밀번호", type="password")
    if st.button("로그인"):
        if verify_user(username, password):
            st.session_state.user = username
            st.session_state.logged_in = True
            st.success("로그인 성공!")
            st.rerun()
        else:
            st.error("잘못된 사용자 이름 또는 비밀번호입니다.")

def signup_user():
    st.subheader("회원가입")
    new_username = st.text_input("새 사용자 이름")
    new_password = st.text_input("새 비밀번호", type="password")
    if st.button("가입하기"):
        if create_user(new_username, new_password):
            st.success("계정이 생성되었습니다. 이제 로그인할 수 있습니다.")
        else:
            st.error("이미 존재하는 사용자 이름입니다.")

def logout_user():
    st.session_state.user = None
    st.session_state.logged_in = False
    st.session_state.conversation = []
    st.session_state.current_stage = 'Trust'
    st.session_state.question_count = 0
    st.session_state.coaching_finished = False
    st.info("로그아웃되었습니다.")
    st.rerun()

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
        generate_first_question()  # 첫 질문을 자동으로 생성

    st.markdown(get_chat_css(), unsafe_allow_html=True)

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
            generate_first_question()  # 초기화 후 첫 질문을 생성
            st.experimental_rerun()

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

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

# TEACHer 모델의 단계 정의
TEACHER_STAGES = ['Trust', 'Explore', 'Action', 'Change', 'Habit', 'Evaluate', 'Reinforce']

def generate_coach_response(conversation, current_stage, question_count, username):
    try:
        # 각 단계별 최소 질문 수 설정
        min_questions_per_stage = 3
        
        # 현재 단계의 인덱스 찾기
        current_stage_index = TEACHER_STAGES.index(current_stage)
        
        # 현재 단계에서 질문 수가 최소 질문 수를 초과하고, 마지막 단계가 아닌 경우
        if question_count >= min_questions_per_stage and current_stage_index < len(TEACHER_STAGES) - 1:
            next_stage = TEACHER_STAGES[current_stage_index + 1]
            return f"이제 {current_stage} 단계를 마치고 {next_stage} 단계로 넘어가겠습니다. {next_stage} 단계에서는 어떤 점을 중점적으로 다루고 싶으신가요?"

        # 마지막 단계이고 최소 질문 수를 충족한 경우
        if current_stage_index == len(TEACHER_STAGES) - 1 and question_count >= min_questions_per_stage:
            return "모든 단계를 완료했습니다. 전체 코칭 과정을 통해 어떤 점을 깨닫거나 배우셨나요? 마지막으로 느낀 점을 공유해 주시겠어요?"

        # 마지막 질문에 대한 사용자의 응답 후 코칭 종료
        if "모든 단계를 완료했습니다." in conversation[-2]:
            st.session_state.coaching_finished = True
            return "코칭 세션이 끝났습니다. 귀하의 성장과 발전을 응원합니다. 추가 코칭이 필요하시면 언제든 새로운 세션을 시작해 주세요."

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

        다음 지침을 엄격히 따르세요:
        1. 응답은 한국어로 작성하고, 레이블이나 마커 없이 자연스럽게 흘러가야 합니다.
        2. 사용자를 단수 형태(예: '당신', '귀하')로 지칭하세요.
        3. 반드시 하나의 질문만 생성하세요. 이 질문은 현재 단계와 관련되어야 하며, 이전 질문들과 중복되지 않아야 합니다. 질문 시에는 대화의 흐름을 자연스럽게 유지하고, 제공된 파일을 창의적으로 응용하여 적절하게 제시합니다.
        4. 현재 단계의 목표 달성도를 평가하고, 필요시 다음 단계로의 전환을 고려하세요.
        5. 각 단계에서 최소 {min_questions_per_stage}개의 질문을 하기 전에는 다음 단계로 넘어가지 마세요.

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

            # 질문 횟수 증가
            st.session_state.question_count += 1

            # 다음 단계로 넘어가는 경우
            if "단계로 넘어가겠습니다" in coach_response:
                next_stage = coach_response.split()[2]  # "Trust 단계로 넘어가겠습니다." 에서 "Trust" 추출
                st.session_state.current_stage = next_stage
                st.session_state.question_count = 0

            save_conversation(st.session_state.user, st.session_state.conversation)
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {str(e)}")
        finally:
            st.session_state.user_input = ""  # 입력창 비우기

        if st.session_state.coaching_finished:
            st.success("코칭이 종료되었습니다. 대화를 초기화하고 새로운 세션을 시작하시겠습니까?")
            if st.button("새 세션 시작"):
                st.session_state.conversation = []
                st.session_state.current_stage = 'Trust'
                st.session_state.question_count = 0
                st.session_state.coaching_finished = False
                generate_first_question()  # 첫 대화 생성
                st.rerun()

# 첫 질문 생성 함수
def generate_first_question():
    st.session_state.conversation.append("안녕하세요, 당신을 위한 라이프 코치입니다. 오늘 기분은 어떠세요?")

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
            st.session_state.conversation = []
            generate_first_question()  # 첫 대화 생성
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

# 메인 앱 로직
def main():
    st.title("AI 코칭 시스템")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        login_user()
        st.markdown("---")
        signup_user()
    else:
        st.write(f"안녕하세요, {st.session_state.user}님!")
        if st.button("로그아웃"):
            logout_user()

        st.markdown(get_chat_css(), unsafe_allow_html=True)

        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
            generate_first_question()  # 첫 대화 생성

        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = 'Trust'
        if 'question_count' not in st.session_state:
            st.session_state.question_count = 0
        if 'coaching_finished' not in st.session_state:
            st.session_state.coaching_finished = False

        st.subheader("현재 질문:")
        current_message = st.session_state.conversation[-1] if st.session_state.conversation else ""
        st.markdown(f'<div class="message current-message">{current_message}</div>', unsafe_allow_html=True)

        if not st.session_state.coaching_finished:
            # 입력 처리와 상태 초기화를 위해 on_change 사용
            st.text_input("메시지를 입력하세요...", key="user_input", max_chars=200, on_change=process_user_input)
        else:
            st.success("코칭이 종료되었습니다. 대화를 초기화하고 새로운 세션을 시작하시겠습니까?")
            if st.button("새 세션 시작"):
                st.session_state.conversation = []
                st.session_state.current_stage = 'Trust'
                st.session_state.question_count = 0
                st.session_state.coaching_finished = False
                generate_first_question()  # 첫 대화 생성
                st.rerun()

        # 대화 초기화 버튼 추가
        if st.button("대화 초기화"):
            st.session_state.conversation = []
            st.session_state.current_stage = 'Trust'
            st.session_state.question_count = 0
            st.session_state.coaching_finished = False
            generate_first_question()  # 첫 대화 생성
            st.rerun()

        # 현재 단계 표시
        st.sidebar.subheader("현재 단계")
        st.sidebar.write(st.session_state.current_stage)

        # 이전 대화 기록을 현재 질문과 채팅창 아래로 이동
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

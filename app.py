import streamlit as st
import pandas as pd
import uuid
from openai import OpenAI
import sqlite3
import json

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# 데이터베이스 연결 및 초기화 함수
def get_db_connection():
    conn = sqlite3.connect('coaching_sessions.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, stage TEXT, question_count INTEGER, conversation TEXT)''')
    conn.commit()
    conn.close()

# 세션 데이터 저장 및 로드 함수
def save_session_data(session_id, stage, question_count, conversation):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
              (session_id, stage, question_count, json.dumps(conversation)))
    conn.commit()
    conn.close()

def load_session_data(session_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row['stage'], row['question_count'], json.loads(row['conversation'])
    return 'Trust', 0, []

# 코칭 질문 데이터 로드
@st.cache_data
def load_coach_data():
    return pd.read_excel('coach.xlsx')

# 데이터베이스 초기화
init_db()

coach_df = load_coach_data()

# Streamlit 앱 설정
st.set_page_config(page_title="GPT 기반 TEACHer 코칭 시스템", layout="wide")

# 세션 관리
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

# 세션 데이터 로드
current_stage, question_count, conversation = load_session_data(session_id)

# TEACHer 모델의 단계
stages = ['Trust', 'Explore', 'Aspire', 'Create', 'Harvest', 'Empower&Reflect']

# GPT를 사용한 코칭 대화 생성 함수
def generate_coach_response(conversation, current_stage, question_count):
    stage_questions = coach_df[coach_df['step'].str.contains(current_stage, case=False, na=False)]
    available_questions = stage_questions.iloc[:, 1:].values.flatten().tolist()
    available_questions = [q for q in available_questions if pd.notnull(q)]
    
    prompt = f"""You are an empathetic life coach using the TEACHer model. 
    Current stage: {current_stage}
    Question count: {question_count}
    Previous conversation: {conversation[-5:] if len(conversation) > 5 else conversation}
    
    Based on the user's responses, generate a natural, empathetic response and a follow-up question.
    Choose from or create a question similar to these for the current stage:
    {available_questions}
    
    Your response should be in Korean and follow this format:
    Coach: [Empathetic response]
    
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

# 채팅 인터페이스
st.title("GPT 기반 TEACHer 코칭 시스템")

# 대화 기록 표시
for i, message in enumerate(conversation):
    if i % 2 == 0:
        st.text_area("You:", value=message, height=100, key=f"msg_{i}", disabled=True)
    else:
        st.text_area("Coach:", value=message, height=100, key=f"msg_{i}", disabled=True)

# 사용자 입력
user_input = st.text_input("메시지를 입력하세요...", key="user_input")

# 엔터 키 감지 및 메시지 제출
if st.session_state.user_input and st.session_state.user_input != st.session_state.get('previous_input', ''):
    conversation.append(user_input)
    
    # 코치 응답 생성
    coach_response = generate_coach_response(conversation, current_stage, question_count)
    conversation.append(coach_response)
    
    # 질문 카운트 증가 및 단계 관리
    question_count += 1
    if question_count >= 3:
        current_stage_index = stages.index(current_stage)
        if current_stage_index < len(stages) - 1:
            current_stage = stages[current_stage_index + 1]
            question_count = 0
    
    # 세션 데이터 저장
    save_session_data(session_id, current_stage, question_count, conversation)
    
    # 입력 필드 초기화 및 페이지 새로고침
    st.session_state.previous_input = st.session_state.user_input
    st.session_state.user_input = ""
    st.experimental_rerun()

# 현재 세션 정보 표시 (개발용, 실제 사용 시 숨김 처리 가능)
st.sidebar.write(f"세션 ID: {session_id}")
st.sidebar.write(f"현재 단계: {current_stage}")
st.sidebar.write(f"질문 수: {question_count}")

import streamlit as st
import pandas as pd
import uuid
from openai import OpenAI
import sqlite3
import json

# Streamlit 페이지 설정
st.set_page_config(page_title="GPT 기반 TEACHer 코칭 시스템", layout="wide")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# 데이터베이스 관련 함수들
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

# 코칭 데이터 로드
@st.cache_data
def load_coach_data():
    return pd.read_excel('coach.xlsx')

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
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

# 대화 초기화 함수
def reset_conversation():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.current_stage = 'Trust'
    st.session_state.question_count = 0
    st.session_state.conversation = []
    st.session_state.user_input = ""

# 메인 앱 로직
def main():
    st.title("GPT 기반 TEACHer 코칭 시스템")

    # 세션 상태 초기화
    initialize_session_state()
    
    # 첫 질문 생성
    if not st.session_state.conversation:
        first_question = generate_coach_response([], st.session_state.current_stage, 0)
        st.session_state.conversation.append(first_question)

    # 대화 기록 표시
    for i, message in enumerate(st.session_state.conversation):
        if i % 2 == 0:
            st.text_area("Coach:", value=message, height=100, key=f"msg_{i}", disabled=True)
        else:
            st.text_area("You:", value=message, height=100, key=f"msg_{i}", disabled=True)

    # 사용자 입력
    user_input = st.text_input("메시지를 입력하세요...", key="user_input", value=st.session_state.user_input)

    # 메시지 제출 버튼
    if st.button("전송"):
        if user_input:
            st.session_state.conversation.append(user_input)
            
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
            
            # 세션 데이터 저장
            save_session_data(st.session_state.session_id, st.session_state.current_stage, st.session_state.question_count, st.session_state.conversation)
            
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

# 데이터베이스 초기화 및 코칭 데이터 로드
init_db()
coach_df = load_coach_data()

# 메인 앱 실행
if __name__ == "__main__":
    main()

import streamlit as st
from openai import OpenAI
import pandas as pd
import uuid

# OpenAI 클라이언트 설정
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# 세션 상태 초기화
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

# Excel 파일 불러오기
@st.cache_data
def load_questions():
    df = pd.read_excel("coach.xlsx")
    questions_by_stage = {}
    for _, row in df.iterrows():
        stage = str(row['step'])  # 단계를 문자열로 변환
        questions = [q for q in row.iloc[1:] if pd.notna(q)]
        questions_by_stage[stage] = questions[:3]  # 각 단계별로 최대 3개의 질문 선택
    return questions_by_stage

questions_by_stage = load_questions()

def get_ai_response(prompt, conversation_history):
    messages = [{"role": "system", "content": "당신은 전문적인 코치입니다. 항상 질문을 통해 대화를 이끌어나가며, 직접적인 조언은 하지 않습니다."}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def main():
    st.title("AI 코칭 봇")

    # 디버깅 정보
    st.write("Debug - questions_by_stage:", questions_by_stage)

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if st.session_state.session_id not in st.session_state.sessions:
        st.session_state.sessions[st.session_state.session_id] = {
            "stage": 0,
            "question_index": 0,
            "conversation": [],
            "agreed": False
        }

    session = st.session_state.sessions[st.session_state.session_id]

    # 디버깅 정보
    st.write("Debug - Current stage:", session["stage"])
    st.write("Debug - Question index:", session["question_index"])

    if session["stage"] == 0:
        st.write("안녕하세요. AI 코칭 세션에 오신 것을 환영합니다.")
        st.write("이 세션은 완전히 비공개로 진행되며, 모든 대화 내용은 세션 종료 후 자동으로 삭제됩니다.")
        st.write("코칭을 시작하기 전에 몇 가지 동의를 구하고자 합니다.")
        agree = st.checkbox("코칭 세션 시작 및 개인정보 보호 정책에 동의합니다.")
        if agree and not session["agreed"]:
            session["agreed"] = True
            session["stage"] = 1
            session["conversation"].append({"role": "assistant", "content": "코칭 세션을 시작하겠습니다. 먼저, 오늘 어떤 주제에 대해 이야기 나누고 싶으신가요?"})
            st.experimental_rerun()

    elif session["stage"] <= len(questions_by_stage):
        for message in session["conversation"]:
            if message['role'] == 'user':
                st.text_area("You:", value=message['content'], height=100, disabled=True)
            else:
                st.text_area("Coach:", value=message['content'], height=100, disabled=True)

        user_input = st.text_input("Your response:")
        if user_input:
            session["conversation"].append({"role": "user", "content": user_input})
            
            current_stage = str(session["stage"])
            if current_stage in questions_by_stage and session["question_index"] < len(questions_by_stage[current_stage]):
                question = questions_by_stage[current_stage][session["question_index"]]
                prompt = f"다음 질문에 대한 코치의 응답을 생성해주세요. 질문: {question}"
                ai_response = get_ai_response(prompt, session["conversation"])
                session["conversation"].append({"role": "assistant", "content": ai_response})
                session["question_index"] += 1
            else:
                session["stage"] += 1
                session["question_index"] = 0
                if str(session["stage"]) in questions_by_stage:
                    ai_response = "다음 단계로 넘어가겠습니다. 준비되셨나요?"
                else:
                    ai_response = "모든 단계를 완료했습니다. 코칭 세션을 마무리하고 싶으신가요?"
                session["conversation"].append({"role": "assistant", "content": ai_response})

            st.experimental_rerun()

    else:
        st.write("코칭 세션이 끝났습니다. 세션을 요약해 드리겠습니다.")
        summary_prompt = "다음은 코칭 세션의 대화 내용입니다. 주요 포인트를 요약해주세요:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in session["conversation"]])
        summary = get_ai_response(summary_prompt, [])
        st.text_area("Session Summary:", value=summary, height=300, disabled=True)
        if st.button("새 세션 시작"):
            st.session_state.session_id = str(uuid.uuid4())
            st.experimental_rerun()

if __name__ == "__main__":
    main()

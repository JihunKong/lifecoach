import streamlit as st
import openai
import pandas as pd
import uuid

# OpenAI 클라이언트 설정
api_key = st.secrets["openai"]["api_key"]
openai.api_key = api_key

# 세션 상태 초기화
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

# Excel 파일 불러오기
@st.cache_data
def load_questions():
    df = pd.read_excel("coach.xlsx")
    questions_by_stage = {}
    for _, row in df.iterrows():
        stage = str(row['step'])
        questions = [q for q in row.iloc[1:] if pd.notna(q)]
        questions_by_stage[stage] = questions
    return questions_by_stage

questions_by_stage = load_questions()

def get_ai_response(prompt, conversation_history, system_message):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message['content'].strip()

def generate_coaching_question(stage, conversation_history, questions):
    system_message = f"""
    당신은 전문적인 코치입니다. 현재 코칭의 {stage}단계에 있습니다. 
    다음 질문 목록을 참고하되, 대화의 흐름과 클라이언트의 답변을 고려하여 
    가장 적절한 다음 질문을 생성해주세요. 질문 전에 클라이언트의 이전 답변에 대한 
    간단한 공감과 재진술을 포함해주세요.
    
    참고할 질문 목록:
    {', '.join(questions)}
    
    출력 형식:
    공감과 재진술: [클라이언트의 이전 답변에 대한 공감과 재진술]
    다음 질문: [생성된 코칭 질문]
    """
    
    prompt = "이전 대화 내용을 바탕으로 적절한 다음 코칭 질문을 생성해주세요."
    response = get_ai_response(prompt, conversation_history, system_message)
    return response

def main():
    st.title("AI 코칭 봇")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if st.session_state.session_id not in st.session_state.sessions:
        st.session_state.sessions[st.session_state.session_id] = {
            "stage": 0,
            "conversation": [],
            "agreed": False
        }

    session = st.session_state.sessions[st.session_state.session_id]

    if not session["agreed"]:
        st.write("안녕하세요. AI 코칭 세션에 오신 것을 환영합니다.")
        st.write("이 세션은 완전히 비공개로 진행되며, 모든 대화 내용은 세션 종료 후 자동으로 삭제됩니다.")
        st.write("코칭을 시작하기 전에 몇 가지 동의를 구하고자 합니다.")
        if st.button("코칭 세션 시작 및 개인정보 보호 정책에 동의합니다", key="agree_button"):
            session["agreed"] = True
            session["stage"] = 1
            initial_question = "코칭 세션을 시작하겠습니다. 먼저, 오늘 어떤 주제에 대해 이야기 나누고 싶으신가요?"
            session["conversation"].append({"role": "assistant", "content": initial_question})
            st.experimental_rerun()

    elif session["stage"] <= len(questions_by_stage):
        for i, message in enumerate(session["conversation"]):
            if message['role'] == 'user':
                st.text_area("You:", value=message['content'], height=100, disabled=True, key=f"user_message_{i}")
            else:
                st.text_area("Coach:", value=message['content'], height=100, disabled=True, key=f"coach_message_{i}")

        user_input = st.text_input("Your response:", key=f"input_{session['stage']}_{len(session['conversation'])}")
        if st.button("Send", key=f"send_{session['stage']}_{len(session['conversation'])}"):
            if user_input:
                session["conversation"].append({"role": "user", "content": user_input})

                current_stage = str(session["stage"])
                if current_stage in questions_by_stage:
                    ai_response = generate_coaching_question(current_stage, session["conversation"], questions_by_stage[current_stage])
                    session["conversation"].append({"role": "assistant", "content": ai_response})

                    session["stage"] += 1
                    next_question = "다음 단계로 넘어가겠습니다. 준비되셨나요?"
                    session["conversation"].append({"role": "assistant", "content": next_question})
                
                st.experimental_rerun()

    else:
        st.write("모든 단계를 완료했습니다. 코칭 세션을 마무리하고 싶으신가요?")
        if st.button("예", key="end_session"):
            st.write("코칭 세션이 끝났습니다. 세션을 요약해 드리겠습니다.")
            summary_prompt = "다음은 코칭 세션의 대화 내용입니다. 주요 포인트를 요약해주세요:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in session["conversation"]])
            summary = get_ai_response(summary_prompt, [], "당신은 전문적인 코치입니다. 코칭 세션의 주요 내용을 간략하게 요약해주세요.")
            st.text_area("Session Summary:", value=summary, height=300, disabled=True, key="summary")
            if st.button("새 세션 시작", key="new_session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.experimental_rerun()

if __name__ == "__main__":
    main()

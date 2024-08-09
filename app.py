import openai
import streamlit as st
import pandas as pd
import uuid

# OpenAI API 키 설정
openai.api_key = st.secrets["openai"]["api_key"]

# 코칭 질문 엑셀 파일 읽기
coach_df = pd.read_excel('coach.xlsx')

# Streamlit 웹 애플리케이션 제목
st.title("GPT 기반 TEACHer 코칭 시스템")

# UUID를 사용하여 개별 대화 세션 관리
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# TEACHer 모델의 단계
stages = ['Trust', 'Explore', 'Aspire', 'Create', 'Havest', 'Empower&Reflect']

# 현재 단계 설정
if f'{session_id}_stage' not in st.session_state:
    st.session_state[f'{session_id}_stage'] = 'Trust'
if f'{session_id}_question_count' not in st.session_state:
    st.session_state[f'{session_id}_question_count'] = 0

# 단계별 코칭 질문을 GPT로 선택하는 함수
def suggest_coaching_question(stage, previous_answers):
    questions = coach_df[coach_df['step'].str.contains(stage)]['Question1':'Question15'].values.flatten().tolist()
    questions = [q for q in questions if pd.notnull(q)]
    
    prompt = f"Given the user's previous answers: {previous_answers}, select the most appropriate question from the following options:\n"
    for question in questions:
        prompt += f"- {question}\n"
    prompt += "Select the most appropriate question."

    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )
    selected_question = response.choices[0].text.strip()
    return selected_question

# 현재 단계의 질문 선택
current_stage = st.session_state[f'{session_id}_stage']
previous_answers = st.session_state.get(f'{session_id}_answers', [])
current_question_count = st.session_state[f'{session_id}_question_count']

if current_question_count < 3:
    current_question = suggest_coaching_question(current_stage, previous_answers)
    st.write(f"**{current_stage} 단계의 질문 {current_question_count + 1}:** {current_question}")
else:
    st.write(f"**{current_stage} 단계의 모든 질문이 완료되었습니다. 다음 단계로 넘어갑니다.**")

# 사용자 입력 받기
user_input = st.text_area("이 질문에 대해 답변해보세요:")

if st.button("다음 질문"):
    if user_input:
        # 사용자의 답변 저장
        previous_answers.append(user_input)
        st.session_state[f'{session_id}_answers'] = previous_answers

        # GPT-4에 입력을 전송하여 사용자의 답변 요약 생성
        summary_prompt = f"사용자가 다음과 같은 내용을 말했습니다: '{user_input}' 이 내용을 간단하게 요약해 주세요."
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=summary_prompt,
            max_tokens=60
        )
        summary = response.choices[0].text.strip()

        # 요약된 내용을 피드백으로 출력
        st.write("**코치의 재진술:**")
        st.write(summary)
        
        # 질문 횟수 증가 또는 다음 단계로 진행
        st.session_state[f'{session_id}_question_count'] += 1
        if st.session_state[f'{session_id}_question_count'] >= 3:
            current_stage_index = stages.index(current_stage)
            if current_stage_index < len(stages) - 1:
                st.session_state[f'{session_id}_stage'] = stages[current_stage_index + 1]
                st.session_state[f'{session_id}_question_count'] = 0
                st.session_state[f'{session_id}_answers'] = []
            else:
                st.write("모든 단계가 완료되었습니다. 코칭이 종료되었습니다.")
    else:
        st.write("먼저 답변을 입력하세요!")

# 사용자 안내
st.write("이 코칭 시스템은 질문과 간단한 재진술을 통해 귀하의 자기 성찰을 지원합니다.")
st.write(f"현재 세션 ID: {session_id}")

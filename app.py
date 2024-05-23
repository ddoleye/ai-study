
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os
import tempfile
import base64
from pydub import AudioSegment
    
# .env
load_dotenv()
os.getenv("OPENAI_API_KEY")

client = OpenAI()

# 임시 파일 생성
def tmp(ext):
    return tempfile.NamedTemporaryFile(prefix='temp_', suffix='.' + ext, delete=False)

# m4a 를 BadRequest 로 반환해서
def mp3(input_file, output_file):
    # m4a 파일 불러오기
    audio = AudioSegment.from_file(input_file, format="m4a")

    # mp3로 변환
    audio.export(output_file, format="mp3")

def speech2text(audio):
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    print (transcription)
    return transcription

def text2speech(text, output):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(output.name)
    
    
def summarize(txt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 유능한 비서입니다."},
            {"role": "user", "content": "다음 내용을 요약해서 보여주고 내용에서 시간을 포함하는 일정과 해야 할일이 있으면 정리해서 보여주세요"},
            {"role": "user", "content": txt}
        ]
    )
    print(completion)
    return completion.choices[0]

st.title("녹음 요약") # title

# file uploader
file = st.file_uploader(
    label = "녹음파일을 선택해주세요...", 
    type = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
)
submit = st.button("요약!") # button object
if submit:
    if file is not None:
        with st.spinner('텍스트 추출...'):
            if file.name.lower().endswith('.m4a'):
                target = tmp('mp3');
                mp3(file, tmp)
                temp = True
            else:
                target = file
                temp = False
            txt = speech2text(target)
            if temp:
                print(f'임시 파일 {target} 삭제')
                os.remove(target)
            div = st.container(border=True)
            div.write(txt.text)

        with st.container():
            with st.spinner('요약 중...'):
                sum = summarize(txt.text)
                div = st.container(border=True)
                div.markdown(sum.message.content)

st.markdown("--------------------------")
st.markdown("### 녹음 샘플이 없다면?")
# 텍스트 입력 받기
text_input = st.text_area("대화 형식의 텍스트를 입력합니다", """버튼을 클릭하면 쇼핑몰 화면에 추천 상품 목록이 나왔으면 좋겠어요.

추천은 얼마 만에 완료되어야 하나요? 추천 상품 목록은 어떤 기준으로 정렬되어야 하나요? 얼마나 자주 추천 목록이 갱신되어야 하나요?

버튼 클릭과 동시에 추천 상품 목록이 반환되어야 하며 목록은 추천도 순으로 정렬되어야 해요. 사용자의 행동 패턴을 분석해서 추천 상품 목록이 자동으로 변경되었으면 좋겠어요.

추천 시스템은 현재 구조에서 매번 호출될 수 없어요. 이 기능 구현은 불가능합니다.
추천 시스템은 현재 구조에서 매 요청마다 호출될 수는 없지만, 갱신이 즉각적일 필요가 없다면 우리는 저장해놓고 이를 사용할 수 있어요. 기획을 조금 바꾼다면 시간 내에 충분히 개발이 가능합니다.
""", height=160)

generate_mp3 = st.button("녹음파일 만들기(mp3)")
# 입력된 텍스트 출력
if generate_mp3:
    with st.spinner('생성중...'):
        target = tmp('mp3')
        sum = text2speech(text_input, target)
        b64 = base64.b64encode(target.read()).decode()  # 파일을 바이트로 변환하고 base64로 인코딩
        target.close()
        os.remove(target.name)
        href = f'<a href="data:file/mp3;base64,{b64}" download="example.mp3">다운로드 MP3</a>'
        st.markdown(href, unsafe_allow_html=True)
        
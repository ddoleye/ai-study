
import streamlit as st
# from transformers import BartForConditionalGeneration
# from tokenizers import PreTrainedTokenizerFast, BartTokenizer
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

# generation_config = GenerationConfig.from_pretrained("openai/whisper-base")
# recognizer = pipeline(task="automatic-speech-recognition",
#                       model="p4b/whisper-small-ko-fl-v2",
#                       config=generation_config,
#                       )
# summarizer = pipeline(task="summarization")

def text(sample):
    
    # config = GenerationConfig(
    #     max_new_tokens=500, 
    #     max_length= 800,
    #     temperature=1.2, 
    #     num_return_sequences=3)
        
    # generator = pipeline(task = 'automatic-speech-recognition', 
    #                      model='p4b/whisper-small-ko-fl-v2', 
    #                      device=0, 
    #                      generation_config = config)
    # result = generator(sample)



    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("모델 로딩")
    model_id = "p4b/whisper-small-ko-fl-v2"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # # sample = dataset[0]["audio"]

    result = pipe(sample)
    print(result["text"])
    return result

    # print(result)
    # return result
    
def summarize(txt):
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
    # model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")
    # inputs = tokenizer(txt, return_tensors="pt", padding="max_length", truncation=True, max_length=1026)
    # print(inputs)
    # summary_text_ids = model.generate(
    #     input_ids=inputs['input_ids'],
    #     attention_mask=inputs['attention_mask'],
    #     bos_token_id=model.config.bos_token_id,
    #     eos_token_id=model.config.eos_token_id,
    #     length_penalty=1.0,
    #     max_length=300,
    #     min_length=12,
    #     num_beams=20,
    #     repetition_penalty=1.5,
    #     no_repeat_ngram_size=15,
    # )
    
    # result = summarizer(txt)
    return txt



st.title("녹음 요약") # title
st.markdown("녹음 파일을 업로드하세요")
st.markdown("---") # division

# file uploader
file = st.file_uploader(
    label = "녹음파일을 선택해주세요...", 
    type = ["m4a"]
)
submit = st.button("요약!") # button object
if submit:
    if file is not None:
        # opencv의 imdecode 인자로 사용하기 위해 byte로 변환한다
        audio = np.asarray(
            bytearray(file.read()), 
            dtype=np.uint8
        )

        with st.spinner('텍스트 추출...'):
            txt = text(audio)
            div = st.container(border=True)
            div.write(txt["text"])

        # with st.container():
        #     with st.spinner('요약 중...'):
        #         summarize(txt)

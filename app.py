"""
Hello! World 

"""
import streamlit as st
from transformers import PreTrainedTokenizerFast,BartTokenizer, BartForConditionalGeneration
st.title('Squeezer')

# Application run : stremalit run Squeezer.py

with st.container():
    st.title("요약할 논문 정보")
    with st.form(key="form"):
        content = st.text_area(label="논문 내용")
        submit = st.form_submit_button(label = "요약하기")
        if submit:
            with st.spinner('요약 중...'):

                tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
                model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")
                input_text = content
                inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1026)
                print(inputs)
                summary_text_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    bos_token_id=model.config.bos_token_id,
                    eos_token_id=model.config.eos_token_id,
                    length_penalty=1.0,
                    max_length=300,
                    min_length=12,
                    num_beams=20,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=15,
                )
                with st.container():
                    st.title("요약 결과")
                    st.write(tokenizer.decode(summary_text_ids[0], skip_special_tokens=True))
                st.success('Done!')


with st.container():
    st.title("Summarization")
    with st.form(key="form2"):
        content = st.text_area(label="Content")
        submit = st.form_submit_button(label = "Summarize")
        if submit:
            with st.spinner('Summarizing...'):
                model_name = "com3dian/Bart-large-paper2slides-summarizer"
                tokenizer = BartTokenizer.from_pretrained(model_name)
                model = BartForConditionalGeneration.from_pretrained(model_name)
                input_text = content
                input_ids = tokenizer.encode(input_text, return_tensors="pt")
                output = model.generate(input_ids)
                summary = tokenizer.decode(output[0], skip_special_tokens=True)

                with st.container():
                    st.title("요약 결과")
                    st.write(summary)
                st.success('Done!')
if submit:
        with st.spinner('번역 중...'):





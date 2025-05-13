import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

# تحميل الموديل والتوكنيزر
model_path = "fine_tuned_ar_en_model"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

st.title("Arabic to English Translation")

# واجهة المستخدم
text = st.text_area("اكتب جملة بالعربي:", "")

if st.button("ترجم"):
    if text.strip() == "":
        st.warning("من فضلك اكتب جملة للترجمة.")
    else:
        # تحويل الجملة لتوكنز
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        
        # الترجمة
        translated = model.generate(**inputs)
        
        # فك التوكنز للجملة المترجمة
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        st.success("الترجمة الإنجليزية:")
        st.write(translated_text)

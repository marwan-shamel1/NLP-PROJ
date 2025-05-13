import tkinter as tk
from transformers import MarianTokenizer, MarianMTModel
import torch

# تحميل الموديل والتوكنيزر من المسار المحلي
model_path = "fine_tuned_ar_en_model"  # عدل المسار لو محتاج
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# دالة الترجمة
def translate_text():
    input_text = input_box.get("1.0", tk.END).strip()
    if not input_text:
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "من فضلك أدخل جملة.")
        return

    # التوكنيز والترجمة
    inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs, max_length=128)
    output_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # عرض الترجمة
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, output_text)

# واجهة التطبيق
window = tk.Tk()
window.title("مترجم عربي - إنجليزي")
window.geometry("600x400")
window.config(bg="#f2f2f2")

tk.Label(window, text="ادخل الجملة بالعربية:", font=("Arial", 14), bg="#f2f2f2").pack(pady=10)
input_box = tk.Text(window, height=5, font=("Arial", 12))
input_box.pack(padx=20)

tk.Button(window, text="ترجم", command=translate_text, font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)

tk.Label(window, text="الترجمة بالإنجليزية:", font=("Arial", 14), bg="#f2f2f2").pack(pady=10)
output_box = tk.Text(window, height=5, font=("Arial", 12))
output_box.pack(padx=20)

window.mainloop()

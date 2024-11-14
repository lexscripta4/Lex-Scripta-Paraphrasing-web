from flask import Flask, render_template, request
from transformers import pipeline

# Initialize Flask application
app = Flask(__name__)

# Initialize paraphrasing model
paraphraser = pipeline("text2text-generation", model="t5-base")

# Main page with input text form
@app.route("/", methods=["GET", "POST"])
def home():
    paraphrased_text = ""
    if request.method == "POST":
        original_text = request.form["original_text"]
        if original_text:
            # Menambahkan beberapa pengaturan parameter untuk menghasilkan paraphrase yang lebih baik
            result = paraphraser(
                original_text,
                max_length=200,        # Perpanjang panjang teks output
                num_beams=5,           # Menggunakan beam search untuk meningkatkan kualitas
                num_return_sequences=1,  # Hanya satu hasil paraphrase
                top_p=0.9,             # Membatasi probabilitas kumulatif untuk sampling
                temperature=0.7        # Menurunkan randomness agar output lebih konsisten
            )
            paraphrased_text = result[0]['generated_text']
    return render_template("index.html", paraphrased_text=paraphrased_text)

if __name__ == "__main__":
    print("Memulai aplikasi...")
    print("Aplikasi berjalan di http://127.0.0.1:5001")
    app.run(debug=True, port=5001)

from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("./finetuned_model")
model = GPT2LMHeadModel.from_pretrained("./finetuned_model")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            inputs.input_ids,
            max_length=100,
            do_sample=True,
            temperature=0.8
        )
        result = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
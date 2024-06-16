from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define a list of available models
available_models = [
    "microsoft/DialoGPT-small",
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-large"
]

# Initialize with the default model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

embedded_messages = []
instructions_sent = []
token_limit = 1000
response_limit = 50
rag_results = []

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template('chat.html', models=available_models)

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    response = get_Chat_response(input)
    return response

@app.route("/edit", methods=["POST"])
def edit_message():
    msg = request.form["msg"]
    response = get_Chat_response(msg)
    return jsonify({"user_msg": msg, "bot_msg": response})

@app.route("/embed", methods=["POST"])
def embed_message():
    user_msg = request.form["user_msg"]
    bot_msg = request.form["bot_msg"]
    embedded_message = {"User response": user_msg, "Chatbot response": bot_msg}
    embedded_messages.append(embedded_message)
    print("Embedding message:", embedded_message)
    print(embedded_messages)
    return "Message embedded successfully!"

@app.route("/sendinstructions", methods=["POST"])
def send_instructions():
    msg = request.form["msg"]
    instructions_sent.append(msg)
    print(instructions_sent)
    return "Instructions sent successfully!"

@app.route("/select_model", methods=["POST"])
def select_model():
    selected_model = request.form["model"]
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    model = AutoModelForCausalLM.from_pretrained(selected_model)
    return f"Model {selected_model} selected successfully!"

@app.route("/set_token_limit", methods=["POST"])
def set_token_limit():
    global token_limit
    try:
        token_limit = int(request.form["token_limit"])
        print(token_limit)
        if token_limit <= 0:
            raise ValueError("Token limit must be a positive integer.")
    except ValueError as e:
        return str(e), 400
    return f"Token limit set to {token_limit} successfully!"

@app.route("/set_response_limit", methods=["POST"])
def set_response_limit():
    global response_limit
    try:
        response_limit = int(request.form["response_limit"])
        print(response_limit)
        if response_limit <= 0:
            raise ValueError("Response limit must be a positive integer.")
    except ValueError as e:
        return str(e), 400
    return f"Response limit set to {response_limit} successfully!"

@app.route("/upload_excel", methods=["POST"])
def upload_excel():
    if 'excel_file' not in request.files:
        return "No file part", 400
    file = request.files['excel_file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Process the file as needed here
        return "Excel file uploaded successfully!", 200
    return "Invalid file type", 400

@app.route("/rag_results", methods=["GET"])
def get_rag_results():
    return jsonify(rag_results)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xls', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_Chat_response(input):
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    app.run()

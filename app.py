import joblib
import gradio as gr
import nltk
import os
import pandas as pd
import numpy as np
from huggingface_hub import HfApi
import plotly.graph_objects as go

# =========================
# Hugging Face Setup
# =========================
DATASET_REPO = "RizwanSayyad/news-feedback"
FEEDBACK_FILE = "feedback.csv"

api = HfApi()

# =========================
# Load model & vectorizer
# =========================
model = joblib.load("ag_news_model.joblib")
vectorizer = joblib.load("ag_news_vectorizer.joblib")

label_map = {
    0: "World",
    1: "Sports",
    2: "Business or Other",
    3: "Sci/Tech"
}

# =========================
# Preprocessing
# =========================
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

def preprocess(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    bigrams = ['_'.join(pair) for pair in ngrams(tokens, 2)]
    return ' '.join(bigrams)

# =========================
# Prediction (Top 3 with Plotly visualization)
# =========================
def predict_news(text):
    processed_text = preprocess(text)
    vectorized_text = vectorizer.transform([processed_text])
    probs = model.predict_proba(vectorized_text)[0]

    # Top 3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(label_map[i], probs[i]) for i in top3_idx]

    # Plotly bar chart
    fig = go.Figure()
    for i, (label, prob) in enumerate(top3):
        fig.add_trace(go.Bar(
            x=[label],
            y=[prob],
            marker_color="green" if i == 0 else "skyblue"
        ))
    fig.update_layout(
        title="Top-3 Prediction Confidence",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )

    # Table data for DataFrame
    table_data = [(label, f"{prob:.2f}") for label, prob in top3]

    return fig, table_data

# =========================
# Feedback function
# =========================
def save_feedback(news_text, top3_preds, user_feedback):
    df_new = pd.DataFrame([{
        "news_text": news_text,
        "top3_predictions": str(top3_preds),
        "user_feedback": user_feedback
    }])

    if os.path.exists(FEEDBACK_FILE):
        df_old = pd.read_csv(FEEDBACK_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(FEEDBACK_FILE, index=False)

    api.upload_file(
        path_or_fileobj=FEEDBACK_FILE,
        path_in_repo=FEEDBACK_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset"
    )

    return "✅ Thanks for your feedback!"

# =========================
# Gradio Interface (Enhanced UI)
# =========================
import gradio as gr

# =========================
# Gradio Interface (Enhanced UI)
# =========================
# Sample articles
sample_articles = {
    "Tech Giant Reports Record Quarterly Profits": 
    "Global tech company Nexon Corp announced record profits this quarter, driven by strong sales in cloud computing and AI products. Analysts predict continued growth as the company expands into emerging markets.",

    "UN Urges Global Cooperation on Climate Change":
    "The United Nations called on countries worldwide to strengthen their efforts to combat climate change. Secretary-General António Guterres emphasized the need for urgent action to reduce carbon emissions and invest in renewable energy.",

    "New AI Tool Helps Students Learn Faster":
    "A startup in Bangalore has launched an AI-powered learning app that adapts to students’ learning styles. The app provides personalized quizzes, feedback, and study materials, helping students improve their grades efficiently. Experts say this could revolutionize online education.",

    "Local Football Team Wins State Championship":
    "The Nagpur Tigers won the state football championship yesterday, defeating the Pune Lions 3-1. Fans celebrated as the team lifted the trophy. Coach Ramesh Patil praised the players for their hard work and teamwork throughout the season."
}

# =========================
# Gradio Interface (with sample articles)
# =========================
with gr.Blocks(title="News Classifier", theme="default") as demo:
    
    # Inject CSS for styling
    gr.HTML("""
    <style>
        .input-box textarea { border-radius: 10px; border: 2px solid #1E90FF; padding: 10px; }
        .feedback-box textarea { border-radius: 10px; border: 2px solid #FFA500; padding: 10px; }
        .predict-btn { background-color: #1E90FF; color: white; font-weight: bold; border-radius: 8px; margin-top: 5px; }
        .submit-btn { background-color: #32CD32; color: white; font-weight: bold; border-radius: 8px; margin-top: 5px; }
        .pred-table table { border: 2px solid #ccc; border-radius: 5px; }
        .feedback-label { color: darkgreen; font-weight: bold; margin-top: 5px; }
        .center-title { margin-bottom: 0px; }
    </style>
    """)

    # Main title
    gr.Markdown("<h1 style='text-align:center; color: darkblue;'>📰 News Classifier</h1>", elem_classes="center-title")
    gr.Markdown("<p style='text-align:center; color: gray;'>Select a sample article or enter your own to see <b>Top-3 predictions</b> and submit your feedback!</p>")

    # Dropdown for sample articles
    article_selector = gr.Dropdown(
        choices=list(sample_articles.keys()),
        label="Select Sample Article",
        value=list(sample_articles.keys())[0]
    )

    # Input textbox
    news_input = gr.Textbox(
        lines=6,
        label="📝 News Article",
        placeholder="Paste or type a news article here...",
        elem_classes="input-box"
    )

    # When dropdown changes, update textbox
    article_selector.change(
        lambda x: sample_articles[x],
        inputs=article_selector,
        outputs=news_input
    )

    predict_btn = gr.Button(
        "Predict 🟢",
        elem_classes="predict-btn"
    )

    # Outputs section
    prediction_chart = gr.Plot(label="📊 Confidence Chart")  # Plotly figure
    predictions = gr.Dataframe(
        headers=["Category", "Confidence"],
        datatype=["str", "str"],
        row_count=3,
        col_count=2,
        label="Top 3 Predictions",
        elem_classes="pred-table"
    )
    feedback_box = gr.Textbox(
        lines=2,
        label="💬 Your Feedback",
        placeholder="Write the correct category or any notes...",
        elem_classes="feedback-box"
    )
    submit_btn = gr.Button(
        "Submit Feedback ✅",
        elem_classes="submit-btn"
    )
    feedback_msg = gr.Label(label="Status", elem_classes="feedback-label")

    # Function bindings
    predict_btn.click(
        fn=predict_news,
        inputs=news_input,
        outputs=[prediction_chart, predictions]
    )

    submit_btn.click(
        fn=save_feedback,
        inputs=[news_input, predictions, feedback_box],
        outputs=feedback_msg
    )

if __name__ == "__main__":
    demo.launch()
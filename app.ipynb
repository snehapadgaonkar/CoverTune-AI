# Install necessary packages
!pip install -q gradio transformers diffusers accelerate torch torchvision torchaudio

# Imports
import gradio as gr
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import tempfile
import os

# Load summarizer and sentiment analyzer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Analyze lyrics for theme
def analyze_lyrics(lyrics):
    summary = summarizer(lyrics, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    sentiment = sentiment_analyzer(summary)[0]['label']
    mood = "uplifting" if sentiment == "POSITIVE" else "melancholic"
    return f"{mood} theme about: {summary}"

# Transcription stub (replace with real model if needed)
def transcribe_audio(audio_path):
    return "Stub: extracted lyrics from audio. Replace this with real transcription."

# Generate image using Stable Diffusion
def generate_image(prompt, style):
    style_prefix = "a highly detailed realistic painting of" if style == "Realistic" else "an animated album cover showing"
    full_prompt = f"{style_prefix} {prompt}"
    image = pipe(full_prompt).images[0]

    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, "cover.png")
    image.save(image_path)
    return image_path

# Main logic
def create_cover(input_type, lyrics_input, audio_file, style):
    if input_type == "Paste Lyrics":
        lyrics = lyrics_input
    elif input_type == "Upload Audio" and audio_file:
        lyrics = transcribe_audio(audio_file)
    else:
        return None, "Please provide valid input."

    analysis = analyze_lyrics(lyrics)
    img = generate_image(analysis, style)
    return img, f"Prompt used: {analysis}"

# Gradio UI
def interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎵 Lyrics to Album Cover Art Generator")

        input_type = gr.Radio(["Paste Lyrics", "Upload Audio"], value="Paste Lyrics", label="Input Type")
        lyrics_input = gr.Textbox(label="Paste Lyrics Here", lines=6, visible=True)
        audio_file = gr.Audio(label="Upload Song Audio", type="filepath", visible=False)
        style = gr.Dropdown(["Animated", "Realistic"], label="Art Style", value="Animated")

        generate_btn = gr.Button("Generate Album Cover")
        output_image = gr.Image(label="Generated Cover Art")
        prompt_text = gr.Textbox(label="Prompt Used", interactive=False)

        def toggle_visibility(choice):
            return gr.update(visible=(choice == "Paste Lyrics")), gr.update(visible=(choice == "Upload Audio"))

        input_type.change(fn=toggle_visibility, inputs=[input_type], outputs=[lyrics_input, audio_file])
        generate_btn.click(fn=create_cover, inputs=[input_type, lyrics_input, audio_file, style], outputs=[output_image, prompt_text])

    return demo

app = interface()
app.launch(share=True)

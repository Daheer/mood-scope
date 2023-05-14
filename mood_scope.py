import cv2
import numpy as np
import tempfile
import time
import streamlit as st
from PIL import Image 
from io import BytesIO
import plotly.graph_objs as go
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"

DEMO_IMAGE = 'demo.jpg'

EMOTION_DICT = {
  0: ['Angry', '😡'],
  1: ['Disgusted', '🤢'],
  2: ['Fearful', '😨'],
  3: ['Happy', '😃'],
  4: ['Neutral', '😐'],
  5: ['Sad', '☹️'],
  6: ['Surprised', '😮']
}

device = 'cpu'

@st.cache_data
def load_model():
  processor = CLIPProcessor.from_pretrained(MODEL_ID)
  model = CLIPModel.from_pretrained(MODEL_ID)
  return processor, model

@st.cache_data
def load_token_embds():
  emotions = list(EMOTION_DICT.values())
  desc = [f'a photo of a {emotion[0]} person' for emotion in emotions]
  tok = processor(text = desc, return_tensors = 'pt', images = None, padding = True).to(device)
  tok_emb = model.get_text_features(**tok)
  tok_emb = tok_emb.detach().cpu().numpy() / np.linalg.norm(tok_emb.detach().cpu().numpy(), axis=0)
  return tok_emb

st.set_page_config(page_title="Mood Scope", page_icon="🎭")
st.title('Mood-Scope')
st.sidebar.title('Options')

app_mode = st.sidebar.selectbox('Choose Page', ['About the App', 'Run Mood Scope'])

st.markdown(
      """
      <style>
        [data-testid = 'stSidebar'][aria-expanded = 'true'] > div:first-child{
          width: 350px
        }
        [data-testid = 'stSidebar'][aria-expanded = 'false'] > div:first-child{
          width: 350px
          margin-left: -350px
        }
      </style>
      """, unsafe_allow_html = True
)

if app_mode == 'About the App':
  st.markdown('Will edit this later!!')

elif app_mode == 'Run Mood Scope':
  
  processor, model = load_model()

  st.sidebar.markdown('---')
  
  with st.columns(3)[1]:
    kpi = st.markdown('**Dominant Detected Emotion**')
    emotion_emoji = st.markdown('-')
    #emotion_text = st.markdown('-')

  img_file_buffer = st.sidebar.file_uploader('Upload an Image', type = ['jpg', 'png', 'jpeg'])
  if img_file_buffer:
    buffer = BytesIO(img_file_buffer.read())
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
  else:
    demo_image = DEMO_IMAGE
    image = cv2.imread(demo_image, cv2.IMREAD_COLOR)
  
  st.sidebar.text('Original Image')
  st.sidebar.image(image, channels = 'BGR')

  im_proc = processor(images=image, return_tensors='pt')['pixel_values']
  im_emb = model.to(device).get_image_features(im_proc.to(device))
  im_emb = im_emb.detach().cpu().numpy()

  tok_emb = load_token_embds()
  score = np.dot(im_emb, tok_emb.T)
  
  output_emoji = EMOTION_DICT[score.argmax(axis = 1).item()][1]
  output_text = EMOTION_DICT[score.argmax(axis = 1).item()][0]

  emotion_emoji.write(f'<h1> {output_emoji} </h1>', unsafe_allow_html = True)
  
  categories = [emotion[0] for emotion in EMOTION_DICT.values()]
  data = list(map(int, (100 * (score / score.sum())).squeeze()))

  trace = go.Scatterpolar(r = data, theta = categories, fill = 'toself', name = 'Emotions')

  layout = go.Layout(
      polar = dict(
          radialaxis = dict(
              visible = False,
              range = [0, 50]
          )
      ),
  )
  fig = go.Figure(data=[trace], layout=layout)
  st.plotly_chart(fig, use_container_width=True)

  #emotion_text.write(f'**{output_text}**')

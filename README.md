# Mood Scope
Mood Scope detects emotions of people in an image using deep learning

# Installation
- Clone this repo ` git clone https://github.com/Daheer/mood-scope.git `
- Install requirements ` pip insatll requirements.txt `
- Launch streamlit app ` streamlit run mood_scope.py `

# Usage

The 'Run Mood Scope' section of the app lets you upload any image. After doing so, it analyzes and detects the mood of the person in the picture. 

The app displays the detected dominant emotion with a suitable emoji. 

It also displays the distribution of the moods using a spider chart. The higher the point, the stronger the presence of that emotion in the image.

### Emotion-emoji guide

| Emotion    | Emoji      |
|------------|------------|
| Angry      | 😡         |
| Disgusted  | 🤢         |
| Fearful    | 😨         |
| Happy      | 😃         |
| Neutral    | 😐         |
| Sad        | ☹️         |
| Surprised  | 😮         |



The app is available and can be accessed via two platforms
- [`Hugging Face Spaces`](https://huggingface.co/spaces/deedax/mood-scope)
- [`Render`](https://mood-scope.onrender.com/)

# Features

- Image upload
- Emotion detection 
- Spider chart display
- Emotion intensity analysis

# Built Using
- [Python](https://python.org)
- [PyTorch](https://pytorch.org)
- [OpenAI CLIP](https://openai.com/research/clip)
- [Streamlit](https://streamlit.io/)
    
# Details

Face facts achieves zero-shot image classification using CLIP. CLIP can be a powerful tool for image classification because it allows you to leverage both visual and language information to classify images. This even means no dataset was used for any training or finetuning. 

First, the emotions (angry, fearful, sad, neutral etc.) were organized using a template to create natural language descriptions for the images. Each emotion was transformed into a template phrase "a photo of a {emotion} person," where {emotion} is one of the emotions in the list. The text descriptions were then tokenized to generate text embeddings that can be processed by the CLIP model.

The image was preprocessed using the CLIPProcessor, which includes resizing and normalization. This prepares the image for feature extraction. The CLIP model then computes features for the image to generate image embeddings that capture the visual features of the image.

To calculate the similarity between each description and the image, a dot product is performed between the image embeddings and text embeddings. This results in a score that indicates how similar the description is to the image. The score is then used to classify the image into one of the emotion categories.

# Contact

Dahir Ibrahim (Deedax Inc) <br>
Email - dahiru.ibrahim@outlook.com <br>
Twitter - https://twitter.com/DeedaxInc <br>
YouTube - https://www.youtube.com/@deedaxinc <br>
Project Link - https://github.com/Daheer/mask-check


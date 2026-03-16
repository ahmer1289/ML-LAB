import os
from pathlib import Path
import uuid
import wave
import struct
import numpy as np
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

try:
    os.environ['USE_TF'] = '0'
    os.environ['USE_TORCH'] = '1'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
   
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
except:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / 'static' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


_sentiment_pipeline = None
_qa_pipeline = None
_text_gen_pipeline = None
_translation_pipeline = None
_cnn_model = None

img_size = (128, 128)



def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            from transformers import pipeline
            _sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Sentiment pipeline error: {e}")
    return _sentiment_pipeline


def _get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            from transformers import pipeline
            _qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        except Exception as e:
            print(f"QA pipeline error: {e}")
    return _qa_pipeline


def _get_text_gen_pipeline():
    global _text_gen_pipeline
    if _text_gen_pipeline is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            from transformers import pipeline
            _text_gen_pipeline = pipeline("text-generation", model="distilgpt2")
        except Exception as e:
            print(f"Text generation pipeline error: {e}")
    return _text_gen_pipeline


def _get_translation_pipeline():
    global _translation_pipeline
    if _translation_pipeline is None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                from transformers import pipeline
                _translation_pipeline = pipeline("translation_en_to_ur", model="Helsinki-NLP/opus-mt-en-ur")
        except Exception as e:
            print(f"Translation pipeline error: {e}")
            print("Translation will use stub responses")
    return _translation_pipeline


def get_cnn_model():

    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    try:
        import pickle
        model_path_pkl = BASE_DIR / 'models' / 'cnn_model.pkl'
        model_path_keras = BASE_DIR / 'models' / 'cnn_model.keras'
        model_path_h5 = BASE_DIR / 'models' / 'cnn_model.h5'

        if model_path_pkl.exists():
            with open(str(model_path_pkl), 'rb') as f:
                _cnn_model = pickle.load(f)
            print(f"CNN model loaded from {model_path_pkl}")
        elif model_path_keras.exists():
            import tensorflow as tf
            _cnn_model = tf.keras.models.load_model(str(model_path_keras))
            print(f"CNN model loaded from {model_path_keras}")
        elif model_path_h5.exists():
            import tensorflow as tf
            _cnn_model = tf.keras.models.load_model(str(model_path_h5))
            print(f"CNN model loaded from {model_path_h5}")
        else:
            print(f"CNN model not found at {model_path_pkl}, {model_path_keras}, or {model_path_h5}")
    except ImportError as e:
        print(f"Import error loading CNN model: {e}")
    except Exception as e:
        print(f"CNN model load error: {e}")
    return _cnn_model


def classify_image(image_path: str):

    try:
        model = get_cnn_model()
        if model is None:
            return {'label': 'unknown', 'score': 0.0, 'note': 'Model not loaded. Train and save model first.'}

        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        arr = np.array(img) / 255.0
        arr = arr.reshape((1, img_size[0], img_size[1], 3))
        preds = model.predict(arr)
        label = int(preds[0].argmax())
        score = float(preds[0].max())
        return {
            'label': 'male' if label == 1 else 'female',
            'score': score,
            'raw_predictions': [float(p) for p in preds[0]]
        }
    except Exception as e:
        return {'error': str(e), 'label': 'unknown'}


def generate_text(prompt: str):
    try:
        pipeline = _get_text_gen_pipeline()
        if pipeline is None:
            return prompt + ' (model not loaded)'

        result = pipeline(
            prompt, 
            max_new_tokens=60,          
            do_sample=True,              
            top_k=50,                    
            top_p=0.95,                  
            temperature=0.8,            
            repetition_penalty=1.2,      
            no_repeat_ngram_size=2,      
            truncation=True,
            pad_token_id=50256          
        )
        
        full_text = result[0]['generated_text']
        
        if "." in full_text:
            full_text = full_text[:full_text.rfind(".")+1]
            
        return full_text
    except Exception as e:
        return f"Error: {str(e)}"

def translate_en_to_ur(text: str):

    try:
        pipeline = _get_translation_pipeline()
        if pipeline is None:
            return text + ' (model not loaded)'
        
        result = pipeline(
            text, 
            max_length=512,         
            num_beams=5,            
            repetition_penalty=1.2, 
            early_stopping=True,   
            truncation=True
        )
        
        return result[0]['translation_text']
    except Exception as e:
        return f"Error: {str(e)}"


def speech_to_text(audio_path: str):
    try:
        import os
        import speech_recognition as sr
        from pydub import AudioSegment

        audio_path = os.path.abspath(audio_path)
        print("STT received path:", audio_path)
        print("Exists:", os.path.exists(audio_path))

        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        recognizer = sr.Recognizer()

        audio = AudioSegment.from_file(audio_path)
        audio = (
            audio
            .set_frame_rate(16000)
            .set_channels(1)
            .set_sample_width(2)
        )

        wav_path = os.path.join(
            UPLOAD_DIR,
            f"temp_{uuid.uuid4().hex}.wav"
        )

        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        return recognizer.recognize_google(audio_data)

    except Exception as e:
        print(f"Speech-to-text error: {e}")
        return None


def sentiment_from_text(text: str):

    try:
        pipeline = _get_sentiment_pipeline()
        if pipeline is None:
            return {'label': 'NEUTRAL', 'score': 0.0, 'note': 'Model not loaded'}
        result = pipeline(text)
        return {
            'label': result[0]['label'],
            'score': result[0]['score']
        }
    except Exception as e:
        return {'error': str(e)}


def answer_question(question: str, context: str = None):

    try:
        pipeline = _get_qa_pipeline()
        if pipeline is None:
            return "Model not loaded"

       
        if context is None:
            context = (
            "The solar system consists of the Sun and everything bound to it by gravity. "
            "There are eight major planets, with Jupiter being the largest and Mercury being the smallest. "
            "Mars is often called the Red Planet because of iron oxide on its surface. "
            "The Milky Way galaxy contains billions of stars and is part of a local group of galaxies "
            "extending millions of light-years across the observable universe."
            )

        result = pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"


def text_to_speech(text: str):
    try:
        from gtts import gTTS

        if not text or len(text.strip()) == 0:
            print("Warning: Empty text for TTS")
            text = "No response generated"

        print(f"Generating speech for: {text[:50]}...")
        out_path = UPLOAD_DIR / f"tts_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(str(out_path))
        print(f"TTS saved to {out_path}")
        return str(out_path)
    except ImportError as e:
        print(f"gTTS not installed: {e}")
        # Create fallback silent WAV
        out_path = UPLOAD_DIR / f"tts_{uuid.uuid4().hex}.wav"
        framerate = 16000
        duration_seconds = 1
        nframes = int(framerate * duration_seconds)
        with wave.open(str(out_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            for _ in range(nframes):
                wf.writeframes(struct.pack('<h', 0))
        return str(out_path)
    except Exception as e:
        print(f"TTS error: {e}")
        # Create fallback silent WAV
        out_path = UPLOAD_DIR / f"tts_{uuid.uuid4().hex}.wav"
        framerate = 16000
        duration_seconds = 1
        nframes = int(framerate * duration_seconds)
        with wave.open(str(out_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            for _ in range(nframes):
                wf.writeframes(struct.pack('<h', 0))
        return str(out_path)

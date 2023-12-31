

import os
import torch
#from google.colab import drive
#drive.mount('/content/drive')
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "jagadeeshjagat/whisper-small-mr"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline("automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="mr", task="transcribe")

#print('Transcription: ', pipe(sample)["text"])   ##सत्यवती अंबालिकेला व्यासांना बघितल्यावर डोळे मिटू नकोस असे सांगते

def speech_to_text(Filename):
  ##path='/content/drive/MyDrive/samples/'+str(Filename)
  prediction=pipe(Filename)['text']
  return prediction

#speech_to_text("common_voice_mr_27591986.wav")
transcribed_text=speech_to_text("./common_voice_mr_27703240.wav")
print(transcribed_text)
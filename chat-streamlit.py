
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import pandas as pd
import speech_recognition as sr
import numpy as np
from pydub import AudioSegment
from st_audiorec import st_audiorec
import numpy as np 
import os



st.title('Voice Ordering System')


wav_audio_data = st_audiorec()
recognizer = sr.Recognizer()


if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'my_recorder_output' not in st.session_state:
    st.session_state.my_recorder_output = {}



def recognize_audio_from_wav(wav_filename):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_filename) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        # Record the audio data
        audio_data = recognizer.record(source)  # Read the entire audio file

    # Recognize the speech in the audio
    try:
        # Using Google Web Speech API
        text = recognizer.recognize_google(audio_data, language='ar-EG')  # Change language as needed
        print("Recognized text:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")



audio_file_path = os.path.join(os.getcwd(),"recorded_audio.wav")
if wav_audio_data:
    #result = recognizer.adjust_for_ambient_noise(wav_audio_data)
    #audio = recognizer.listen(wav_audio_data)
    audio_segment = AudioSegment(
        data=wav_audio_data,
        sample_width=4,  # 16-bit audio = 2 bytes per sample
        frame_rate=44100,  # Assuming 44100 Hz sample rate
        channels=1  # Mono
    )
    audio_segment.export(audio_file_path, format="wav")
 
    try:
        text = recognize_audio_from_wav(audio_file_path)
        prompt = """
            You are part of a system that receives Arabic speech input, converts it into text, and processes the extracted text. Your task is to structure the text you receive into a dictionary like the example below format with specific keys: 'item_name', 'quantity', 'weight', 'notes'.

            Here's an example of the structure you should follow:
            [
                {{
                    "Item": "كشري",
                    "Brand" : "هند"
                    "Quantity": 3,
                    "Weight": null,
                    "Notes": "سعر 25"
                }},
                {{
                    "Item": "دجاج",
                    "Brand" : "ماجي"
                    "Quantity": 1,
                    "Weight": "كيلو و نص",
                    "Notes": null
                }}
            ]
            ----------------------------------------------------------
            NOTES:
                item -> هو اسم الصنف اللي هيطلبه العميل.
                brand -> هو اسم الشركه المنتجه لهذا الصنف .
                quantity -> العدد المطلوب من الصنف.
                weight -> وزن الصنف المطلوب.
                notes -> الملحوظات حول الصنف زي مثلا شطة زيادة.
            
            Given the text below, identify and extract relevant details to fill in the values for 'item_name', 'quantity', 'weight', and 'notes'. If any information is missing or unclear, leave the corresponding value as null.

            Here is the text you will work with and you have to structure it well : {text}

            - Ensure that the values match the provided speech content for each key. If any of the information is missing, leave the corresponding value as null.
            - Ensure that the item_name and the brand are written correctly in native arabic. 
                for example :
                    1) روز = أرز
                    2) فرخات  =  دجاج
                    3) مايه = مياه
                    4) شعريه = شعيرة
                    5) ازايز = قارورة
            - Ensure that you write in the item in a singular name not a plural names, for example:
                1) بيضات = بيضة
                2) فرخات = دجاج  
            - If the quantity is not mention, the default is 1.
            IMPORTANT: Do not provide any additional thoughts; only output the organized data in the array of dictionary without any qoutes or backticks before or after it
            """
                    
        prompt_template = PromptTemplate(input_variables=["text"], template=prompt)
        google_api_key = 'AIzaSyBwNNnYm_4TqjwuFZHytOdgmBE1U0dzke4'
        model = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key, temperature=0)
            
        #Create chain for the LLM and pass the recognized text
        chain = prompt_template | model | StrOutputParser()
        llm_answer = chain.invoke({'text':text})
        print("llm response")
        print(llm_answer)
        cleaned_string = llm_answer.strip("```json\n")
       
        data = json.loads(cleaned_string)
       
        df = pd.DataFrame(data)
        st.write(text)

        st.table(df)
    except sr.UnknownValueError:
        st.write("Could not understand the audio.")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
os.remove(audio_file_path)
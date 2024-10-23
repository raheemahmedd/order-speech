import streamlit as st
import speech_recognition as sr
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# Initialize Streamlit app
st.title("Arabic Speech to Text & Information Extraction")

# Initialize the recognizer


if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Process audio if microphone is used
use_microphone = st.button("Use Microphone")
from st_audiorec import st_audiorec



# if use_microphone:
#     st.write("Recording using microphone...")
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source)  # Reduces background noise
#         st.write("Please say something in Arabic...")
#         try:
#             audio = recognizer.listen(source)
#             # Recognize speech using Google Web Speech API (Arabic language)
#             text = recognizer.recognize_google(audio, language="ar-EG")
#             print(text)
            
#             # Append user input to conversation
#             st.session_state.conversation.append({"role": "user", "content": text})
            
#             # Define the model and prompt template
#             google_api_key = 'AIzaSyBwNNnYm_4TqjwuFZHytOdgmBE1U0dzke4'
#             model = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key, temperature=0)
            
#             # interactive_prompt = """
#             # You are an interactive assistant designed to engage users in meaningful conversations. Your goal is to provide helpful information while encouraging the user to elaborate on their thoughts.

#             # 1. Start by greeting the user warmly and asking how you can assist them today.
#             # 2. If the user asks a question, provide a detailed answer, and then ask a follow-up question related to their inquiry to deepen the conversation.
#             # 3. If the user shares an opinion or experience, acknowledge it and ask for more details to encourage them to share further.
#             # 4. Use phrases like "That's interesting!" or "Can you tell me more about that?" to show enthusiasm.
#             # 5. Offer suggestions or options when appropriate, asking the user for their preferences.
#             # 6. Keep the tone friendly, supportive, and engaging, aiming to create a dialogue rather than a simple Q&A.

#             # Here’s an example interaction:

#             # User: "I'm interested in learning more about machine learning."
#             # Bot: "That's a fascinating field! Are you looking for resources, or do you have specific topics in mind? For instance, are you more interested in supervised learning, unsupervised learning, or perhaps deep learning?"

#             # Now, let's start the conversation! How can I assist you today?
            
#             # ----------------------
#             # user_input : {text}
#             # """
#             prompt="""
#             You are part of a system that receives Arabic speech input, converts it into text, and processes the extracted text. Your task is to structure the text you receive into a dictionary format with specific keys: 'item_name','quantity','weight','notes'. 
#             Here’s an example of the structure you should follow:
#             [{{item_name:"كشري",
#                 quantity:3,
#                 "weight":null,
#                 "notes":"سعر 25"
#                 }},
#                 {{"item_name":" فرخه",
#                 "quantity":1,
#                 "weight":"كيلو و نص",
#                 "notes":null
#                 }}
                
#                 ]
            
#             NOTES:
#                 1- item_name -> هو اسم الصنف اللي هيطلبه العميل .
#                 2- quantity -> العدد المطلوب من الصنف .
#                 3- weight -> وزن الصنف المطلوب .
#                 4- notes -> الملحوظات حول الصنف زي مثلا شطة زيادة .

#             --------------------------
#             Given the text, identify and extract relevant details to fill in the values for 'item_name', 'quantity', 'weight', and 'notes'. If any information is missing or unclear, leave the corresponding value as None. 
#             Here is the text you will work with: {text}
#             Ensure that the values match the provided speech content for each key. If any of the information is missing, leave the corresponding value as None.
            
            
#             VERY IMPORTANT:
#             Dont give me any additional thoughts , you have only to well organize the the output as df only.
            
#             """
                    
#             prompt_template = PromptTemplate(input_variables=["text"], template=prompt)
            
#             #Create chain for the LLM and pass the recognized text
#             chain = prompt_template | model | StrOutputParser()
#             llm_answer = model.invoke(text)
#             st.session_state.conversation.append({"role": "assistant", "content": llm_answer})

#         except sr.UnknownValueError:
#             st.error("Could not understand the audio.")
#         except sr.RequestError as e:
#             st.error(f"Could not request results from the speech recognition service; {e}")


# for message in st.session_state.conversation:
#     is_arabic = message["content"][0].isascii() is False  # Check if the first character is non-ASCII (likely Arabic)
#     alignment = "right" if is_arabic else "left"  # Align right for Arabic, left for English
#     with st.chat_message(message["role"]):
#         st.markdown(
#             f"<div style='text-align: {alignment};'>{message['content']}</div>",
#             unsafe_allow_html=True
#         )
import wave  
import numpy as np 
from pydub import AudioSegment
import os
wav_audio_data = st_audiorec()
recognizer = sr.Recognizer()
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np


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

def audiosegment_to_speech_recognition_data(audio_segment):
    # Convert AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    
    # Normalize the audio samples to float32 between -1 and 1
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))  # Reshape if stereo
    normalized_samples = samples.astype(np.float32) / np.iinfo(audio_segment.array_type).max
    
    # Convert to bytes
    audio_bytes = normalized_samples.tobytes()
    
    # Create an AudioData object
    audio_data = sr.AudioData(audio_bytes, audio_segment.frame_rate, audio_segment.channels * 2)
    return audio_data
import json
import pandas as pd
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
    audio_data = audiosegment_to_speech_recognition_data(audio_segment=audio_segment)
    print("--------------------------------------------------")
    print(type(audio_data))
    # Export to a temporary WAV file
    # with sr.AudioFile(audio_file_path) as source:
    #     recognizer.adjust_for_ambient_noise(source)
    #     audio = recognizer.record(source)  # Use record instead of listen

#     # Try recognizing the speech
    try:
        text = recognize_audio_from_wav(audio_file_path)
        prompt = """
            You are part of a system that receives Arabic speech input, converts it into text, and processes the extracted text. Your task is to structure the text you receive into a dictionary like the example below format with specific keys: 'item_name', 'quantity', 'weight', 'notes'.

            Here's an example of the structure you should follow:

            
            [
                {{
                    "item_name": "كشري",
                    "quantity": 3,
                    "weight": null,
                    "notes": "سعر 25"
                }},
                {{
                    "item_name": "فرخه",
                    "quantity": 1,
                    "weight": "كيلو و نص",
                    "notes": null
                }}
            ]
            ----------------------------------------------------------
            NOTES:

            item_name -> هو اسم الصنف اللي هيطلبه العميل.
            quantity -> العدد المطلوب من الصنف.
            weight -> وزن الصنف المطلوب.
            notes -> الملحوظات حول الصنف زي مثلا شطة زيادة.
            Given the text below, identify and extract relevant details to fill in the values for 'item_name', 'quantity', 'weight', and 'notes'. If any information is missing or unclear, leave the corresponding value as null.

            Here is the text you will work with and you have to structure it well : {text}

            Ensure that the values match the provided speech content for each key. If any of the information is missing, leave the corresponding value as null.

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
        #print("cleaned_string")
        #print(cleaned_string)
        data = json.loads(cleaned_string)
        print("before data")
        print(data)
        df = pd.DataFrame(data)
        #print(df)
        #st.write(text)
        #st.write()
        st.write(text)

        st.table(df)
    except sr.UnknownValueError:
        st.write("Could not understand the audio.")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
#os.remove(audio_file_path)
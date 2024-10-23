import streamlit as st
import speech_recognition as sr
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import json
import sounddevice as sd
import pandas as pd
from double_function_btn import AudioRecorder 
# Initialize Streamlit app
st.title("Arabic Speech to Text & Information Extraction")

# Initialize the recognizer
recognizer = sr.Recognizer()

# Fixed audio file path
#fixed_audio_file = "E://Electro Pi//sury//record_out.wav"

# Display fixed audio template
#st.write("The template of record...")
#st.audio(fixed_audio_file, format="audio/wav")

# Button to use the microphone

# user_input = st.chat_input("Please enter your question here")
# if user_input:
    
# st.session_state.conversation = []
# Create session state for conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Process audio if microphone is used
use_microphone = st.button("Use Microphone")
if use_microphone:
    st.write("Recording using microphone...")
    recorder = AudioRecorder()

    # Start recording
    
    recorder.start_recording()

    # Wait for user input to stop recording
    stop = st.button("Stop Recording")
    # Stop recording
    if stop:
        recorder.stop_recording()
        # Save the recorded audio
        recorder.save_audio("recorded_audio.wav")
  
        try:
            audio_data = recorder.get_audio_as_data()
            if audio_data:
                print("Audio data recorded successfully.")
            else:
                print("No audio data was recorded.")
            recognizer = sr.Recognizer()
            text = recognizer.recognize_google(audio_data, language="ar-EG")
            print(text)
            st.session_state.conversation.append({"role": "user", "content": text})
            google_api_key = 'AIzaSyBwNNnYm_4TqjwuFZHytOdgmBE1U0dzke4'
            model = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key, temperature=0)
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
            st.chat_message("user").markdown(text)

            st.table(df)
            
            #st.session_state.conversation.append({"role": "assistant", "content": df})

        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from the speech recognition service; {e}")
   
    # 
# Display conversation
# if st.session_state.conversation:
#     for message in st.session_state.conversation:
#         if message["role"] == "user":
# else:
#     st.write("Start a conversation by using the microphone.")

# for message in st.session_state.conversation:
#     is_arabic = message["content"][0].isascii() is False  # Check if the first character is non-ASCII (likely Arabic)
#     alignment = "right" if is_arabic else "left"  # Align right for Arabic, left for English
#     with st.chat_message(message["role"]):
#         st.markdown(
#             f"<div style='text-align: {alignment};'>{message['content']}</div>",
#             unsafe_allow_html=True
#         )

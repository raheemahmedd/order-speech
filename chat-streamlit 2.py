import streamlit as st
import speech_recognition as sr
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import json
import pandas as pd

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
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Reduces background noise
        st.write("Please say something in Arabic...")
        try:
            audio = recognizer.listen(source,phrase_time_limit=60,timeout=10)
            # Recognize speech using Google Web Speech API (Arabic language)
            text = recognizer.recognize_google(audio, language="ar-EG")
            #print(text)
            
            # Append user input to conversation
            st.session_state.conversation.append({"role": "user", "content": text})
            
            # Define the model and prompt template
            google_api_key = 'AIzaSyBwNNnYm_4TqjwuFZHytOdgmBE1U0dzke4'
            model = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key, temperature=0)
            
            # interactive_prompt = """
            # You are an interactive assistant designed to engage users in meaningful conversations. Your goal is to provide helpful information while encouraging the user to elaborate on their thoughts.

            # 1. Start by greeting the user warmly and asking how you can assist them today.
            # 2. If the user asks a question, provide a detailed answer, and then ask a follow-up question related to their inquiry to deepen the conversation.
            # 3. If the user shares an opinion or experience, acknowledge it and ask for more details to encourage them to share further.
            # 4. Use phrases like "That's interesting!" or "Can you tell me more about that?" to show enthusiasm.
            # 5. Offer suggestions or options when appropriate, asking the user for their preferences.
            # 6. Keep the tone friendly, supportive, and engaging, aiming to create a dialogue rather than a simple Q&A.

            # Here’s an example interaction:

            # User: "I'm interested in learning more about machine learning."
            # Bot: "That's a fascinating field! Are you looking for resources, or do you have specific topics in mind? For instance, are you more interested in supervised learning, unsupervised learning, or perhaps deep learning?"

            # Now, let's start the conversation! How can I assist you today?
            
            # ----------------------
            # user_input : {text}
            # """
            # prompt="""
            # You are part of a system that receives Arabic speech input, converts it into text, and processes the extracted text. Your task is to structure the text you receive into a dictionary format with specific keys: 'item_name','quantity','weight','notes'. 
            # Here’s an example of the structure you should follow:
            # [{{item_name:"كشري",
            #     quantity:3,
            #     "weight":null,
            #     "notes":"سعر 25"
            #     }},
            #     {{"item_name":" فرخه",
            #     "quantity":1,
            #     "weight":"كيلو و نص",
            #     "notes":null
            #     }}
                
            #     ]
            
            # NOTES:
            #     1- item_name -> هو اسم الصنف اللي هيطلبه العميل .
            #     2- quantity -> العدد المطلوب من الصنف .
            #     3- weight -> وزن الصنف المطلوب .
            #     4- notes -> الملحوظات حول الصنف زي مثلا شطة زيادة .

            # --------------------------
            # Given the text, identify and extract relevant details to fill in the values for 'item_name', 'quantity', 'weight', and 'notes'. If any information is missing or unclear, leave the corresponding value as None. 
            # Here is the text you will work with: {text}
            # Ensure that the values match the provided speech content for each key. If any of the information is missing, leave the corresponding value as None.
            
            
            # VERY IMPORTANT:
            # Dont give me any additional thoughts , you have only to well organize the the output as df only.
            
            # """
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

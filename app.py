# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import time
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64

# Importing modules
import numpy as np
import streamlit as st
import cv2
# ... [other imports] ...


# Add this right after your imports at the very top
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 20px auto;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 25px;
        background-color: white;
    }
    
    /* Bot message */
    .bot-message {
        background-color: #e3f2fd;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        max-width: 70%;
        display: inline-block;
        color: #333;
        font-size: 16px;
    }
    
    /* User message */
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0 8px 30%;
        max-width: 70%;
        display: inline-block;
    }
    
    /* Song cards */
    .song-card {
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 15px 0;
        background-color: #f8f9fa;
        border-radius: 0 8px 8px 0;
        transition: all 0.3s;
    }
    
    .song-card:hover {
        transform: translateX(5px);
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        border: none;
        font-weight: 500;
        transition: all 0.3s;
        font-size: 16px;
        margin: 10px 0;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .header img {
        width: 80px;
        margin-bottom: 10px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #666;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for song recommendations
if 'new_df' not in st.session_state:
    st.session_state.new_df = pd.DataFrame(columns=['name', 'emotional', 'pleasant', 'link', 'artist'])


df = pd.read_csv("C:\\Users\\Devan\\Desktop\\emotion\\ok\\muse_v3.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]
print(df)

df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
print(df)

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):

    data = pd.DataFrame()

    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':             
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)): 
            v = list[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':             
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:      
                data = pd.concat([df_sad.sample(n=t)])


    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data =pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):           
            v = list[i]         
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':           
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':           
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':          
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul
    




model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


model.load_weights('C:\\Users\\Devan\\Desktop\\emotion\\ok\\model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier('C:\\Users\\Devan\\Desktop\\emotion\\ok\\haarcascade_frontalface_default.xml')
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

# Replace everything from page_bg_img to the column definitions with this:

# App header with logo
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.image("https://cdn-icons-png.flaticon.com/512/2721/2721620.png", width=100)
st.markdown("<h1 style='text-align: center; color: #333;'>MoodTunes</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Your personal emotion-based music assistant</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Initialize empty list for emotions
if 'emotion_list' not in st.session_state:
    st.session_state.emotion_list = []
# Replace everything from col1, col2, col3 definition to the end with this:

def convert_spotify_api_url(api_url):
    """Convert Spotify API URL to regular web player URL"""
    if isinstance(api_url, str):
        if 'api.spotify.com/v1/tracks/' in api_url:
            track_id = api_url.split('api.spotify.com/v1/tracks/')[1].split('?')[0]
            return f"https://open.spotify.com/track/{track_id}"
        elif 'spotify:track:' in api_url:
            track_id = api_url.split('spotify:track:')[1]
            return f"https://open.spotify.com/track/{track_id}"
    return api_url
    
# Main chat container
with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Bot greeting message
    st.markdown("<div class='bot-message'>Hi there! 😊 I can recommend music based on your current mood. Click the button below to scan your emotions.</div>", unsafe_allow_html=True)
    

    # Emotion scanning button
    if st.button('🎤 Scan My Emotions', key='scan_button'):
        with st.spinner('Analyzing your emotions...'):
            # Your existing emotion detection code
            count = 0
            st.session_state.emotion_list.clear()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")          
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
                faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)          
                count = count + 1

                for (x, y, w, h) in faces:               
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)               
                    roi_gray = gray[y:y + h, x:x + w]               
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)             
                    prediction = model.predict(cropped_img)             
                    max_index = int(np.argmax(prediction))

                    st.session_state.emotion_list.append(emotion_dict[max_index])

                    cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)              
                    cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
                if count >= 20:
                    break
            
            cap.release()
            cv2.destroyAllWindows()

            # Process emotions and get recommendations
            processed_emotions = pre(st.session_state.emotion_list)
            st.session_state.new_df = fun(processed_emotions)
            
            # Bot response with detected emotions
            emotions = ", ".join(set(processed_emotions))  # Get unique emotions
            st.markdown(f"<div class='bot-message'>I detected these emotions: <b>{emotions}</b>. Here are some perfect song recommendations for you:</div>", unsafe_allow_html=True)
            
            # Display songs in beautiful cards
            for i, row in st.session_state.new_df.head(10).iterrows():
                processed_link = convert_spotify_api_url(row['link'])
                with st.container():
                    st.markdown(f"""
                    <div class='song-card'>
                        <h4>{i+1}. <a href="{processed_link}" target="_blank" style="color: #007bff; text-decoration: none;">{row['name']}</a></h4>
                        <p style="color: #666; margin-bottom: 0;"><i>{row['artist']}</i></p>
                        {"<small>🎵 Play on Spotify</small>" if 'spotify.com' in str(processed_link).lower() else ""}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add confetti celebration
            st.balloons()
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close chat container

# Footer
st.markdown("<div class='footer'>🎶 MoodTunes - Music for every emotion</div>", unsafe_allow_html=True)


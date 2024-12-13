import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
import streamlit as st
import cv2
import face_recognition
import numpy as np
import sqlite3
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize database
conn = sqlite3.connect('faces.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding BLOB)''')
conn.commit()

# Transformer for video frames
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.known_face_encodings = []
        self.load_faces()

    def load_faces(self):
        c.execute("SELECT name, encoding FROM users")
        for name, encoding in c.fetchall():
            self.known_face_encodings.append((name, np.frombuffer(encoding, dtype=np.float64)))

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        for encoding in face_encodings:
            matches = face_recognition.compare_faces([enc for _, enc in self.known_face_encodings], encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_encodings[first_match_index][0]
                cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(img, "Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img

st.title("Face Login App")

option = st.selectbox("Choose Action", ["Login", "Register"])

if option == "Register":
    name = st.text_input("Enter your name")
    if st.button("Register"):
        ctx = webrtc_streamer(key="register", video_transformer_factory=FaceRecognitionTransformer)
        if ctx.video_transformer:
            img = ctx.video_transformer.transform(ctx.video_transformer.frame)
            rgb = img[:, :, ::-1]
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                encoding = encodings[0].tobytes()
                c.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, encoding))
                conn.commit()
                st.success("Registered successfully")
else:
    webrtc_streamer(key="login", video_transformer_factory=FaceRecognitionTransformer)
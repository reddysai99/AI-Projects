import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

print("Initializing imports...")
try:
    import tensorflow as tf
    print("TensorFlow imported successfully")
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
except ImportError as e:
    print(f"TensorFlow import error: {str(e)}")
    print("Please ensure TensorFlow is installed correctly.")
    print("Try running: pip install tensorflow==2.10.0")
    exit(1)

try:
    import cv2
    import numpy as np
    import speech_recognition as sr
    from transformers import pipeline
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    import threading
    import time
    import mediapipe as mp
    import pyttsx3
    print("All other dependencies imported successfully")
except ImportError as e:
    print(f"Import error: {str(e)}")
    print("Please install missing dependencies:")
    print("pip install opencv-python mediapipe==0.10.0 protobuf==3.20.0 speechrecognition pyttsx3 transformers torch pyaudio")
    exit(1)

class AIAssistant:
    def __init__(self):
        try:
            print("Initializing AI Assistant...")
            self.setup_models()
            self.setup_gui()
            print("AI Assistant initialized successfully")
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize: {str(e)}")
            exit(1)
        
    def setup_models(self):
        """Initialize AI models"""
        print("\nInitializing AI models...")
        
        try:
            print("Loading ResNet50 model...")
            self.image_model = ResNet50(weights='imagenet')
            print("✓ ResNet50 model loaded successfully")
            
            print("Loading NLP models...")
            try:
                self.nlp_sentiment = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
                print("✓ NLP models loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load NLP models: {str(e)}")
                print("NLP features will be disabled")
                self.nlp_sentiment = None
            
            print("Initializing video components...")
            self.mp_face_mesh = mp.solutions.face_mesh
            print("✓ Video components initialized successfully")
            
            print("Setting up speech components...")
            self.recognizer = sr.Recognizer()
            self.engine = pyttsx3.init()
            # Configure speech properties
            self.engine.setProperty('rate', 150)    # Speed of speech
            self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            print("✓ Speech components initialized successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
        
        print("✓ All models initialized successfully!")

    def setup_gui(self):
        """Setup the main GUI"""
        self.root = tk.Tk()
        self.root.title("AI Assistant")
        self.root.geometry("800x600")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True)

        # Create tabs
        self.image_tab = ttk.Frame(self.notebook)
        self.text_tab = ttk.Frame(self.notebook)
        self.speech_tab = ttk.Frame(self.notebook)
        self.video_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.image_tab, text="Image Processing")
        self.notebook.add(self.text_tab, text="NLP")
        self.notebook.add(self.speech_tab, text="Speech")
        self.notebook.add(self.video_tab, text="Video")

        self.setup_image_tab()
        self.setup_text_tab()
        self.setup_speech_tab()
        self.setup_video_tab()

    def setup_image_tab(self):
        """Setup image processing interface"""
        # Image upload button
        ttk.Button(
            self.image_tab, 
            text="Upload Image",
            command=self.process_image
        ).pack(pady=10)

        # Result display
        self.image_result = tk.Text(self.image_tab, height=10, width=50)
        self.image_result.pack(pady=10)

    def setup_text_tab(self):
        """Setup NLP interface"""
        # Text input
        self.text_input = tk.Text(self.text_tab, height=5, width=50)
        self.text_input.pack(pady=10)

        # Analysis button
        ttk.Button(
            self.text_tab,
            text="Analyze Text",
            command=self.process_text
        ).pack(pady=10)

        # Result display
        self.text_result = tk.Text(self.text_tab, height=10, width=50)
        self.text_result.pack(pady=10)

    def setup_speech_tab(self):
        """Setup speech interface"""
        ttk.Button(
            self.speech_tab,
            text="Start Recording",
            command=self.process_speech
        ).pack(pady=10)

        self.speech_result = tk.Text(self.speech_tab, height=10, width=50)
        self.speech_result.pack(pady=10)

    def setup_video_tab(self):
        """Setup video analytics interface"""
        ttk.Button(
            self.video_tab,
            text="Start Video",
            command=self.process_video
        ).pack(pady=10)

        self.video_result = tk.Text(self.video_tab, height=10, width=50)
        self.video_result.pack(pady=10)

    def process_image(self):
        """Process uploaded image"""
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                # Load and preprocess image
                img = load_img(file_path, target_size=(224, 224))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # Make prediction
                preds = self.image_model.predict(x)
                results = decode_predictions(preds, top=3)[0]

                # Display results
                self.image_result.delete(1.0, tk.END)
                self.image_result.insert(tk.END, "Predictions:\n\n")
                for pred in results:
                    self.image_result.insert(tk.END, f"{pred[1]}: {pred[2]:.2%}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_text(self):
        """Process input text"""
        try:
            text = self.text_input.get(1.0, tk.END).strip()
            if text:
                # Perform sentiment analysis
                sentiment = self.nlp_sentiment(text)[0]

                # Display results
                self.text_result.delete(1.0, tk.END)
                self.text_result.insert(tk.END, f"Sentiment: {sentiment['label']}\n")
                self.text_result.insert(tk.END, f"Confidence: {sentiment['score']:.2%}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_speech(self):
        """Process speech input"""
        try:
            self.speech_result.delete(1.0, tk.END)
            self.speech_result.insert(tk.END, "Listening...\n")
            
            # Adjust for ambient noise
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.speech_result.insert(tk.END, "Ready! Speak now...\n")
                
                try:
                    # Record audio with timeout
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Convert to text
                    text = self.recognizer.recognize_google(audio)
                    
                    # Display results
                    self.speech_result.insert(tk.END, f"You said: {text}\n")
                    
                    # Respond with speech
                    self.engine.say(f"I heard you say: {text}")
                    self.engine.runAndWait()
                    
                except sr.WaitTimeoutError:
                    self.speech_result.insert(tk.END, "No speech detected within timeout\n")
                except sr.UnknownValueError:
                    self.speech_result.insert(tk.END, "Could not understand audio\n")
                except sr.RequestError as e:
                    self.speech_result.insert(tk.END, f"Could not request results; {e}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_video(self):
        """Process video from webcam"""
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            self.is_video_running = True
            
            def video_loop():
                try:
                    # Initialize face mesh
                    face_mesh = self.mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        static_image_mode=False
                    )
                    
                    while self.is_video_running:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Flip the frame horizontally
                        frame = cv2.flip(frame, 1)
                        
                        # Convert to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process frame
                        results = face_mesh.process(rgb_frame)
                        
                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                # Draw the face mesh
                                mp.solutions.drawing_utils.draw_landmarks(
                                    image=frame,
                                    landmark_list=face_landmarks,
                                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(0, 255, 0), thickness=1, circle_radius=1),
                                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(0, 255, 0), thickness=1)
                                )

                            # Update result text
                            self.video_result.delete(1.0, tk.END)
                            self.video_result.insert(tk.END, 
                                f"Detected {len(results.multi_face_landmarks)} faces\n")

                        # Display frame
                        cv2.imshow('Face Detection', frame)
                        
                        # Break loop on 'q' press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                except Exception as e:
                    print(f"Error in video loop: {str(e)}")
                finally:
                    face_mesh.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    self.stop_video()

            # Add stop button if it doesn't exist
            if not hasattr(self, 'stop_button'):
                self.stop_button = ttk.Button(
                    self.video_tab,
                    text="Stop Video",
                    command=self.stop_video
                )
                self.stop_button.pack(pady=5)

            # Start video processing in a separate thread
            self.video_thread = threading.Thread(target=video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start video: {str(e)}")
            self.stop_video()

    def stop_video(self):
        """Stop video processing"""
        self.is_video_running = False
        cv2.destroyAllWindows()
        if hasattr(self, 'stop_button'):
            self.stop_button.configure(state='disabled')

    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    print("Starting AI Assistant...")
    app = AIAssistant()
    app.run()

if __name__ == "__main__":
    main() 
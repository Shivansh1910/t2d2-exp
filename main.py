from kivy.app import App

# import kivy UI elements
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button

# Import other kivy stuff
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# import other libraries
import os
import cv2

import tensorflow as tf
import numpy as np

from kivy.config import Config

Config.set("kivy", "log_level", "error")
Config.write()


class T2D2(App):
    def build(self):
        # Main Layout
        self.camera = Image(size_hint=(1, 0.8), allow_stretch=True, keep_ratio=True)
        self.button = Button(text="Start", font_size=14, size_hint=(1, 0.1))
        self.verification = Label(text="Started", font_size=14, size_hint=(1, 0.1))

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.camera)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # video capture
        self.capture = cv2.VideoCapture(0)
        
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path="models/yolov8n_float32.tflite")
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Run update function in every 1/30  second
        Clock.schedule_interval(self.update, 1 / 30)

        return layout

    # Run continuously to get the camera frames
    def update(self, *args):
        try:
            # Read the camera frame
            ret, frame = self.capture.read()
            
            # Object detection
            input_data = self.preprocess_input(frame)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process and display the detected objects
            # detected_objects = self.postprocess_output(output_data)
            self.postprocess_output(frame, output_data)

            # # Draw bounding boxes on the frame
            # self.draw_boxes(frame, detected_objects)

            # Flip the frame horizontally
            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
            )
            img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.camera.texture = img_texture
        except Exception as e:
            print("Error : Cannot read frame")
            print("Exception : ", e)
            pass
        
    def preprocess_input(self, frame):
        # Preprocess the input frame for the model
        # Resize the frame to match the input size of the model
        input_size = tuple(self.input_details[0]['shape'][1:3])
        input_data = cv2.resize(frame, input_size)

        # Convert the input data type to FLOAT32
        input_data = input_data.astype(np.float32)


        # Add a batch dimension to the input data
        input_data = np.expand_dims(input_data, axis=0)

        # Normalize the input data if required (adapt this based on your model)
        input_data /= 255.0
        
        return input_data
        
        # # Preprocess the input frame for the TFLite model
        # input_shape = self.input_details[0]['shape']
        # input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
        # input_data = np.expand_dims(input_data, axis=0)
        # input_data = input_data.astype(np.float32) / 255.0
        # return input_data
    
    def postprocess_output(self, frame, output_data):
        # Postprocess the output of the model
                
        for detection in output_data[0]:
            # detection = detection.reshape(1, -1)
            print("Detection dimensions:", detection.shape)
            print("Detection values:", detection)
            
            # Reshape the detection array to a 2D array with shape (num_boxes, num_values_per_box)
            detection = detection.reshape(-1, detection.shape[-1])
            print("Detection dimensions:", detection.shape)
            
            try:     
                print(detection)
                class_id = np.argmax(detection)
                score = detection[0, class_id]
                # class_probs = detection[5:]
                # class_id = np.argmax(class_probs)
                # score = class_probs[class_id]
                print(class_id)
                label = f"Class: {class_id}, Score: {score:.2f}"
                print("label: ", label)
            except Exception as e:
                print("Error : postprocess_output failed", e)
                pass                
        
    def draw_boxes(self, frame, detected_objects):
        # Draw bounding boxes on the frame based on the detected objects
        for obj in detected_objects:
            x, y, w, h = obj[:4]

            # Convert the coordinates to integers
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


if __name__ == "__main__":
    T2D2().run()
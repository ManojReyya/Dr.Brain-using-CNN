from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model('tumor_type.h5')  # Load your trained model
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Class labels

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded!"
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No selected file!"
            else:
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)  # Save uploaded file

                # Preprocess the image
                img = image.load_img(file_path, target_size=(64, 64))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0  # Normalize the image

                # Make prediction
                prediction = np.argmax(model.predict(x))
                result = f"The uploaded image is classified as: {class_labels[prediction]}"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')  # Create uploads folder if it doesn't exist
    app.run(debug=True)

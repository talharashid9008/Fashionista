# from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request, redirect, url_for,jsonify, send_file
import os
import sys
import base64
from werkzeug.utils import secure_filename
# sys.path.append('C:\\Users\\ahmer\\Desktop\FYP1\\model')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))
from model.segmentation import work
from guess_name import predict_random_image
#rom guess_name import run_example
STATIC_DIR = os.path.abspath('static/')
                             
app = Flask(__name__, template_folder='templates', static_folder=STATIC_DIR)


# Set allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg'}


# Set input image folder
INPUT_IMAGE_FOLDER = 'C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\model\\input_images'
app.config['INPUT_IMAGE_FOLDER'] = INPUT_IMAGE_FOLDER

# Set output image folder
OUTPUT_IMAGE_FOLDER = "C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\static\\segmentation_output"
app.config['OUTPUT_IMAGE_FOLDER'] = OUTPUT_IMAGE_FOLDER
# Root page or landing page
@app.route('/')
def index():
    return render_template('upload.html')



def allowed_file(filename):
    print(filename)
    """
    Check if the file type is allowed for upload
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/prediction' , methods=['POST','GET'])
def prediction_result():
    if request.method == 'POST':
            """
            Handle file upload request
            """
            # Check if file is present in the request
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'})

            file = request.files['file']

            # Check if file name is empty
            if file.filename == '':
                return jsonify({'error': 'No file selected'})

            # Check if file type is allowed
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'})
            
            input_image_folder = app.config['INPUT_IMAGE_FOLDER']#C:\Users\Talha\Downloads\FYP-F22-108-D-Fashionista\model\input_images

            # Remove all files in input image folder
            for input_filename in os.listdir(input_image_folder):
                file_path = os.path.join(input_image_folder, input_filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            filename = secure_filename(file.filename)
            file.save(os.path.join(input_image_folder, filename))
            top_classes = predict_random_image(input_image_folder, num_classes=10)
            for i, (class_name, probability) in enumerate(top_classes):
                print("*")
                print(f'Top {i+1} class: {class_name}, probability: {probability:2f}')
            return render_template("prediction_Results.html",top_classes=top_classes)
    return render_template('prediction.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/upload', methods=['POST'])
def upload():
        if request.method == 'POST':
            """
            Handle file upload request
            """
            # Check if file is present in the request
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'})

            file = request.files['file']

            # Check if file name is empty
            if file.filename == '':
                return jsonify({'error': 'No file selected'})

            # Check if file type is allowed
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'})
            
            input_image_folder = app.config['INPUT_IMAGE_FOLDER']#C:\Users\Talha\Downloads\FYP-F22-108-D-Fashionista\model\input_images

            # Remove all files in input image folder
            for input_filename in os.listdir(input_image_folder):
                file_path = os.path.join(input_image_folder, input_filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

            # Remove all files in output image folder
            output_image_folder = app.config['OUTPUT_IMAGE_FOLDER']#C:\Users\Talha\Downloads\FYP-F22-108-D-Fashionista\model\output_images
            for out_filename in os.listdir(output_image_folder):
                file_path = os.path.join(output_image_folder, out_filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

            # Save file to input image folder
            filename = secure_filename(file.filename)
            file.save(os.path.join(input_image_folder, filename))

            # calling model function here
            work()

            # here below test code
            image_files = os.listdir(os.path.join(output_image_folder, output_image_folder))
            # here new
            # for image_name in image_files[1:]:
            #     run_example(image_name)
            # ----------
            image_urls = [url_for('static', filename=f'segmentation_output/{f}') for f in image_files]
            # test code end here

            return render_template("result.html",image_urls=image_urls, show=1)

        return render_template("upload.html",show=1)




# @app.route('/output/<filename>', methods=['GET'])
# def output_image(filename):
#     """
#     Serve output image to frontend
#     """
#     filename = filename.split('.')[0]
#     filename = filename +'.png'
#     # output_image_folder = app.config['INPUT_IMAGE_FOLDER']
#     output_image_folder = app.config['OUTPUT_IMAGE_FOLDER'] 
#     return send_file(os.path.join(output_image_folder, filename), mimetype='image/jpeg')

# @app.route('/output/<directory>', methods=['GET'])
# def output_image(directory):
#     """
#     Serve output images to frontend
#     """
#     output_image_folder = app.config['OUTPUT_IMAGE_FOLDER']
    # image_files = os.listdir(os.path.join(output_image_folder, directory))
    # image_urls = [url_for('static', filename=f'output_images/{directory}/{f}') for f in image_files]
#     return render_template('gallery.html', image_urls=image_urls)


if __name__ == '__main__':
    app.run(debug=True)




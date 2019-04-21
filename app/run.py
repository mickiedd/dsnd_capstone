import os
from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename
from classifier import DogBreedClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '{}/uploads'.format(os.getcwd())
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

html = '''
    <!DOCTYPE html>
    <title>Creature Detector</title>
    <h1>Creature Detector</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=upload>
    </form>
    '''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in set(['png', 'jpg', 'jpeg'])
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            
            c = DogBreedClassifier()
            pred = c.predict("." + file_url)
            
            return html + '<label>' +  pred + '</label><br><img src=' + file_url + '>'
    return html

if __name__ == '__main__':
    server = '0.0.0.0'
    port = 3001
    app.run(host = server,  port = port, debug = True)
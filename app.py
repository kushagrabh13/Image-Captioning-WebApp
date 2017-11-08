import os
import _pickle as cPickle
from flask import Flask, render_template, request
from image_rnn_predict import captioning, andro_cap
import requests

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
path =  os.path.join(APP_ROOT+ "\\static\\img\\test\\")

            
def Image_Caption(image_path):
    return captioning(image_path)

def Android_Caption(image_path):
    return andro_cap(image_path)

@app.route('/')
def index():
    return render_template("upload.html")


@app.route("/android", methods=['GET'])
def android_caption():
    image_andro = request.args.get('image')
    img_data = requests.get("{}".format(image_andro)).content
    with open('static/img/android/test.jpg', 'wb') as handler:
        handler.write(img_data)
    capt = Android_Caption("static/img/android/test.jpg")
    return render_template("android.html", filename = capt)

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    image_path = []
    images = []
    captions = []
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            os.unlink(fp)
        
    for file in request.files.getlist("file"):
        filename = file.filename
        images.append(filename)
        file_path = os.path.join(path + filename)
        file.save(os.path.join(path, filename))
        image_path.append(file_path)
    predicts = Image_Caption([img for img in image_path])

    print(images)
    image_path = ["../static/img/test/" + i for i in images]
    #image_path = ['"{}"' .format(i) for i in image_path]
    print(image_path)
    
    for (image, predict) in zip(image_path, predicts):
        captions.append((image, predict))

##      with open(caption_file, 'wb') as f:
##          cPickle.dump(captions, f)

    display = captions[-5:]

    print (display)
    print(display[0][0])
    return render_template("index.html", image1 = display[0][0], image2 = display[1][0], image3 = display[2][0], image4 = display[3][0],
                           image5 = display[4][0], caption1 = display[0][1], caption2 = display[1][1], caption3 = display[2][1],
                           caption4 = display[3][1], caption5 = display[4][1])


if __name__ == '__main__':
    app.run(port=4555,debug=True)

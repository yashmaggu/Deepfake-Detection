import os
import logging
from flask import Flask,request,Response,redirect,flash,jsonify
from flask_cors import CORS,cross_origin
from PredictModel import PredictModel
from werkzeug.utils import secure_filename
from prometheus_flask_exporter import PrometheusMetrics

os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')


UPLOAD_FOLDER = 'predict_videos/'
ALLOWED_EXTENSIONS = {'mp4'}

logging.basicConfig(level=logging.INFO)
logging.info("Setting LOGLEVEL to INFO")

app= Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
metrics = PrometheusMetrics(app)


# static information as metric
metrics.info('Deepfake_detection_Model', 'Detecting deepfakes', version='1.0.0')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
# @metrics.counter('invocation_by_type', 'Number of invocations by type',
#          labels={'item_type': lambda: request.view_args['file']})
# @metrics.summary('requests_by_status', 'Request latencies by status',
#                  labels={'status': lambda r: r.status_code})
# @metrics.histogram('requests_by_status_and_path', 'Request latencies by status and path',
#                    labels={'status': lambda r: r.status_code, 'path': lambda: request.path})
# @metrics.gauge('in_progress', 'Long running requests in progress')
def prediction():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            y_hat,confi=model.doPrediction()
            y_hat="Real" if y_hat==1  else "Fake"
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(y_hat,confi)
            data={
                "result":y_hat,
                "confidence":confi,
            }
            return jsonify(data)
    #         return f""" <!doctype html>
    #         <title>Result</title>
    #         <h1>Result:{y_hat} with confidence of {confi}</h1>
    #     </form>
    # """
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':#makes sure the server only runs 
    #if the script is executed directly from the Python interpreter and not used as an imported module.
    model=PredictModel()
    app.run(host="0.0.0.0",port=5000)



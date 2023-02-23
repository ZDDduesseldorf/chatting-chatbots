import json
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import transformer
import sentence_processing

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def postprocess(sentence):
    """Process sentence."""

    sentence = sentence_processing.remove_repetitions(sentence)
    sentence = sentence_processing.remove_spaces(sentence)
    sentence = sentence_processing.replace_after_sentence_sign(sentence)
    sentence = sentence_processing.handle_following_appostrophs(sentence)
    sentence = sentence[0].upper() + sentence[1:]
    return sentence


@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    inp = request.form['message']
    response = postprocess(transformer.predict(inp))
    return json.dumps(response)


app.run(port=4000)

from flask import Flask, send_file, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_file('index_updated.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
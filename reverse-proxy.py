from flask import Flask,request,redirect,Response
import requests, os
#RAY_HEAD_IP=os.environ['RAY_HEAD_IP']

with open("RAY_HEAD_IP") as fp:
    RAY_HEAD_IP=fp.read()

app = Flask(__name__)
SITE_NAME = f'http://{RAY_HEAD_IP}:9000/'

@app.route('/')
def index():
    return 'Flask is now running!'
@app.route('/<path:path>',methods=['GET','POST',"DELETE"])
def proxy(path):
    global SITE_NAME
    if request.method=='GET':
        resp = requests.get(f'{SITE_NAME}{path}')
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in  resp.raw.headers.items() if name.lower() not in excluded_headers]
        response = Response(resp.content, resp.status_code, headers)
        return response
    elif request.method=='POST':
        resp = requests.post(f'{SITE_NAME}{path}',json=request.get_json())
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
        response = Response(resp.content, resp.status_code, headers)
        return response
    elif request.method=='DELETE':
        resp = requests.delete(f'{SITE_NAME}{path}').content
        response = Response(resp.content, resp.status_code, headers)
        return response
if __name__ == '__main__':
    from waitress import serve
    serve(app, host="127.0.0.1", port=8100)
    #app.run(debug = False,port=8100)

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ Flask 正常運作"

@app.route('/video_feed')
def video_feed():
    return Response("這裡是影片串流內容", mimetype='text/plain')

if __name__ == '__main__':
    print("🚀 Flask 開始運行在 http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)

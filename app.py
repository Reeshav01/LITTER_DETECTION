import requests
import threading, time
import cv2, subprocess
import gradio as gr
from ultralytics import YOLO

model = YOLO('best.pt')

def detect_image(img, conf):
    if img is None:
        return None, "Please upload a photo first"
    r = model.predict(img[:, :, ::-1], conf=conf, verbose=False)[0]
    return r.plot()[:, :, ::-1], f"Litter found: {len(r.boxes)}"

def detect_video(path, conf):
    if path is None:
        return None, "Please upload a video first"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw, out_path = 'video_raw.mp4', 'video_out.mp4'
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    most, i, last = 0, 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 3 == 0:
            r = model.predict(frame, conf=conf, imgsz=480, verbose=False)[0]
            most = max(most, len(r.boxes))
            last = r.plot()
        writer.write(last if last is not None else frame)
        i += 1
    cap.release(); writer.release()
    try:
        subprocess.run(['ffmpeg', '-y', '-i', raw, '-vcodec', 'libx264', out_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path, f"Most litter seen in one frame: {most}"
    except Exception:
        return raw, f"Most litter seen in one frame: {most}"
def live_stream(url, conf):
    if not url:
        yield None, "Enter the phone address first"
        return
    latest = {"frame": None}
    stop = {"flag": False}

    def reader():
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not stop["flag"]:
            ok, frame = cap.read()
            if ok:
                latest["frame"] = frame        # keep only the newest frame
            else:
                time.sleep(0.05)
        cap.release()

    threading.Thread(target=reader, daemon=True).start()

    t0 = time.time()
    while latest["frame"] is None:
        if time.time() - t0 > 10:
            stop["flag"] = True
            yield None, "Could not connect. Check the address and the WiFi."
            return
        time.sleep(0.1)

    try:
        while True:
            frame = latest["frame"].copy()
            r = model.predict(frame, conf=conf, imgsz=416, verbose=False)[0]
            yield r.plot()[:, :, ::-1], f"Litter: {len(r.boxes)}"
    finally:
        stop["flag"] = True
def read_sensors(esp_ip):
    try:
        d = requests.get(f"http://{esp_ip}/data", timeout=2).json()
        return d.get("temperature"), d.get("humidity"), d.get("distance_cm"), "Connected"
    except Exception:
        return None, None, None, "Not connected"
with gr.Blocks(title="Litter Detection") as app:
    gr.Markdown("# Litter Detection\nUpload a photo or video and the model marks the litter.")
    conf = gr.Slider(0.1, 0.9, value=0.25, label="Confidence (higher = fewer false alarms)")
    with gr.Tab("Photo"):
        img_in = gr.Image(type="numpy", label="Upload photo")
        img_out = gr.Image(label="Result")
        img_txt = gr.Textbox(label="Count")
        gr.Button("Detect").click(detect_image, [img_in, conf], [img_out, img_txt])
    with gr.Tab("Video"):
        vid_in = gr.Video(label="Upload video")
        vid_out = gr.Video(label="Result")
        vid_txt = gr.Textbox(label="Count")
        gr.Button("Detect").click(detect_video, [vid_in, conf], [vid_out, vid_txt])
    with gr.Tab("Live stream"):
        gr.Markdown("On the phone: open IP Webcam, tap Start server, and type the address below. It must end in /video.")
        url = gr.Textbox(label="Stream address", value="http://192.168.1.5:8080/video")
        live_img = gr.Image(label="Live result")
        live_txt = gr.Textbox(label="Count")
        start = gr.Button("Start")
        stop_btn = gr.Button("Stop")
        ev = start.click(live_stream, [url, conf], [live_img, live_txt])
        stop_btn.click(None, None, None, cancels=[ev])
    with gr.Tab("Sensors"):
        gr.Markdown("Live readings from the ESP32 sensor node (DHT22 + ultrasonic) over WiFi.")
        esp_ip = gr.Textbox(label="ESP32 address (shown in the Arduino serial monitor)", value="192.168.1.50")
        temp = gr.Number(label="Temperature (°C)")
        hum = gr.Number(label="Humidity (%)")
        dist = gr.Number(label="Height above ground (cm)")
        status = gr.Textbox(label="Status")
        timer = gr.Timer(1)
        timer.tick(read_sensors, esp_ip, [temp, hum, dist, status])

app.launch()
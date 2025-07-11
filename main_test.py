from ultralytics import YOLO

# Load your YOLO model
model = YOLO('yolov8n.pt')

# Export to TensorRT format
model.export(format='engine')

# Load the TensorRT model
trt_model = YOLO('yolov8n.engine')

# Run inference using the TensorRT model
results = trt_model('test/suitcase3.mp4')
from ultralytics import YOLO

# Load your YOLO model
model = YOLO('yolo11m-seg.pt')


# Export to TensorRT format
model.export(format='engine')

# Load the TensorRT model
trt_model = YOLO('yolo11m-seg.engine')

# Run inference using the TensorRT model
results = trt_model('test/suitcase3.mp4')
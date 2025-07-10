import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2


class YOLOv8Seg_TRT_Jetson:
    def __init__(self, engine_path, conf=0.3, iou=0.7):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.conf = conf
        self.iou = iou

        # 讀取 engine
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 相容 TensorRT 8/9/10 API
        if hasattr(self.engine, "get_tensor_index"):
            self.get_index = self.engine.get_tensor_index
            self.get_name = self.engine.get_tensor_name
            self.get_shape = self.engine.get_tensor_shape
            self.get_mode = self.engine.get_tensor_mode
            self.num_io = self.engine.num_io_tensors
        else:
            self.get_index = self.engine.get_binding_index
            self.get_name = self.engine.get_binding_name
            self.get_shape = self.engine.get_binding_shape
            self.get_mode = lambda i: "INPUT" if self.engine.binding_is_input(i) else "OUTPUT"
            self.num_io = self.engine.num_bindings

        print(f"Engine has {self.num_io} IO tensors:")
        self.output_tensor_names = []
        for i in range(self.num_io):
            name = self.get_name(i)
            shape = self.get_shape(i)
            mode = self.get_mode(i)
            print(f"  - Tensor {i}: name={name}, shape={shape}, mode={mode}")
            if mode != "INPUT":
                self.output_tensor_names.append(name)
            else:
                self.input_tensor_name = name

        # 取得 input/output shape 與 dtype
        self.input_shape = tuple(self.get_shape(self.get_index(self.input_tensor_name)))
        self.output_shapes = [
            tuple(self.get_shape(self.get_index(name))) for name in self.output_tensor_names
        ]

        # 根據 dtype 判斷要配置多少記憶體（float32:4 / float16:2）
        def get_dtype_size(name):
            dtype = self.engine.get_tensor_dtype(name) if hasattr(self.engine, "get_tensor_dtype") else trt.DataType.FLOAT
            return np.dtype(np.float16).itemsize if dtype == trt.DataType.HALF else np.dtype(np.float32).itemsize

        input_dtype_size = get_dtype_size(self.input_tensor_name)
        output_dtype_sizes = [get_dtype_size(name) for name in self.output_tensor_names]

        # 記憶體配置
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape)) * input_dtype_size)
        self.d_outputs = [
            cuda.mem_alloc(int(np.prod(shape)) * size)
            for shape, size in zip(self.output_shapes, output_dtype_sizes)
        ]

        self.bindings = [0] * self.num_io
        self.context.set_tensor_address(self.input_tensor_name, int(self.d_input))
        self.bindings[self.get_index(self.input_tensor_name)] = int(self.d_input)

        for name, d_out in zip(self.output_tensor_names, self.d_outputs):
            self.context.set_tensor_address(name, int(d_out))
            self.bindings[self.get_index(name)] = int(d_out)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (640, 640))
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return np.ascontiguousarray(img.astype(np.float32))

    def postprocess(self, outputs: list[np.ndarray], original_shape: tuple[int, int]) -> list[dict]:
        output0 = outputs[0]
        preds = output0[0]
        conf_mask = preds[:, 4] > self.conf
        preds = preds[conf_mask]

        boxes = preds[:, :4]
        class_ids = preds[:, 5:].argmax(axis=1)
        confidences = preds[:, 4] * preds[:, 5:].max(axis=1)

        boxes_xywh = boxes.copy()
        boxes_xywh[:, 0] -= boxes[:, 2] / 2
        boxes_xywh[:, 1] -= boxes[:, 3] / 2
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0]
        boxes_xyxy[:, 1] = boxes_xywh[:, 1]
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), confidences.tolist(), self.conf, self.iou
        )

        output_results = []
        if indices is not None and len(indices) > 0:
            for i in np.array(indices).flatten():
                output_results.append({
                    "box": boxes_xyxy[i],
                    "conf": confidences[i],
                    "class_id": int(class_ids[i])
                })
        return output_results

    def infer(self, image: np.ndarray) -> list[dict]:
        input_tensor = self.preprocess(image)
        cuda.memcpy_htod(self.d_input, input_tensor)

        # 推論（若 context 無效需 rebuild engine）
        self.context.execute_v2(self.bindings)

        host_outputs = []
        for shape, d_out in zip(self.output_shapes, self.d_outputs):
            h_out = np.empty(shape, dtype=np.float32)
            cuda.memcpy_dtoh(h_out, d_out)
            host_outputs.append(h_out)

        return self.postprocess(host_outputs, image.shape[:2])

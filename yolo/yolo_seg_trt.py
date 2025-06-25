import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')
import cv2


class YOLOv8Seg_TRT:
    def __init__(self, engine_path, conf=0.3, iou=0.7):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.conf = conf
        self.iou = iou

        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        print(f"Engine has {self.engine.num_bindings} bindings:")
        for i in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            io = "input" if self.engine.binding_is_input(i) else "output"
            print(f"  Binding {i}: name={name}, shape={shape}, {io}")

        # 取得輸入輸出索引
        self.input_binding_idx = next(i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i))
        self.output_binding_idx = next(i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i))

        self.input_shape = tuple(self.engine.get_binding_shape(self.input_binding_idx))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_binding_idx))

        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape)) * 4)
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape)) * 4)

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_binding_idx] = int(self.d_input)
        self.bindings[self.output_binding_idx] = int(self.d_output)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (640, 640))
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return np.ascontiguousarray(img.astype(np.float32))

    def postprocess(self, output: np.ndarray, original_shape: tuple[int, int]) -> list[np.ndarray]:
        preds = output[0]
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

        # 1. 不用 set_input_shape (視 engine 是否為動態形狀)
        #    如果你的 engine 是固定形狀，可省略

        # 2. 用固定的 input/output shape 分配記憶體
        output = np.empty(self.output_shape, dtype=np.float32)

        # 3. 複製 input 到 GPU
        cuda.memcpy_htod(self.d_input, input_tensor)

        # 4. 執行推論時，帶入 bindings（GPU address list）
        self.context.execute_v2(self.bindings)

        # 5. 拷回輸出結果
        cuda.memcpy_dtoh(output, self.d_output)

        return self.postprocess(output, image.shape[:2])




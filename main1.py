import argparse

import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

gpu_frame = cv2.cuda_GpuMat()
stream = cv2.cuda_Stream()

class YOLOv8SegGPU:
    def __init__(self, onnx_model: str, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640):
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider"]
        )
        self.imgsz = (imgsz, imgsz)
        self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]
        self.conf = conf
        self.iou = iou
        print("ONNX Runtime Providers:", self.session.get_providers())

    def __call__(self, img: np.ndarray) -> list[Results]:
        prep_img = self.preprocess_gpu(img)
        ort_input = ort.OrtValue.ort_value_from_torch_tensor(prep_img)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: ort_input})
        return self.postprocess(img, prep_img, outputs)

    def letterbox(self, img: torch.Tensor, new_shape: tuple[int, int]) -> torch.Tensor:
        _, h, w = img.shape
        scale = min(new_shape[1] / h, new_shape[0] / w)
        new_unpad = (int(round(w * scale)), int(round(h * scale)))
        pad_w = new_shape[0] - new_unpad[0]
        pad_h = new_shape[1] - new_unpad[1]
        img = F.interpolate(img.unsqueeze(0), size=(new_unpad[1], new_unpad[0]), mode='bilinear', align_corners=False).squeeze(0)
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.447)
        return img

    def preprocess_gpu(self, img: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().cuda() / 255.0
        img_tensor = self.letterbox(img_tensor, self.imgsz)
        return img_tensor.unsqueeze(0)  # shape: (1, 3, H, W)

    def postprocess(self, img: np.ndarray, prep_img: torch.Tensor, outs: list) -> list[Results]:
        preds, protos = [torch.from_numpy(p).float() for p in outs]
        preds = ops.non_max_suppression(preds, self.conf, self.iou, nc=len(self.classes))

        results = []
        for i, pred in enumerate(preds):
            if pred is None or pred.shape[0] == 0:
                results.append(Results(img, path="", names=self.classes, boxes=torch.empty((0, 6)), masks=None))
                continue
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names=self.classes, boxes=pred[:, :6], masks=masks))
        return results

    def process_mask(self, protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        c, mh, mw = protos.shape
        masks = (masks_in @ protos.view(c, -1)).view(-1, mh, mw)
        masks = ops.scale_masks(masks[None], shape)[0]
        masks = ops.crop_mask(masks, bboxes)
        return masks.gt_(0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    args = parser.parse_args()

    try:
        model = YOLOv8SegGPU(args.model, args.conf, args.iou)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)
    video = cv2.cudacodec.createVideoReader('/home/eden/ultralytics/examples/YOLOv8-Segmentation-ONNXRuntime-Python/pics/IMG_2894.mov')
    # cap = cv2.VideoCapture('/home/eden/ultralytics/examples/YOLOv8-Segmentation-ONNXRuntime-Python/pics/IMG_2894.mov')
    # cap = cv2.VideoCapture(args.source)


    while True:
        ret, frame = video.nextFrame()
        if not ret:
            break

        start = time.time()
        gpu_resized = cv2.cuda.resize(frame, (640, 480), stream=stream)
        stream.waitForCompletion()
        results = model(gpu_resized)
        output = results[0].plot() if results[0].masks is not None else frame_cpu

        fps = 1.0 / (time.time() - start)
        print(f"FPS: {fps:.2f}")

        cv2.imshow("YOLOv8 GPU Inference", output)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break
    cv2.destroyAllWindows()

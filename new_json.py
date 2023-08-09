import json

data = {
	"source": "streams.txt",
	"weights": "../runs/yolov5/origin/weights/best.pt",
	"device": 0, # 使用的device类别，如是GPU，可填"0"
	"imgsz": 640,  # 输入图像的大小
	"stride": 32,  # 步长
	"conf_thres": 0.35, # 置信值阈值
	"iou_thres": 0.45,  # iou阈值
	"augment": False
}

with open('yolov5_config.json', 'w') as f:
    json.dump(data, f)


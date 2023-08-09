import time
import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh, set_logging, check_requirements
from utils.plots import colors, Annotator  # plot_one_box
from utils.torch_utils import select_device  # time_synchronized
from utils.grabsreen import grab_screen
from PIL import Image
import pyautogui
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

pyautogui.FAILSAFE = False


@torch.no_grad()
def detect(
        # --------------------这里更改配置--------------------
        # ---------------------------------------------------
        weights='yolov5s.pt',  # 训练好的模型路径
        imgsz=640,  # 训练模型设置的尺寸
        cap=0,  # 摄像头
        conf_thres=0.25,  # 置信度
        iou_thres=0.45,  # NMS IOU 阈值
        max_det=1000,  # 最大侦测的目标数
        device='',  # 设备
        crop=True,  # 显示预测框
        classes=None,  # 种类
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # 是否扩充推理
        half=False,  # 使用FP16半精度推理
        hide_labels=False,  # 是否隐藏标签
        hide_conf=False,  # 是否隐藏置信度
        line_thickness=3  # 预测框的线宽
):
    # #--------------------这里更改配置--------------------
    # -----------------------------------------------------

    # -----初始化-----
    set_logging()
    # 设置设备
    device = select_device(device)
    # CUDA仅支持半精度
    half &= device.type != 'cpu'

    # -----加载模型-----
    # 加载FP32模型
    model = attempt_load(weights, map_location=device)
    # 模型步幅
    stride = int(model.stride.max())
    # 检查图像大小
    imgsz = check_img_size(imgsz, s=stride)
    # 获取类名
    names = model.module.names if hasattr(model, 'module') else model.names
    # toFP16
    if half:
        model.half()

        # ------运行推理------
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 跑一次

    # -----进入循环：ESC退出-----
    picnum = 0
    while (True):
        image_array = grab_screen(region=(0, 0, 1280, 720))

        array_to_image = Image.fromarray(image_array, mode='RGB')  # 将array转成图像，才能送入yolo进行预测
        img = np.asarray(array_to_image)  # 将图像转成array

        # 设置labels--记录标签/概率/位置
        labels = []
        # 计时
        t0 = time.time()
        img0 = img
        # 填充调整大小
        img = letterbox(img0, imgsz, stride=stride)[0]
        # 转换
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        # uint8 to fp16/32
        img = img.half() if half else img.float()
        # 0 - 255 to 0.0 - 1.0
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推断
        # t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # 添加 NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # t2 = time_synchronized()

        # 目标进程
        for i, det in enumerate(pred):  # 每幅图像的检测率
            s, im0 = '', img0.copy()
            # 输出字符串
            s += '%gx%g ' % img.shape[2:]
            # 归一化增益
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 将框从img_大小重新缩放为im0大小
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                # 输出结果
                for c in det[:, -1].unique():
                    # 每类检测数
                    n = (det[:, -1] == c).sum()
                    # 添加到字符串
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    # 结果输出
                for *xyxy, conf, cls in reversed(det):
                    # 归一化xywh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # 标签格式
                    line = (cls, *xywh, conf)
                    # 整数类
                    c = int(cls)
                    # 建立标签
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    # 画预测框
                    if crop:
                        # print('right')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    # 记录标签/概率/位置
                    labels.append([names[c], conf, xyxy])
                    # print(labels)

                    # 设定延迟时间，以画面中的圆圈数来区分速度，画面中只有一个圈的时候就要慢一点，反之则快
                    ys = 0
                    if len(labels) < 2:
                        ys = 0.17
                    elif len(labels) < 4:
                        ye = 0.14
                    elif len(labels) < 6:
                        ys = 0.12
                    else:
                        ys = 0.1

                    # 获取中心点，即分别求横、纵坐标的中间点
                    pointx = int((xyxy[0] + xyxy[2]) / 2)
                    pointy = int((xyxy[1] + xyxy[3]) / 2)
                    # 移动鼠标到中心点位置，并点击
                    pyautogui.moveTo(pointx, pointy, duration=ys)
                    pyautogui.click()
                    pyautogui.click()

            # 显示图片
            imshow = cv2.cvtColor(im0, cv2.COLOR_RGB2BGRA)
            cv2.imshow("copy window", imshow)
            # 输出计算时间
            print(f'消耗时间: ({time.time() - t0:.3f}s)')

        key = cv2.waitKey(2)

        # 这里设置ESC退出
        if key == 'q':
            break
        # --------------------END--------------------
        # -------------------------------------------------
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
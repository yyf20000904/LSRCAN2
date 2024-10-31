import cv2
import numpy as np
import math
import os


def get_AS(masks):
    #mask_pos = np.zeros((len(masks), h, w))
    smes = 0
    for i in range(len(masks)): #n个实例
        mask = masks[i]  # torch.cuda.FloatTensor
        #mask是tensor类型的 转换成非tensor

        # padded_mask[:, :] = mask.cpu().numpy()  # 从第二个取到倒数第二个
        padded_mask= mask
        contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)  # contours, hierarchy  只检测外轮廓  压>缩水平方向，垂直方向，对角线方向的元素，
        
        for contour in contours:
            if cv2.contourArea(contour) < 1:
                continue
            length = cv2.arcLength(curve=contour, closed=False)
            contour_approxed = cv2.approxPolyDP(curve=contour, epsilon=1, closed=False)
            if length > 0:
                sm = len(contour_approxed)/math.sqrt(length)
            else:
                sm = 0
            smes += sm
    return smes/len(masks)


if __name__ == "__main__":
    folder_path = 'RE/rcan/modelX4new'
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.lower().endswith(('.png', '.jpg', '.jpeg','.bmp'))]
    masks_list = []

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #image = cv2.imread(path)
        masks_list.append(image)


    as_value = get_AS(masks_list)
    print("AS value:", as_value)

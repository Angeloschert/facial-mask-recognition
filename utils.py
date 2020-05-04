import numpy as np

# Compute every scaling value of the input image
def calculateScales(img):
    copy_img = img.copy()
    pr_scale = 1.0
    h, w, _ = copy_img.shape

    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)
    w, h = int(w * pr_scale), int(h * pr_scale)

    scales = []
    factor, factor_count = 0.709, 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T

    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([1], 2, axis=0).T
    return rectangles

# NMS (Non-maximum supression algorithm) to select the bounding box with the largest probability
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles

    boxes = np.array(rectangles)
    x1, y1, x2, y2, s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    pick = []

    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.maximum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.maximum(y2[I[-1]], y2[I[0:-1]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)

        pick.append(I[-1])
        I = I[np.where(o < threshold)[0]]

    result_rectangle = boxes[pick].tolist()
    return result_rectangle


def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)
    stride = 0

    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
    (x, y) = np.where(cls_prob >= threshold)

    boundingbox = np.array([x, y]).T

    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)

    boundingbox = np.concatenate((bb1, bb2), axis=1)
    dx1, dx2, dx3, dx4 = roi[0][x, y], roi[1][x, y], roi[2][x, y], roi[3][x, y]
    score, offset = np.array([cls_prob[x, y]]).T, np.array([dx1, dx2, dx3, dx4]).T

    boundingbox = boundingbox + offset * 12.0 * scale
    rectangles = np.concatenate((boundingbox, score), axis=1).rect2square()

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)




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



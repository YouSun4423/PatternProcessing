import cv2
import numpy as np


def main():
    img = cv2.imread("images/Parrots.bmp", 0)
    cv2.imshow("image", img)
    noise = np.random.normal(0, 10, img.shape)
    img_noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imshow("noisy", img_noisy)
    # 平均化フィルタ、3×3
    img_out_averaging_3 = cv2.blur(img_noisy, (3, 3))
    cv2.imshow("img_out_averaging_3", img_out_averaging_3)
    # 平均化フィルタ、5×5
    img_out_averaging_5 = cv2.blur(img_noisy, (5, 5))
    cv2.imshow("img_out_averaging_5", img_out_averaging_5)
    # ガウシアンフィルタ、5×5
    img_out_gaussian = cv2.GaussianBlur(img_noisy, (5, 5), 0)
    cv2.imshow("img_out_gaussian", img_out_gaussian)
    # メジアンフィルタ、5×5
    img_out = cv2.medianBlur(img_noisy, 5)
    cv2.imshow("img_out", img_out)

    cv2.waitKey()


if __name__ == "__main__":
    main()
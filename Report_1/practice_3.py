import cv2
import numpy as np


def main():

    # 入力画像の読み込み
    img = cv2.imread("images/Lenna.bmp", 0)
    cv2.imshow("Lenna_img.bmp", img)

    dst = resize(img, 0.75)

    # 結果を出力
    cv2.imshow("Lenna_out_img.bmp", dst)
    cv2.imwrite("out/output.bmp", dst)

    cv2.waitKey()


def resize(src, mag):
    hi, wi = src.shape[0], src.shape[1]

    width = mag * wi
    height = mag * hi

    dst = np.empty(
        (round(height), round(width)),
        dtype=np.uint8,
    )

    for y in range(round(height)):
        for x in range(round(width)):
            xi, yi = round(x / mag), round(y / mag)
            # 存在しない座標の処理
            if xi > wi - 1:
                xi = wi - 1
            if yi > hi - 1:
                yi = hi - 1

            dst[y][x] = src[yi][xi]

    return dst


if __name__ == "__main__":
    main()
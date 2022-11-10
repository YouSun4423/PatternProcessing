import cv2
import numpy as np


def main():
    img = cv2.imread("Lenna.bmp", 0)
    pos_x, pos_y = 100, 100  # 切り出す位置(左上座標)
    t_height, t_width = 64, 64 # 切り出すサイズ
    template = img[pos_y:pos_y+t_height, pos_x:pos_x+t_width].copy()
    cv2.imshow("template", template)

    # パターンマッチング
    # 差分二乗和
    result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)

    # 相互相関
    # result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)

    # 最大値と最小値，その位置を取得
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = min_loc  # 差分二条和の場合
    # top_left = max_loc  # 相互相関の場合
    bottom_right = (top_left[0] + t_width, top_left[1] + t_height)

    cv2.imshow("result", result)

    cv2.rectangle(img, top_left, bottom_right, 255, 1)
    cv2.imshow("image", img)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
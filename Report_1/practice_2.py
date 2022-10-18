import cv2


def main():
    # 画像の読み込み
    img_in = cv2.imread("images/Lenna.bmp")
    # 読み込んだ画像の表示
    cv2.imshow("input", img_in)
    h, w = img_in.shape[0], img_in.shape[1]

    # 2次元回転のアフィン変換行列を作成し画像に適応
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1)
    img_rotation = cv2.warpAffine(img_in, rotation_matrix, (w, h))
    # 90度回転した画像の表示，出力
    cv2.imshow("output_rotation", img_rotation)
    cv2.imwrite("out/output_rotation.bmp", img_rotation)

    # 画像を左右反転する
    img_flip = cv2.flip(img_in, 1)
    # 左右反転した画像の表示
    cv2.imshow("output_flip", img_flip)
    cv2.imwrite("out/output_flip.bmp", img_flip)

    cv2.waitKey()


if __name__ == "__main__":
    main()
import cv2


def main():
    img = cv2.imread("images/Lenna.bmp")
    cv2.imshow("image", img)
    
    # 補間あり
    img_resize = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow("image_resize", img_resize)
    cv2.imwrite("out/output_resize.bmp", img_resize)

    # 補間なし
    nearest = cv2.resize(
        img,
        dsize=None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow('image_nearest', nearest)
    cv2.imwrite("out/output_nearest.bmp", nearest)
    
    cv2.waitKey()


if __name__ == "__main__":
    main()
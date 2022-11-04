import cv2
import numpy as np

GAMMA = 1e-3


def main():
    img = cv2.imread("images/Parrots.bmp", 0)
    cv2.imshow("image", img)

    height, width = 128, 128
    img_in = cv2.resize(img, (height, width))
    x = np.reshape(img_in, (width * height, 1))  # 画像のベクトル化
    H = np.zeros((width * height, width * height), np.float32)
    for i in range(width * height):
        if i - width - 1 >= 0 and i + width + 1 < width * height:
            H[i, i - width - 1] = 1 / 16  # h_-1,-1
            H[i, i - width] = 2 / 16      # h_0,-1
            H[i, i - width + 1] = 1 / 16  # h_1,-1
            H[i, i - 1] = 2 / 16          # h_-1,0
            H[i, i] = 4 / 16              # h_0,0
            H[i, i + 1] = 2 / 16          # h_1,0
            H[i, i + width - 1] = 1 / 16  # h_-1,1
            H[i, i + width] = 2 / 16      # h_0,1
            H[i, i + width + 1] = 1 / 16  # h_1,1
    y = np.matmul(H, x)

    noise = np.random.normal(0, 1, y.shape)
    y = np.clip(y + noise, 0, 255).astype(np.uint8)

    x_estimate = np.linalg.solve(
        np.matmul(H.T, H) + GAMMA * np.identity(height * width),
        np.matmul(H.T, y),
    )
    x_estimate = np.clip(x_estimate, 0, 255).astype(np.uint8)

    img_blur = np.reshape(y, (height, width)).astype(np.uint8)
    img_output = np.reshape(x_estimate, (height, width)).astype(np.uint8)

    cv2.imshow("img_blur", img_blur)
    cv2.imshow("img_output", img_output)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
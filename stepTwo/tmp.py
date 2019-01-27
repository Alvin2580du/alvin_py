import cv2


def seed_fill(img):
    ret, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    label = 100
    stack_list = []
    h, w = img.shape
    for i in range(1, h - 1, 1):
        for j in range(1, w - 1, 1):
            if (img[i][j] == 255):
                img[i][j] = label
                stack_list.append((i, j))
                while len(stack_list) != 0:
                    cur_i = stack_list[-1][0]
                    cur_j = stack_list[-1][1]
                    img[cur_i][cur_j] = label
                    stack_list.remove(stack_list[-1])
                    ####### 四邻域，可改为八邻域
                    if (img[cur_i - 1][cur_j] == 255):
                        stack_list.append((cur_i - 1, cur_j))
                    if (img[cur_i][cur_j - 1] == 255):
                        stack_list.append((cur_i, cur_j - 1))
                    if (img[cur_i + 1][cur_j] == 255):
                        stack_list.append((cur_i + 1, cur_j))
                    if (img[cur_i][cur_j + 1] == 255):
                        stack_list.append((cur_i, cur_j + 1))
    cv2.imwrite('./result.jpg', img)
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    img = cv2.imread('test.jpeg', 0)
    seed_fill(img)

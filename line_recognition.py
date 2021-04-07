# -*- coding: cp949 -*-
# -*- coding: utf-8 -*-
import cv2  # import OpenCV
import numpy as np # import Numpy


def grayscale(img):  # Canny Edge Detection ������ ���� ����̹����� ��ȯ
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny Edge Detection
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # Gaussian Filter
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI ����

    mask = np.zeros_like(img)  # mask = img�� ���� ũ���� �� �̹���

    # ROI �κ��� mask���� ������� ä���ֱ� ���ؼ� ������
    if len(img.shape) > 2:  # Color �̹���(3ä��)��� :
        color = color3
    else:  # ��� �̹���(1ä��)��� :
        color = color1

    # vertices�� ���� ����� �̷��� �ٰ����κ�(ROI �����κ�)�� color�� ä��
    cv2.fillPoly(mask, vertices, color)

    # �̹����� color�� ä���� ROI�� ��ħ
    ROI_image = cv2.bitwise_and(img, mask)
    # cv2.imshow("mask", mask)
    # cv2.imshow("ROI", ROI_image)
    return ROI_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):  # Draw Line
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# �𸣰���
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # Hough Transformation
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def weighted_img(img, initial_img, ��=1, ��=1., ��=0.):  # �� �̹��� operlap �ϱ�
    return cv2.addWeighted(initial_img, ��, img, ��, ��)

if __name__ == "__main__":
    cap = cv2.VideoCapture('solidWhiteRight.mp4')  # ������ �ҷ�����

    while (cap.isOpened()):
        ret, image = cap.read()

        height, width = image.shape[:2]  # �̹��� ����, �ʺ�

        gray_img = grayscale(image)  # ����̹����� ��ȯ

        blur_img = gaussian_blur(gray_img, 3)  # Blur ȿ��

        canny_img = canny(blur_img, 70, 210)  # Canny edge �˰���

        vertices = np.array(
            [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
            dtype=np.int32)
        ROI_img = region_of_interest(canny_img, vertices)  # ROI ����

        hough_img = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # ���� ��ȯ

        result = weighted_img(hough_img, image)  # ���� �̹����� ����� �� overlap
        cv2.imshow('result', result)  # ��� �̹��� ���
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release
    cap.release()
    cv2.destroyAllWindows()
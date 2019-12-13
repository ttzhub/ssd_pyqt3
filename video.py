import cv2
import os


img_dir = 'D:\VOC2007\VOCdevkit\VOC2007\JPEGImages'
img_list = os.listdir(img_dir)[:50]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_write = cv2.VideoWriter('Voc_test.avi', fourcc, 5, (720, 480))

for img_path in img_list:
    img_path_abs = os.path.join(img_dir, img_path)
    print(img_path_abs)
    try:
        img = cv2.imread(img_path_abs)
        img = cv2.resize(img, (720, 480))
        video_write.write(img)
        cv2.imshow('asd', img)
        cv2.waitKey(10)
    except:
        video_write.release()

cv2.destroyAllWindows()
video_write.release()
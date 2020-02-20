import dlib,os,glob,time
import  cv2
import numpy as np

# @author 许翔
# @function 利用opencv和dlib实现人脸检测
# @time 2020-2-13
# 声明各个资源路径
resources_path = os.path.abspath(".")+"\Resources\\"
predictor_path = resources_path + "shape_predictor_68_face_landmarks.dat"
model_path = resources_path + "dlib_face_recognition_resnet_model_v1.dat"
video_path =resources_path + "face_recognition.mp4"
resources_vResult=resources_path+"video\\"
# 加载视频,加载失败则退出
video = cv2.VideoCapture(video_path)
# 获得视频的fps
fps = video.get(cv2.CAP_PROP_FPS)
if not video.isOpened():
    print("video is not opened successfully!")
    exit(0)

# # 加载模型
#人脸特征提取器
detector = dlib.get_frontal_face_detector()
#人脸关键点标记
predictor= dlib.shape_predictor(predictor_path)
#生成面部识别器
facerec = dlib.face_recognition_model_v1(model_path)
#定义视频创建器
video_writer = cv2.VideoWriter(resources_vResult+"result1.avi",
                               cv2.VideoWriter_fourcc(*'XVID'), int(fps),
                               (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 创建窗口
cv2.namedWindow("Face Recognition", cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("Face Recognition", 720,576)

descriptors = []
faces = []
# 处理视频，按帧处理
ret,frame = video.read()
flag = True                  # 标记是否是第一次迭代
i = 0                        # 记录当前迭代到的帧位置
while ret:
    if i % 6== 0:           # 每6帧截取一帧
        # 转为灰度图像处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)        # 检测帧图像中的人脸
      #  for i in range(len(dets)):
        #    landmarks = np.matrix([[p.x, p.y] for p in predictor(gray,dets[i]).parts()])
        # 处理检测到的每一张人脸
        for index,value in enumerate(dets):
            #获取面部关键点
            shape = predictor(gray,value)
            #pos = (value[0, 0], value[0, 1])

            #标记人脸
            cv2.rectangle(frame, (value.left(), value.top()), (value.right(), value.bottom()), (0, 255, 0), 2)
            #标记关键点
            for pti,pt in enumerate(shape.parts()):
                pos=(pt.x,pt.y)
                cv2.circle(frame, pos, 1, color=(0, 255, 0))
            # 提取特征-图像中的68个关键点转换为128D面部描述符，其中同一人的图片被映射到彼此附近，并且不同人的图片被远离地映射。
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            v = np.array(face_descriptor)
            # 将第一张人脸照片直接保存
            if flag:
                descriptors.append(v)
                faces.append(frame)
                flag = False
            else:
                sign = True                 # 用来标记当前人脸是否为新的
                l = len(descriptors)
                for i in range(l):
                    distance = np.linalg.norm(descriptors[i] - v)    # 计算两张脸的欧式距离，判断是否是一张脸
                    # 取阈值0.5，距离小于0.5则认为人脸已出现过
                    if distance < 0.5:
                        # print(faces[i].shape)
                        face_gray = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                        # 比较两张人脸的清晰度，保存更清晰的人脸
                        if cv2.Laplacian(gray, cv2.CV_64F).var() > cv2.Laplacian(face_gray, cv2.CV_64F).var():
                            faces[i] = frame
                        sign = False
                        break
                # 如果是新的人脸则保存
                if sign:
                    descriptors.append(v)
                    faces.append(frame)
        cv2.imshow("Face Recognition", frame)      # 在窗口中显示
        exitKey= cv2.waitKey(1)
        if exitKey == 27:
            video.release()
            video_writer.release()
            cv2.destroyWindow("Face Recognition")
            break
    video_writer.write(frame)
    ret,frame = video.read()
    i += 1

print(len(descriptors))     # 输出不同的人脸数
print(len(faces))          #输出的照片数
# 将不同的比较清晰的人脸照片输出到本地
j = 1
for fc in faces:
    cv2.imwrite(resources_path + "\pictures\\" + str(j) +".jpg", fc)
    j += 1
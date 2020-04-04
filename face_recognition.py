import dlib,os,glob,time
import  cv2
import numpy as np
import csv
import pandas as pd


# @author 许翔
# @function 利用opencv和dlib实现人脸识别
# @time 2020-2-13
# 声明各个资源路径
resources_path = os.path.abspath(".")+"\Resources\\"
predictor_path = resources_path + "shape_predictor_68_face_landmarks.dat"
model_path = resources_path + "dlib_face_recognition_resnet_model_v1.dat"
video_path =resources_path + "face_recognition.mp4"
resources_vResult=resources_path+"video\\"
faceDB_path="Resources/featureMean/"
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
#定义视频创建器,用于输出视频
video_writer = cv2.VideoWriter(resources_vResult+"result1.avi",
                               cv2.VideoWriter_fourcc(*'XVID'), int(fps),
                               (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
#读取本地人脸库
head = []
for i in range(128):
    fe = "feature_" + str(i + 1)
    head.append(fe)
face_path=faceDB_path+"feature_all.csv"
face_feature=pd.read_csv(face_path,names=head)
print(face_feature.shape)
face_feature_array=np.array(face_feature)
print(face_feature_array.shape)
face_list=["Chandler","Joey","Monica","phoebe","Rachel","Ross"]
# 创建窗口
cv2.namedWindow("Face Recognition", cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("Face Recognition", 720,576)

#计算128D描述符的欧式距离
def compute_dst(feature_1,feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

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
        if len(dets)>0:
            for index,value in enumerate(dets):
                #获取面部关键点
                shape = predictor(gray,value)
                #pos = (value[0, 0], value[0, 1])

                #标记人脸
                cv2.rectangle(frame, (value.left(), value.top()), (value.right(), value.bottom()), (0, 255, 0), 2)
                #进行人脸识别并打上姓名标签
                # 提取特征-图像中的68个关键点转换为128D面部描述符，其中同一人的图片被映射到彼此附近，并且不同人的图片被远离地映射。
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                v = np.array(face_descriptor)
                print(v.shape)
                l = len(descriptors)
                Flen=len(face_list)
                flag=0
                for j in range(Flen):
                    # 人脸匹配，距离小于阈值，表示识别成功，打上标签
                    if(compute_dst(v,face_feature_array[j])<0.56):
                        flag=1
                        cv2.putText(frame,face_list[j],(value.left(), value.top()),cv2.FONT_HERSHEY_COMPLEX,0.8, (0, 255, 255), 1, cv2.LINE_AA)
                        break
                if(flag==0):
                    cv2.putText(frame,"Unknonw", (value.left(), value.top()), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1,
                                cv2.LINE_AA)

                #标记关键点
                for pti,pt in enumerate(shape.parts()):
                    pos=(pt.x,pt.y)
                    cv2.circle(frame, pos, 1, color=(0, 255, 0))
                #faces.append(frame)
               # 将第一张人脸照片直接保存
                if flag:
                    descriptors.append(v)
                    faces.append(frame)
                    flag = False
                else:
                    sign = True                 # 用来标记当前人脸是否为新的
                    for i in range(l):
                        distance = compute_dst(descriptors[i] , v)    # 计算两张脸的欧式距离，判断是否是一张脸
                        # 取阈值0.5，距离小于0.5则认为人脸已出现过
                        if distance < 0.4:
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

# Face_Recognition
使用Opencv和Dlib实现基于视频的人脸识别

# 文件夹介绍
1、Resources\pictures    
此文件夹下存放人脸保存结果  
2、Resources\video       
此文件夹下存放带标注视频保存结果  
3、Resources\faceS  
此文件夹下存放各个人物的图片，用于人脸库的建立  
4、Resources\featureDB  
此文件下为各个人物的128D描述符的数据存储文件  
5、Resources\featureMean\feature_all.csv  
此文件为人脸特征库文件  
6、Resources\shape_predictor_68_face_landmarks.dat       
人脸关键点标记模型文件  
7、Resources\dlib_face_recognition_resnet_model_v1.dat    
面部识别模型文件  
8、face_recognition.mp4   
待检测的视频  
9、face_recognition.py  
人脸识别代码文件  
10、detection.py  
人脸检测代码文件  
11、face_recognition.py  
建立本地人脸库代码文件  
# 介绍
##  思路介绍
无论是基于视频或者调用摄像头来完成人脸识别，其实是一样，通过使用opencv，来捕获视频或者摄像头传来的图像，每隔若干帧取一帧做人脸识别，调用Dlib中的人脸检测器来检测人脸，并通过Dlib的人脸关键点预测器来获得人脸的关键点，接下来使用Dlib的面部识别模型将获得的68个关键点转换成128D面部描述符，我们通过计算人脸的128D面部描述符与本地人脸库(需要自己建立人脸库)中的人脸128D面部描述符的欧氏距离，来判断是否为同一人，当距离小于特定阈值时，认定识别成功，打上标签。  
![Image text](https://github.com/coderXuXiang/Face_Recognition/blob/master/%E6%80%9D%E7%BB%B4%E8%BF%87%E7%A8%8B%E5%9B%BE/%E8%BF%87%E7%A8%8B%E5%9B%BE.png)
## 运行环境介绍
操作系统版本：Windows10  
运行环境：python3.6+opencv4.1.2+dlib19.8.1  
软件:PyCharm  

(注:这里下载dlib包最好下载.whl文件，不需要安装cmake以及boost这些麻烦的东西。因为dilib包的没有python3.7版的whl文件，所以建议使用python3.6)
**附上opencv和dlib包链接**：https://pan.baidu.com/s/1Z33r7SoD5Z0faH96wr7Ecw 
提取码：a8gl 

## 模型介绍
这里的人脸识别使用了Dlib已训练成功的两个模型--人脸关键点预测器和面部识别模型。使用时需要加载模型，文件分别为shape_predictor_68_face_landmarks.dat和dlib_face_recognition_resnet_model_v1.dat
**模型文件下载地址**  http://dlib.net/files/

### 人脸关键点预测器
Dlib中标记68个特征点采用的是ERT算法，是一种基于回归树的人脸对齐算法，这种方法通过建立一个级联的残差回归树来使人脸形状从当前形状一步一步回归到真实形状。每一个GBDT的每一个叶子节点上都存储着一个残差回归量，当输入落到一个节点上时，就将残差加到改输入上，起到回归的目的，最终将所有残差叠加在一起，就完成了人脸对齐的目的。


**用法:**

```python
predictor_path = resources_path + "shape_predictor_68_face_landmarks.dat"
#加载人脸关键点预测器
predictor= dlib.shape_predictor(predictor_path)
#获取面部关键点，gary为灰度化的图片
shape = predictor(gray,value)
```
### 人脸识别模型
Dlib中使用的人脸识别模型是基于深度残差网络，深度残差网络通过残差块来构建，它有效的解决了梯度消失以及梯度爆炸问题。当网络深度很大时，普通网络的误差会增加，而深度残差网络却有较小的误差。这里的人脸识别通过训练深度残差网络将人脸的68个特征关键点转换成128D面部描述符，用于人脸的识别。

```python
model_path = resources_path + "dlib_face_recognition_resnet_model_v1.dat"
#生成面部识别器
facerec = dlib.face_recognition_model_v1(model_path)
 # 提取特征-图像中的68个关键点转换为128D面部描述符，其中同一人的图片被映射到彼此附近，并且不同人的图片被远离地映射。
face_descriptor = facerec.compute_face_descriptor(frame, shape)
```
# 效果展示
![Image text](https://github.com/coderXuXiang/Face_Recognition/blob/master/%E6%95%88%E6%9E%9C%E5%B1%95%E7%A4%BA%E5%9B%BE/%E5%9B%BE%E7%89%872.png)  
![Image text](https://github.com/coderXuXiang/Face_Recognition/blob/master/%E6%95%88%E6%9E%9C%E5%B1%95%E7%A4%BA%E5%9B%BE/%E5%9B%BE%E7%89%871.png)  
![Image text](https://github.com/coderXuXiang/Face_Recognition/blob/master/%E6%95%88%E6%9E%9C%E5%B1%95%E7%A4%BA%E5%9B%BE/%E5%9B%BE%E7%89%873.png)  
# 识别过程
1、本地人脸库建立  
![Image text](https://github.com/coderXuXiang/Face_Recognition/blob/master/%E6%80%9D%E7%BB%B4%E8%BF%87%E7%A8%8B%E5%9B%BE/%E4%BA%BA%E8%84%B8%E5%BA%93%E5%BB%BA%E7%AB%8B%E5%9B%BE.png)  
2 、视频处理  
通过opencv提供的VideoCapture()函数对视频进行加载，并计算视频的fps，以方便人脸标记之后的视频的输出。  
3、加载模型  
 将已经训练好的模型加载进来，将人脸关键点标记模型和面部识别模型加载进来，以便后续使用。    
4、人脸检测  
 对视频进行读取，每隔6帧，取一帧进行人脸检测，先将取得的照片进行灰度处理，然后进行人脸检测，并绘画人脸标记框进行展示，然后通过加载的人脸关键点标记模型识别图像中的人脸关键点，并且标记。  
5、人脸识别  
将获取的人脸关键点转换成128D人脸描述符，将其与人脸库中的128D面部描述符进行欧氏距离计算，当距离值小于某个阈值时，认为人物匹配，识别成功，打上标签。当无一小于该阈值，打上Unknown标签  
![Image text](https://github.com/coderXuXiang/Face_Recognition/blob/master/%E6%80%9D%E7%BB%B4%E8%BF%87%E7%A8%8B%E5%9B%BE/%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%E5%9B%BE.png)  
6、 保存人脸标记视频  
 将整个处理过程进行输出，将人脸标记过程保存下来。

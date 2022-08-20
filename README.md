# Phát hiện bất thường trên MRI não với mô hình Pix2Pix
Trong bài luận, mình sử dụng các mô hình phân lớp như [Resnet50](https://keras.io/api/applications/resnet/#resnet50-function), [InceptionV3](https://keras.io/api/applications/inceptionv3/), và [EfficientNetB6](https://keras.io/api/applications/efficientnet/#efficientnetb6-function) để huấn luyện mô hình phân lớp lắt cắt MRI não có bất thường hay không. Sau đó, mình sử dụng mô hình [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf) để huấn luyện mô hình tìm vùng bất thường trên các ảnh có bất thường. Cuối cùng mình sử dụng tkinter để thiết kế giao diện để sử dụng các mô hình đã huấn luyện được.

**Các thư viện sử dụng**
- [Tensorflow 2.3.4](https://pypi.org/project/tensorflow/2.3.4/)
- [labelme](https://github.com/wkentaro/labelme)
- [tkinter](https://docs.python.org/3/library/tkinter.html)

**Môi trường**
- [Google Colaboratory](https://research.google.com/colaboratory/)
- Pycharm
- Python 3.9.2
- CUDA Version: 11.2

**Run demo**
```
https://github.com/liemkg1234/Pix2Pix_MRIBrain
cd Pix2Pix_MRIBrain
pip install -r requirements.txt
python app/pred.py
```

## Sơ đồ tổng quát
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/model.PNG)
## Mô hình Pix2Pix phát hiện vùng bất thường trên MRI não
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/Pix2Pix.PNG)
### Generator
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/G.PNG)
### Discriminator
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/D.PNG)

## Tập dữ liệu
- MRI Brain DHYDCT: Gồm 2812 MRI não (604 ảnh có bất thường/khối u) của 139 bệnh nhân với chuỗi xung T2FLAIR được Bệnh viện Trường Đại học Y Dược Cần Thơ cung cấp cùng với nhãn bất thường được các bác sĩ tại bệnh viện thực hiện gán nhãn bằng ứng dụng labelme. Tập dữ liệu được chia với tỷ lệ Train/Validation/Test là 65%/15%/20%.
- [Brain MRI segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation): Gồm 3929 ảnh MRI não (1373 lát cắt có bất thường/khối u) với chuỗi xung FLAIR. Tập dữ liệu được sử dụng để tăng cường cho tập Train để huấn luyện mô hình.

## Tiền xử lý và tăng cường dữ liệu
**Tiền xử lý ảnh**

- Bước 1: Chuyển mặt nạ từ .JSON sang .JPG
- Bước 2: Lọc các bệnh nhân không có bất thường (33 bệnh nhân)
- Bước 3: Cắt bỏ viền ảnh (Dùng Yolov5)
- Chuẩn hóa giá trị từ [0,255] về [-1,1] (Pix2Pix) hoặc [0,1] (Efficient)

**Tăng cường hình ảnh**

Để tập dữ liệu cân bằng, cần tăng cường các hình ảnh có bất thường, sau đó các hình ảnh bất thường tăng lên 2 lần. Các phương pháp gồm:
- flip_left_right: Tỷ lệ 30%
- flip_up_down: Tỷ lệ 30%
- rotation 90: Tỷ lệ 40%
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/preprocessing.PNG)

## Kết quả thực nghiệm
Các thông số chung:
- Img_size: 256x256
- Batch_size: 64
- Steps: 20000
- Generator_lr: 0.0002
- Discriminator_lr: 0.0002
- Loss Distance: Euclid (L2)
- Lambda: 200

**Kết quả phân lớp**
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/kq1.PNG)
**Kết quả phân vùng bất thường**
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/kq2.PNG)
## Demo
![samples](https://github.com/liemkg1234/Pix2Pix_MRIBrain/blob/master/images/Demo.PNG)

### Liên hệ
Email:  liemkg1234@gmail.com

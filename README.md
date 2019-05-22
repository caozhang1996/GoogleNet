# GoogleNet

1、搭建模型的py文件使用的是TensorFlow官方的yu源码。

2、python create_tfrecords_files.py: 创建tfrecord文件

3、python create_txt_files.py: 创建class_name到class_id的映射文件

4、python inception_v1_train_val.py: 训练训练集中的数据并验证测试集中的数据

5、python predict.py: 预测test_image文件夹下的图片

1、整理原始数据，保存到trainSet.xlsx文件中
2、按不同情感分类分割数据集，保存到data文件夹下面，代码里面执行get_train_data函数，
在程序184行修改method的值为get_train_data，运行完毕后data目录下面就会有数据。
3、下载百度云链接：https://pan.baidu.com/s/1jee2p5rGBv9VFdTsEuwZfA 提取码：mhwd，下载预训练词向量，保存到data目录下。
然后修改184行method为build运行bilstm情感分类模型，模型会输出情感分类的准确率以及模型文件。
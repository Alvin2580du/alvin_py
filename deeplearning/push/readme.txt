
请根据样本和网络实际进行调整
CLASS_NUM = 2  总共有几类
INPUT_WIDTH = 224  网络输入width大小
INPUT_HEIGHT = 224 网络输入height大小
INPUT_CHANNEL = 3  通道为多少
BATCH_SIZE = 8   训练时用的batch size大小，请根据显存大小进行调整



先生成tfrecord
python make_tf.py --images_path ./data/train --record_path ./train_reocrd
指定目录及其子目录下的所有图片都会被加进去 label设定请看 generate_record方法
可以指定某一类图片如 *.jpg或 prefix*.png进行过滤，具体请看get_files方法


第一次使用resnet50预训练模型
python main.py --train_reocrd ./train_reocrd --resnet50_model_path ./models/resnet_v1_50.ckpt

当训练一段时间有logs目录下有checkpoint生成时，可以中途停止然后指定ckpt文件继续训练
python main.py --train_reocrd ./train_reocrd  --checkpoint_prefix ./checkpoint/model.ckpt-495


导出训练模型方便部署，会在指定的文件夹下生成 frozen_inference_graph.pb 可以供python及其它语言调用
python export_graph.py --checkpoint_prefix ./checkpoint/model.ckpt-495 --output_dir ./models

test_one.py  指定一个文件测试

test_batch.py 指定一个目录测试

preprocess.py  图片数据增广常用操作



日志查看及网络结构可视
完成一个epoch后可以开始看训练相关日志或网络结构可视化
需要安装tensorboard 
pip install tensorboard 

在训练目录下执行 端口随意 只要没有被其它服务占用
tensorboard --logdir ./logs --port 5555
或当前目录非训练目录，指定日志目录的绝对路径即可
tensorboard --logdir /root/logs --port 5555
然后浏览器访问  http://127.0.0.1:5555

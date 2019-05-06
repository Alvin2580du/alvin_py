import os.path
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import sys
from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
from sklearn.externals import joblib

ports = sys.argv[1]
define("port", default=ports, help="run on the given port", type=int)

# 加载模型
imdb_w2v = Word2Vec.load('w2v_model.pkl')
clf = joblib.load('svm_model.pkl')


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 构建待预测句子的向量
def get_predict_vecs(words, n_dim=300):
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    return train_vecs


# 对单个句子进行情感判断
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    result = clf.predict(words_vecs)
    if int(result[0]) == 1:
        return "positive"
    else:
        return "negative"


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class UserHandler(tornado.web.RequestHandler):
    def post(self):
        message = self.get_argument("message")
        print("输入的句子是：{}".format(message))
        res = svm_predict(message)
        self.render("message.html", message="{}的情感极性是：\n{}".format(message, res))


handlers = [
    (r"/", IndexHandler),
    (r"/user", UserHandler)
]

if __name__ == "__main__":
    """ 测试句子
    坐牢，怎么可能轻易放过
    把携程亲子园所有的老师全部全家处死一个不留
    妈呀，光看视频就已经哭的不行，这些人还有没有人性啊，希望法律严惩，给家长们一个交代。
    认错已经不是原谅的理由，必须严惩，孩子的伤害是无法弥补的
    中国改改法律吧，就是因为他们以前这种幼师犯罪判个一两年就了事，才有这么多人更甚，最少十年以上，严重判死刑，看有几个还敢的
    真应该给这些人判死刑啊
    真的是心疼到无法呼吸！！！！！啊啊啊啊啊啊妈的比
    没有职业道德就不用当幼师，承受不了孩子的吵闹各种调皮就不要当幼师，真的别当幼师，你都没爱心了，何必去当幼师，可怜的孩子遇见你真的是很可怜
    打死都不可惜
    我也是位母亲，看到这样的视频，真的是很揪心
    简直不配做人！简直无法理解！谁招来的这畜生也得负责任吧！不，畜生都比她强！
    这种人希望被国家拉黑
    """
    template_path = os.path.join(os.path.dirname(__file__), "template")

    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers, template_path)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

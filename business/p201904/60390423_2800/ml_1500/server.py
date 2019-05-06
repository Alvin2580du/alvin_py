import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import tornado
import os
import pandas as pd
from collections import Counter


def wordFreq(data_name):
    # 统计词频
    data = pd.read_excel("results/{}_Cut.xlsx".format(data_name))
    content = data['comment_cut'].values.tolist()
    content_list = []

    for i in content:
        if isinstance(i, str):
            for j in i.split():
                if len(j) == 1:
                    continue
                content_list.append(j)

    rows = []
    for x, y in Counter(content_list).most_common(50):
        res = "{}:{}\n".format(x, y)
        rows.append(res)
    return " ".join(rows)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class UserHandler(tornado.web.RequestHandler):
    def post(self):
        message = self.get_argument("message")
        print("要统计的事件是：{}".format(message))
        rows = wordFreq(message)
        self.render("message.html", message="{}".format(message), results='{}'.format(rows))


if __name__ == '__main__':
    ports = 8994
    define("port", default=ports, help="run on the given port", type=int)

    handlers = [
        (r"/", IndexHandler),
        (r"/user", UserHandler)
    ]
    template_path = os.path.join(os.path.dirname(__file__), "template")
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers, template_path)
    http_server = tornado.httpserver.HTTPServer(app)
    print("服务已启动")
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

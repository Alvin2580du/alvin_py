import json
import sys
import os
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpserver
import traceback
import tornado.template
import tornado.websocket
import time
import random
import re

# set default port


class PredictHandler(tornado.web.RequestHandler):
    def initialize(self):
        print("init predict handler")

    def get(self):
        # 拿到参数后返回
        dict = {}
        try:
            res = json.dumps(dict)
            self.finish(res)
        except Exception as e:
            print(e)
            traceback.print_exc()

    def post(self):

        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        try:
            id = self.get_argument("id", None)
            ret = {'id': id}
            self.write(json.dumps(ret))
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.write("exception {}".format(e.message))
        self.finish()


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/predict", PredictHandler), (r"/", MainHandler), (r"/chatsocket", ChatSocketHandler)]
        settings = dict(
            cookie_secret="XCZSANTBOT",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
        )
        print("application init finished")
        tornado.web.Application.__init__(self, handlers, **settings)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            self.render("index.html", messages=ChatSocketHandler.cache)
        except:
            traceback.print_exc()
            print("Error sending message", exc_info=True)


class ChatSocketHandler(tornado.websocket.WebSocketHandler):
    waiters = set()
    cache = []
    cache_size = 200

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):
        ChatSocketHandler.waiters.add(self)

    def on_close(self):
        ChatSocketHandler.waiters.remove(self)

    @classmethod
    def update_cache(cls, chat):
        cls.cache.append(chat)
        if len(cls.cache) > cls.cache_size:
            cls.cache = cls.cache[-cls.cache_size:]

    @classmethod
    def send_updates(cls, chat):
        print("sending message to %d waiters", len(cls.waiters))
        for waiter in cls.waiters:
            try:
                waiter.write_message(chat)
            except:
                print("Error sending message", exc_info=True)

    @classmethod
    def send_updates_xsrf(cls, xsrf, chat):
        print("sending message to %d waiters", len(cls.waiters))
        xsrf = xsrf.split('|')[-1]
        for waiter in cls.waiters:
            try:
                if str(waiter.__dict__['request'].__dict__).find(xsrf) > -1 or len(cls.waiters) == 1:
                    waiter.write_message(chat)
            except:
                print("Error sending message", exc_info=True)

    def on_message(self, message):
        parsed = tornado.escape.json_decode(message)
        print(parsed)
        # xsrf = parsed['_xsrf']
        ip = self.request.remote_ip
        msg = parsed['body'].strip()
        print("get msg: {}".format(msg))
        try:
            out, answer = generate_answer(msg)
            answer = re.sub(r'\[[^\[\]\n]*\]', '', answer)
            chat = {"msg": answer, "ip": ip, "out": out}
            # do report
            s_time = random.randint(1, 2)
            time.sleep(s_time)
            ChatSocketHandler.send_updates(chat)
        except Exception as e:
            print(str(e))


class ServerException(Exception):
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message
        print(message)


def create_server(port):
    print("server port {} starting...".format(port))
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()


def generate_answer(mess):
    return mess


# start a service
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("please input port: ex. 8990")
        exit(1)
    port = sys.argv[1]
    create_server(port)

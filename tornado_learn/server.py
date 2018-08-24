import os.path

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
import logging

define("port", default=8992, help="run on the given port", type=int)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            self.render("index.html")
        except:
            logging.error("Error sending message", exc_info=True)


class UserHandler(tornado.web.RequestHandler):
    def post(self):
        user_name = self.get_argument("username")
        user_email = self.get_argument("email")
        user_website = self.get_argument("website")
        user_language = self.get_argument("language")
        self.render("user.html", username=user_name, email=user_email, website=user_website, language=user_language)
        logging.info("{},{},{},{}".format(user_name, user_email, user_website, user_language))


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/user", UserHandler)
        ]

        settings = dict(
            cookie_secret="XCZSANTBOT",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            xsrf_cookies=False,
        )
        print()
        print("----------------------- Service Success start -----------------------")
        print()
        tornado.web.Application.__init__(self, handlers, **settings)


def create_server(port):
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()
    logging.info("server port {} starting...".format(port))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logging.warning("please input port: ex. 8990")
        exit(1)
    port = sys.argv[1]
    create_server(port)

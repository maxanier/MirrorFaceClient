from http.server import HTTPServer, BaseHTTPRequestHandler
import constants as const
import config

class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type',"application/json")
        self.end_headers()
    def do_GET(self):
        self._set_headers()
        if config.state == const.STATUS_RUNNING:
            config.state = const.STATUS_PAUSE
        else:
            config.state = const.STATUS_RUNNING

        self.wfile.write(bytes("{}","utf8"))
    def do_HEAD(self):
        self._set_headers()

def run_http_api(server_class=HTTPServer, handler_class=RequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print("Starting HTTP")
    httpd.serve_forever()
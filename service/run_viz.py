import http.server
import socketserver
import webbrowser
import os
import sys

# Define the port
PORT = 5647
DIRECTORY = "rules"

# Change directory to serve files from
if os.path.isdir(DIRECTORY):
    os.chdir(DIRECTORY)
else:
    print(f"Directory {DIRECTORY} not found!")
    sys.exit(1)

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print("Opening browser...")
        webbrowser.open(f"http://localhost:{PORT}/visualize_layers.html")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()
except OSError as e:
    if e.errno == 98 or e.errno == 10048: # Address already in use
        print(f"Port {PORT} is already in use. Trying port {PORT+1}...")
        with socketserver.TCPServer(("", PORT+1), Handler) as httpd:
            print(f"Serving at http://localhost:{PORT+1}")
            webbrowser.open(f"http://localhost:{PORT+1}/visualize_layers.html")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    else:
        raise
except KeyboardInterrupt:
    print("\nServer stopped.")


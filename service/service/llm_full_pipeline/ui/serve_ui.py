import http.server
import socketserver
import json
import os
from pathlib import Path

# Configuration
PORT = 8085
UI_DIR = Path(__file__).parent
REPORTS_DIR = UI_DIR.parent / "evaluator" / "reports"

class AuditHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/audit_dashboard.html"
            
        if self.path == "/audit_dashboard.html":
            return super().do_GET()
            
        if self.path == "/api/reports":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            reports = []
            if REPORTS_DIR.exists():
                reports = sorted(
                    [f.name for f in REPORTS_DIR.glob("*.json")],
                    reverse=True
                )
            self.wfile.write(json.dumps(reports).encode())
            return

        if self.path.startswith("/reports/"):
            filename = self.path.replace("/reports/", "")
            file_path = REPORTS_DIR / filename
            
            if file_path.exists() and file_path.parent == REPORTS_DIR:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(file_path.read_bytes())
                return
            else:
                self.send_error(404, "Report not found")
                return

        # Serve static files from UI dir
        return super().do_GET()

    def translate_path(self, path):
        # Override to serve from UI_DIR by default
        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        # Don't mess with /reports/ logic which is handled in do_GET before this if not caught
        # But for static files (html, css, js), serve from UI_DIR
        return str(UI_DIR / path.lstrip("/"))

def run():
    # Ensure reports dir exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    os.chdir(UI_DIR) # Set CWD to UI dir for convenience
    
    with socketserver.TCPServer(("", PORT), AuditHandler) as httpd:
        print(f"Serving Audit Dashboard at http://localhost:{PORT}")
        print(f"Reports directory: {REPORTS_DIR}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run()




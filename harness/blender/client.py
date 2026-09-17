import sys
import os
import json
import socket
import argparse

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876

def send_blender_code(code: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 300.0):
    """Sends python code to Blender socket server and returns response dict."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
    except Exception as e:
        return {"status": "error", "message": f"Could not connect to Blender at {host}:{port}. Is Blender running and server started? Details: {e}"}

    command = {
        "type": "execute_code",
        "params": {
            "code": code
        }
    }
    
    sock.sendall(json.dumps(command).encode('utf-8'))
    
    chunks = []
    while True:
        chunk = sock.recv(8192)
        if not chunk:
            break
        chunks.append(chunk)
        try:
            data = b''.join(chunks)
            response = json.loads(data.decode('utf-8'))
            sock.close()
            return response
        except json.JSONDecodeError:
            continue

    sock.close()
    return {"status": "error", "message": "Incomplete response from Blender"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Python scripts directly inside running Blender instance.")
    parser.add_argument("script_file", nargs="?", help="Path to Python script file to execute in Blender")
    parser.add_argument("-c", "--code", help="Inline python code string to execute")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Blender MCP host (default: localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Blender MCP port (default: 9876)")
    
    args = parser.parse_args()
    
    code_to_run = ""
    if args.code:
        code_to_run = args.code
    elif args.script_file and os.path.exists(args.script_file):
        with open(args.script_file, "r", encoding="utf-8") as f:
            code_to_run = f.read()
    else:
        print("Usage error: Provide a script file path or use -c 'python code'")
        sys.exit(1)

    print(f"Sending code to Blender ({args.host}:{args.port})...")
    res = send_blender_code(code_to_run, host=args.host, port=args.port)
    print("Response:", json.dumps(res, indent=2))

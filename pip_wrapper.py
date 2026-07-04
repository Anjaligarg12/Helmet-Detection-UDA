import socket
import sys
import runpy

old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    ipv4_responses = [response for response in responses if response[0] == socket.AF_INET]
    return ipv4_responses if ipv4_responses else responses
socket.getaddrinfo = new_getaddrinfo

sys.argv[0] = 'pip'
runpy.run_module('pip', run_name='__main__')

#Author: Everett Stenberg
#Description:   functionality for networking 


import socket
import os 

#Colorful display on terminal
class Color:
    os.system("")
    blue    = '\033[94m'
    tan     = '\033[93m'
    green   = '\033[92m'
    red     = '\033[91m'
    bold    = '\033[1m'
    end     = '\033[0m'    



def send_bytes(sock:socket.socket,bytes_message:bytes,pack_len:int) -> bool:

    #Confirm with the recipient that were sending them data
    sock.send('sendbytes'.encode())
    confirmation            = sock.recv(32).decode()
    if not confirmation == 'ready':
        print(f"{Color.red}\trecipient failed confirmation, sent: '{confirmation}'{Color.end}")
        return False
    
    #Confirm with recipient the message to be sent 
    bytes_len               = len(bytes_message)
    sock.send(str(bytes_len).encode())
    confirmation            = int(sock.recv(32).decode())
    if not confirmation == bytes_len:
        print(f"{Color.red}recipient failed length check, sent: '{confirmation}' != {bytes_len}{Color.end}")
        return False
    
    #Send data 
    window                  = 0 
    while window < bytes_len:

        data_packet         = bytes_message[window:window+pack_len]
        sock.send(data_packet)
        window              += pack_len
    
    #Confirm with they finished up
    confirmation            = sock.recv(32).decode()
    
    #Send confirmation back
    sock.send('sentbytes'.encode())
    if not confirmation == 'recieved':
        print(f"recipient failed receipt, sent: '{confirmation}'")
        return False
    return True


def recieve_bytes(sock:socket.socket,pack_len:int):
    #Confirm with sender that they're sending bytes 
    send_intent             = sock.recv(32).decode()
    if not send_intent == 'sendbytes':
        print(f"unexpected message from sender: '{send_intent}'")
        return False
    sock.send('ready'.encode())

    #Get and confirm message length
    bytes_len               = int(sock.recv(32).decode())
    sock.send(str(bytes_len).encode())

    #Download bytes
    bytes_message           = bytes() 
    while len(bytes_message) < bytes_len:

        data_packet         = sock.recv(pack_len)
        bytes_message       += data_packet

    #Let em know were done
    sock.send('recieved'.encode())

    #Recieve confirmation from sender 
    confirmation            = sock.recv(32).decode()
    if not confirmation == 'sentbytes':
        print(f"Expected end of transmission, got '{confirmation}'")
        return False

    return bytes_message
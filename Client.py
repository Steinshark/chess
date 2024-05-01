import alg_train
from model import ChessModel
from net_chess import Color
import torch
import json
import socket
import time
from hashlib import md5
from io import BytesIO
from threading import Thread
from parallel_mctree import MCTree_Handler

class Client(Thread):


    def __init__(self,address='localhost',port=15555,device=None,pack_len=8192):
        super(Client,self).__init__()

        #Setup socket 
        self.client_socket          = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.address                = address
        self.port                   = port
        self.running                = True
        self.pack_len               = pack_len

        #game related variables
        self.model_state            = None
        self.model_hash             = None
        self.device                 = device
        self.lookup_dict            = {}
        self.max_lookup_len         = 100_000


    #Runs the client.
    #   loops getting game type and running
    #   that type of game 
    def run(self):

        #Initialize the connection with the server and receive id
        self.client_socket.connect((self.address,self.port))
        self.id                     = int(self.client_socket.recv(32).decode())
        print(f"\t{Color.green}client connected to {Color.tan}{self.address}{Color.green} with id:{self.id}{Color.end}")


        #Generate game_handler 
        self.mctree_handler         = MCTree_Handler(8,self.device,160,800)
        #Do for forever until we die
        while self.running:

            #Run training game 
            self.execute_game()
            print(f"\t\t{Color.green}executed game{Color.end}")


    #Gets 1 model's parameters from 
    #   the client_manager
    def recieve_model_params(self):

        #Check not running 
        if not self.running:
            return

        #Receive params hash and check against current  
        params_hash                         = self.client_socket.recv(128)
        if params_hash == self.model_hash:
            #Send skip
            self.client_socket.send('skip'.encode())

            #Get confirmation of skip
            server_confirmation                 = self.client_socket.recv(32).decode()
            if not server_confirmation == 'skip':
                print(f"\t{Color.red}server did not confirm skip: sent '{server_confirmation}'{Color.end}")
            else:
                print(f"\t{Color.tan}skipping model download{Color.end}")

        else:

            #Send 'send'
            self.client_socket.send('send'.encode())

            #Received length 
            message_len                         = int(self.client_socket.recv(32).decode())

            #Confirm packet_len
            self.client_socket.send(str(message_len).encode())

            #Cumulative add to message until full size recieved 
            message                     = bytes()
            while len(message) < message_len:
                #Recieve the next pack_len bytes 
                message                 += self.client_socket.recv(self.pack_len)
            buffer                      = BytesIO(message)
            params_as_bytes             = buffer.getvalue()
            print(f"\t{Color.tan}recieved {len(params_as_bytes)} bytes{Color.end}")
            self.model_state            = torch.load(BytesIO(params_as_bytes))
            self.model_hash             = md5(params_as_bytes).digest()

            #Reset mctree handler dict 
            self.mctree_handler.lookup_dict = {}


        #Check model state works
        self.current_model          = ChessModel(19,16).cpu()
        self.current_model.load_state_dict(self.model_state)
        self.current_model.float()

        self.client_socket.send("done".encode())


    #Runs a game based on the type of 
    #   recieved by the client_manager
    def execute_game(self):

        #Get/check model params
        self.recieve_model_params()
        self.mctree_handler.load_dict(self.current_model)

        #Get game_params
        data_packet             = self.client_socket.recv(1024).decode()
        game_parameters         = json.loads(data_packet)
        max_game_ply            = game_parameters['ply']
        n_iters                 = game_parameters['n_iters']
        n_experiences           = game_parameters['n_exp']
        n_parallel              = game_parameters['n_parallel']

        #Update game manager 
        self.mctree_handler.update_game_params(max_game_ply,n_iters,n_parallel)
        
        #Generate data 
        t0                      = time.time()
        training_data           = self.mctree_handler.collect_data(n_exps=n_experiences)
        print(f"\t\t{Color.tan}{(time.time()-t0)/len(training_data):.2f}s/move{Color.end}")
        
        #Upload data
        self.upload_data(training_data)

        #Take memory off the cuda device 
        torch.cuda.empty_cache()


    #Defines the protocol for sending bytes (of any length)
    def send_bytes(self,bytes_message:bytes) -> None:

        #Confirm with the recipient that were sending them data
        self.client_socket.send('sendbytes'.encode())
        confirmation            = self.client_socket.recv(32).decode()
        if not confirmation == 'ready':
            print(f"recipient failed confirmation, sent: '{confirmation}'")
            exit()
        
        #Confirm with recipient the message to be sent 
        bytes_len               = len(bytes_message)
        self.client_socket.send(str(bytes_len).encode())
        confirmation            = int(self.client_socket.recv(32).decode())
        if not confirmation == bytes_len:
            print(f"recipient failed length check, sent: '{confirmation}' != {bytes_len}")
            exit()
        
        #Send data 
        window                  = 0 
        while window < bytes_len:

            data_packet         = bytes_message[window:window+self.pack_len]
            self.client_socket.send(data_packet)

            window              += self.pack_len
        
        #Confirm with the recipient that we sent the data
        time.sleep(.1)
        self.client_socket.send('sentbytes'.encode())
        confirmation            = self.client_socket.recv(32).decode()
        if not confirmation == 'recieved':
            print(f"recipient failed receipt, sent: '{confirmation}'")
            exit()
        
        return


    #Defines the protocol for receiving bytes (of any length)
    def recieve_bytes(self) -> bytes:

        #Confirm with sender that they're sending bytes 
        send_intent             = self.client_socket.recv(32).decode()
        if not send_intent == 'sendbytes':
            print(f"unexpected message from sender: '{send_intent}'")
            exit()
        self.client_socket.send('ready'.encode())

        #Get and confirm message length
        bytes_len               = int(self.client_socket.recv(32).decode())
        self.client_socket.send(str(bytes_len).encode())

        #Download bytes
        bytes_message           = bytes() 
        while len(bytes_message) < bytes_len:

            data_packet         = self.client_socket.recv(self.pack_len)
            bytes_message       += data_packet
        
        #Recieve confirmation from sender 
        confirmation            = self.client_socket.recv(32).decode()
        if not confirmation == 'sentbytes':
            print(f"Expected end of transmission, got '{confirmation}'")
            exit()
        
        return bytes_message


    #Uploads data from the client to the client_manager
    #   handles both 'Train' and 'Test' modes
    def upload_data(self,data):

        #Check for dead 
        if not self.running:
            #print(f"detected not running")
            return

        bytes_data                      = json.dumps(data).encode()

        self.send_bytes(bytes_data)

        #Reset 
        self.mctree_handler.dataset     = []


    #Closes down the socket and everything else
    def shutdown(self):
        print(f"Closing socket")
        self.running = False
        self.mctree_handler.stop_sig     = True
        self.client_socket.close()
        print(f"joined and exiting")
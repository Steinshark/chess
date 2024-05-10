#Author: Everett Stenberg
#Description:   Handles client side computing and interfacing with the 
#              client_manager (found in net_chess.py)

from model import ChessModel
from utilities import Color
import torch
import json
import socket
import time
from hashlib import md5
from io import BytesIO
from threading import Thread
from parallel_mctree import MCTree_Handler
import networking
import settings 

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
        params_hash                             = networking.recieve_bytes(self.client_socket,self.pack_len)
        if params_hash == self.model_hash:
            #Send skip
            networking.send_bytes(self.client_socket,'skip'.encode(),self.pack_len)

            #Get confirmation of skip
            server_confirmation                 = networking.recieve_bytes(self.client_socket,self.pack_len).decode()
            
            if not server_confirmation == 'skip':
                print(f"\t{Color.red}server did not confirm skip: sent '{server_confirmation}'{Color.end}")
            else:
                print(f"\t{Color.tan}skipping model download{Color.end}")

        else:
            #Send 'send'
            networking.send_bytes(self.client_socket,'send'.encode(),self.pack_len)

            #receive model params
            params_as_bytes                     = networking.recieve_bytes(self.client_socket,self.pack_len)    
            #Convert to model state and hashvalue
            self.model_state                    = torch.load(BytesIO(params_as_bytes))
            self.model_hash                     = md5(params_as_bytes).digest()

            print(f"\t{Color.tan}downloaded {len(params_as_bytes)} bytes{Color.end}")
            #Reset mctree handler dict 
            self.mctree_handler.lookup_dict = {}


        #Check model state works
        self.current_model                      = ChessModel(17,16).cpu().type(settings.DTYPE)
        self.current_model.load_state_dict(self.model_state)


    #Runs a game based on the type of 
    #   recieved by the client_manager
    def execute_game(self):

        #Get/check model params
        self.recieve_model_params()
        self.mctree_handler.load_dict(self.current_model)

        #Get game_params
        data_packet             = networking.recieve_bytes(self.client_socket,self.pack_len).decode()
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
        if len(training_data) == 0:
            exit()
        print(f"\t\t{Color.tan}{(time.time()-t0)/len(training_data):.2f}s/move{Color.end}")
        
        #Upload data
        self.upload_data(training_data)

        #Take memory off the cuda device 
        torch.cuda.empty_cache()


    #Uploads data from the client to the client_manager
    #   handles both 'Train' and 'Test' modes
    def upload_data(self,data):

        #Check for dead 
        if not self.running:
            #print(f"detected not running")
            return

        bytes_data                      = json.dumps(data).encode()

        networking.send_bytes(self.client_socket,bytes_data,self.pack_len)

        #Reset 
        self.mctree_handler.dataset     = []


    #Closes down the socket and everything else
    def shutdown(self):
        print(f"Closing socket")
        networking.send_bytes(self.client_socket,"kill".encode(),self.pack_len)
        self.running = False
        self.mctree_handler.stop_sig     = True
        self.client_socket.close()
        print(f"joined and exiting")
        exit()
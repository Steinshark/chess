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


    #Uploads data from the client to the client_manager
    #   handles both 'Train' and 'Test' modes
    def upload_data(self,data):

        #Check for dead 
        if not self.running:
            #print(f"detected not running")
            return


        self.client_socket.send('data send'.encode())

        #Send exps
        for packet_i,exp in enumerate(data):

            #Wait for "Ready" from server
            confirm         = self.client_socket.recv(32).decode()
            if not confirm == "Ready":
                print(f"{Color.red}didnt get Send, got {confirm}{Color.end}")
                break

            #separate fen,move_data,and outcome and 
            #   json convert to strings to send over network
            fen:str         = exp[0]
            move_stats:str  = json.dumps(exp[1])
            outcome:str     = str(exp[2])
            q_score:str     = str(exp[3])

            #Combine strings 
            data_packet     = (fen,move_stats,outcome,q_score)
            data_packet     = json.dumps(data_packet)

            #encode to bytes and send to server 
            self.client_socket.send(data_packet.encode())


        #Receive the last "Ready"
        confirm         = self.client_socket.recv(32).decode()

        #Let server know thats it
        self.client_socket.send("End".encode())

        #Reset 
        self.mctree_handler.dataset     = []


    #Closes down the socket and everything else
    def shutdown(self):
        print(f"Closing socket")
        self.running = False
        self.client_socket.close()
        print(f"joined and exiting")
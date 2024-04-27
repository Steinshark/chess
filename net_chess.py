#Author: Everett Stenberg
#Description:   defines functions for training the network from
#               start to finish WITH NETWORK and THREADING CONSIDERATIONS



from collections import OrderedDict
import torch 
from mctree import MCTree
import socket
import json
import numpy
from queue import Queue
from threading import Thread
from model import ChessModel2
import alg_train
from io import BytesIO
import random
import trainer 
from copy import deepcopy
import time 
import os
from hashlib import md5

#Colorful display on terminal
class Color:
    os.system("")
    blue    = '\033[94m'
    tan     = '\033[93m'
    green   = '\033[92m'
    red     = '\033[91m'
    bold    = '\033[1m'
    end     = '\033[0m'    


#Runs on the client machine and generates training data 
#   by playing games againts itself
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
        
<<<<<<< HEAD
        try:
            #Initialize the connection with the server and receive id
            self.client_socket.connect((self.address,self.port))
            self.id                     = int(self.client_socket.recv(32).decode())
            print(f"\t{Color.green}client connected to {Color.tan}{self.address}{Color.green} with id:{self.id}{Color.end}")
                #Do for forever until we die
            while self.running:
                
                #Recieve game type 
                self.receive_game_type()
                print(f"\n\t{Color.tan}recieved game type - {self.game_mode}{Color.end}")

                #Execute that game 
                self.execute_game()
                print(f"\t\t{Color.green}executed game{Color.end}")
        
        except TimeoutError:
            print(f"{Color.red}\t\tClient timeout{Color.end}")
            self.shutdown() 
        except ConnectionResetError:
            print(f"{Color.red}\t\tClient timeout{Color.end}")
            self.shutdown() 
=======
        #Initialize the connection with the server and receive id
        self.client_socket.connect((self.address,self.port))
        self.id                     = int(self.client_socket.recv(32).decode())
        print(f"\t{Color.green}client connected to {Color.tan}{self.address}{Color.green} with id:{self.id}{Color.end}")
    
        #Do for forever until we die
        while self.running:

            #Run training game 
            self.execute_game()
            print(f"\t\t{Color.green}executed game{Color.end}")
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7


 
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

<<<<<<< HEAD
        #attempt to instantiate model with them
        self.current_model          = ChessModel2()
        self.current_model.load_state_dict(model_parameters)
        self.current_model.float().to(self.device)
        self.client_socket.send("Recieved".encode())
=======
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


        #Check model state works
        self.current_model          = ChessModel2(19,24).cpu()
        self.current_model.load_state_dict(self.model_state)
        self.current_model.float()

        self.client_socket.send("done".encode())
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7



    #Runs a game based on the type of 
    #   recieved by the client_manager
    def execute_game(self):

        #Get/check model params
        self.recieve_model_params()

        #Get game_params
        data_packet             = self.client_socket.recv(1024).decode()
        game_parameters         = json.loads(data_packet)
        max_game_ply            = game_parameters['ply']
        n_iters                 = game_parameters['n_iters']

        #Run game 
        t0                      = time.time()
        training_data           = alg_train.play_game(self.model_state,max_game_ply,n_iters,self,self.device,self.lookup_dict)
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

<<<<<<< HEAD
                #separate fen,move_data,and outcome and 
                #   json convert to strings to send over network
                fen:str         = exp[0]
                move_stats:str  = json.dumps(exp[1])
                outcome:str     = str(exp[2])
                q_value         = str(exp[3])

                #Combine strings 
                data_packet     = (fen,move_stats,outcome,q_value)
                data_packet     = json.dumps(data_packet)
=======
            #separate fen,move_data,and outcome and 
            #   json convert to strings to send over network
            fen:str         = exp[0]
            move_stats:str  = json.dumps(exp[1])
            outcome:str     = str(exp[2])

            #Combine strings 
            data_packet     = (fen,move_stats,outcome)
            data_packet     = json.dumps(data_packet)
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7

            #encode to bytes and send to server 
            self.client_socket.send(data_packet.encode())


        #Receive the last "Ready"
        confirm         = self.client_socket.recv(32).decode()

        #Let server know thats it
        self.client_socket.send("End".encode())


    #Closes down the socket and everything else
    def shutdown(self):
        print(f"Closing socket")
        self.running = False 
        self.client_socket.close()
        print(f"joined and exiting")



#Handles a single client and gives it a model to use,
#   tells it to play games, and translates it back
#   to the Server 
class Client_Manager(Thread):

    def __init__(self,client_socket:socket.socket,connection:str,client_id:str,client_queue:Queue,model_params:OrderedDict,game_params,pack_len=8192):
        super(Client_Manager,self).__init__()
        self.client_socket          = client_socket
        self.client_address         = connection[0]
        self.client_port            = connection[1]
        self.id                     = client_id
        self.queue                  = client_queue
        self.running                = True
        self.pack_len               = pack_len

        #Game vars 
        self.game_params            = game_params

        #Model vars 
        self.top_model_params       = model_params
        self.model1_params          = None
        self.model2_params          = None
        self.in_game                = False
        self.lock                   = False

        #Set 500 sec timeout 
        self.client_socket.settimeout(500)

        print(f"\n\t{Color.green}launched a new client manager for {Color.tan}{self.client_address}\n{Color.end}")


    def is_alive(self):
        if not isinstance(self.client_address,socket.socket):
            return False
        return True
    

    def recieve_model(self,new_model):
        self.model_state                    = new_model
    

    def run_training_game(self):

        #Send model parameters to generate training data 
        self.send_model_params(self.top_model_params)

        #Send game parameters to play with
        self.send_game_params(self.game_params)
        
        #CLIENT PLAYS GAME 

        #Receive training data from game 
        self.recieve_client_result()


    def run(self):
        try:
    
            #First, send job worker id to client 
            self.client_socket.send(str(self.id).encode())

            while self.running:
                self.in_game            = True

                #Execute game and get results
                self.run_training_game()
                
                self.in_game                = False

                while self.lock:
                    time.sleep(1)
                            


                #client_socket.send("Kill".encode())
        except TimeoutError:
            print(f"\n\t{Color.red}client shutting down{Color.end}")
            self.shutdown()
        except OSError:
            print(f"\n\t{Color.red}Lost communication with client{Color.end}")
<<<<<<< HEAD
            self.shutdown()
 
=======
            return False
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7

    
    def send_model_params(self,parameters:OrderedDict):

        #Create a BytesIO buffer to load model into
        data_buffer             = BytesIO()

        #Load params into data_buffer as bytes
        torch.save(parameters,data_buffer)

        #Get bytes out of buffer
        params_as_bytes         = data_buffer.getvalue()

        #Get hash 
        params_hash                     = md5(params_as_bytes).digest()

        #PROTOCOL:
        #   send client hash
        #   if client already has model it will reply 'skip'
        #       confirm skip
        #   otherwise, it will reply 'send
        #       send client n_bytes it will recieve 
        #       get client confirmation of n_bytes
        #       send packets of pack_len bytes until end 
        
        #   wait for Done signal


        #Check if client has model
        self.client_socket.send(params_hash)
        client_response                         = self.client_socket.recv(32).decode()

        if client_response == 'skip':
            self.client_socket.send('skip'.encode())
        
        elif client_response == 'send':
            data_len                            = len(params_as_bytes)
            self.client_socket.send(str(data_len).encode())

            data_len_confirmation               = int(self.client_socket.recv(32).decode())
            
            if not data_len_confirmation == data_len:
                print(f"client didnt recieve proper byte len, got {data_len_confirmation}")
            window                  = 0 
            while window < data_len:
                
                #Prep and send data packet
                data_packet         = params_as_bytes[window:window+self.pack_len]
                self.client_socket.send(data_packet)
                window              += self.pack_len

        #Get confirmation from client "Recieved"
        self.client_socket.recv(32)


    def send_game_params(self,parameters:dict):
               
        #Send game parameters to client 
        data_packet                 = json.dumps(parameters)
        self.client_socket.send(data_packet.encode())


    def recieve_client_result(self):

        #Check what client is saying
        send_confirmation                   = self.client_socket.recv(32).decode()

        #If 'Result', then only outcome is sent
        if send_confirmation == 'data send':

            #Recieve data until "End" signal
            while True:
                
                #Send 'Ready' to recieve and get next client response
                self.client_socket.send("Ready".encode())
                client_response     = self.client_socket.recv(32768).decode()

                #Break if recieve 'End'
                if client_response  == "End":
                    break 

                #Process data if else
                else:
                    #Add experience to the experience queue to transfer
                    data_packet     = json.loads(client_response)
                    #Re-convert             FEN                   MOVE_DISTR                      OUTCOME
                    client_response = (data_packet[0],json.loads(data_packet[1]),float(data_packet[2]),float(data_packet[3]))
                    
                    #Place in the data queue for the Server thread to fetch 
                    self.queue.put(client_response)

        
        #if 'Experience', then will recieve a list of experiences
        else:
            print(f"client send strange answer: '{send_confirmation}'")
    

    def recieve_test_game(self,game_packet):
        self.p1             = game_packet[0]
        self.p2             = game_packet[1]

        self.model1_params  = game_packet[2]
        self.model2_params  = game_packet[3]
        self.game_uid       = game_packet[4]


        #unlock and go for it 
        self.unlock()
        self.in_game        = True


    def set_lock(self):
        self.lock   = True 
    

    def unlock(self):
        self.lock   = False


    def shutdown(self):
        self.running                = False
        self.client_socket.close()


#Handles the server (Aka in charge of the training algorithm)
#   each client that connects will get its own Client_Manager 
#   and generate trainin games
class Server(Thread):


    def __init__(self,address='localhost',port=15555,pack_len=8192):
        
        #Init super
        super(Server,self).__init__()

        #Network items
        self.client_managers:list[Client_Manager]   = [] 
        self.build_socket(address,port)
        self.pack_len                               = pack_len
        self.running                                = True 
        self.lock                                   = False

        #Model items 
<<<<<<< HEAD
        self.model_params                   = {0:ChessModel2().cpu().state_dict()}
        self.top_model                      = 0 
        self.game_params                    = {"ply":80,"n_iters":400}
        self.test_params                    = {"ply":120,"n_iters":400,'n_games':12}

        #Training items 
        self.current_generation_data        = [] 
        self.all_training_data              = []
        self.window_size                    = 65536
        self.train_thresh                   = 32768

        self.bs                             = 4096
        self.lr                             = .002
        self.wd                             = .01 
        self.betas                          = (.5,.8)
        self.n_epochs                       = 1 
        self.gen                            = 0 

        self.update_iter                    = 30
        self.next_update_t                  = time.time() + self.update_iter
        self.game_stats                     = []
        self.lr_mult                        = .9

        self.max_models                     = 4   
=======
        self.chess_model                            = ChessModel2(19,24).eval().cpu().float()
        self.model_state                            = self.chess_model.state_dict()
        self.device                                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Game items
        self.game_params                            = {"ply":120,"n_iters":50}

        #Training vars
        self.data_pool                              = [] 
        self.train_every                            = 4096
        self.exp_counter                            = 0
        self.bs                                     = 1024        
        self.lr                                     = .001
        self.wd                                     = .01 
        self.betas                                  = (.5,.8)
        self.n_epochs                               = 1 
        self.train_step                             = 0 
        self.lr_mult                                = .8 

        #Telemtry vars
        self.update_iter                            = 30
        self.next_update_t                          = time.time() + self.update_iter
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7


    #Run the server
    def run(self):
        
        #Always on 
        while self.running:

            #Check for a new clients
            self.check_for_clients()

            #Update server training state
            self.update_training_state()

            #Update client states
            self.update_clients()


    #Implement this to load models
    def load_models(self):
        
        pass

    #Creates the server socket and sets 
    #   various server settings
    def build_socket(self,address,port):
        
        #Set class vars
        self.address                = address 
        self.port                   = port

        #Create socket 
        self.server_socket          = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server_socket.bind((address,port))
        self.server_socket.settimeout(.1)
        self.server_socket.listen(16)   #Accept up to 16 simultaneous connections
        print(f"\t{Color.green}server listening on {Color.tan}address{Color.end}")


    #Assign the next available client id
    def get_next_id(self):
        candidate_id    = 0
        while candidate_id in [client.id for client in self.client_managers]:
            candidate_id += 1
        return candidate_id
    

    #Check all clients and remove those that are dead 
    def check_dead_clients(self):

        #Find all dead clients
        hitlist:list[Client_Manager]    = [] 
        for client in self.clients: 
            if not client.running:
                hitlist.append(client)

        #Remove all dead clients
        for dead_client in hitlist:
            self.clients.remove(dead_client)
            print(f"\t{Color.red}remove client: {dead_client.id}{Color.end}")


    #Updates clients with the most recent parameters and 
    def update_clients(self):
        
<<<<<<< HEAD
        self.check_dead_clients()

        for client in self.clients:
            #Pass-on most recent model 
            client.recieve_model(self.model_params[self.top_model])
=======
        hitlist         = [] 
        for client_index,client in enumerate(self.client_managers):

            #Check client is alive 
            if not client.running:
                hitlist.append(client_index)
            else:
                #Pass-on most recent model 
                client.recieve_model(self.model_state)
                
        for client_index in hitlist:
            self.client_managers.pop(client_index)
            print(f"\tremove client: {client_index}")
        pass 
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7


    #Blocks until all clients have finished their game and 
    #   the client_managers have passed them back to the server
    def sync_all_clients(self):

        #Lock server 
        self.lock                                   = True 
        found_running_game                          = True 

        while found_running_game:
            found_running_game                      = False

<<<<<<< HEAD
            self.check_dead_clients()
            for client in self.clients:
                client.lock             = True 
=======
            for client in self.client_managers:
                client.lock                         = True 
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7
                
                if client.in_game:
                    found_running_game              = True 
                
   
   #Unblocks client managers and lets them generate games 
   #    until sync'd again
    def unsync_all_clients(self):

        #Unlock all clients
        for client in self.client_managers:
            client.unlock()


    #Curates the list of training examples for a 
    #   given training run
    def add_experiences_to_set(self):

        #If not, then apply window algorithm 
        self.all_training_data          = self.all_training_data + self.current_generation_data
        self.all_training_data          = self.all_training_data[int(-self.window_size*min(1,(self.gen+1)/3)):]


    #Tests a model agains a stockfish dataset 
    def test_model(self,chess_model:ChessModel2):
        p_loss,v_loss                           = trainer.check_vs_stockfish(chess_model)
        print(f"\t\t{Color.blue}PRE_TRAIN:\n\t\tp_loss:{Color.tan}{p_loss:.4f}{Color.blue}\t\tv_loss:{Color.tan}{v_loss:.4f}\n\n")
        
    
    #Trains the current model
    def train_current_model(self,chess_model:ChessModel2):
        
        print(f"\t\t{Color.blue}TRAINING:")

        #Ensure proper state 
        chess_model.cuda().train()  

        #Train loop
        for epoch_num in range(self.n_epochs):

            #Select training batch 
            if len(self.all_training_data) == len(self.current_generation_data):
                dataset                         = self.all_training_data
            else:
                dataset                         = random.sample(self.all_training_data,k=int(len(self.all_training_data)/3))
            
            dataset                             = trainer.TrainerExpDataset(dataset)
            #Train new model 
            p_losses,v_losses                   = trainer.train_model(chess_model,dataset,bs=self.bs,lr=self.lr,wd=self.wd,betas=self.betas,n_epochs=self.n_epochs)
            print(f"\t\t{Color.blue}[{epoch_num}]: p_loss:{Color.tan}{p_losses[-1]:.4f}\t{Color.blue}v_loss:{Color.tan}{v_losses[-1]:.4f}\t{Color.blue}size:[{len(dataset)}]{Color.end}\n")

        return chess_model
    

    #Finds the latest top_model
    def find_best_model(self):
        #Generate lists for the algorithm
        remaining_models                = [(id,self.model_params[id]) for id in self.model_params]
        bracket                         = []
        byes                            = []

        #Create the first bracket (recursive)
        self.create_test_bracket(remaining_models,bracket,byes)

        #Run, then generate new until done
        while len(remaining_models) > 1:
            outcomes,winners            = self.run_bracket(bracket)
            remaining_models            = [(id,self.model_params[id]) for id in winners+byes]
            bracket                     = []
            byes                        = []
            self.create_test_bracket(remaining_models,bracket,byes)

        #Set new top model
        self.top_model                  = winners[0]
        print(f"\t{Color.tan}Top Model: {self.top_model}{Color.end}")


        #Increment generation
        self.gen                        += 1

        #If params > 5, drop lowest id that didnt win 
        while len(self.model_params) > self.max_models:
            id                          = 0
            while not id in self.model_params or self.top_model == id:
                id += 1
            print(f"{Color.tan}\t dropped model {id}{Color.end}")
            del self.model_params[id]
                
        return


    #Add experiences to server's list and 
    #   Check if its time to train 
    def update_training_state(self):

        #Snatch data from queues and update total exps gathered
        for client in self.client_managers:
            while not client.queue.empty():
                self.data_pool.append(client.queue.get())
                self.exp_counter                        += 1

        #Print statistics 
        if time.time() - self.next_update_t > 0:    
<<<<<<< HEAD
            
            #Add leading zeros to len
            cur_len_string                  = str(len(self.current_generation_data))
            while len(cur_len_string) < len(str(self.train_thresh)):
                cur_len_string = "0" + cur_len_string

            print(f"\t{Color.tan}Generation [{self.gen}]:\t[{cur_len_string}/{self.train_thresh}] samples accumulated{Color.green} - running!{Color.end}")
            self.next_update_t              = time.time() + self.update_iter


        #Once over self.train_thresh, train and update model 
        if len(self.current_generation_data) > self.train_thresh:
            
            #Set in test mode
            self.test_mode                  = True 
            
            #Dont train until all games are done
            print(f"\n\t{Color.tan}Syncing Clients",end='')
            self.sync_all_clients()

            #Snatch all experiences still in queues
            for client in self.clients:
                while not client.queue.empty():
                    self.current_generation_data.append(client.queue.get())
        
            #Show some data
            print(f" - collected [{len(self.current_generation_data)}]{Color.end}")
            print(f"\n\n\t{Color.blue}Training Gen {Color.tan}{self.gen}:\n{Color.end}")
            print(f"\t{Color.blue}TRAIN PARAMS{Color.end}")
            print(f"\t\t{Color.blue}bs:\t{Color.tan}{self.bs}{Color.end}")
            print(f"\t\t{Color.blue}lr:\t{Color.tan}{self.lr:.4}{Color.end}")
            print(f"\t\t{Color.blue}betas:\t{Color.tan}{self.betas}{Color.end}")
            print(f"\t\t{Color.blue}iters:\t{Color.tan}{(self.game_params['n_iters'])}{Color.end}")
            print(f"\t\t{Color.blue}ply:\t{Color.tan}{(self.game_params['ply'])}\n\n{Color.end}")

            #Handle data 
            self.add_experiences_to_set()

            #Prep next model with current best params
            next_gen_model                  = ChessModel2()
            next_gen_model.load_state_dict(self.model_params[self.top_model])
            next_gen_model.cuda().eval().float()

            #View performance vs stockfish before
            p_loss,v_loss                   = trainer.check_vs_stockfish(next_gen_model)
            print(f"\t\t{Color.blue}PRE_TRAIN:\n\t\tp_loss:{Color.tan}{p_loss:.4f}{Color.blue}\t\tv_loss:{Color.tan}{v_loss:.4f}\n\n")
            
            #Train new model
            self.train_current_model(next_gen_model)

            #View performance vs stockfish after
            p_loss,v_loss                   = trainer.check_vs_stockfish(next_gen_model)
            print(f"\t\t{Color.blue}PRE_TRAIN:\n\t\tp_loss:{Color.tan}{p_loss:.4f}{Color.blue}\t\tv_loss:{Color.tan}{v_loss:.4f}\n\n")
            


            #place in params  
            next_gen_parameters             = next_gen_model.half().cpu().state_dict()
            self.model_params[self.gen+1]   = next_gen_parameters


            #Find best model 
            print(f"\t\t{Color.blue}MODEL SHOWDOWN\n{Color.end}")
            self.find_best_model()

        
            #Reset training data pool
            self.current_generation_data    = []

            #Reduce lr 
            self.lr                         *= self.lr_mult
=======
            print(f"\t{Color.tan}Train Step [{self.train_step}]:\tdata_pool size: {len(self.data_pool)}{Color.green} - running!{Color.end}")
            self.next_update_t                      = time.time() + self.update_iter

        #Train if next iteration
        if self.exp_counter > self.train_every:
                
            #Sync clients so no new games are created
            self.sync_all_clients()
            
            #Snatch experiences from games after sync
            for client in self.client_managers:
                while not client.queue.empty():
                    self.data_pool.append(client.queue.get())
                    self.exp_counter += 1
                
            #Update window and reset train counter
            self.apply_window()
            self.train_step                             += 1
            self.exp_counter                            = 0 

            #Print stats
            print(f"\n\n\t{Color.blue}Training Step {Color.tan}{self.train_step}:{Color.end}")
            print(f"\t\t{Color.blue}bs:\t{Color.tan}{self.bs}{Color.end}")
            print(f"\t\t{Color.blue}lr:\t{Color.tan}{self.lr:.5}{Color.end}")
            print(f"\t\t{Color.blue}betas:\t{Color.tan}{self.betas}\n\n{Color.end}")

            #Sample dataset to train         
            training_batch                              = random.sample(self.data_pool,k=self.train_every)
            training_dataset                            = trainer.TrainerExpDataset(training_batch)

            #Train model on data
            self.chess_model.float().train().to(self.device)
            trainer.train_model(self.chess_model,training_dataset,bs=self.bs,lr=self.lr,wd=self.wd,betas=self.betas)

            #Check stockfish baseline 
            p_loss,v_loss                               = trainer.check_vs_stockfish(self.chess_model)
            print(f"\t\t{Color.blue}STOCKFISH BASELINE:\n\t\tv_loss: {Color.tan}{v_loss:.4f}{Color.blue}\tp_loss:{Color.tan}{p_loss:.4f}{Color.end}\n\n")

            #Step lr 
            self.lr                                     *= self.lr_mult

            #Place model back in eval mode and get state dict
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7
            
            # CPU ONLY TEST 
            #self.chess_model.half().eval().cpu()
            self.chess_model.eval().cpu()
            #/CPU ONLT TEST 


            self.model_state                            = self.chess_model.state_dict()

            #Save models 
            if not os.path.exists("generations/"):
                os.mkdir('generations')
<<<<<<< HEAD
            torch.save(self.model_params[self.top_model],f"generations/gen_{self.gen}.dict")

            #Go back to a training state
=======
            torch.save(self.model_state,f"generations/gen_{self.train_step}.dict")

            #Unlock clients
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7
            self.return_client_to_train_state()
            
            #Update game params throughout training 
            if self.gen < 5:
                self.game_params['n_iters']     = self.game_params['n_iters'] + 50 
                self.test_params['n_iters']     = self.test_params['n_iters'] + 50 

                self.game_params['ply']         = self.game_params['ply'] + 5 
                self.test_params['ply']         = self.test_params['ply'] + 5 


    #Performs the work of checking if a new client is looking
    #   to connect
    def check_for_clients(self):

        try:

            #Look for a connecting client 
            client_socket,addr                  = self.server_socket.accept() 

            #Create a new client_manager for them 
            new_client_manager                  = Client_Manager(client_socket,
                                                         addr,
                                                         self.get_next_id(),
                                                         Queue(),
                                                         self.model_state,
                                                         self.game_params,
                                                         pack_len=self.pack_len)
            
            #If were in test mode, make sure to let client_manager know 
            if self.lock:
                new_client_manager.set_lock()

            #Add to clients list 
            self.client_managers.append(new_client_manager)

            #Start em up
            new_client_manager.start()
        
        except TimeoutError:
            pass


    #Applies the window alg to the datapool
    def apply_window(self):

        #Remove first step after 3 experience gathers
        if self.train_step == 2:
            self.data_pool                          = self.data_pool[-self.train_every*2:]      
        
        #Increase window to 3 after first few iters
        elif self.train_step in [4,5]:
            self.data_pool                          = self.data_pool[-self.train_every*3:]  

        #Increase to 5
        elif self.train_step in [6,7]:
            self.data_pool                          = self.data_pool[-self.train_every*5:]
        
        #All further gens have 8
        else:
<<<<<<< HEAD
            #if single item, then it has a bye
            if len(remaining_models) == 1:
                byes.append(remaining_models[0][0])
                return
            
            else:

                #Create game_packets 
                #Get data 
                p1_id,p1_params     = remaining_models[0]
                p2_id,p2_params     = remaining_models[1]

                #Queue up half was W 
                for _ in range(self.test_params['n_games']//2):
                    game_packets.append((p1_id,p2_id,p1_params,p2_params,random.randint(10000,99999)))  #last item is games uid
                for _ in range(self.test_params['n_games']//2):
                    game_packets.append((p2_id,p1_id,p2_params,p1_params,random.randint(10000,99999)))


    #Recursively create bracket and play out models 
    #Model lists is a list of (id#,params) tuples 
    def run_bracket(self,bracket:list):

        #Keep track of games we need to hear back from 
        outstanding_games       = [pack[-1] for pack in bracket]
        game_results            = {}
        player_to_key           = {}

        #Create game results based on players 
        for pack in bracket:
            p1  = pack[0]
            p2  = pack[1]

            #Key will always be lowers number first 
            key     = (p2,p1) if p1 > p2 else (p1,p2)

            game_results[key]       = {p1:0,p2:0,'draws':0}
            player_to_key[p1]       = key
            player_to_key[p2]       = key


        #Create queue of games to be played 
        remaining_games         = Queue()
        for game_packet in bracket:
            remaining_games.put(game_packet)

        
        #While games remain, get a client manager to play them         
        while bracket:
            
            self.check_dead_clients()

            #Pass next game to the next available client
            self.pass_test_game_to_client_manager(bracket.pop())

        
        #Wait for all games to finish but not longer than 'reset' seconds
        t0                      = time.time()
        reset                   = 500
        while outstanding_games and (time.time()-t0) < reset:

            #Check for items in all clients queues 
            finished_games      = [client.queue.get_nowait() for client in self.clients if client.queue.qsize()]

            for game_data in finished_games:
                outcome         = game_data['outcome']
                players         = game_data['players']
                uid             = game_data['uid']

                #get key 
                p1,p2           = players 

                #Keys must be in same order (i.e. 1v0 == 0v1), so key will always be lower number first 
                #   if we end up switching the players to make the key, then the outcome is also opposite 
                key     = (p2,p1) if p1 > p2 else (p1,p2)

                #Update player stats
                if outcome == 1:
                    game_results[key][p1]       += 1

                elif outcome == -1:
                    game_results[key][p2]       += 1
                else:
                    game_results[key]['draws']  += 1


                outstanding_games.remove(uid)

                #Reset time 
                t0                  = time.time()
        
        matches         = {}
        winners         = [] 
        for matchup in game_results:
            p1,p2           = matchup 

            p1_wins         = game_results[matchup][p1]
            p2_wins         = game_results[matchup][p2]

            if p1_wins > p2_wins:
                winners.append(p1)
            else:
                winners.append(p2)
            
        res_string  = str(game_results).replace(" ","").replace("'draws'",'tie')
        print(f"\t{Color.tan}matches are {game_results}{Color.end}")
        self.train_thresh   = 16384
        return matches, winners
=======
            self.data_pool                          = self.data_pool[-self.train_every*8:]
>>>>>>> 61c6c08cd9b235f3db67875f869bdc2e81c3c2b7


    #Give a game_packet to a client manager for it's client 
    #   to play out and return back 
    def pass_test_game_to_client_manager(self,game_packet):
        passed_to_client                = False

        while not passed_to_client:
            
            #Check for new clients here 
            self.check_for_clients()

            #Give to client if one is available 
            for client in self.client_managers:

                #Pass only if locked and waiting and on same device
                if client.lock and not client.in_game:
                    client.recieve_test_game(game_packet)
                    passed_to_client    = True
                    break 

        return client


    #Return clients to generate training games 
    #   after determining top model 
    def return_client_to_train_state(self):

        #Return back to test mode
        self.test_mode                  = False
        
        #Send updated model and unlock
        for client_manager in self.client_managers:
            client_manager.state_dict                   = self.model_state
            client_manager.unlock()



    #Safely close up shop
    def shutdown(self):
        self.running    =  False 

        for client_manager in self.client_managers:
            client_manager.shutdown()
        
        self.server_socket.close()

        self.join()




#TODO 
#   - put all dictionaries in their own folder
#   - ensure test games are only played on the same machine (check ip)


#DEBUG purposes
if __name__ == "__main__":

    server  = Server()

    server.model_params[1] = ChessModel2().state_dict()
    server.model_params[2] = ChessModel2().state_dict()
    server.model_params[3] = ChessModel2().state_dict()
    server.model_params[4] = ChessModel2().state_dict()
    server.model_params[5] = ChessModel2().state_dict()
    server.model_params[6] = ChessModel2().state_dict()

    games   =    []
    server.create_test_bracket([(i,server.model_params[i]) for i in server.model_params],games)
    print([(pack[0],pack[1]) for pack in games])
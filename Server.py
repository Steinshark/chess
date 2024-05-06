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
from model import ChessModel
from io import BytesIO
import random
import trainer 
from copy import deepcopy
import time 
import os
from hashlib import md5
import networking
from networking import Color

#Runs on the client machine and generates training data 
#   by playing games againts itself
#Handles a single client and gives it a model to use,
#   tells it to play games, and translates it back
#   to the Server 
class Client_Manager(Thread):

    def __init__(self,client_socket:socket.socket,connection:str,client_id:str,client_queue:Queue,model_dict:OrderedDict,game_settings,pack_len=8192):
        super(Client_Manager,self).__init__()
        self.client_socket          = client_socket
        self.client_address         = connection[0]
        self.client_port            = connection[1]
        self.id                     = client_id
        self.queue                  = client_queue
        self.running                = True
        self.pack_len               = pack_len

        #Game vars 
        self.game_settings          = game_settings

        #Model vars 
        self.model_dict             = model_dict
        self.in_game                = False
        self.lock                   = False

        #Set 500 sec timeout 
        self.client_socket.settimeout(3600/2)

        print(f"\n\t{Color.green}launched a new client manager for {Color.tan}{self.client_address}\n{Color.end}")


    def is_alive(self):
        if not isinstance(self.client_socket,socket.socket):
            return False
        return True
    

    def recieve_model(self,new_model):
        self.model_state                    = new_model
        self.top_model_params               = new_model
    

    def run_training_game(self):

        #Send model parameters to generate training data 
        self.send_model_params(self.top_model_params)

        #Send game parameters to play with
        self.send_game_params(self.game_settings)
        
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
                
                self.in_game            = False
                while self.lock:
                    time.sleep(1)
                            


                #client_socket.send("Kill".encode())
        except TimeoutError:
            print(f"\n\t{Color.red}client shutting down{Color.end}")
            self.shutdown()
        except OSError:
            print(f"\n\t{Color.red}Lost communication with client{Color.end}")
            self.shutdown()
            return False

    
    def send_model_params(self,parameters:OrderedDict):

        #Create a BytesIO buffer to load model into
        data_buffer                     = BytesIO()

        #Load params into data_buffer as bytes
        torch.save(parameters,data_buffer)

        #Get bytes out of buffer
        params_as_bytes                 = data_buffer.getvalue()

        #Get hash 
        params_hash                     = md5(params_as_bytes).digest()

        #Check if client has model
        networking.send_bytes(self.client_socket,params_hash,self.pack_len)
        client_response                 = networking.recieve_bytes(self.client_socket,self.pack_len).decode()
        if client_response == 'skip':
            networking.send_bytes(self.client_socket,'skip'.encode(),self.pack_len)
        
        elif client_response == 'send':
            networking.send_bytes(self.client_socket,params_as_bytes,self.pack_len)
    

    def send_game_params(self,parameters:dict):
               
        #Send game parameters to client 
        data_packet                 = json.dumps(parameters)
        networking.send_bytes(self.client_socket,data_packet.encode(),self.pack_len)


    def recieve_client_result(self):

        bytes_message                       = networking.recieve_bytes(self.client_socket,self.pack_len)

        experience_set                      = json.loads(bytes_message.decode())

        for exp in experience_set:
            self.queue.put_nowait(exp)
        

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
        self.chess_model                            = ChessModel(17,16).eval().cpu().float()
        self.model_dict                            = self.chess_model.state_dict()
        #self.model_state                            = torch.load("generations/gen_16.dict")
        self.device                                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Game items
        self.game_settings                            = {"ply":30,"n_iters":50,"n_exp":128,"n_parallel":8}

        #Training vars
        self.data_pool                              = [] 
        self.train_every                            = 65536
        self.exp_counter                            = 0
        self.bs                                     = 1024
        self.lr                                     = .0001
        self.wd                                     = 0
        self.betas                                  = (.5,.9)
        self.n_epochs                               = 1 
        self.train_step                             = 0 
        self.lr_mult                                = 1
        self.gen                                    = 0

        #Telemtry vars
        self.update_iter                            = 60
        self.next_update_t                          = time.time() + self.update_iter


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
        print(f"\t{Color.green}server listening on {Color.tan}{address}{Color.end}")


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
        
        hitlist         = [] 
        for client_index,client in enumerate(self.client_managers):

            #Check client is alive 
            if not client.running:
                hitlist.append(client_index)
            else:
                #Pass-on most recent model 
                client.recieve_model(self.model_dict)
                
        for client_index in hitlist:
            self.client_managers.pop(client_index)
            print(f"\tremove client: {client_index}")
        pass 


    #Blocks until all clients have finished their game and 
    #   the client_managers have passed them back to the server
    def sync_all_clients(self):
        
        print(f"\n\t{Color.green}Syncing Clients {[cm.id for cm in self.client_managers]}{Color.end}\n")
        #Lock server 
        self.lock                                   = True 
        found_running_game                          = True 

        while found_running_game:
            found_running_game                      = False

            for client in self.client_managers:
                client.lock                         = True 
                
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
    def test_model(self,chess_model:ChessModel):
        p_loss,v_loss                           = trainer.check_vs_stockfish(chess_model)
        print(f"\t\t{Color.blue}PRE_TRAIN:\n\t\tp_loss:{Color.tan}{p_loss:.4f}{Color.blue}\t\tv_loss:{Color.tan}{v_loss:.4f}\n\n")
        

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
            print(f"\t{Color.tan}Train Step [{self.train_step}]:\tdata_pool size: {len(self.data_pool)}{Color.green} - running!{Color.end}")
            self.next_update_t                      = time.time() + self.update_iter

        #Train if next iteration
        if self.exp_counter > self.train_every:
                
            #Sync clients so no new games are created
            self.sync_all_clients()
            
            #Snatch experiences from games after sync
            self.exp_counter                            = 0 
            for client in self.client_managers:
                while not client.queue.empty():
                    self.data_pool.append(client.queue.get())
                    self.exp_counter += 1
                
            #Update window and reset train counter
            self.apply_window()
            self.train_step                             += 1

            #Print stats
            print(f"\n\n\t{Color.blue}Training Step {Color.tan}{self.train_step}:{Color.end}")
            print(f"\t\t{Color.blue}bs:\t{Color.tan}{self.bs}{Color.end}")
            print(f"\t\t{Color.blue}lr:\t{Color.tan}{self.lr:.5}{Color.end}")
            print(f"\t\t{Color.blue}betas:\t{Color.tan}{self.betas}\n\n{Color.end}")

            #Sample dataset to train         
            training_batch                              = random.sample(self.data_pool,k=self.train_every)
            training_dataset                            = trainer.TrainerExpDataset(training_batch)

            #Train model on data
            self.chess_model.train().to(self.device)
            trainer.train_model(self.chess_model,training_dataset,bs=self.bs,lr=self.lr,wd=self.wd,betas=self.betas)

            #Check stockfish baseline 
            p_loss,v_loss                               = trainer.check_vs_stockfish(self.chess_model)
            print(f"\t\t{Color.blue}STOCKFISH BASELINE:\n\t\tv_loss: {Color.tan}{v_loss:.4f}{Color.blue}\tp_loss:{Color.tan}{p_loss:.4f}{Color.end}\n\n")

            #Step lr 
            self.lr                                     *= self.lr_mult

            #Place model back on cpu, in eval mode, and get state dict
            self.chess_model.eval().cpu()
            self.model_dict                            = self.chess_model.state_dict()

            #Save models 
            if not os.path.exists("generations/"):
                os.mkdir('generations')
            torch.save(self.model_dict,f"generations/gen_{self.train_step}.dict")

            #Unlock clients
            self.return_client_to_train_state()
            
            #Update game params throughout training 
            if self.gen < 4:
                self.game_settings['n_iters']     = self.game_settings['n_iters'] + 50


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
                                                         self.model_dict,
                                                         self.game_settings,
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
            self.data_pool                          = self.data_pool[-self.train_every*4:]
        
        #All further gens have 8
        else:
            self.data_pool                          = self.data_pool[-self.train_every*5:]


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
            client_manager.top_model_params                 = self.model_dict
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

    server.model_params[1] = ChessModel().state_dict()
    server.model_params[2] = ChessModel().state_dict()
    server.model_params[3] = ChessModel().state_dict()
    server.model_params[4] = ChessModel().state_dict()
    server.model_params[5] = ChessModel().state_dict()
    server.model_params[6] = ChessModel().state_dict()

    games   =    []
    server.create_test_bracket([(i,server.model_params[i]) for i in server.model_params],games)
    print([(pack[0],pack[1]) for pack in games])
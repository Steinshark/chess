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

#Runs on the client machine and generates training data 
#   by playing games againts itself
class Client(Thread):


    def __init__(self,address='localhost',port=15555,device_id=None):
        super(Client,self).__init__()

        #Setup socket 
        self.client_socket          = socket.socket(socket.AF_INET,socket.SOCK_STREAM)  
        self.address                = address
        self.port                   = port 
        self.running                = True
        self.n_moves                = 0 

        #game related variables
        self.current_model_params   = None
        self.device_id              = device_id
        self.game_mode              = "Train"
        self.lookup_dict            = {}
        self.max_lookup_len         = 100_000


    #Runs the client.
    #   loops getting game type and running
    #   that type of game 
    def run(self):
        
        #Initialize the connection with the server and receive id
        self.client_socket.connect((self.address,self.port))
        self.id                     = int(self.client_socket.recv(32).decode())
        print(f"\tclient connected to {self.address} with id:{self.id}")
    
        #Do for forever until we die
        while self.running:
            
            #Recieve game type 
            self.receive_game_type()
            print(f"received game type - {self.game_mode}")

            #Execute that game 
            self.execute_game()
            print(f"\texecuted game")


    #Gets the game type from client_manager
    #   either 'Test' or 'Train'
    #   Reset added to reset lookup dict after 
    #   new model release 
    def receive_game_type(self):
        game_type                   = self.client_socket.recv(32).decode()

        if game_type == "Train":
            self.game_mode          = "Train"
        elif game_type == "Test":
            self.game_mode          = "Test"
        elif game_type == 'Reset':
            self.lookup_dict        = {}
            self.game_mode          = 'Test'

        #Acknowledge 
        self.client_socket.send('Ready'.encode())


    #Gets 1 model's parameters from 
    #   the client_manager
    def recieve_model_params(self):

        #Check not running 
        if not self.running:
            return 
        
        #Receive 64MB of data 
        data_packet                 = self.client_socket.recv(1024*1024*64) 
        buffer                      = BytesIO(data_packet)                 
        params_as_bytes             = buffer.getvalue()
        model_parameters            = torch.load(BytesIO(params_as_bytes))

        #attempt to instantiate model with them
        self.current_model          = ChessModel2(19,24).cuda()
        self.current_model.load_state_dict(model_parameters)
        self.client_socket.send("Recieved".encode())

        #Return the params 
        return model_parameters


    #Runs a game based on the type of 
    #   received by the client_manager
    def execute_game(self):
        if self.game_mode == "Train":

            #Get model params
            model_params            = self.recieve_model_params()

            #Get game_params
            data_packet             = self.client_socket.recv(1024).decode()
            game_parameters         = json.loads(data_packet)
            max_game_ply            = game_parameters['ply']
            n_iters                 = game_parameters['n_iters']

            #Run game 
            t0                      = time.time()
            training_data           = alg_train.play_game(model_params,max_game_ply,n_iters,self,self.device_id,self.lookup_dict)
            print(f"\t{(time.time()-t0)/len(training_data):.2f}s/move")
            
            #Upload data
            self.upload_data(training_data,mode='Train')

            #Take memory off the cuda device 
            torch.cuda.empty_cache()
        
        elif self.game_mode == "Test":
            model1_params           = self.recieve_model_params()
            model2_params           = self.recieve_model_params()

            #Get game_params
            data_packet             = self.client_socket.recv(1024).decode()
            game_parameters         = json.loads(data_packet)
            max_game_ply            = game_parameters['ply']
            n_iters                 = game_parameters['n_iters']

            #Run game 
            game_outcome            = alg_train.showdown_match((model1_params,model2_params,max_game_ply,n_iters))
        
            #Upload data
            self.upload_data(game_outcome,mode='Test')


    #DEPRECATED in favor of execute_game  
    def run_game(self):
        print(f"beginning game")
        #Recieve 1024 bytes of data 
        data_packet                 = self.client_socket.recv(1024).decode()
        game_parameters             = json.loads(data_packet)

        #Decode game parameters
        max_game_ply                = game_parameters['ply']
        n_iters                     = game_parameters['n_iters']
        game_type                   = game_parameters['type']

        #Run game 
        training_data               = alg_train.play_game(self.current_model,max_game_ply,n_iters,self,self.device_id)
        
        #Upload 

        #Play game 
        if game_type == "Train":
            training_data               = alg_train.play_game(self.current_model,max_game_ply,n_iters,self,self.device_id)
            self.current_data_batch     = training_data

        elif game_type == "Test":

            #Will be recieving another model dict 
            model_dict1                 = self.current_model_params
            self.recieve_model_params()
            model_dict2                 = self.current_model_params

            game_outcome                = alg_train.showdown_match(model_dict1,model_dict2,max_game_ply,n_iters,self,self.device_id)
            self.current_data_batch     = game_outcome
            print(f"outcome was {self.current_data_batch}")

        print(f"finished game\n")


    #Uploads data from the client to the client_manager
    #   handles both 'Train' and 'Test' modes
    def upload_data(self,data,mode='Train'):
        
        #Check for dead 
        if not self.running:
            return 
        
        #If Result game, send only the outcome 
        if mode == 'Test':
            self.client_socket.send(mode.encode())
            self.client_socket.recv(32)             #Get "Ready"
            self.client_socket.send(str(data).encode())

        #Otherwise send full list
        elif mode == 'Train':
            self.client_socket.send(mode.encode())
            
            #Send exps
            for packet_i,exp in enumerate(data):

                #Wait for "Ready" from server
                confirm         = self.client_socket.recv(32).decode()
                if not confirm == "Ready":
                    print(f"didnt get Send, got {confirm}")
                    break

                #separate fen,move_data,and outcome and 
                #   json convert to strings to send over network
                fen:str         = exp[0]
                move_stats:str  = json.dumps(exp[1])
                outcome:str     = str(exp[2])

                #Combine strings 
                data_packet     = (fen,move_stats,outcome)
                data_packet     = json.dumps(data_packet)

                #encode to bytes and send to server 
                self.client_socket.send(data_packet.encode())


        #Receive the last "Ready"
        confirm         = self.client_socket.recv(32).decode()

        #Let server know thats it
        self.client_socket.send("End".encode())


    #Closes down the socket and everything else
    def shutdown(self):
        self.running = False 
        self.client_socket.close()



#Handles a single client and gives it a model to use,
#   tells it to play games, and translates it back
#   to the Server 
class Client_Manager(Thread):

    def __init__(self,client_socket:socket.socket,connection:str,client_id:str,client_queue:Queue,model_params:OrderedDict,game_params,test_params):
        super(Client_Manager,self).__init__()
        self.client_socket          = client_socket
        self.client_address         = connection[0]
        self.client_port            = connection[1]
        self.id                     = client_id
        self.queue                  = client_queue
        self.running                = True

        #Game vars 
        self.game_params            = game_params
        self.test_params            = test_params

        #Model vars 
        self.top_model_params       = model_params
        self.model1_params          = None
        self.model2_params          = None
        self.in_game                = False
        self.lock                   = False
        self.game_mode              = "Train"

        print(f"\n\tlaunched a new client manager for {self.client_address}\n")


    def is_alive(self):
        if not isinstance(self.client_address,socket.socket):
            return False
        return True
    

    def recieve_model(self,new_model):
        self.top_model_params    = new_model
    

    def run_training_game(self):

        #Send model parameters to generate training data 
        self.send_model_params(self.top_model_params)

        #Send game parameters to play with
        self.send_game_params(self.game_params)
        
        #CLIENT PLAYS GAME 

        #Receive training data from game 
        self.recieve_client_result()


    def run_test_game(self):
        
        #send both model params 
        self.send_model_params(self.model1_params)
        self.send_model_params(self.model2_params)

        #Send game parameters to play with
        self.send_game_params(self.test_params)

        #CLIENT PLAYS GAME 

        #Receive training data from game 
        self.recieve_client_result()

    
    def run(self):
        try:
    
            #First, send job worker id to client 
            self.client_socket.send(str(self.id).encode())

            while self.running:
                self.in_game            = True

                #Send type of game 
                self.send_game_type()

                #Execute game and get results
                self.run_game_type()
                
                self.in_game                = False


                #If client_manager is in a 'Test' mode, then lock after each game 
                if self.game_mode == 'Test':
                    self.set_lock()

                while self.lock:
                    time.sleep(1)
                            


                #client_socket.send("Kill".encode())
        except OSError:
            print(f"Lost communication with client")
            return False
            

    def send_game_type(self):

        #Send text to client 
        self.client_socket.send(self.game_mode.encode())

        #place mode back to "train" if in 'Reset'
        if self.game_mode == "Reset":
            self.game_mode          = 'Train'

        #Get confirmation
        confirmation            = self.client_socket.recv(32).decode()

        if not confirmation == "Ready":
            print(f"\tClient did not confirm game mode: '{confirmation}'")


    def run_game_type(self):
        
        if self.game_mode == "Train":
            self.run_training_game()

        elif self.game_mode == "Test":
            self.run_test_game()
       
        elif self.game_mode == 'Reset':
            self.run_training_game()

        else:
            print(f"\tbad game_mode: '{self.game_mode}")
    
    
    def send_model_params(self,parameters:OrderedDict):

        #Create a BytesIO buffer to load model into
        data_buffer             = BytesIO()

        #Create copy of parameters 
        parameters_copy         = deepcopy(parameters)

        torch.save(parameters_copy,data_buffer)

        #Get bytes out of buffer
        params_as_bytes         = data_buffer.getvalue()

        #Send bytes to client
        self.client_socket.send(params_as_bytes)

        #Get confirmation from client "Recieved"
        self.client_socket.recv(32)


    def send_game_params(self,parameters:dict):
               
        #Send game parameters to client 
        data_packet                 = json.dumps(parameters)
        self.client_socket.send(data_packet.encode())


    def shutdown(self):
        self.client_socket.close()
        self.running                = False


    def recieve_client_result(self):

        #Check what client is saying
        game_mode                   = self.client_socket.recv(32).decode()

        #If 'Result', then only outcome is sent
        if game_mode == "Test":

            #Send 'Ready' to recieve
            self.client_socket.send("Ready".encode())

            #Get and process result 
            outcome                 = int(self.client_socket.recv(32).decode())

            #in the queue, put the data package
            self.queue.put({'outcome':outcome,'players':(self.p1,self.p2),'uid':self.game_uid})

            #Send last 'Ready' signal
            self.client_socket.send("Ready".encode())

            #Recieve 'End' signal
            confirm_end             = self.client_socket.recv(32).decode()

        
        #if 'Experience', then will recieve a list of experiences
        elif game_mode == "Train":

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
                    client_response = (data_packet[0],json.loads(data_packet[1]),float(data_packet[2]))
                    
                    #Place in the data queue for the Server thread to fetch 
                    self.queue.put(client_response)
    

    def recieve_test_game(self,game_packet):
        self.p1             = game_packet[0]
        self.p2             = game_packet[1]

        self.model1_params  = game_packet[2]
        self.model2_params  = game_packet[3]
        self.game_uid       = game_packet[4]

        #Set to test mode 
        self.game_mode      = "Test"

        #unlock and go for it 
        self.unlock()
        self.in_game        = True


    def set_lock(self):
        self.lock   = True 
    

    def unlock(self):
        self.lock   = False



#Handles the server (Aka in charge of the training algorithm)
#   each client that connects will get its own Client_Manager 
#   and generate trainin games
class Server(Thread):


    def __init__(self,address='localhost',port=15555):
        
        #Init super
        super(Server,self).__init__()

        #Establish items 
        self.clients:list[Client_Manager]   = [] 
        self.build_socket(address,port)

        self.running                        = True 
        self.test_mode                      = False

        #Model items 
        self.model_params                   = {0:ChessModel2(19,24).cpu().state_dict()}
        self.top_model                      = 0 
        self.game_params                    = {"ply":90,"n_iters":400}
        self.test_params                    = {"ply":100,"n_iters":800,'n_games':10}

        #Training items 
        self.current_generation_data        = [] 
        self.train_thresh                   = 8192
        self.train_size                     = 4096
        self.bs                             = 2048
        self.lr                             = .001
        self.wd                             = .01 
        self.betas                          = (.5,.8)
        self.n_epochs                       = 1 
        self.gen                            = 0 

        self.update_iter                    = 30
        self.next_update_t                  = time.time() + self.update_iter
        self.game_stats                     = []
        self.lr_mult                        = .8 

        self.game_outcomes                  = {}     
        self.max_models                     = 4   


    #Run the server
    def run(self):
        
        #Always on 
        while self.running:

            #Check for a new client
            self.check_for_clients()

            #Update server training state
            self.update_training_state()

            #Pass data to clients 
            self.update_clients()


    #Load the most recent models
    def load_models(self):
        
        #Load all gen_x.dict files
        filenames                   = [file for file in os.listdir() if "gen" in file and ".dict" in file]
        if not filenames:
            print(f"\tServer loaded no state_dicts")
            return
        #Find the top 5, and assume most recent is best model
        filenames.sort(key=lambda x: int(x.replace('gen_','').replace('.dict','')),reverse=True)
        top_params                  = filenames[:self.max_models]

        #build param list 
        self.model_params           = {int(fname.replace('gen_','').replace('.dict','')):torch.load(fname) for fname in top_params}
        self.top_model              = max(self.model_params)
        
        #set current gen to max of params + 1 
        self.gen                    = max(self.model_params) + 1
        print(f"\tServer loaded state_dicts {list(self.model_params.keys())}")


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
        print(f"\tserver listening on {address}")


    #Assign the next available client id
    def get_next_id(self):
        candidate_id    = 0
        while candidate_id in [client.id for client in self.clients]:
            candidate_id += 1
        return candidate_id
    

    #Updates clients with the most recent parameters and 
    def update_clients(self):
        
        hitlist         = [] 
        for client_index,client in enumerate(self.clients):

            #Check client is alive 
            if not client.running:
                hitlist.append(client_index)
            else:
                #Pass-on most recent model 
                client.recieve_model(self.model_params[self.top_model])
                
        for client_index in hitlist:
            self.clients.pop(client_index)
            print(f"\tremove client: {client_index}")
        pass 


    #Blocks until all clients have finished their game and 
    #   the client_managers have passed them back to the server
    def sync_all_clients(self):
        found_running_game  = True 

        while found_running_game:
            found_running_game = False

            for client in self.clients:
                client.lock             = True 
                
                if client.in_game:
                    found_running_game  = True 
                
   
   #Unblocks client managers and lets them generate games 
   #    until sync'd again
    def unsync_all_clients(self):

        #Unlock all clients
        for client in self.clients:
            client.unlock()


    #Add experiences to server's list and 
    #   Check if its time to train 
    def update_training_state(self):

        #Snatch all experiences still in queues
        for client in self.clients:
            while not client.queue.empty():
                self.current_generation_data.append(client.queue.get())

        if time.time() - self.next_update_t > 0:    
            
            #Add leading zeros to len
            cur_len_string              = str(len(self.current_generation_data))
            while len(cur_len_string) < len(str(self.train_thresh)):
                cur_len_string = "0" + cur_len_string

            print(f"\tGeneration [{self.gen}]:\t[{cur_len_string}/{self.train_thresh}] samples accumulated")
            self.next_update_t = time.time() + self.update_iter


        #Once over self.train_thresh, train and update model 
        
        if len(self.current_generation_data) > self.train_thresh:
            
            #Set in test mode
            self.test_mode                  = True 
            
            #Dont train until all games are done
            self.sync_all_clients()

            #Snatch all experiences still in queues
            for client in self.clients:
                while not client.queue.empty():
                    self.current_generation_data.append(client.queue.get())
                


            #Prep for saving game outcomes
            self.game_outcomes[self.gen]    = []

            print(f"\n\n\tTraining Gen {self.gen}:\n")
            print(f"\tTRAIN PARAMS")
            print(f"\t\tbs:\t{self.bs}")
            print(f"\t\tlr:\t{self.lr:.4}")
            print(f"\t\tbetas:\t{self.betas}\n\n")
            #Select dataset to train on 
            training_batch                  = random.choices(self.current_generation_data,k=self.train_size)
            training_dataset                = trainer.TrainerExpDataset(training_batch)

            #Clone current best model 
            next_gen_model                  = ChessModel2(19,24).cpu()              #Always pass stuff on the cpu
            next_gen_model.load_state_dict(self.model_params[self.top_model])

            #View performance vs stockfish before
            p_loss,v_loss                   = trainer.check_vs_stockfish(next_gen_model)
            print(f"\t\tPRE_TRAIN:\n\t\tp_loss:{p_loss:.4f}\t\tv_loss:{v_loss:.4f}\n\n")
            #Train it 
            print(f"\t\tTRAINING:")
            p_losses,v_losses               = trainer.train_model(next_gen_model,training_dataset,bs=self.bs,lr=self.lr,wd=self.wd,betas=self.betas,n_epochs=self.n_epochs)
            epoch                           = 0 
            for p,v in zip(p_losses,v_losses):
                print(f"\t\tp_loss:{p:.4f}\t\tv_loss:{v:.4f}\n")
            self.model_params[self.gen+1]   = next_gen_model.cpu().state_dict()

            #View performance vs stockfish after 
            p_loss,v_loss                   = trainer.check_vs_stockfish(next_gen_model)
            print(f"\n\t\tPOST_TRAIN:\n\t\tp_loss:{p_loss:.4f}\t\tv_loss:{v_loss:.4f}\n\n")

            #Find best model 
            print(f"\t\tMODEL SHOWDOWN\n")

            remaining_models                = [(id,self.model_params[id]) for id in self.model_params]
            bracket                         = []
            byes                            = []
            self.create_test_bracket(remaining_models,bracket,byes)

            while len(remaining_models) > 1:
                outcomes,winners            = self.run_bracket(bracket)
                remaining_models            = [(id,self.model_params[id]) for id in winners+byes]
                bracket                     = []
                byes                        = []
                self.create_test_bracket(remaining_models,bracket,byes)

            self.top_model                  = winners[0]
            print(f"\t\tTop Model: {self.top_model}")
            for match in self.game_outcomes[self.gen]:
                print(f"\t\t\t{match}")
            print("")
            self.gen                        += 1

            #If params > 5, drop lowest id that didnt win 
            while len(self.model_params) > self.max_models:
                id                          = 0
                while not id in self.model_params or self.top_model == id:
                    id += 1
                print(f"\t dropped {id}")
                del self.model_params[id]
                

        
            #Reset training data pool
            self.current_generation_data    = []
            #Reduce lr 
            self.lr                         *= self.lr_mult
            
            #Save models 
            torch.save(self.model_params[self.top_model],f"gen_{self.gen}.dict")
            self.test_mode                  = False
            self.return_client_to_train_state()
            

    #Performs the work of checking if a new client is looking
    #   to connect
    def check_for_clients(self):

        try:

            #Look for a connecting client 
            client_socket,addr              = self.server_socket.accept() 

            #Create a new client_manager for them 
            new_client_manager                  = Client_Manager(client_socket,
                                                         addr,
                                                         self.get_next_id(),
                                                         Queue(),
                                                         self.model_params[self.top_model],
                                                         self.game_params,
                                                         self.test_params)
            
            #If were in test mode, make sure to let client_manager know 
            if self.test_mode:
                new_client_manager.game_mode    = 'Test'
                new_client_manager.set_lock()

            #Add to clients list 
            self.clients.append(new_client_manager)
            #Start em up
            new_client_manager.start()
        
        except TimeoutError:
            pass


    #Creates a bracket of all remaining models that 
    #   will be run. called several times until 1 model remains.
    #   this one is used for data generation 
    def create_test_bracket(self,remaining_models,game_packets,byes):
        if len(remaining_models) > 2:
            midpoint    = len(remaining_models) //2 
            self.create_test_bracket(remaining_models[:midpoint],game_packets,byes)
            self.create_test_bracket(remaining_models[midpoint:],game_packets,byes)

        else:
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
            
            #Pass next game to the next available client
            self.pass_test_game_to_client_manager(bracket.pop())

        
        #Wait for all games to finish but not longer than 'reset' seconds
        t0                      = time.time()
        reset                   = 250
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
            

        print(f"\tmatches are {game_results}")
        return matches, winners


    #Give a game_packet to a client manager for it's client 
    #   to play out and return back 
    def pass_test_game_to_client_manager(self,game_packet):
        passed_to_client                = False

        while not passed_to_client:
            
            #Check for new clients here 
            self.check_for_clients()

            #Give to client if one is available 
            for client in self.clients:

                #Pass only if locked and waiting and on same device
                if client.lock and not client.in_game and client.client_address == self.address:
                    client.recieve_test_game(game_packet)
                    passed_to_client    = True
                    break 

        return client


    #Return clients to generate training games 
    #   after determining top model 
    def return_client_to_train_state(self):
        
        for client in self.clients:
            client.game_mode        = "Reset"
            client.in_game          = False
            client.unlock()



    #Safely close up shop
    def shutdown(self):
        self.running    =  False 

        for client_manager in self.clients:
            client_manager.shutdown()
        
        self.server_socket.close()

        self.join()




#TODO 
#   - put all dictionaries in their own folder
#   - ensure test games are only played on the same machine (check ip)


#DEBUG purposes
if __name__ == "__main__":

    server  = Server()

    server.model_params[1] = ChessModel2(19,24).state_dict()
    server.model_params[2] = ChessModel2(19,24).state_dict()
    server.model_params[3] = ChessModel2(19,24).state_dict()
    server.model_params[4] = ChessModel2(19,24).state_dict()
    server.model_params[5] = ChessModel2(19,24).state_dict()
    server.model_params[6] = ChessModel2(19,24).state_dict()

    games   =    []
    server.create_test_bracket([(i,server.model_params[i]) for i in server.model_params],games)
    print([(pack[0],pack[1]) for pack in games])
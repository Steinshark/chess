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

import time 


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
        
        #game related variables
        self.current_model_params   = None
        self.device_id              = device_id
        self.game_mode              = "Train"
        print(f"creating client")


    def run(self):
        
        #Initialize the connection with the server and receive id
        self.client_socket.connect((self.address,self.port))
        self.id                     = int(self.client_socket.recv(32).decode())
        print(f"\tworker connected with id:{self.id}")
    
        #Do for forever until we die
        while self.running:
            
            #Recieve game type 
            self.receive_game_type()

            #Execute that game 
            self.execute_game()

         
    def receive_game_type(self):
        game_type                   = self.client_socket.recv(32)

        if game_type == "Train":
            self.game_mode          = "Train"
        elif game_type == "Test":
            self.game_mode          = "Test"

        #Acknowledge 
        self.client_socket.send('Ready'.encode())


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
            training_data           = alg_train.play_game(model_params,max_game_ply,n_iters,self,self.device_id)

            #Upload data
            self.upload_data(training_data,mode='Train')
        
        elif self.game_mode == "Test":
            model1_params           = self.recieve_model_params()
            model2_params           = self.recieve_model_params()

            #Get game_params
            data_packet             = self.client_socket.recv(1024).decode()
            game_parameters         = json.loads(data_packet)
            max_game_ply            = game_parameters['ply']
            n_iters                 = game_parameters['n_iters']

            #Run game 
            game_outcome            = alg_train.showdown_match(model1_params,model2_params,max_game_ply,n_iters,self,self.device_id)
        
            #Upload data
            self.upload_data(game_outcome,mode='Test')


    #DEPRECATED      
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


    def shutdown(self):
        self.running = False 
        self.client_socket.close()



#Handles a single client and gives it a model to use,
#   tells it to play games, and translates it back
#   to the Server 
class Client_Manager(Thread):

    def __init__(self,client_socket:socket.socket,address:str,client_id:str,client_queue:Queue,model_params:OrderedDict,game_params):
        super(Client_Manager,self).__init__()
        self.client_socket          = client_socket
        self.client_address         = address
        self.id                     = client_id
        self.queue                  = client_queue
        self.running                = True

        self.top_model_params       = model_params
        self.game_params            = game_params

        self.model1_params          = None
        self.model2_params          = None
        self.in_game                = False
        self.lock                   = False
        self.game_mode              = "Train"


    def is_alive(self):
        if not isinstance(self.client_address,socket.socket):
            return False
        return True
    

    def recieve_model(self,new_model):
        self.current_model_params    = new_model
    

    def run_training_game(self):

        #Send model parameters to generate training data 
        self.send_model_params(self.current_model_params)

        #Send game parameters to play with
        self.send_game_params(self.current_game_params)
        
        #CLIENT PLAYS GAME 

        #Receive training data from game 
        self.recieve_client_result()


    def run_test_game(self):
        
        #send both model params 
        self.send_model_params(self.model1_params)
        self.send_model_params(self.model2_params)

        #Send game parameters to play with
        self.send_game_params(self.current_game_params)

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
                    time.sleep(.1)
                            


                #client_socket.send("Kill".encode())
        except OSError:
            print(f"Lost communication with client")
            return False
            

    def send_game_type(self):

        #Send text to client 
        self.client_socket.send(self.game_mode.encode())

        #Get confirmation
        confirmation            = self.client_socket.recv(32).decode()

        if not confirmation == "Ready":
            print(f"\tClient did not confirm game mode: '{confirmation}'")


    def run_game_type(self):
        
        if self.game_mode == "Train":
            self.run_training_game()

        elif self.game_mode == "Test":
            self.run_test_game()

        else:
            print(f"\tbad game_mode: '{self.game_mode}")
    
    
    def send_model_params(self,parameters:OrderedDict):

        #Create a BytesIO buffer to load model into
        data_buffer             = BytesIO()
        torch.save(parameters,data_buffer)

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
            self.queue.put(outcome)
        
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
    

    def recieve_test_params(self,model1_params,model2_params):
        self.model1_params  = model1_params
        self.model2_params  = model2_params


    def set_lock(self):
        self.lock   = True 
    

    def unlock(self):
        self.lock   == False


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

        #Model items 
        self.model_params                   = {0:ChessModel2(19,24).cuda().state_dict()}
        self.top_model                      = 0 
        self.game_params                    = {"ply":100,"n_iters":50}
        self.test_params                    = {"ply":160,"n_iters":500,'n_games':20}

        #Training items 
        self.current_generation_data        = [] 
        self.train_thresh                   = 2048
        self.train_size                     = 768
        self.bs                             = 128   
        self.lr                             = .001 
        self.wd                             = .01 
        self.betas                          = (.5,.8)
        self.n_epochs                       = 1 
        self.gen                            = 0 

        self.update_iter                    = 30
        self.next_update_t                  = time.time() + self.update_iter
        self.game_stats                     = []
        self.lr_mult                        = .75      

        self.game_outcomes                  = {}       


    def build_socket(self,address,port):
        
        #Set class vars
        self.address                = address 
        self.port                   = port

        #Create socket 
        self.server_socket          = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server_socket.bind((address,port))
        self.server_socket.settimeout(.1)
        self.server_socket.listen(16)   #Accept up to 16 simultaneous connections


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


    def recieve_data_from_clients(self):
        for client in self.clients:
            continue
        pass
    

    def sync_all_clients(self):
        print(f"locking")
        found_running_game  = True 

        while found_running_game:
            found_running_game = False

            for client in self.clients:
                client.lock             = True 
                
                if client.in_game:
                    found_running_game  = True 
                
        print(f"all clients locked\n")

   
    def unsync_all_clients(self):

        #Unlock all clients
        for client in self.clients:
            client.lock                 = False 
        print(f"all clients unlocked")


    def unlock_client(self,client:Client_Manager):
        client.lock         = False 
    

    def lock_client(self,client:Client_Manager):
        client.lock         = True


    def update_training_state(self):

        #Snatch all experiences still in queues
        for client in self.clients:
            while not client.queue.empty():
                self.current_generation_data.append(client.queue.get())

        if time.time() - self.next_update_t > 0:    
            
            #Prep for saving game outcomes
            self.game_outcomes[self.gen]    = []

            #Add leading zeros to len
            cur_len_string              = str(len(self.current_generation_data))
            while len(cur_len_string) < len(str(self.train_thresh)):
                cur_len_string = "0" + cur_len_string

            print(f"\tGeneration [{self.gen}]:\t[{cur_len_string}/{self.train_thresh}] samples accumulated")
            self.next_update_t = time.time() + self.update_iter
        #Once over self.train_thresh, train and update model 
        if len(self.current_generation_data) > self.train_thresh:
            #Dont train until all games are done
            self.sync_all_clients()

            print(f"\n\n\tTraining Gen {self.gen}:\n")
            print(f"\tTRAIN PARAMS")
            print(f"\t\tbs:\t{self.bs}")
            print(f"\t\tlr:\t{self.lr}")
            print(f"\t\tbetas:\t{self.betas}\n\n")
            #Select dataset to train on 
            training_batch                  = random.choices(self.current_generation_data,k=self.train_size)
            training_dataset                = trainer.TrainerExpDataset(training_batch)

            #Clone current best model 
            next_gen_model                  = ChessModel2(19,24).cuda()
            next_gen_model.load_state_dict(self.model_params[self.top_model])

            #View performance vs stockfish before
            p_loss,v_loss                   = trainer.check_vs_stockfish(next_gen_model)
            print(f"\t\tPRE_TRAIN:\n\t\tp_loss:{p_loss:.4f}\t\tv_loss:{v_loss:.4f}\n\n")
            #Train it 
            print(f"\t\tTRAINING:")
            p_losses,v_losses               = trainer.train_model(next_gen_model,training_dataset,bs=self.bs,lr=self.lr,wd=self.wd,betas=self.betas,n_epochs=self.n_epochs)
            epoch                           = 0 
            for p,v in zip(p_losses,v_losses):
                print(f"\t\tEPOCH [{epoch}]\n\t\t\tp_loss:{p:.4f}\t\tv_loss:{v:.4f}\n")
            self.model_params[self.gen+1]   = next_gen_model.state_dict()
            print(f"\n")

            #View performance vs stockfish after 
            p_loss,v_loss                   = trainer.check_vs_stockfish(next_gen_model)
            print(f"\n\t\tPOST_TRAIN:\n\t\tp_loss:{p_loss:.4f}\t\tv_loss:{v_loss:.4f}\n\n")

            #Find best model 
            print(f"\t\tMODEL SHOWDOWN\n")
            self.top_model,matchups         = alg_train.find_best_model(self.model_params,40,50)
            print(f"\t\tTop Model: {self.top_model}")
            for match in matchups:
                print(f"\t\t\t{match}")
            print("")
            self.gen                        += 1

        
            #Reset training data pool
            self.current_generation_data    = []
            #Reduce lr 
            self.lr                         *= self.lr_mult

            self.unsync_all_clients()
    

    #Will pass the test params to the next available client 
    #   and return which client this is 
    def pass_test_params_to_client(self,model1_params,model2_params):

        passed_params                   = False

        while not passed_params:
            for client in self.clients:
                #Pass only if locked and waiting
                if client.lock:
                    client.recieve_test_params(model1_params,model2_params)
                    passed_params           = True 
                    break 
        return client
    

    #Run the server
    def run(self):
        
        #ALways on 
        while self.running:

            #Check for a new client
            try:
                client_socket,addr      = self.server_socket.accept() 
                
                #Create and start client
                new_client              = Client_Manager(client_socket,addr,self.get_next_id(),Queue(),self.model_params[self.top_model],self.game_params)
                new_client.start()
                print(f"\n\tServer started client id:{new_client.id}\n")

                #Keep track of in client list
                self.clients.append(new_client)
                
            except TimeoutError:
                pass 
            
            self.update_training_state()

            #Pass data to clients 
            self.update_clients()

            # #Recieve dtaa from clients
            # self.recieve_data_from_clients()


    #Recursively create bracket and play out models 
    #Model lists is a list of (id#,params) tuples 
    def run_testing_games(self,model_lists):

        if len(model_lists) > 2:
            midpoint    = len(model_lists) //2 
            winner1     = self.run_testing_games(model_lists[:midpoint])
            winner2     = self.run_testing_games(model_lists[midpoint:])

            return self.run_testing_games([winner1,winner2])

        else:
            #if single item, return it as 'winner
            if len(model_lists) == 1:
                return model_lists[0] 
            else:
                #Get data 
                p1_id,p1_params     = model_lists[0]
                p2_id,p2_params     = model_lists[1]

                #Play games 
                p1_wins,p2_wins     = alg_train.showdown_match(p1_params,p2_params,self.test_params['n_games'],self.test_params['ply'],self.test_params['n_iters'],self)

                self.game_outcomes[self.gen].append(f"{p1_id}vs{p2_id}\t{p1_wins}:{p2_wins}|{self.test_params['n_games']-(p1_wins+p2_wins)}")

                if p1_wins > p2_wins:
                    return model_lists[0] 
                
                #Return second model on tie (no reason?...)
                else:
                    return model_lists[1]

    def shutdown(self):
        self.running    =  False 

        for client_manager in self.clients:
            client_manager.shutdown()
        

        self.server_socket.close()

        self.join()


#Plays one game using the specified model and returns all experiences from that game
#   max_game_ply is the max moves per game 
#   n_iters is the number of iterations run by the MCTree to evaluate the position
#   kill is the way for the spawning process to kill the game immediately and return
def play_game(model_dict:str|OrderedDict|torch.nn.Module,
              model_id:str,
              kill,
              max_game_ply=160,
              n_iters=800):

    #Create board and tree
    print(f"creating engine with model_id:{model_id}")
    engine              = MCTree(max_game_ply=max_game_ply)
    game_experiences    = []
    result              = None
    engine.load_dict(model_dict)

    #Play out game
    while result is None and not kill:

        #Evaluate move 
        move_probs      = engine.evaluate_root(n_iters=n_iters)

        #Add experiences
        game_experiences.append([engine.board.fen(),{m.uci():n for m,n in move_probs.items()},0])

        #get best move by visit count
        top_move        = None
        top_visits      = -1 
        for move,n_visits in move_probs.items():
            if n_visits > top_visits:
                top_move    = move 
                top_visits  = n_visits
        
        #Push move to board and setup engine for next mve
        result          = engine.make_move(top_move)

    if kill:
        input(f"recieved direction to terminal")
    #update game outcome in set of experiences
    #   current behavior if killed in the middle of the game 
    #   is to assume a tie but still return
    for i in range(len(game_experiences)):
        game_experiences[i][2]  = result

    return game_experiences



#Streams the set of game data from a client to the server.
def stream_exp_to_server(game_experiences:list,client_socket:socket.socket):
    print(f"streaming data to server")
    #Let server know incoming
    client_socket.send("Start".encode())

    for packet_i,exp in enumerate(game_experiences):
        #Get a "Send" instruction
        confirm         = client_socket.recv(32).decode()
        if not confirm == "Send":
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
        client_socket.send(data_packet.encode())
        #print(f"\tsent packet [{packet_i}/{len(game_experiences)}]")#DEBUG

    #Let server know thats it
    confirm         = client_socket.recv(32).decode()
    client_socket.send("End".encode())
    print(f"end streaming to server")
    return True


#Asks server what should be done next
def get_workload(client_socket:socket.socket):

    #Client_Manager socket will recieve a string max length 1024
    next_job:str        = client_socket.recv(32).decode()

    #Decode job 
    #   will be either:
    #       - switch to new model (will require downloading model dictionary)
    #       - play another game 
    if next_job == "New Model":
        print(f"got job New Model")
    elif next_job == "New Game":
        print(f"got job New Game")
    elif next_job == "Kill":
        print(f"got job Kill")
    else:
        print(f"got strange response: '{next_job}'")
    input(f"execute? ")
    return next_job


#Handles a client (to allow for threading)
def handle_client(client_socket:socket.socket,address,idw,communication_var,job_queue:Queue):
    
    try:
        #First, send job worker id to client 
        client_socket.send(str(idw).encode())
        while True:

            #Next, send workload
            client_socket.send("New Game".encode())
            #print(f"sending workload")

            #Receive the 'Start' signal 
            client_response         = client_socket.recv(32)
            experiences             = []

            #Recieve data until "End" signal
            #print(f"streaming data")
            while True:
                #Send go ahead 
                client_socket.send("Send".encode())
                client_response     = client_socket.recv(32768).decode()
                if client_response  == "End":
                    break 
                else:
                    try:
                        #Add experience to the experience queue to transfer
                        #   back to server thread
                        client_response = json.loads(client_response)
                        experiences.append(client_response)
                    except json.JSONDecodeError:
                        print(f"recieved data\n{client_response}")
                        input(f"ERROR continue?")
                        

            # print(f"ending data stream")
            # print(f"queue is {job_queue.qsize()}\n\n")

            #client_socket.send("Kill".encode())
    except OSError:
        print(f"Lost communication with client")
        return False
            
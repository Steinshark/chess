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
        print(f"creating client")


    def run(self):
        
        #Initialize the connection with the server and receive id
        self.client_socket.connect((self.address,self.port))
        self.id                     = int(self.client_socket.recv(32).decode())
        print(f"\tworker connected with id:{self.id}")
    
        #
        while self.running:

            #Recieve a model_dictionary
            self.recieve_model_params()

            #Run a game
            self.run_game()

            #Return experiences 
            self.upload_data()
         
    
    def recieve_model_params(self):

        #Check not running 
        if not self.running:
            return 
        
        #Receive 64MB of data 
        data_packet                 = self.client_socket.recv(1024*1024*64) 
        buffer                      = BytesIO(data_packet)                 
        params_as_bytes             = buffer.getvalue()
        model_parameters            = torch.load(BytesIO(params_as_bytes))
        #model_parameters            = json.loads(data_packet.decode())

        #attempt to instantiate model with them
        self.current_model          = ChessModel2(19,24).cuda()
        self.current_model.load_state_dict(model_parameters)
        self.current_model_params   = model_parameters

        self.client_socket.send("Recieved".encode())


    def run_game(self):
        print(f"beginning game")
        #Recieve 1024 bytes of data 
        data_packet                 = self.client_socket.recv(1024).decode()
        game_parameters             = json.loads(data_packet)

        #Decode game parameters
        max_game_ply                = game_parameters['ply']
        n_iters                     = game_parameters['n_iters']

        #Play game 
        training_data               = alg_train.play_game(self.current_model,
                                                          max_game_ply=max_game_ply,
                                                          n_iters=n_iters,
                                                          wildcard=self,
                                                          device_id=self.device_id)

        self.current_data_batch     = training_data
        print(f"finished game\n")


    def upload_data(self):
        
        #Check for dead 
        if not self.running:
            return 
        
        #Alert server   
        self.client_socket.send("Start".encode())

        #Send exps
        for packet_i,exp in enumerate(self.current_data_batch):

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
            #print(f"\tsent packet [{packet_i}/{len(game_experiences)}]")#DEBUG


        #Receive the "Ready"
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

        self.current_model_params   = model_params
        self.current_game_params    = game_params
        self.in_game                = False
        self.lock                   = False


    def is_alive(self):
        if not isinstance(self.client_address,socket.socket):
            return False
        return True
    

    def recieve_model(self,new_model):
        self.current_model_params    = new_model
    

    def run(self):
        try:
            #First, send job worker id to client 
            self.client_socket.send(str(self.id).encode())


            while self.running:
                self.in_game            = True
                #Send model parameters to client 
                #   serialize params to buffer, then send buffer    
                buffer                  = BytesIO()
                torch.save(self.current_model_params,buffer)
                params_as_bytes         = buffer.getvalue()
                #data_packet             = json.dumps(self.current_model_params)
                self.client_socket.send(params_as_bytes)

                #Get confirmation from client "Recieved"
                self.client_socket.recv(32)

                #Send game parameters to client 
                data_packet             = json.dumps(self.current_game_params)
                self.client_socket.send(data_packet.encode())


                #   *CLIENT PLAYS GAME* 

                #Receive the 'Start' signal 
                client_response         = self.client_socket.recv(32)

                #Recieve data until "End" signal
                while True:
                    #Send Ready Signal
                    self.client_socket.send("Ready".encode())
                    client_response     = self.client_socket.recv(32768).decode()
                    if client_response  == "End":
                        break 
                    else:
                        try:
                            #Add experience to the experience queue to transfer
                            #   back to server thread
                            client_response = json.loads(client_response)

                            #Clean back up            FEN                   MOVE_DISTR                      OUTCOME
                            client_response = (client_response[0],json.loads(client_response[1]),float(client_response[2]))
                            self.queue.put(client_response)
                        except json.JSONDecodeError:
                            print(f"ERROR IN DATAT recieved:\n{client_response}")
                            input(f"ERROR continue?")
                
                self.in_game                = False

                while self.lock:
                    time.sleep(.1)
                            


                #client_socket.send("Kill".encode())
        except OSError:
            print(f"Lost communication with client")
            return False
            pass


    def shutdown(self):
        self.client_socket.close()
        self.running                = False



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
            
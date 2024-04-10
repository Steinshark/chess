#Author: Everett Stenberg
#Description:   defines functions for training the network from
#               start to finish WITH NETWORK and THREADING CONSIDERATIONS



from collections import OrderedDict
import torch 
from mctree import MCTree
import socket
import json
import numpy

#Plays one game using the specified model and returns all experiences from that game
#   max_game_ply is the max moves per game 
#   n_iters is the number of iterations run by the MCTree to evaluate the position
#   kill is the way for the spawning process to kill the game immediately and return
def play_game(model_dict:str|OrderedDict|torch.nn.Module,
              kill,
              max_game_ply=160,
              n_iters=800):

    #Create board and tree
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

    #Client socket will recieve a string max length 1024
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
def handle_client(client_socket:socket.socket,address,idw,communication_var):
    #First, send job worker id to client 
    client_socket.send(str(idw).encode())
    while True:

        #Next, send workload
        client_socket.send("New Game".encode())
        print(f"sending workload")

        #Receive the 'Start' signal 
        client_response         = client_socket.recv(32)
        experiences             = []

        #Recieve data until "End" signal
        print(f"streaming data")
        while True:
            #Send go ahead 
            client_socket.send("Send".encode())
            print(f"\tSEND")
            client_response     = client_socket.recv(32768).decode()
            print(f"\t recieve {client_response[:10]}")
            if client_response  == "End":
                print(f"Received END signal")
                break 
            else:
                try:
                    client_response = json.loads(client_response)
                    experiences.append(client_response)
                except json.JSONDecodeError:
                    print(f"recieved data\n{client_response}")
                    

        print(f"ending data stream")
        print(f"comm var is {communication_var}\n\n")

        #client_socket.send("Kill".encode())
        
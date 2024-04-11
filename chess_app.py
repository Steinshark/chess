#Author: Everett Stenberg
#Description: a GUI application to streamline all chess-related activities (train,play,observe)



#Chess related
import chess
import chess.svg 

#Utility related 
import random
import time
import math
import json

#Window related
import tkinter as tk 
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk    import Checkbutton, Button,Entry, Label
from tkinter.ttk    import Combobox, Progressbar
from ttkthemes import ThemedTk
from cairosvg import svg2png
import chess_player
from tkinter.ttk import Frame 

#System related 
import sys
import os
import threading

#Debug related 
from matplotlib import pyplot as plt
from pprint import pp

#ML related
import torch 
import numpy
import net_chess
from model import ChessModel2
import mctree

#Network related 
import socket
import threading
from queue import Queue
#Do stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("C:/gitrepos")


PLAYER_TYPES            = {"Human":chess_player.SteinChessPlayer,"Engine":chess_player.HoomanChessPlayer}
WORKING_MODEL           = ChessModel2(19,24).cuda().state_dict()
WORKING_MODEL_ID        = 0
SERVER_IP               = '192.168.5.10'

#A GUI program to make life easier
class ChessApp:
    global WORKING_MODEL
    global WORKING_MODEL_ID
    
    def __init__(self):
        print(f"create app")
        #Chess related variables
        self.tournament_board   = chess.Board()
        self.player_white       = chess_player.ChessPlayer(self.tournament_board)
        self.player_black       = chess_player.ChessPlayer(self.tournament_board)

        
        #Network variabls 
        self.server_socket      = None 
        self.client_sockets     = {}
        self.client_threads     = {}

        #Window related variables
        self.window             = ThemedTk(theme='adapta')
        self.kill_var           = False
        self.comm_var           = "Normal"
        print(f"created window")
        self.setup_window()
        
        self.moves = list()
        self.game_over = False
        self.players = ['white','black']
        self.counter = 0
        

        #Thread related vars 


    #Create the gui windows and menus, etc...
    #   define behavior for drag and on_close
    def setup_window(self,window_x=1920,window_y=1080):

        #Set up window
        self.window.geometry(f"{window_x}x{window_y}")
        self.window.resizable()
        self.window.title("Chess Showdown v0.1")

        #Define behavior on drag 
        self.drag_action        = False
        def drag_action(event):

            if event.widget == self.window and not self.drag_action:
                self.window.geometry(f"{400}x{50}") 
                self.drag_action = True

        self.window.bind('<Configure>',drag_action)
        self.window.protocol('WM_DELETE_WINDOW',self.on_close)

        #Create menu
        self.main_menu      = tk.Menu(self.window)
        #   File
        self.file_menu      = tk.Menu(self.main_menu,tearoff=False) 
        self.file_menu.add_command(label='New') 
        self.file_menu.add_command(label='Reset') 
        self.file_menu.add_command(label='Players') 
        self.file_menu.add_command(label='Edit')
        #   Game
        self.game_menu      = tk.Menu(self.main_menu,tearoff=False) 
        self.game_menu.add_command(label='New') 
        self.game_menu.add_command(label='Reset') 
        self.game_menu.add_command(label='Players') 
        self.game_menu.add_command(label='Edit')
        #   Players
        self.players_menu      = tk.Menu(self.main_menu,tearoff=False) 
        self.players_menu.add_command(label='Configure',command=self.setup_players) 
        self.players_menu.add_command(label='-') 
        self.players_menu.add_command(label='-') 
        self.players_menu.add_command(label='-')
        #   Train 
        self.train_menu         = tk.Menu(self.main_menu,tearoff=False) 
        self.train_menu.add_command(label='Start Server',command=self.run_as_server) 
        self.train_menu.add_command(label='Start Client',command=self.run_as_worker) 
        self.train_menu.add_command(label='Start Client[4060]',command=lambda: self.run_as_worker(device=0)) 
        self.train_menu.add_command(label='Start Client[3060]',command=lambda: self.run_as_worker(device=1)) 
        self.train_menu.add_command(label='-') 
        self.train_menu.add_command(label='-')

        #Add cascades 
        self.main_menu.add_cascade(label='File',menu=self.file_menu)
        self.main_menu.add_cascade(label='Game',menu=self.game_menu)
        self.main_menu.add_cascade(label='Players',menu=self.players_menu)
        self.main_menu.add_cascade(label='Train',menu=self.train_menu)

        self.window.config(menu=self.main_menu)



        #Run
        self.window.mainloop()
    

    #Define what each player will be 
    #   either engine or human player
    def setup_players(self,winx=400,winy=625):
        dialogue_box    = ThemedTk()
        dialogue_box.title("Player Setup")
        dialogue_box.geometry(F"{winx}x{winy}")

        p1title_frame   = Frame(dialogue_box)
        p1title_frame.pack(side='top',expand=True,fill='x')
        p1name_frame    = Frame(dialogue_box)
        p1name_frame.pack(side='top',expand=True,fill='x')
        p1type_frame    = Frame(dialogue_box)
        p1type_frame.pack(side='top',expand=True,fill='x')
        p1blank_frame    = Frame(dialogue_box)
        p1blank_frame.pack(side='top',expand=True,fill='x')
        p2title_frame   = Frame(dialogue_box)
        p2title_frame.pack(side='top',expand=True,fill='x')
        p2name_frame    = Frame(dialogue_box)
        p2name_frame.pack(side='top',expand=True,fill='x')
        p2type_frame    = Frame(dialogue_box)
        p2type_frame.pack(side='top',expand=True,fill='x')
        p2blank_frame    = Frame(dialogue_box)
        p2blank_frame.pack(side='top',expand=True,fill='x')
        dataentry_frame    = Frame(dialogue_box)
        dataentry_frame.pack(side='top',expand=True,fill='x')



        #Player1 Title
        p1label                 = Label(p1title_frame,text='PLAYER1',font=('Helvetica', 16, 'bold'))
        p1label.pack(expand=True,fill='x')
        #Player1 Name
        p1name_label            = Label(p1name_frame,text='Player1 Name',width=25)
        p1name_entry            = Entry(p1name_frame,width=45)
        p1name_label.pack(side='left',expand=False,fill='x',padx=5)
        p1name_entry.pack(side='right',expand=True,fill='x',padx=5)
        #Player1 Type
        p1type_label            = Label(p1type_frame,text='Player1 Type',width=25)
        p1type_entry            = Combobox(p1type_frame,state='readonly',width=45)
        p1type_entry['values']  = list(PLAYER_TYPES.keys())
        p1type_entry.current(0)
        p1type_label.pack(side='left',expand=False,fill='x',padx=5)
        p1type_entry.pack(side='right',expand=True,fill='x',padx=5)
        #Player2 Title
        p2label                 = Label(p2title_frame,text='PLAYER2',font=('Helvetica', 16, 'bold'))
        p2label.pack(expand=True,fill='x')
        #Player2 Name
        p2name_label            = Label(p2name_frame,text='Player2 Name',width=25)
        p2name_entry            = Entry(p2name_frame,width=45)
        p2name_label.pack(side='left',expand=False,fill='x',padx=5)
        p2name_entry.pack(side='right',expand=True,fill='x',padx=5)
        #Player2 Type
        p2type_label            = Label(p2type_frame,text='Player2 Type',width=25)
        p2type_entry            = Combobox(p2type_frame,state='readonly',width=45)
        p2type_entry['values']  = list(PLAYER_TYPES.keys())
        p2type_entry.current(0)
        p2type_label.pack(side='left',expand=False,fill='x',padx=5)
        p2type_entry.pack(side='right',expand=True,fill='x',padx=5)
        #Dataentry
        dataenter_button        = Button(dataentry_frame,text="Submit",command=self.save_player_info)
        dataenter_button.pack(expand=False,fill='x',padx=20)
        

        #PACK
        self.dialog_pointer     = dialogue_box
        dialogue_box.mainloop()


    #Bring data over to main window
    def save_player_info(self):
        self.dialog_pointer.destroy()
        pass 


    #Run this as a server (handles training algorithm)
    def run_as_server(self):

        self.server                 = net_chess.Server(address=SERVER_IP)
        self.server.start()                 
        print(f"started Server")
    

    #Run this as a worker (Generates training data)
    def run_as_worker(self,device=None):
        self.client                     = net_chess.Client(device_id=device,address=SERVER_IP)
        self.client.start()
        print(f"started client thread\n\n")


    #DEPRECATED
    def play(self):

            self.board = chess.Board(chess.STARTING_FEN)
            self.game_over = False
            while not self.game_over:
                print(self.board)
                print(f"{self.get_legal_moves()}")
                move = input("mv: ")
                self.board.push_san(move)
                self.check_game()
            res = self.board.outcome()
            print(res)

    #DEPRECATED
    def check_move_from_board(board,move):
        #is move legal?
        return move in [board.uci(move)[-5:] for move in iter(board.legal_moves)]

    #DEPRECATED
    def check_game(self):
        if self.board.outcome() is None:
            return
        else:
            self.game_over = True


    #Ensure proper behavior when closing window
    def on_close(self):

        print(f"closing")

        #Attempt server shutdown
        try:
            self.server.shutdown()
            print(f"shutdown server")
        except AttributeError:
            pass 
        
        #Attempt client shutdown 
        try:
            self.client.shutdown()
        except AttributeError:
            pass
        self.kill_var       = True 
        
        #Ensure all servers and clients are closed
        try:
            self.server_socket.close()
        except OSError:
            pass 
        except AttributeError:
            pass

        for wid in self.client_sockets:
            try:
                self.client_sockets[wid].close()
            except OSError:
                pass

        #Return 
        for wid in self.client_threads:
            try:
                self.client_threads[wid].join()
            except OSError:
                pass
        try:
            self.server_thread.join()
        except AttributeError:
            pass
        print(f"on_close finished")
        self.window.destroy()


if __name__ == "__main__":  
    app     = ChessApp()
    
#Author: Everett Stenberg
#Description: a GUI application to streamline all chess-related activities (train,play,observe)



#Chess related
import PIL.Image
import chess
import chess.svg 
import cairosvg

#Utility related 
import random
import time
import math
import json
import io 

#Window related
import tkinter as tk 
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk    import Checkbutton, Button,Entry, Label
from tkinter.ttk    import Combobox, Progressbar
from ttkthemes import ThemedTk
from cairosvg import svg2png
import chess_player
from tkinter.ttk import Frame, Style

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
import PIL
from PIL import ImageTk

#Do stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("C:/gitrepos")


PLAYER_TYPES            = {"Human":chess_player.SteinChessPlayer,"Engine":chess_player.HoomanChessPlayer}
MODEL_DICTS             = [fname for fname in os.listdir() if "gen_" in fname and ".dict" in fname]

SERVER_IP               = '169.254.34.168'

#A GUI program to make life easier
class ChessApp:

    def __init__(self):
        #Chess related variables
        self.board              = chess.Board()
        self.player_white       = chess_player.ChessPlayer(self.board)
        self.player_black       = chess_player.ChessPlayer(self.board)
        self.current_moves      = list(self.board.generate_legal_moves())
        
        #Network variabls 
        self.server_socket      = None 
        self.client_sockets     = {}
        self.client_threads     = {}

        #Window related variables
        self.window             = ThemedTk(theme='adapta')
        self.kill_var           = False
        self.comm_var           = "Normal"
        
        self.moves = list()
        self.game_over = False
        self.players = ['white','black']
        self.counter = 0
        

        #explore related vars 
        self.fill               = {} 
        self.prev_click         = None


        self.setup_window()

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
            if event.widget == self.window:
                if 'pil_img' in self.__dict__:
                    self.create_board_img()


        self.window.bind('<Configure>',drag_action)
        self.window.protocol('WM_DELETE_WINDOW',self.on_close)

        #Create menu 
        self.setup_menu()

        #Run
        self.window.mainloop()
    
    
    #Creates the menu 
    def setup_menu(self):
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
        self.train_menu.add_command(label='Start Client[4060]',command=lambda: self.run_as_worker(device=torch.device('cuda:0'))) 
        self.train_menu.add_command(label='Start Client[3060]',command=lambda: self.run_as_worker(device=torch.device('cuda:1'))) 
        self.train_menu.add_command(label='Explore',command=self.explore_training) 
        self.train_menu.add_command(label='Model Config',command=self.setup_model) 
        self.train_menu.add_command(label='-')

        #Add cascades 
        self.main_menu.add_cascade(label='File',menu=self.file_menu)
        self.main_menu.add_cascade(label='Game',menu=self.game_menu)
        self.main_menu.add_cascade(label='Players',menu=self.players_menu)
        self.main_menu.add_cascade(label='Train',menu=self.train_menu)

        self.window.config(menu=self.main_menu)
    
    
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


    #Define what each player will be 
    #   either engine or human player
    def setup_model(self,winx=400,winy=625):
        dialogue_box    = ThemedTk()
        dialogue_box.title("Model Setup")
        dialogue_box.geometry(F"{winx}x{winy}")

        p1title_frame   = Frame(dialogue_box)
        p1title_frame.pack(side='top',expand=True,fill='x')
        p1name_frame    = Frame(dialogue_box)
        p1name_frame.pack(side='top',expand=True,fill='x')
        p1type_frame    = Frame(dialogue_box)
        p1type_frame.pack(side='top',expand=True,fill='x')
        dataentry_frame    = Frame(dialogue_box)
        dataentry_frame.pack(side='top',expand=True,fill='x')



        #Model Setup title
        setuplabel                 = Label(p1title_frame,text='Model Setup',font=('Helvetica', 16, 'bold'))
        setuplabel.pack(expand=True,fill='x')

        #Model Dict File
        setupdict_label            = Label(p1name_frame,text='Model File',width=25)
        setupdict_entry            = Combobox(p1name_frame,state='readonly',width=45)
        setupdict_label.pack(side='left',expand=False,fill='x',padx=5)
        setupdict_entry.pack(side='right',expand=True,fill='x',padx=5)
        setupdict_entry['values']  = list(MODEL_DICTS)
        setupdict_entry.current(0)

        #Model default iters
        setupiters_label            = Label(p1type_frame,text='Default iters',width=25)
        setupiters_label.pack(side='left',expand=False,fill='x',padx=5)
        setupiters_entry            = Entry(p1type_frame,width=45)
        setupiters_entry.pack(side='right',expand=True,fill='x',padx=5)

        dataenter_button        = Button(dataentry_frame,text="Submit",command=lambda : self.save_model_info(setupdict_entry,setupiters_entry,dialogue_box))
        dataenter_button.pack(expand=False,fill='x',padx=20)
        

        #PACK
        self.dialog_pointer     = dialogue_box
        dialogue_box.mainloop()


    #Bring data over to main window
    def save_player_info(self):
        self.dialog_pointer.destroy()
        pass 


    def save_model_info(self,fname_cbox:Combobox,iters_entry:Entry,window):
        filename            = fname_cbox.get()
        n_iters             = int(iters_entry.get())

        self.model_file     = filename
        self.n_iters        = n_iters

        #Create model 
        self.create_model(filename)

        window.destroy()


    def create_model(self,fname):

        #Create model 
        self.cur_file       = fname
        self.model          = mctree.MCTree(from_fen=self.board.fen())
        self.model.load_dict(fname)

    
    #Run this as a server (handles training algorithm)
    def run_as_server(self):
        self.server                 = net_chess.Server(address=SERVER_IP)
        self.server.load_models()       
        self.server.start()          
    

    #Run this as a worker (Generates training data)
    def run_as_worker(self,device=torch.device('cuda')):
        self.client                     = net_chess.Client(device=device,address=SERVER_IP)
        self.client.start()


    #Setup the app to explore chess games
    def explore_training(self):
        
        #Start board bigger
        self.window.geometry(f"{800}x{600}")
        #Start a board object 
        #Clear the screen
        markdel                 = [] 
        for child in self.window.children:
            markdel.append(self.window.children[child])
        for frame in markdel:
            frame.destroy()    
        
        #Rebuild menu 
        self.setup_menu()

        #Sertup window for exploring
        style                   = Style()
        style.configure('ExploreFrame1.TFrame',background='#9494aa')
        style.configure('ExploreFrame2.TFrame',background='blue')
        self.button_frame       = Frame(self.window,style='ExploreFrame1.TFrame',width=250)
        self.button_frame.pack(side='left',fill='y',expand=False)
        self.explore_frame      = Frame(self.window,style='ExploreFrame2.TFrame')
        self.explore_frame.pack(side='right',fill='both',expand=True)
        self.board_canvas       = tk.Canvas(self.explore_frame,width=20,height=20)
        self.board_canvas.pack(fill='both',padx=10,pady=10,expand=True)
        self.window.update()

        self.model_move_f       = Frame(self.button_frame)
        self.model_move_f.pack(expand=True,fill='x')
        self.model_move_l       = Label(self.model_move_f,text='n_iters',width=20)
        self.model_move_l.pack(side='left',fill='x')
        self.model_move_e       = Entry(self.model_move_f,text='',width=25)
        self.model_move_e.pack(side='left',fill='x')
        self.model_move_b       = Button(self.model_move_f,text='run',command=self.make_model_move)
        self.model_move_b.pack(side='right',fill='x')


        #Setup click listeners
        self.board_canvas.bind('<Button-1>',self.handle_click)
        self.create_board_img()
        pass

    
    #Return a PIL img of the board, scaled to the size of the board_canvas
    def create_board_img(self):

        #Create the svg
        svg_img                 = chess.svg.board(self.board,fill=self.fill)

        #Write to bytes
        png_bytesio             = io.BytesIO()
        cairosvg.svg2png(bytestring=svg_img,write_to=png_bytesio)
        png_bytes               = png_bytesio.getvalue()

        self.pil_img            = PIL.Image.open(io.BytesIO(png_bytes))
        self.canv_w             = self.board_canvas.winfo_width()
        self.canv_h             = self.board_canvas.winfo_height()
        square_dim              = min(self.canv_w,self.canv_h)
        self.pil_image          = self.pil_img.resize((square_dim,square_dim))

        # scaled_bytesio          = io.BytesIO()
        # scaled_img.save(scaled_bytesio,format='png')
        # scaled_bytes            = scaled_bytesio.getvalue()
        #self.pil_img.show()
        self.image              = ImageTk.PhotoImage(self.pil_image)#data=scaled_bytes,format='png')

        #Create x-offset to center square 
        self.x_offset           = (self.canv_w - square_dim)  // 2 if self.canv_w > self.canv_h else 0
        self.board_canvas.create_image(self.x_offset,0,image=self.image,anchor=tk.NW)
        

    #Handle clicks on board when exploration mode 
    def handle_click(self,mouse_event):

        click_x         = mouse_event.x 
        click_y         = mouse_event.y

        #Figure out relative to board 
        board_offset    = self.pil_image.size[0] * 3 / 80
        board_real_size = self.pil_image.size[0] - 2 * board_offset

        #Figure out click location by percent of board
        real_board_x    = 100 * (click_x - self.x_offset - board_offset) / board_real_size
        real_board_y    = 100 * (click_y - board_offset) / board_real_size

        #Figure out square 
        click_file      = int(real_board_x/(100/8))
        click_rank      = 7 - int(real_board_y/(100/8))


        #If first click, then display cur moves 
        clicked_sq      = chess.square(click_file,click_rank)

        if self.prev_click is None:
            #Get dict of colors 
            legal_moves     = [mv.to_square for mv in self.current_moves if mv.from_square == clicked_sq]
            print(f"legal moves are {legal_moves}")
            self.fill       = dict.fromkeys(legal_moves,"#959595")
        
        #if second click, and in moves from last_click, play that move 
        elif clicked_sq in [mv.to_square for mv in self.current_moves if mv.from_square == self.prev_click]:
            print(f"making move ")
            self.push_move_to_board(chess.Move(self.prev_click,clicked_sq))
            self.fill       = {}

        #If click again, clear piece
        elif clicked_sq == self.prev_click:
            self.fill       = {}

        #Print the squares like normal
        else:
            #Get dict of colors 
            legal_moves     = [mv.to_square for mv in self.current_moves if mv.from_square == clicked_sq]
            print(f"legal moves are {legal_moves}")
            self.fill       = dict.fromkeys(legal_moves,"#959595")

        self.prev_click     = clicked_sq
        print(f"board click: {(click_file,click_rank)}, board size is {self.pil_image.size}")
        self.create_board_img()
        
    
    #Make a model move on the current board 
    def make_model_move(self):

        #Create model        
        top_move    = None 
        top_count   = 0 

        move_counts     = self.model.evaluate_root(n_iters=int(self.model_move_e.get())).items()
        for move,count in move_counts:
            if count > top_count:
                top_move    = move 
                top_count   = top_count
        print(f"moves are {move_counts}")
        self.push_move_to_board(top_move)


    #Push a move to the apps board
    def push_move_to_board(self,move:chess.Move):

        if move in self.current_moves:
            self.board.push(move)
        else:
            print(f"not a legal move! {move}")
        
        if self.model:
            self.model.make_move(move)
        self.current_moves  = list(self.board.generate_legal_moves())
        self.create_board_img()
   
    
    def reset_board(self):
        self.board          = chess.Board()
        self.current_moves  = list(self.board.generate_legal_moves())

        if "model" in self.__dict__:
            self.create_model(self.cur_file)

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
    
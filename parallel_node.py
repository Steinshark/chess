#Author: Everett Stenberg
#Description:   Acts as the nodes in the MCTree. Contains information about the position,
#               move, score, visit count, and other data



import chess
import utilities
import numpy
import math


class Node:

    #Determine exploration tendency
    c           = 4.

    #For easy game outcome mapping
    RESULTS     = {"1/2-1/2":0,
                   "*":0,
                   "1-0":1,
                   "0-1":-1}



    def __init__(self,move:chess.Move,parent,prior_p:float,depth,turn:bool):

        #Postition related vars
        self.move                   = move
        self.turn                   = 1. if turn else -1.
        self.top_score              = -1_000_000 if turn else 1_000_000
        self.op                     = self.maximize if turn else self.minimize
        self.depth                  = depth

        #Tree related vars
        self.parent:Node            = parent
        self.children:list[Node]    = []

        #Node related vars and such
        self.n_visits               = 0.
        self.prior_p                = float(prior_p)
        self.cumulative_score       = 0.
        self.key                    = None

        #precompute val for score computation (good speedup)
        self.pre_compute()


    #Re-pre-compute (when applying dirichlet after first expansion from root, must do this or
    # pre compute will be off)
    def pre_compute(self):
        self.precompute         = -1 * self.turn * self.prior_p * self.c


    #Make mctree code cleaner by wrapping this
    def is_leaf(self):
        return not bool(self.children)


    #Used when finding best node. Maximizing if parent node is White els Min
    def maximize(self,x,y):
        return x > y


    #See above comment, I just want a comment above each fn
    def minimize(self,x,y):
        return x < y


    #Picks best child from the perspective of the node before it.
    #   If current node is turn 1, then looking to maximize next node score
    def pick_best_child(self):

        #Set top and best score vars for sort
        top_node    = None
        best_score  = self.top_score

        #Traverse and find best next node
        for package in [(node,node.get_score()) for node in self.children]:
            curnode,score  = package
            if self.op(score,best_score):
                best_score      = score
                top_node        = curnode

        # if top_node is None:
        #     input([(node,node.get_score()) for node in self.children])

        return top_node


    #Score is evaluated in a revised PUCT manner. Uses average result as well as exploration tendency and move counts
    def get_score(self):
        return (self.cumulative_score / (self.n_visits+1)) + self.precompute *(math.sqrt(self.parent.n_visits)/(self.n_visits+1))


    #Return just the q_score (for training)
    def get_q_score(self):
        return self.cumulative_score / (self.n_visits+1)


    #Passes up the value recieved at the leaf and updates the visit count
    def bubble_up(self,outcome):
        outcome                 = float(outcome)

        self.cumulative_score   += outcome
        self.n_visits           += 1.
        #self.computed_score     = self.get_score()

        if not self.parent is None:
            self.parent.bubble_up(outcome)


    #Used once a move is pushed onto the actual game board.
    #   This will be called on MCTree root to traverse to the move played and
    #   keep the computations
    def traverse_to_child(self,move:chess.Move):
        for child in self.children:
            if child.move == move:
                return child
        return -1


    #A node is represented as the sequence of moves that kets it there
    def __repr__(self):
        if self.parent == None:
            return "root"
        return str(self.parent) + " -> " + str(self.move)



if __name__ == '__main__':

    board   = chess.Board()
    root    = Node(None,None,0,board.turn)


    #ALGORITHM - Get to leaf
    curnode     = root
    print(f"is leaf = {curnode.is_leaf()}")
    while not curnode.is_leaf():
        curnode     = curnode.pick_best_child()

    print(f"root is {curnode}")
    print(f"board is\n{board}")
    root.expand(board)

    print(f"is leaf = {curnode.is_leaf()}")

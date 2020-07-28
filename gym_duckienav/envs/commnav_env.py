import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+-----------------+",
    "| | | : : : : : : |",
    "| | | | | | | | | |",
    "| | : | | | | | | |",
    "| : | | | : : : : |",
    "| | | | | | | | | |",
    "| : : : : : : | | |",
    "| | | | | | | | | |",
    "| | | : : | | : : |",
    "| | | | | | | | | |",
    "| : : : : : : | | |",
    "| | | | | | | | | |",
    "| : : : : : : | | |",
    "| | | | | | | | | |",
    "| : : : : : : : : |",
    "+-----------------+",
]   

ACTIONS = ["South", "North", "East", "West", "Deploy"]
nS = 16003008 #8001504
nR = 14
nC = 9
nI = 4
maxR = nR-1
maxC = nC-1

class CommNavEnv(discrete.DiscreteEnv):
    """
    Created by James Ewe (zlewe)
    
    Modified from The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    Robot need to place 2 nodes on the map to maximize the coverage while staying in the covered area. When 2 nodes is placed, episode end.
    Node 0 will be place on spawned location.
    
    Observation:
    4 initial location choices, (14*9)^3 discrete states since there are 14*9 taxi positions, (14*9)^2 states for location of 2 nodes. Total 8001504.
    
    Actions:
    5 discrete actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: deploy

    Rewards:
    per-step reward of -1 if inside coverage, -50 if outside coverage.
    except for deploying the nodes, which is constant*extra coverage.

    Rendering:
    - blue: current location
    - letters (X,Y,Z): locations of placed node(s)
    
    state space is represented by:
        (init, taxi_row, taxi_col, node2_row, node2_col, node3_row, node3_col)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.coverage = np.zeros((14,9), dtype='bool')
        self.desc = np.asarray(MAP,dtype='c')
        
        isd = np.zeros(nS)
        nA = 5
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for init in range(4):
            for dly in range (2):
                for row in range(nR):
                    for col in range(nC):
                        for row1 in range(nR):
                            for col1 in range(nC):
                                for row2 in range(nR):
                                    for col2 in range(nC):
                                        state = self.encode(init, dly, row, col, row1, col1, row2, col2)
                                        if (row == init and col == init) and (row1 == row and col1 == col) and (row2 == row and col2 == col) and (dly == 0): 
                                            isd[state] += 1                                                   
                                        for a in range(nA):
                                            # defaults
                                            newinit, newdly, newrow, newcol, newrow1, newcol1, newrow2, newcol2 = init, dly, row, col, row1, col1, row2, col2
                                            reward = -1
                                            done = False
                                            robotloc = (row, col)

                                            if a == 0:
                                                newrow = min(row+1, maxR)
                                            elif a == 1:
                                                newrow = max(row-1, 0)
                                            elif a == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                                newcol = min(col + 1, maxC)
                                            elif a == 3 and self.desc[1 + row, 2 * col] == b":":
                                                newcol = max(col - 1, 0)
                                            elif a == 4: # deploy
                                                if (dly == 0): #deploy node 1
                                                    newrow1 = row
                                                    newcol1 = col
                                                    newdly += 1
                                                    reward = 10*self.calculate_coverage_diff(init, dly, row, col, row1, col1, row2, col2)
                                                    if reward == 0:
                                                        reward = -100
                                                else: #deploy node 2
                                                    newrow2 = row
                                                    newcol2 = col
                                                    reward = 10*self.calculate_coverage_diff(init, dly, row, col, row1, col1, row2, col2)
                                                    if reward == 0:
                                                        reward = -100
                                                    done = True

                                            self.fill_coverage(init, init, newrow1, newcol1, newrow2, newcol2)
                                            if self.coverage[newrow][newcol] == False:
                                                reward = -50

                                            newstate = self.encode(newinit, newdly, newrow, newcol, newrow1, newcol1, newrow2, newcol2)
                                            P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, init, dly, robotrow, robotcol, row1, col1, row2, col2):
        # (4), 2, 14, 9, 14, 9, 14, 9
        i = init
        i *= 2
        i += dly
        i *= 14
        i += robotrow
        i *= 9 
        i += robotcol
        i *= 14
        i += row1
        i *= 9
        i += col1
        i *= 14
        i += row2 
        i *= 9
        i += col2
        return i

    def decode(self, i):
        out = []
        out.append(i % 9)
        i = i // 9
        out.append(i % 14)
        i = i // 14
        out.append(i % 9)
        i = i // 9
        out.append(i % 14)
        i = i // 14
        out.append(i % 9)
        i = i // 9 
        out.append(i % 14)
        i = i // 14
        out.append(i % 2)
        i = i // 2
        out.append(i)
        assert 0 <= i < 4 
        return reversed(out)

    def calculate_coverage_diff(self, init, dly, row, col, row1, col1, row2, col2):
        
        if (dly==0): #deploy node 1
            self.fill_coverage(init, init, init, init, init, init)
            old = np.count_nonzero(self.coverage)
            for i in range(max(row-2, 0),min(row+2, maxR)+1):
                for j in range(max(col-2,0), min(col+2, maxC)+1):
                    self.coverage[i][j] = True
            new = np.count_nonzero(self.coverage)
            return(new - old)
        
        else: #deploy node 2
            self.fill_coverage(init, init, row1, col1, init, init)
            old = np.count_nonzero(self.coverage)
            for i in range(max(row-2, 0),min(row+2, maxR)+1):
                for j in range(max(col-2,0), min(col+2, maxC)+1):
                    self.coverage[i][j] = True
            new = np.count_nonzero(self.coverage)
            return(new - old)
    
    def fill_coverage(self, row, col, row1, col1, row2, col2):
        self.coverage = np.zeros((14,9), dtype='bool')
        for i in range(max(row-2, 0),min(row+2, maxR)+1):
            for j in range(max(col-2,0), min(col+2, maxC)+1):
                self.coverage[i][j] = True
        for i in range(max(row1-2, 0),min(row1+2, maxR)+1):
            for j in range(max(col1-2,0), min(col1+2, maxC)+1):
                self.coverage[i][j] = True
        for i in range(max(row2-2, 0),min(row2+2, maxR)+1):
            for j in range(max(col2-2,0), min(col2+2, maxC)+1):
                self.coverage[i][j] = True
    
    def reset(self):
        while True:        
            s = super(CommNavEnv,self).reset()
            p = list(self.decode(s))
            self.initial_loc = [p[0], p[1]]
            return s

    def render(self, mode='human'):
        print(self.initial_loc)
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        init, dly, robotrow, robotcol, row1, col1, row2, col2 = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        
        out[1+robotrow][2*robotcol+1] = utils.colorize(ul(out[1+robotrow][2*robotcol+1]), 'blue', highlight=True)
        self.fill_coverage(init, init, row1, col1, row2, col2)
        for row in range(nR):
            for col in range(nC):
                if (self.coverage[row][col] == True) and (row!=robotrow or col != robotcol):
                    out[1+row][2*col+1] = utils.colorize(ul(out[1+row][2*col+1]), 'yellow', highlight=True)
        
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Deploy"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile

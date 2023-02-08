# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy
from tabulate import tabulate
from scipy.special import erfcinv


class Signal_information:
    pass
    def __init__(self, signal_power,path):
        self.signal_power=signal_power
        self.noise_power=0
        self.latency=0
        self.path=path
        self.copypath=path
        self.index=0
    def change_signal_power(self,signal_power):
        self.signal_power=signal_power
    def change_noise_power(self,noise_power):
        self.noise_power=noise_power
    def change_latency(self,latency):
        self.latency=latency
    def change_path(self,path):
        self.path=path
class Lightpath(Signal_information):
    pass
    def __init__(self, signal_power,path,channel):
        super().__init__(signal_power,path)
        self.channel=channel
        self.R_s=32
        self.df=50
class Node:
    pass
    def __init__(self,node_dict,strategy):
        self.successive={}
        self.label=next(iter(node_dict))
        self.connected_nodes=node_dict[self.label]["connected_nodes"]
        self.position=node_dict[self.label]["position"]
        self.switching_matrix=dict()
        self.strategy=strategy


    def propagate(self,signal_information):
        index=signal_information.index
        copypath=signal_information.copypath
        if len(signal_information.path)>=2 and index>=1:
            self.switching_matrix[copypath[index-1]][copypath[index+1]][signal_information.channel-1]=0
            try:
               self.switching_matrix[copypath[index - 1]][copypath[index + 1]][signal_information.channel - 2] = 0
            except:
               pass
            try:
               self.switching_matrix[copypath[index - 1]][copypath[index + 1]][signal_information.channel] = 0
            except:
               pass

        signal_information.change_path(signal_information.path[1:])
        signal_information.index=signal_information.index+1
        if signal_information.path:
            next_node=signal_information.path[0]
            next_line=self.label+next_node
            self.successive[next_line].optimized_launch_power()
            self.successive[next_line].propagate(signal_information)


        if not signal_information.path:
            return

    def probe(self, signal_information):
        signal_information.change_path(signal_information.path[1:])
        if signal_information.path:
            next_node = signal_information.path[0]
            next_line = self.label + next_node
            self.successive[next_line].probe(signal_information)

        if not signal_information.path:
            return

        #

class Line:
    pass
    def __init__(self,label,length):
        self.successive={}
        self.label=label
        self.length=length
        self.state=10*[1]
        self.n_amplifiers=np.divmod(length,80*10**3)[0]
        self.gain=10**1.6
        self.noise_figure=10**0.55
        self.physical_feat=[2.3025851*10**-5,2.13*10**-26,1.27*10**-3]
        self.P_opt=0
    def latency_generation(self,signal_information):
        latency=self.length/((2/3)*3e8)
        return latency

    def noise_generation(self,signal_information):
        noise=self.ase_generation()+self.nli_generation(signal_information.signal_power)
        return noise


    def propagate(self,signal_information):
        latency=self.latency_generation(signal_information)
        noise=self.noise_generation(signal_information)
        signal_information.change_latency(signal_information.latency + latency)
        signal_information.change_noise_power(signal_information.noise_power + noise)
        next_node=signal_information.path[0]
        self.state[signal_information.channel-1]=0
        self.successive[next_node].propagate(signal_information)

    def probe(self, signal_information):
        latency=self.latency_generation(signal_information)
        noise=self.noise_generation(signal_information)
        signal_information.change_latency(signal_information.latency + latency)
        signal_information.change_noise_power(signal_information.noise_power + noise)
        next_node=signal_information.path[0]
        self.successive[next_node].probe(signal_information)
    def ase_generation(self):
        h=6.62607015*10**-34
        B_n=12.5*10**9#Ghz
        f=193.414*10**12
        ase=self.n_amplifiers*(h*f*self.noise_figure*(self.gain-1))
        return ase
    def nli_generation(self,power):
        Nspan=self.n_amplifiers-1
        B_n=12.5*10**9

        R_s=32*10**9
        df=50*10**9
        alfa=self.physical_feat[0]
        beta=self.physical_feat[1]
        gamma=self.physical_feat[2]
        L_eff = 1 / (2 * alfa)
        first_part=16/(27*np.pi)*np.log10(np.pi**2/2*beta*R_s**2/alfa*10**(2*R_s/df))
        sec_part=alfa/beta*gamma**2*L_eff**2/R_s**3
        eta=first_part*sec_part
        Nli=power**3*eta*Nspan*B_n
        return Nli
    def optimized_launch_power(self):
        P_opt=(self.ase_generation()/(2*self.physical_feat[0]))**(1/3)
        self.P_opt=P_opt
class Network:
    pass
    def __init__(self,name_file):
        self.node={}
        self.line={}
        with open(name_file) as json_file:
            data = json.load(json_file)

        for i in range(len(data)):
            try:
                strategy=data[list(data.keys())[i]]["transceiver"]
            except:
                strategy="fixed_rate"
            node=Node({list(data.keys())[i]:list(data.values())[i]},strategy)
            self.node[list(data.keys())[i]]=node
            self.node[list(data.keys())[i]].switching_matrix=data[list(data.keys())[i]]["switching_matrix"]
            conn_nodes=data[list(data.keys())[i]]["connected_nodes"]
            for j in range(len(conn_nodes)):
                label=list(data.keys())[i]+conn_nodes[j]
                dist=np.linalg.norm(np.subtract(data[list(data.keys())[i]]["position"],data[conn_nodes[j]]["position"]))
                line=Line(label,dist)
                self.line[label]=line
        self.connect()
        df = pd.DataFrame(columns=['Path', 'Accumulated latency', 'Accumulated noise', 'signal to noise ratio'])
        for keys in self.node.keys():
            for keys2 in self.node.keys():
                if keys != keys2:
                    paths = self.find_all_paths(keys, keys2)
                    for path in paths:
                        signal_power=0.001#added 28/12
                        signal_information = Signal_information(signal_power, path)
                        self.probe(signal_information)
                        df1 = pd.DataFrame({'Path': [path], 'Accumulated latency': [signal_information.latency],
                                            "Accumulated noise": [signal_information.noise_power],
                                            "signal to noise ratio": [10 * np.log10(
                                                signal_information.signal_power / signal_information.noise_power)]})
                        df = pd.concat([df, df1])
                        del signal_information
        self.weighted_paths=df
        df2 = pd.DataFrame(
            columns=['Path', 'ch 1', 'ch 2', 'ch 3', 'ch 4', 'ch 5', 'ch 6', 'ch 7', 'ch 8', 'ch 9', 'ch 10'])

        for keys in self.node.keys():
            for keys2 in self.node.keys():
                if keys != keys2:
                    paths = self.find_all_paths(keys, keys2)
                    for path in paths:
                        df22 = pd.DataFrame({'Path': [path]})
                        for ch in range(1, 11):
                            flag = [1] * 11
                            for n in range(len(path) - 1):
                                if self.line[path[n] + path[n + 1]].state[ch - 1] == 0:
                                    flag[ch] = 0
                                    break;

                        df22 = pd.DataFrame({'Path': [path], 'ch 1': [flag[1]], 'ch 2': [flag[2]], 'ch 3': [flag[3]],
                                             'ch 4': [flag[4]], 'ch 5': [flag[5]], 'ch 6': [flag[6]], 'ch 7': [flag[7]],
                                             'ch 8': [flag[8]], 'ch 9': [flag[9]], 'ch 10': [flag[10]]})
                        df2 = pd.concat([df2, df22])

        self.route_space = df2

    def update_route_space(self):

        for path in self.route_space['Path']:
            x = [1] * len(self.line[list(self.line.keys())[0]].state)
            for n in range(len(path)-1):
                state = self.line[path[n] + path[n + 1]].state
                x=[x * state for x, state in zip(x, state)]
                if n!=0 and n!=(len(path)-1):
                    y=self.node[path[n]].switching_matrix[path[n-1]][path[n+1]]
                    x=[x * y for x, y in zip(x, y)]

            for i in range(len(x)):
                if x[i]==0:
                    self.route_space.loc[
                        self.route_space['Path'].map(tuple) == tuple(path), 'ch ' + str(
                            i + 1)] = 0


               #create weithte path with dataframe
    def connect(self):# this function has to set the successive attributes of all
    #the network elements as dictionaries
        for i in self.node:
           for j in self.node:
               if (i+j) in self.line.keys():
                   self.node[i].successive[i+j]=self.line[i+j]
        for p in self.line:
            self.line[p].successive[p[1]]=self.node[p[1]]

    def find_all_paths(self, start, end, path=[]):#adapted from stackoverflow,given two node labels, this function returns all the paths that connect the two nodes as list of node labels.
    #The admissible paths have to cross any node at most once;
        path = path + [start]
        if start == end:
            return [path]
        if start not in self.node.keys():
            return []
        paths = []
        for line in self.node[start].successive:
            if line[1] not in path:
                newpaths = self.find_all_paths(line[1], end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def propagate (self,signal_information):#this function has to propagate the
    #signal information through the path specified in it and returns the
    #modified spectral information;
        self.route_space.loc[self.route_space['Path'].map(tuple) == tuple(signal_information.path), 'ch ' + str(signal_information.channel)] = 0
        first_node=signal_information.path[0]
        self.node[first_node].propagate(signal_information)

        return signal_information

    def probe (self,signal_information):#this function has to propagate the
    #signal information through the path specified in it and returns the
    #modified spectral information;
        first_node=signal_information.path[0]
        self.node[first_node].probe(signal_information)
        return signal_information

    def draw(self):#this function has to draw the network using matplotlib
    #(nodes as dots and connection as lines)
        x=[]
        y=[]
        for keys in self.node.keys():
            plt.plot((self.node[keys].position[0]), (self.node[keys].position[1]), 'bo')
            plt.annotate(keys, (self.node[keys].position[0], self.node[keys].position[1]),fontsize=20)
            x.append(self.node[keys].position)
            for keys2 in self.node.keys():
                if keys+keys2 in self.line.keys():
                    y.append((self.node[keys].position))
                    y.append((self.node[keys2].position))

                    plt.plot([self.node[keys].position[0],self.node[keys2].position[0]],[self.node[keys].position[1],self.node[keys2].position[1]], color="blue")

        plt.show()
        return
    def find_best_snr(self,node1,node2,ch):#Define a method find best latency() in the class Network that, given
    #a pair of input and output nodes, returns the path that connects the two
    #nodes with the best (lowest) latency introduced by the signal propagation.

        mask = (self.weighted_paths.iloc[:, 0].str[0]).str.startswith(node1.label, na=False)
        df2=self.weighted_paths.loc[mask]
        mask2 = (df2.iloc[:, 0].str[-1]).str.startswith(node2.label, na=False)
        df2=df2.loc[mask2]

        best_path=[]
        start=node1.label
        end=node2.label

        paths=self.find_all_paths(start,end)
        best_snr=0


        for path in df2["Path"]:


            flag = 0

            for n in range(len(path)-1):
                if self.line[path[n]+path[n+1]].state[ch-1]==0:
                    flag=1
                    break;
            if flag:
                continue;

            if self.route_space.loc[self.route_space['Path'].map(tuple) == tuple(path), 'ch ' + str(ch)].values[0]==0:
                continue;

            if df2.loc[df2['Path'].map(tuple) == tuple(path), 'signal to noise ratio'].values[0] > best_snr:
                best_snr=df2.loc[df2['Path'].map(tuple) == tuple(path), 'signal to noise ratio'].values[0]
                best_path=path

        return best_path
    def find_best_latency(self,node1,node2,ch):#Define a method find best latency() in the class Network that, given
    #a pair of input and output nodes, returns the path that connects the two
    #nodes with the best (lowest) latency introduced by the signal propagation.
        mask = (self.weighted_paths.iloc[:, 0].str[0]).str.startswith(node1.label, na=False)
        df2 = self.weighted_paths.loc[mask]
        mask2 = (df2.iloc[:, 0].str[-1]).str.startswith(node2.label, na=False)
        df2 = df2.loc[mask2]

        start = node1.label
        end = node2.label

        best_path = []
        paths = self.find_all_paths(start, end)
        best_inv_latency = 0

        for path in df2["Path"]:
            flag = 0

            for n in range(len(path) - 1):
                if self.line[path[n] + path[n + 1]].state[ch-1] == 0:
                    flag = 1
                    break;
            if flag:
                continue;

            if self.route_space.loc[self.route_space['Path'].map(tuple) == tuple(path), 'ch ' + str(ch)].values[0]==0:
                continue;

            if df2.loc[df2['Path'].map(tuple) == tuple(path), 'Accumulated latency'].values[0] > best_inv_latency:
                best_snr = df2.loc[df2['Path'].map(tuple) == tuple(path), 'Accumulated latency'].values[0]
                best_path = path

        return best_path
    def stream(self,list_conn,label="latency"):#Define the method stream in the class Network that, for each element
    #of a given list of instances of the class Connection, sets its latency
    #and snr attribute. These values have to be calculated propagating a
    #SignalInformation instance that has the path that connects the input
    #and the output nodes of the connection and that is the best snr or latency
    #1path between the considered nodes. The choice of latency or snr has to
    #be made with a label passed as input to the stream function. The label
    #default value has to be set as latency
        origin_switching_matrices=[0]*len(self.node)
        for n in range(len(self.node)):
            origin_switching_matrices[n]=copy.deepcopy(self.node[list(self.node.keys())[n]].switching_matrix)


        path=[]
        if label=="snr":
            for conn in list_conn:
                self.update_route_space()
                ch = random.randrange(1, 11, 1)

                path=self.find_best_snr(self.node[conn.input],self.node[conn.output],ch)

                if path:
                    signal_information = Lightpath(conn.signal_power, path, ch)
                    conn.bit_rate = self.calculate_bit_rate(signal_information, self.node[path[0]].strategy)


                    self.propagate(signal_information)
                    conn.latency=signal_information.latency
                    conn.snr=10*np.log10(signal_information.signal_power/signal_information.noise_power)
                if not path:
                    conn.latency=0
                    conn.snr="None"

        if label=="latency":
            for conn in list_conn:
                self.update_route_space()
                ch = random.randrange(1, 11, 1)

                path = self.find_best_latency(self.node[conn.input],self.node[conn.output],ch)
                if path:

                    signal_information = Lightpath(conn.signal_power, path,ch)
                    conn.bit_rate = self.calculate_bit_rate(signal_information, self.node[path[0]].strategy)
                    self.propagate(signal_information)
                    conn.latency = signal_information.latency
                    conn.snr = 10 * np.log10(signal_information.signal_power / signal_information.noise_power)
                if not path:
                    conn.latency = 0
                    conn.snr = "None"

        for n in range(len(self.node)):
            self.node[list(self.node.keys())[n]].switching_matrix = copy.deepcopy(origin_switching_matrices[n])

    def calculate_bit_rate(self,light_path,strategy):
        Bert=1*10**-3
        path=light_path.path
        r_s=light_path.R_s
        b_n=12.5#Ghz
        gsnr=self.weighted_paths.loc[self.route_space['Path'].map(tuple) == tuple(path), 'signal to noise ratio'].values[0]
        if strategy=="fixed_rate":
            if gsnr>=2*(erfcinv(2*Bert))**2*r_s/b_n:
                Rb=100
            else:
                Rb=0
        if strategy=="flex_rate":
            if gsnr<2*(erfcinv(2*Bert))**2*r_s/b_n:
                Rb=0
            if 2*(erfcinv(2*Bert))**2*r_s/b_n <= gsnr < 14/3*(erfcinv(3/2*Bert))**2*r_s/b_n:
                Rb=100
            if 14/3*(erfcinv(3/2*Bert))**2*r_s/b_n <= gsnr < 10*(erfcinv(8/3*Bert))**2*r_s/b_n:
                Rb=200
            if gsnr >= 10*(erfcinv(8/3*Bert))**2*r_s/b_n:
                Rb=400
        if strategy=="shannon":
            Rb=2*r_s*np.log2(1+gsnr*r_s/b_n)

        return Rb
    def stream_with_matrix(self,matrix,signal_power,label):

        if not np.any(matrix):#the matrix is empty
            return
            print("the uniform matrix is empty")

        while 1:
            [input, output] =random.sample(list(pp.node), 2)
            inmat=list(pp.node.keys()).index(input)
            outmat=list(pp.node.keys()).index(output)
            if matrix[inmat][outmat]:
                break;

        conn = Connection(signal_power, input, output)
        self.stream([conn], label)
        matrix[inmat][outmat]=matrix[inmat][outmat]-conn.bit_rate

        return








class Connection:
    pass

    def __init__(self, signal_power, input, output):
        self.input = input
        self.output = output
        self.signal_power = signal_power
        self.latency = 0
        self.snr = 0
        self.bit_rate=0

        #plt.plot(*zip(*x), marker='o', color='r', ls='')







    # Press the green button in the gutter to run the script.


if __name__ == '__main__':#Create a main that constructs the network defined by ’nodes.json’ and
#runs its method stream over 100 connections with signal power equal
#to 1 mW and the input and output nodes randomly chosen. This run has
#to be performed in turn for latency and snr path choice. Accordingly, plot
#the distribution of all the latencies orr the snrs.


    # pp = Network("nodes_not_full.json")
    # # #
    signal_power = 0.001
    label="snr"
    list_snr3 = []
    list_snr4=[]
    list_snr3_br=[]
    list_snr4_br=[]
    pp = Network("nodes_full_fixed_rate.json")
    M=5
    N=len(pp.node)
    T_ij=np.array([100*M]*N**2).reshape(N,N)
    for i in range(N):
        T_ij[i][i]=0
    for j in range(10):
        pp.stream_with_matrix(T_ij,signal_power,label)
    print(T_ij)
    # list_latency = []
    # list_snr = []
    # list_snr2 = []
    #
    #
    # for i in range(150):
    #     conn = []
    #     [input, output] = random.sample(list(pp.node), 2)
    #     ch=random.randrange(1, 11, 1)
    #     label="snr"
    #     conn = Connection(signal_power, input, output)
    #     pp.stream([conn], ch,label)
    #
    #     list_snr2.append(conn.snr)
    # plt.figure(3)
    # plt.hist(list_snr2)
    # plt.xlabel("Snr")
    # plt.ylabel("Counts")
    # plt.show()
    # print(list_snr2)
    #
    #
    #

    # conn = []
    # for i in range(100):
    #
    #     [input, output] = random.sample(list(pp.node), 2)
    #
    #     label = "snr"
    #     conn.append(Connection(signal_power, input, output))
    #
    #
    # pp.stream(conn, label)
    #
    #
    # for i in range(100):
    #     list_snr3.append(conn[i].snr)
    #     list_snr3_br.append(conn[i].bit_rate)
    # plt.figure(3)
    # plt.hist(list_snr3)
    # plt.xlabel("Snr")
    # plt.ylabel("Counts")
    # plt.show()
    # print(list_snr3)
    # print(list_snr3_br)
    #
    # pp2=Network("nodes_full_flex_rate.json")
    #
    # pp2.stream(conn,label)
    #
    #
    # for i in range(100):
    #     list_snr4.append(conn[i].snr)
    #     list_snr4_br.append(conn[i].bit_rate)
    # plt.figure(3)
    # plt.hist(list_snr3)
    # plt.xlabel("Snr")
    # plt.ylabel("Counts")
    # plt.show()
    # print(list_snr4)
    # print(list_snr4_br)

# pp=Network("Nodes_full.json")
    # pp.stream(conn, label)
    # for i in range(100):
    #     list_snr4.append(conn[i].snr)
    # plt.figure(3)
    # plt.hist(list_snr4)
    # plt.xlabel("Snr")
    # plt.ylabel("Counts")
    # plt.show()
    # print(list_snr4)

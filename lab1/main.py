# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import random
from tabulate import tabulate

class Signal_information:
    pass
    def __init__(self, signal_power,path):
        self.signal_power=signal_power
        self.noise_power=0
        self.latency=0
        self.path=path
    def change_signal_power(self,signal_power):
        self.signal_power=signal_power
    def change_noise_power(self,noise_power):
        self.noise_power=noise_power
    def change_latency(self,latency):
        self.latency=latency
    def change_path(self,path):
        self.path=path

class Node:
    pass
    def __init__(self,node_dict):
        self.successive={}
        self.label=next(iter(node_dict))
        self.connected_nodes=node_dict[self.label]["connected_nodes"]
        self.position=node_dict[self.label]["position"]
    def propagate(self,signal_information):
        signal_information.change_path(signal_information.path[1:])
        if signal_information.path:
            next_node=signal_information.path[0]
            next_line=self.label+next_node
            self.successive[next_line].propagate(signal_information)

        if not signal_information.path:
            return


        #

class Line:
    pass
    def __init__(self,label,length):
        self.successive={}
        self.label=label
        self.length=length
        self.state=1

    def latency_generation(self,signal_information):
        latency=self.length/((2/3)*3e8)
        return latency

    def noise_generation(self,signal_information):
        noise=1e-9*signal_information.signal_power*self.length
        return noise


    def propagate(self,signal_information):
        latency=self.latency_generation(signal_information)
        noise=self.noise_generation(signal_information)
        signal_information.change_latency(signal_information.latency + latency)
        signal_information.change_noise_power(signal_information.noise_power + noise)
        next_node=signal_information.path[0]
        self.successive[next_node].propagate(signal_information)


class Network:
    pass
    def __init__(self,name_file):
        self.node={}
        self.line={}
        with open(name_file) as json_file:
            data = json.load(json_file)
        for i in range(len(data)):
            node=Node({list(data.keys())[i]:list(data.values())[i]})
            self.node[list(data.keys())[i]]=node
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
                        signal_power=1#added 28/12
                        signal_information = Signal_information(signal_power, path)
                        self.propagate(signal_information)
                        df1 = pd.DataFrame({'Path': [path], 'Accumulated latency': [signal_information.latency],
                                            "Accumulated noise": [signal_information.noise_power],
                                            "signal to noise ratio": [10 * np.log10(
                                                signal_information.signal_power / signal_information.noise_power)]})
                        df = pd.concat([df, df1])
                        del signal_information
        self.weighted_paths=df



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
        first_node=signal_information.path[0]
        self.node[first_node].propagate(signal_information)
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
    def find_best_snr(self,node1,node2):#Define a method find best latency() in the class Network that, given
    #a pair of input and output nodes, returns the path that connects the two
    #nodes with the best (lowest) latency introduced by the signal propagation.
        start=node1.label
        end=node2.label
        signal_power=1
        paths=self.find_all_paths(start,end)
        best_snr=0

        for path in paths:
            flag = 0
            signal_information=[]
            for n in range(len(path)-1):
                if self.line[path[n]+path[n+1]].state==0:
                    flag=1
                    break;
            if flag:
                continue;

            signal_information=Signal_information(signal_power,path)
            self.propagate(signal_information)
            snr=signal_information.signal_power / signal_information.noise_power
            if snr>best_snr:
                best_snr=snr
                best_path=path

        return best_path
    def find_best_latency(self,node1,node2):#Define a method find best latency() in the class Network that, given
    #a pair of input and output nodes, returns the path that connects the two
    #nodes with the best (lowest) latency introduced by the signal propagation.
        start=node1.label
        end=node2.label
        signal_power=1
        paths=self.find_all_paths(start,end)
        latency=[]
        flag=0
        inv_best_latency=0
        best_path=[]
        for path in paths:
            flag = 0
            signal_information = []
            for n in range(len(path) - 1):
                if self.line[path[n] + path[n + 1]].state == 0:
                    flag = 1
                    break;
            if flag:
                continue;

            signal_information=Signal_information(signal_power,path)
            self.propagate(signal_information)
            latency=signal_information.latency

            if (1/latency)>inv_best_latency:
                    inv_best_latency=(1/latency)
                    best_path=path
        return best_path
    def stream(self,list_conn,label="latency"):#Define the method stream in the class Network that, for each element
    #of a given list of instances of the class Connection, sets its latency
    #and snr attribute. These values have to be calculated propagating a
    #SignalInformation instance that has the path that connects the input
    #and the output nodes of the connection and that is the best snr or latency
    #1path between the considered nodes. The choice of latency or snr has to
    #be made with a label passed as input to the stream function. The label
    #default value has to be set as latency

        path=[]
        if label=="snr":
            for conn in list_conn:
                path=self.find_best_snr(self.node[conn.input],self.node[conn.output])
                if path:
                    signal_information=Signal_information(conn.signal_power,path)
                    self.propagate(signal_information)
                    conn.latency=signal_information.latency
                    conn.snr=10*np.log10(signal_information.signal_power/signal_information.noise_power)
                if not path:
                    conn.latency=0
                    conn.snr="None"

        if label=="latency":
            for conn in list_conn:
                path = self.find_best_latency(self.node[conn.input],self.node[conn.output])
                if path:
                    signal_information = Signal_information(conn.signal_power, path)
                    self.propagate(signal_information)
                    conn.latency = signal_information.latency
                    conn.snr = 10 * np.log10(signal_information.signal_power / signal_information.noise_power)
                if not path:
                    conn.latency = 0
                    conn.snr = "None"

class Connection:
    pass

    def __init__(self, signal_power, input, output):
        self.input = input
        self.output = output
        self.signal_power = signal_power
        self.latency = 0
        self.snr = 0


        #plt.plot(*zip(*x), marker='o', color='r', ls='')







    # Press the green button in the gutter to run the script.


if __name__ == '__main__':#Create a main that constructs the network defined by ’nodes.json’ and
#runs its method stream over 100 connections with signal power equal
#to 1 mW and the input and output nodes randomly chosen. This run has
#to be performed in turn for latency and snr path choice. Accordingly, plot
#the distribution of all the latencies orr the snrs.
    pp = Network("nodes.json")

    signal_power=0.001
    list_latency=[]
    list_snr=[]
    list_snr2=[]
    for i in range(100):
        conn=[]
        if i%2:
            label="snr"
        else:
            label="latency"
        [input,output]=random.sample(list(pp.node),2)

        conn=Connection(signal_power,input,output)
        pp.stream([conn],label)
        if label=="snr":
            list_snr.append(conn.snr)
        else:
            list_latency.append(conn.latency)

    plt.hist((list_latency))
    plt.xlabel("latency")
    plt.ylabel("Counts")
    plt.show()

    plt.figure(2)
    plt.hist(list_snr)
    plt.xlabel("Snr")
    plt.ylabel("Counts")
    plt.show()

    for i in range(100):
        conn = []
        [input, output] = random.sample(list(pp.node), 2)
        label="snr"
        conn = Connection(signal_power, input, output)
        pp.stream([conn], label)

        list_snr2.append(conn.snr)
    plt.figure(3)
    plt.hist(list_snr2)
    plt.xlabel("Snr")
    plt.ylabel("Counts")
    plt.show()


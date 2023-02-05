# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import random

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
                                            "accumulated noise": [signal_information.noise_power],
                                            "signal to noise ratio": [10 * np.log10(
                                                signal_information.signal_power / signal_information.noise_power)]})
                        df = pd.concat([df, df1])



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




        #plt.plot(*zip(*x), marker='o', color='r', ls='')
        plt.show()
        return






    # Press the green button in the gutter to run the script.


if __name__ == '__main__':#Create a main that constructs the network defined by ’nodes.json’ and
#runs its method stream over 100 connections with signal power equal
#to 1 mW and the input and output nodes randomly chosen. This run has
#to be performed in turn for latency and snr path choice. Accordingly, plot
#the distribution of all the latencies or the snrs.
    pp=Network("nodes.json")
    df = pd.DataFrame(columns=['Path', 'Accumulated latency', 'Accumulated noise', 'signal to noise ratio'])
    for keys in pp.node.keys():
        for keys2 in pp.node.keys():
            if keys != keys2:
                paths = pp.find_all_paths(keys, keys2)
                for path in paths:
                    signal_power=1#added 28/12
                    signal_information = Signal_information(signal_power, path)
                    pp.propagate(signal_information)
                    df1 = pd.DataFrame({'Path': [path], 'Accumulated latency': [signal_information.latency],
                                        "accumulated noise": [signal_information.noise_power],
                                        "signal to noise ratio": [10 * np.log10(
                                            signal_information.signal_power / signal_information.noise_power)]})
                    df = pd.concat([df, df1])

    df.style
    pp.draw()

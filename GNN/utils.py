import numpy as np
import networkx as nx  

STRUCTURES = 5


def process_clamp(clamp_file):   
    data = clamp_file.readline()
    if not data:
        return None
    line = data.strip().split()
    score = float(line[0])
    seq = line[1]
    return (seq, score)

def process_rnacontext(rnacontext_file):
    data = rnacontext_file.readline()
    if not data:
        return None
    seq_line = data.strip()
    assert (seq_line[0] == '>')
    seq = seq_line[1:]
    matrix = list()
    for structure_index in range(STRUCTURES):
        structure_line = rnacontext_file.readline().strip()
        matrix_line = [float(elem) for elem in structure_line.split()]
        matrix.append(matrix_line)
    return (seq, matrix)

def read_combined_data(sequences_path, structures_path, max_seq_len):
    data = []
    lengths = []
    labels = []
    counter = 0
    with open(sequences_path, 'r') as sequences, open(structures_path, 'r') as structures:
        while True:
            counter += 1
            seq_data = process_clamp(sequences)
            structure_data = process_rnacontext(structures)
            if not seq_data or not structure_data:
                return np.array(data), np.array(lengths), np.array(labels), counter-1
            
            labels.append(seq_data[1])
            
            G = nx.Graph()
            for i, base in enumerate(structure_data[0]):
                if i >= max_seq_len:
                    break
                G.add_node(i, label=base)
                G.add_edge(i, i + 1)  
            
            for j in range(STRUCTURES):
                for i in range(len(structure_data[1])):
                    G.nodes[i][f'structure_{j}'] = structure_data[1][i][j]
            
            data.append(G)  

            curr_seq_len = G.number_of_nodes()
            lengths.append(curr_seq_len)
            if curr_seq_len < max_seq_len:
                padd_len = max_seq_len - curr_seq_len
                for _ in range(padd_len):
                    G.add_node(curr_seq_len)
                    G.add_edge(curr_seq_len - 1, curr_seq_len)  
                    curr_seq_len += 1
            
    if __name__ == "__main__":
    
        TRAIN_STRUCTURE_FILE = "path_to_train_structure_file.RNAcontext"
        TRAIN_SEQUENCE_FILE = "path_to_train_sequence_file.clamp" 

        MAX_SEQ_LEN = 250

        data, lengths, labels = read_combined_data(TRAIN_SEQUENCE_FILE, TRAIN_STRUCTURE_FILE, MAX_SEQ_LEN)
        print(len(data))  
        print(lengths)  
        print(labels)  
        print(data[0].nodes(data=True))  

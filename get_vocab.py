import sys
import argparse 
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

def process(data):
    vocab = set()
    for line in data:
        # try:
        s = line.strip("\r\n ")
        print(s)
        hmol = MolGraph(s)
        # except:
        #     print(f'fail\t{s}')
        #     smiles = s.split('.')
        #     hmol = [MolGraph(x) for x in smiles]
        # if not isinstance(hmol, list):
        #     hmol = [hmol]
        # for hmol_ in hmol: 
        #     for node, attr in hmol_.mol_tree.nodes(data=True):
        #         smiles = attr['smiles']
        #         vocab.add( attr['label'] )
        #         for i,s in attr['inter_label']:
        #             vocab.add( (smiles, s) )
    return vocab

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='vocab.txt')

    args = parser.parse_args()

    if args.input is not None:
        with open(args.input, 'r') as f:
            data = f.readlines()
    else:
        data = [mol for line in sys.stdin for mol in line.split()[:2]]
    data = list(set(data))

    batch_size = len(data) // args.ncpu + 1 if args.ncpu != 0 else len(data)
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    if args.ncpu == 0:
        vocab_list = process(batches[0])
    else:
        pool = Pool(args.ncpu)
        vocab_list = pool.map(process, batches)
        pool.close()
        pool.join()
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    with open(args.output, 'w') as f:
        f.write('\n'.join([f"{x} {y}\t" for x,y in sorted(vocab)]))

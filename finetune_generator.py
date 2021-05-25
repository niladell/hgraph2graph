from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from valence.utils.sascorer import calculateScore as synth_score

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED

import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

import hgraph
from hgraph import HierVAE, common_atom_vocab, PairVocab
from valence.models.activity_prediction import TransformerStyleModel
from valence.data.dataloaders import SimpleTokenizer


param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(
    sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None])
)


def load_txt(filename: str) -> List[str]:
    with open(filename, "r") as f:
        return f.read().strip().split("\n")


class PredictorModel(object):
    def __init__(self, loadfile: str, device: str):
        self.device = torch.device(device)
        self.model = TransformerStyleModel.load(loadfile, map_location=self.device)
        self.activation = nn.Sigmoid()
        self.tokenizer = SimpleTokenizer(
            smiles_list=None, savefile="../valence_challenge/data/tokenizer.json"
        )
        vocab = load_txt("./valence_vocab.dot.txt")
        self.vocab, vocab_inter = zip(*[v.split() for v in vocab])
        self.vocab = [v for v in self.vocab if len(v) > 6]
        self.set_vocab = list(set(self.vocab))

    def predict(self, smiles_list: List[str]):
        discarded_mols = []
        preds = len(smiles_list) * [0.0]
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                discarded_mols.append(i)
                continue
            try:
                # Check wheter it has any known scaffolds
                has_scaffold = self.check_scaffolds(mol)
                if not has_scaffold:
                    discarded_mols.append(i)
                    continue
                drug_like = self.drug_likeness(mol)
                synthetizable = self.synthetizability(mol)
                if not drug_like or not synthetizable:
                    discarded_mols.append(i)
                    continue

                tokens = self.tokenizer("[CLS]" + smiles + "[SEP]")
                tokens = tokens.to(self.device).unsqueeze(0)
                preds[i] = self.activation(self.model(tokens, None)).item()
            except KeyError:
                discarded_mols.append(i)
        return preds

    def drug_likeness(self, mol):
        return QED.qed(mol) >= .6

    def synthetizability(self, mol):
        return synth_score(mol) <= 4

    def check_scaffolds(self, mol):
        if mol is None:
            return False
        checks = []
        for s in self.set_vocab:
            patt = Chem.MolFromSmiles(s)
            checks.append(mol.HasSubstructMatch(patt))
        return any(checks)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--atom_vocab", default=common_atom_vocab)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--generative_model", required=True)
    parser.add_argument("--pred-model", required=True)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--hidden_size", type=int, default=250)
    parser.add_argument("--embed_size", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--depthT", type=int, default=15)
    parser.add_argument("--depthG", type=int, default=15)
    parser.add_argument("--diterT", type=int, default=1)
    parser.add_argument("--diterG", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=5.0)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--inner_epoch", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min_similarity", type=float, default=0.1)
    parser.add_argument("--max_similarity", type=float, default=0.5)
    parser.add_argument("--nsample", type=int, default=10000)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    with open(args.train) as f:
        train_smiles = [line.strip("\r\n ") for line in f]

    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
    args.vocab = PairVocab(vocab)

    if args.pred_model:
        score_func = PredictorModel(args.pred_model, "cuda")
    good_smiles = train_smiles
    train_mol = [Chem.MolFromSmiles(s) for s in train_smiles]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in train_mol]

    model = HierVAE(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Loading from checkpoint " + args.generative_model)
    model_state, optimizer_state, _, beta = torch.load(args.generative_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    for epoch in range(args.epoch):
        good_smiles = sorted(set(good_smiles))
        random.shuffle(good_smiles)
        dataset = hgraph.MoleculeDataset(
            good_smiles, args.vocab, args.atom_vocab, args.batch_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda x: x[0],
            shuffle=True,
            num_workers=0,
        )

        print(f"Epoch {epoch} training...")
        for _ in range(args.inner_epoch):
            meters = np.zeros(6)
            for batch in tqdm(dataloader):
                model.zero_grad()
                loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                meters = meters + np.array(
                    [
                        x.cpu().item() if isinstance(x, torch.Tensor) else x
                        for x in [
                            kl_div,
                            loss,
                            wacc * 100,
                            iacc * 100,
                            tacc * 100,
                            sacc * 100,
                        ]
                    ]
                )
            meters /= len(dataset)
            print(
                "Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f"
                % (
                    beta,
                    meters[0],
                    meters[1],
                    meters[2],
                    meters[3],
                    meters[4],
                    meters[5],
                    param_norm(model),
                    grad_norm(model),
                )
            )

        ckpt = (model.state_dict(), optimizer.state_dict(), epoch, beta)
        torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{epoch}"))

        print(f"Epoch {epoch} decoding...")
        decoded_smiles = []
        with torch.no_grad():
            for _ in tqdm(range(args.nsample // args.batch_size)):
                outputs = model.sample(args.batch_size, greedy=True)
                decoded_smiles.extend(outputs)

        if score_func is not None:
            print(f"Epoch {epoch} filtering...")
            scores = score_func.predict(decoded_smiles)
            outputs = [
                (s, p) for s, p in zip(decoded_smiles, scores) if p >= args.threshold
            ]
            print(f"Discovered {len(outputs)} active molecules")

            novel_entries = []
            good_entries = []
            for s, p in outputs:
                mol = Chem.MolFromSmiles(s)
                fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                sims = np.array(DataStructs.BulkTanimotoSimilarity(fps, train_fps))
                good_entries.append((s, p, sims.max()))
                if args.min_similarity <= sims.max() <= args.max_similarity:
                    novel_entries.append((s, p, sims.max()))
                    good_smiles.append(s)

            print(f"Discovered {len(novel_entries)} novel active molecules")
            with open(os.path.join(args.save_dir, f"new_molecules.{epoch}"), "w") as f:
                for s, p, sim in novel_entries:
                    print(s, p, sim, file=f)

            with open(os.path.join(args.save_dir, f"good_molecules.{epoch}"), "w") as f:
                for s, p, sim in good_entries:
                    print(s, p, sim, file=f)

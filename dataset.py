import torch
from torch.utils.data import Dataset
import h5py
import os
from tqdm import tqdm
import pickle
import pandas as pd
import collections


def tokenize(data):
    encoder_dict = dict(zip(['P','A','T','C','G'], range(5)))
    tokenized_data = list()
    for gene in tqdm(data):
        tokenized_data.append([encoder_dict[token] for token in gene])

    return torch.as_tensor(tokenized_data)


def load_vocab_kmer(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def tokenize_kmer(data, kmer=3, stride=1):
    vocab3 = load_vocab_kmer('dna_vocab3mer.txt')
    tokenized_data = list()
    for gene in tqdm(data):
        tokenized_data.append([vocab3[''.join(gene[i:i + kmer])] if ''.join(gene[i:i + kmer]) in vocab3 else 0
                               for i in range(0, len(gene), stride)])

    return torch.as_tensor(tokenized_data)


def load_dna_letters(datadir, file, data_bool):
    if os.path.isfile(os.path.join(datadir, file)):
        with open(os.path.join(datadir, file), 'rb') as fp:
            dna_letters = pickle.load(fp)
        print('xpresso %s dataset dna letters loaded.', file)
    else:
        print('xpresso %s dataset needs to be converted to letters and saved.', file)
        exit()

    return dna_letters


def load_tokenized_data(datadir, file_tokenized, file_dna_letters, data_bool):
    if os.path.isfile(os.path.join(datadir, file_tokenized)):
        with open(os.path.join(datadir, file_tokenized), 'rb') as fp:
            data_dna_tokens = pickle.load(fp)
    else:
        data_dna_letters = load_dna_letters(datadir, file_dna_letters, data_bool)
        data_dna_tokens = tokenize(data_dna_letters)
        with open(os.path.join(datadir, file_tokenized), 'wb') as fp:
            pickle.dump(data_dna_tokens, fp)
        print('xpresso %s dataset tokenized and saved.', file_tokenized)

    return data_dna_tokens


class XpressoTrain(Dataset):
    def __init__(self, datadir, kmer=1):
        self.datadir = datadir
        self.trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
        self.X_trainhalflife = torch.from_numpy(self.trainfile['data'][:])
        self.X_trainpromoter_bool = torch.from_numpy(self.trainfile['promoter'][:])
        self.y_train = torch.from_numpy(self.trainfile['label'][:])
        self.geneName_train = self.trainfile['geneName'][:]

        self.X_trainpromoter_bool = self.X_trainpromoter_bool.to(torch.float32)

        if kmer == 1:
            self.X_trainpromoter_dna = load_tokenized_data(datadir, 'promoter_dna_train_tokenized.pkl',
                                                           'promoter_dna_train.pkl', self.X_trainpromoter_bool)
        else:
            self.X_trainpromoter_dna = load_dna_letters(datadir, 'promoter_dna_train.pkl', self.X_trainpromoter_bool)
            self.X_trainpromoter_dna = tokenize_kmer(self.X_trainpromoter_dna, kmer=kmer, stride=1)

    def __len__(self):
        return len(self.X_trainhalflife)

    def __getitem__(self, idx):
        return self.X_trainhalflife[idx], self.X_trainpromoter_bool[idx], self.X_trainpromoter_dna[idx], self.y_train[idx]


class XpressoVal(Dataset):
    def __init__(self, datadir, kmer=1):
        self.validfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
        self.X_validhalflife = torch.from_numpy(self.validfile['data'][:])
        self.X_validpromoter_bool = torch.from_numpy(self.validfile['promoter'][:])
        self.y_valid = torch.from_numpy(self.validfile['label'][:])
        self.geneName_valid = self.validfile['geneName'][:]

        self.X_validpromoter_bool = self.X_validpromoter_bool.to(torch.float32)

        if kmer == 1:
            self.X_validpromoter_dna = load_tokenized_data(datadir, 'promoter_dna_val_tokenized.pkl',
                                                           'promoter_dna_val.pkl', self.X_validpromoter_bool)
        else:
            self.X_validpromoter_dna = load_dna_letters(datadir, 'promoter_dna_val.pkl', self.X_validpromoter_bool)
            self.X_validpromoter_dna = tokenize_kmer(self.X_validpromoter_dna, kmer=kmer, stride=1)

    def __len__(self):
        return len(self.X_validhalflife)

    def __getitem__(self, idx):
        return self.X_validhalflife[idx], self.X_validpromoter_bool[idx], self.X_validpromoter_dna[idx], self.y_valid[idx]


class XpressoTest(Dataset):
    def __init__(self, datadir, kmer=1):
        self.testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
        self.X_testhalflife = torch.from_numpy(self.testfile['data'][:])
        self.X_testpromoter_bool = torch.from_numpy(self.testfile['promoter'][:])
        self.y_test =  torch.from_numpy(self.testfile['label'][:])
        self.geneName_test = self.testfile['geneName'][:]

        self.X_testpromoter_bool = self.X_testpromoter_bool.to(torch.float32)

        if kmer == 1:
            self.X_testpromoter_dna = load_tokenized_data(datadir, 'promoter_dna_test_tokenized.pkl',
                                                           'promoter_dna_test.pkl', self.X_testpromoter_bool)
        else:
            self.X_testpromoter_dna = load_dna_letters(datadir, 'promoter_dna_test.pkl', self.X_testpromoter_bool)
            self.X_testpromoter_dna = tokenize_kmer(self.X_testpromoter_dna, kmer=kmer, stride=1)

    def __len__(self):
        return len(self.X_testhalflife)

    def __getitem__(self, idx):
        return self.X_testhalflife[idx], self.X_testpromoter_bool[idx], self.X_testpromoter_dna[idx], \
                   self.y_test[idx]


def load_train_sequences_files(datadir, kmer=1):
    with open(os.path.join(datadir, 'proteome/train_proteome_all_patients_bool.pkl'), 'rb') as fp:
        X_trainpromoter_bool = pickle.load(fp)
    with open(os.path.join(datadir, 'proteome/train_proteome_all_patients_halflife.pkl'), 'rb') as fp:
        X_trainhalflife = pickle.load(fp)

    if kmer == 1:
        with open(os.path.join(datadir, 'proteome/train_proteome_all_patients_dna_tokenized.pkl'), 'rb') as fp:
            X_trainpromoter_dna = pickle.load(fp)
    else:
        X_trainpromoter_dna = load_dna_letters(datadir, 'promoter_dna_train.pkl', X_trainpromoter_bool)
        X_trainpromoter_dna = tokenize_kmer(X_trainpromoter_dna, kmer=kmer, stride=1)
        df_gene_train_ordered = pd.read_pickle('data/train_mRNA_and_proteome_all_patients.pkl')
        valid_index = df_gene_train_ordered.index[
            ~df_gene_train_ordered[list(df_gene_train_ordered.filter(regex='C3'))].isnull().all(axis=1)]
        X_trainpromoter_dna = X_trainpromoter_dna[valid_index]

    return X_trainpromoter_bool, X_trainhalflife, X_trainpromoter_dna


def load_val_sequences_files(datadir, kmer=1):
    with open(os.path.join(datadir, 'proteome/val_proteome_all_patients_bool.pkl'), 'rb') as fp:
        X_valpromoter_bool = pickle.load(fp)
    with open(os.path.join(datadir, 'proteome/val_proteome_all_patients_halflife.pkl'), 'rb') as fp:
        X_valhalflife = pickle.load(fp)

    if kmer == 1:
        with open(os.path.join(datadir, 'proteome/val_proteome_all_patients_dna_tokenized.pkl'), 'rb') as fp:
            X_valpromoter_dna = pickle.load(fp)
    else:
        X_valpromoter_dna = load_dna_letters(datadir, 'promoter_dna_val.pkl', X_valpromoter_bool)
        X_valpromoter_dna = tokenize_kmer(X_valpromoter_dna, kmer=kmer, stride=1)
        df_gene_val_ordered = pd.read_pickle('data/val_mRNA_and_proteome_all_patients.pkl')
        valid_index = df_gene_val_ordered.index[
            ~df_gene_val_ordered[list(df_gene_val_ordered.filter(regex='C3'))].isnull().all(axis=1)]
        X_valpromoter_dna = X_valpromoter_dna[valid_index]

    return X_valpromoter_bool, X_valhalflife, X_valpromoter_dna


def load_test_sequences_files(datadir, kmer=1):
    with open(os.path.join(datadir, 'proteome/test_proteome_all_patients_bool.pkl'), 'rb') as fp:
        X_testpromoter_bool = pickle.load(fp)
    with open(os.path.join(datadir, 'proteome/test_proteome_all_patients_halflife.pkl'), 'rb') as fp:
        X_testhalflife = pickle.load(fp)

    if kmer == 1:
        with open(os.path.join(datadir, 'proteome/test_proteome_all_patients_dna_tokenized.pkl'), 'rb') as fp:
            X_testpromoter_dna = pickle.load(fp)
    else:
        X_testpromoter_dna = load_dna_letters(datadir, 'promoter_dna_test.pkl', X_testpromoter_bool)
        X_testpromoter_dna = tokenize_kmer(X_testpromoter_dna, kmer=kmer, stride=1)
        df_gene_test_ordered = pd.read_pickle('data/test_mRNA_and_proteome_all_patients.pkl')
        valid_index = df_gene_test_ordered.index[
            ~df_gene_test_ordered[list(df_gene_test_ordered.filter(regex='C3'))].isnull().all(axis=1)]
        X_testpromoter_dna = X_testpromoter_dna[valid_index]

    return X_testpromoter_bool, X_testhalflife, X_testpromoter_dna


def load_train_sequences_glio(datadir, kmer=1):
    with open('data/glio/dnasequences/train_glio_dna_bool.pkl', 'rb') as fp:
        X_trainpromoter_bool = pickle.load(fp)
    with open('data/glio/dnasequences/train_glio_halflife.pkl', 'rb') as fp:
        X_trainhalflife = pickle.load(fp)

    if kmer == 1:
        with open('data/glio/dnasequences/train_glio_dna_tokenized.pkl', 'rb') as fp:
            X_trainpromoter_dna = pickle.load(fp)
    else: # to debug if it still working (we change directory of data)
        X_trainpromoter_dna = load_dna_letters(datadir, 'promoter_dna_train.pkl', X_trainpromoter_bool)
        X_trainpromoter_dna = tokenize_kmer(X_trainpromoter_dna, kmer=kmer, stride=1)
        df_gene_train_ordered = pd.read_pickle('data/glio/train_mRNA_prot_onlyAvg_norm_ordered.pkl')
        valid_index = df_gene_train_ordered.index[~df_gene_train_ordered.isnull().any(axis=1)]
        X_trainpromoter_dna = X_trainpromoter_dna[valid_index]

    return X_trainpromoter_bool, X_trainhalflife, X_trainpromoter_dna


def load_val_sequences_glio(datadir, kmer=1):
    with open('data/glio/dnasequences/val_glio_dna_bool.pkl', 'rb') as fp:
        X_valpromoter_bool = pickle.load(fp)
    with open('data/glio/dnasequences/val_glio_halflife.pkl', 'rb') as fp:
        X_valhalflife = pickle.load(fp)

    if kmer == 1:
        with open('data/glio/dnasequences/val_glio_dna_tokenized.pkl', 'rb') as fp:
            X_valpromoter_dna = pickle.load(fp)
    else:
        X_valpromoter_dna = load_dna_letters(datadir, 'promoter_dna_val.pkl', X_valpromoter_bool)
        X_valpromoter_dna = tokenize_kmer(X_valpromoter_dna, kmer=kmer, stride=1)
        df_gene_val_ordered = pd.read_pickle('data/glio/val_mRNA_prot_onlyAvg_norm_ordered.pkl')
        valid_index = df_gene_val_ordered.index[~df_gene_val_ordered.isnull().any(axis=1)]
        X_valpromoter_dna = X_valpromoter_dna[valid_index]

    return X_valpromoter_bool, X_valhalflife, X_valpromoter_dna


def load_test_sequences_glio(datadir, kmer=1):
    with open('data/glio/dnasequences/test_glio_dna_bool.pkl', 'rb') as fp:
        X_testpromoter_bool = pickle.load(fp)
    with open('data/glio/dnasequences/test_glio_halflife.pkl', 'rb') as fp:
        X_testhalflife = pickle.load(fp)

    if kmer == 1:
        with open('data/glio/dnasequences/test_glio_dna_tokenized.pkl', 'rb') as fp:
            X_testpromoter_dna = pickle.load(fp)
    else:
        X_testpromoter_dna = load_dna_letters(datadir, 'promoter_dna_test.pkl', X_testpromoter_bool)
        X_testpromoter_dna = tokenize_kmer(X_testpromoter_dna, kmer=kmer, stride=1)
        df_gene_test_ordered = pd.read_pickle('data/glio/test_mRNA_prot_onlyAvg_norm_ordered.pkl')
        valid_index = df_gene_test_ordered.index[~df_gene_test_ordered.isnull().all(axis=1)]
        X_testpromoter_dna = X_testpromoter_dna[valid_index]

    return X_testpromoter_bool, X_testhalflife, X_testpromoter_dna


class XpressoDNA_Lung_ProteomeLabel(Dataset):
    def __init__(self, datadir, norm_data=False, kmer=1):
        file_train_ordered = 'proteome/train_mRNA_and_proteome_all_patients_ordered_norm.pkl' if norm_data \
            else 'proteome/train_mRNA_and_proteome_all_patients_ordered.pkl'
        df_gene_train_ordered = pd.read_pickle(os.path.join(datadir, file_train_ordered))
        file_val_ordered = 'proteome/val_mRNA_and_proteome_all_patients_ordered_norm.pkl' if norm_data \
            else 'proteome/val_mRNA_and_proteome_all_patients_ordered.pkl'
        file_test_ordered = 'proteome/test_mRNA_and_proteome_all_patients_ordered_norm.pkl' if norm_data \
            else 'proteome/test_mRNA_and_proteome_all_patients_ordered.pkl'
        df_gene_val_ordered = pd.read_pickle(os.path.join(datadir, file_val_ordered))
        df_gene_test_ordered = pd.read_pickle(os.path.join(datadir, file_test_ordered))
        all_data = pd.concat([df_gene_train_ordered, df_gene_val_ordered, df_gene_test_ordered]) # cat train val e test

        all_patients_columns = list(all_data.filter(regex='C3'))
        all_patients_number = list(set([s.split('_')[0].split('.')[1] for s in all_patients_columns]))
        print('All patients: ', all_patients_number)

        train_patients_cols = [col for col in all_patients_columns]
        train_patients_cols_rna = [col for col in train_patients_cols if col.split('_')[1] == 'rna']
        train_patients_cols_prot = [col for col in train_patients_cols if col.split('_')[1] == 'prot']

        y_mRNA = torch.tensor(df_gene_train_ordered[train_patients_cols_rna].mean(axis=1, skipna=True).values)
        y_proteome = torch.tensor(df_gene_train_ordered[train_patients_cols_prot].mean(axis=1, skipna=True).values)
        y_train = torch.cat([y_mRNA.unsqueeze(1), y_proteome.unsqueeze(1)], dim=1)

        y_mRNA = torch.tensor(df_gene_val_ordered[train_patients_cols_rna].mean(axis=1, skipna=True).values)
        y_proteome = torch.tensor(df_gene_val_ordered[train_patients_cols_prot].mean(axis=1, skipna=True).values)
        y_val = torch.cat([y_mRNA.unsqueeze(1), y_proteome.unsqueeze(1)], dim=1)

        y_mRNA = torch.tensor(df_gene_test_ordered[train_patients_cols_rna].mean(axis=1, skipna=True).values)
        y_proteome = torch.tensor(df_gene_test_ordered[train_patients_cols_prot].mean(axis=1, skipna=True).values)
        y_test = torch.cat([y_mRNA.unsqueeze(1), y_proteome.unsqueeze(1)], dim=1)

        self.y_train = torch.cat([y_train, y_val, y_test], dim=0)

        self.X_trainpromoter_bool, self.X_trainhalflife, \
        self.X_trainpromoter_dna = load_train_sequences_files(datadir, kmer=kmer)

        self.X_valpromoter_bool, self.X_valhalflife, \
        self.X_valpromoter_dna = load_val_sequences_files(datadir, kmer=kmer)

        self.X_testpromoter_bool, self.X_testhalflife, \
        self.X_testpromoter_dna = load_test_sequences_files(datadir, kmer=kmer)

        self.X_trainpromoter_bool = torch.cat([self.X_trainpromoter_bool, self.X_valpromoter_bool, self.X_testpromoter_bool], dim=0)
        self.X_trainhalflife = torch.cat([self.X_trainhalflife, self.X_valhalflife, self.X_testhalflife], dim=0)
        self.X_trainpromoter_dna = torch.cat([self.X_trainpromoter_dna, self.X_valpromoter_dna, self.X_testpromoter_dna], dim=0)

    def __len__(self):
        return len(self.X_trainhalflife)

    def __getitem__(self, idx):
        return self.X_trainhalflife[idx], self.X_trainpromoter_bool[idx], self.X_trainpromoter_dna[idx], \
               self.y_train[idx]


class XpressoDNA_Glio_ProteomeLabel(Dataset):
    def __init__(self, datadir, norm_data=False, kmer=1):
        df_gene_train_ordered = pd.read_pickle('data/glio/train_mRNA_prot_onlyAvg_norm_ordered.pkl')
        df_gene_val_ordered = pd.read_pickle('data/glio/val_mRNA_prot_onlyAvg_norm_ordered.pkl')
        df_gene_test_ordered = pd.read_pickle('data/glio/test_mRNA_prot_onlyAvg_norm_ordered.pkl')

        mRNA_column = 'mRNA_avg_global_log2_z' if norm_data else 'mRNA_avg_global_log2'
        prot_column = 'proteome_avg_global'

        y_mRNA = torch.tensor(df_gene_train_ordered[mRNA_column].values)
        y_proteome = torch.tensor(df_gene_train_ordered[prot_column].values)
        y_train = torch.cat([y_mRNA.unsqueeze(1), y_proteome.unsqueeze(1)], dim=1)

        y_mRNA = torch.tensor(df_gene_val_ordered[mRNA_column].values)
        y_proteome = torch.tensor(df_gene_val_ordered[prot_column].values)
        y_val = torch.cat([y_mRNA.unsqueeze(1), y_proteome.unsqueeze(1)], dim=1)

        y_mRNA = torch.tensor(df_gene_test_ordered[mRNA_column].values)
        y_proteome = torch.tensor(df_gene_test_ordered[prot_column].values)
        y_test = torch.cat([y_mRNA.unsqueeze(1), y_proteome.unsqueeze(1)], dim=1)

        self.y_train = torch.cat([y_train, y_val, y_test], dim=0)

        self.X_trainpromoter_bool, self.X_trainhalflife, \
        self.X_trainpromoter_dna = load_train_sequences_glio(datadir, kmer=kmer)

        self.X_valpromoter_bool, self.X_valhalflife, \
        self.X_valpromoter_dna = load_val_sequences_glio(datadir, kmer=kmer)

        self.X_testpromoter_bool, self.X_testhalflife, \
        self.X_testpromoter_dna = load_test_sequences_glio(datadir, kmer=kmer)

        self.X_trainpromoter_bool = torch.cat([self.X_trainpromoter_bool, self.X_valpromoter_bool, self.X_testpromoter_bool], dim=0)
        self.X_trainhalflife = torch.cat([self.X_trainhalflife, self.X_valhalflife, self.X_testhalflife], dim=0)
        self.X_trainpromoter_dna = torch.cat([self.X_trainpromoter_dna, self.X_valpromoter_dna, self.X_testpromoter_dna], dim=0)

    def __len__(self):
        return len(self.X_trainhalflife)

    def __getitem__(self, idx):
        return self.X_trainhalflife[idx], self.X_trainpromoter_bool[idx], self.X_trainpromoter_dna[idx], \
               self.y_train[idx]


class ProteinSequencesDataset(Dataset):
    def __init__(self, datadir, proteomic_label="lung", max_len=10000):
        if proteomic_label == 'lung':
            dataset = pd.read_pickle(os.path.join(datadir, "lung_protSeq_labels_proteomic.pkl"))
            self.X_protein_seq_letters = torch.load(os.path.join(datadir, "lung_protSeq_letters_tokenized.pt"))
        else:
            dataset = pd.read_pickle(os.path.join(datadir, "glio_protSeq_labels_proteomic.pkl"))
            self.X_protein_seq_letters = torch.load(os.path.join(datadir, "glio_protSeq_letters_tokenized.pt"))

        self.y_proteome = torch.tensor(dataset["proteome_avg_global"].values)
        self.max_len = max_len

    def __len__(self):
        return len(self.y_proteome)

    def __getitem__(self, idx):
        return self.X_protein_seq_letters[idx][:self.max_len], \
               torch.nn.functional.one_hot(self.X_protein_seq_letters[idx][:self.max_len], 23), self.y_proteome[idx]


class XpressoDNA_and_ProteinSequences_dataset(Dataset):
    def __init__(self, datadir, proteomic_label="lung", max_len_prot=10000, max_len_dna=10000):
        if proteomic_label == 'lung':
            dataset = pd.read_pickle(os.path.join(datadir, "lung_protSeq_labels_proteomic.pkl"))
            dataset['gene_id'] = dataset['gene_id'].str.split('.', expand=True)[0]
            self.X_protein_seq_letters = torch.load(os.path.join(datadir, "lung_protSeq_letters_tokenized.pt"))
        else:
            dataset = pd.read_pickle(os.path.join(datadir, "glio_protSeq_labels_proteomic.pkl"))
            self.X_protein_seq_letters = torch.load(os.path.join(datadir, "glio_protSeq_letters_tokenized.pt"))

        self.y_proteome = torch.tensor(dataset["proteome_avg_global"].values)
        self.prot_max_len = max_len_prot
        self.max_len_dna = max_len_dna

        with open('data/pM10Kb_1KTest/promoter_dna_all_tokenized.pkl', 'rb') as fp:
            X_promoter_dna_all = pickle.load(fp)
        with open('data/pM10Kb_1KTest/halflife_all.pkl', 'rb') as fp:
            X_halflife_all = pickle.load(fp)
        with open('data/pM10Kb_1KTest/gene_name_all.pkl', 'rb') as fp:
            gene_name_all = pickle.load(fp)
            gene_name_all = pd.DataFrame(gene_name_all, columns=['geneName'])
            self.gene_name_all = gene_name_all['geneName'].str.decode("utf-8")

        self.dna_seq_dict = dict(zip(self.gene_name_all, X_promoter_dna_all))
        self.hlife_seq_dict = dict(zip(self.gene_name_all, X_halflife_all))

        gene_dna_valid_ids = pd.DataFrame(self.dna_seq_dict.keys(), columns=['geneName'])
        dataset_prot_with_dna = pd.merge(dataset, gene_dna_valid_ids, left_on='gene_id', right_on='geneName', how='left')

        index_not_dna = dataset_prot_with_dna.index[~dataset_prot_with_dna.isnull().any(axis=1)]

        self.X_protein_seq_letters = self.X_protein_seq_letters[index_not_dna]
        self.y_proteome = self.y_proteome[index_not_dna]
        self.gene_id_prot = dataset['gene_id']
        self.gene_id_prot = self.gene_id_prot[index_not_dna]
        self.gene_id_prot = self.gene_id_prot.reset_index(drop=True)

    def __len__(self):
        return len(self.y_proteome)

    def __getitem__(self, idx):
        gene_id = self.gene_id_prot[idx]
        return self.X_protein_seq_letters[idx][:self.prot_max_len], \
               torch.nn.functional.one_hot(self.X_protein_seq_letters[idx][:self.prot_max_len], 23), self.y_proteome[idx], \
               self.dna_seq_dict[gene_id], self.hlife_seq_dict[gene_id], \
               torch.nn.functional.one_hot(self.dna_seq_dict[gene_id], 5)[:, 1:]
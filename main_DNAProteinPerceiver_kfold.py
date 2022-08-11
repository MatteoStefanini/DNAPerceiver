import argparse
import numpy as np
import logging
import wandb
from dataset import *
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from scipy import stats
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from models import DNAProteinPerceiver
from utils import Lamb, CyclicCosineDecayLR
from sklearn.model_selection import KFold
import json

torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train DNA&Protein Perceiver for proteome')
_print_freq = 50
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

def train(model, dataloader, optim, loss_fn, scheduler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Training Epoch: [{}]'.format(e)
    for it, (prot_seq, prot_seq_bool, y, dna_seq, halflife, dna_seq_bool) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
        prot_seq_bool, prot_seq, y = prot_seq_bool.to(device), prot_seq.to(device), y.to(device)
        dna_seq, halflife, dna_seq_bool = dna_seq.to(device), halflife.to(device), dna_seq_bool.to(device)

        model_output = model(prot_seq, prot_seq_bool, dna_seq, halflife, dna_seq_bool)

        loss = loss_fn(model_output, y.float()) * args.loss_prot_mult

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler is not None and args.scheduler == 'LambdaLR' or args.scheduler == 'Cos':
            scheduler.step()

        metric_logger.update(loss=loss)

    metric_logger.synchronize_between_processes()

    return metric_logger.loss.global_avg


def evaluate_metrics(model, dataloader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    predictions_test_proteome = list()
    y_test_proteome = list()
    loss_list = list()
    header = 'Evaluation Epoch: [{}]'.format(e)
    with torch.no_grad():
        for it, (prot_seq, prot_seq_bool, y, dna_seq, halflife, dna_seq_bool) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            prot_seq_bool, prot_seq, y = prot_seq_bool.to(device), prot_seq.to(device), y.to(device)
            dna_seq, halflife, dna_seq_bool = dna_seq.to(device), halflife.to(device), dna_seq_bool.to(device)

            model_output = model(prot_seq, prot_seq_bool, dna_seq, halflife, dna_seq_bool)

            loss = loss_fn(model_output, y.float()) * args.loss_prot_mult

            predictions_test_proteome.extend(model_output.detach().cpu().numpy())
            y_test_proteome.extend(y.float().detach().cpu().numpy())

            metric_logger.update(loss=loss)
            loss_list.append(loss.detach().cpu().numpy())

    slope, intercept, r_value_proteome, p_value, std_err = stats.linregress(predictions_test_proteome, y_test_proteome)

    return r_value_proteome ** 2, np.mean(loss_list)


if __name__ == '__main__':
    _logger.info('perceiver_dna')
    parser = argparse.ArgumentParser(description='DNAProteinPerceiver')
    parser.add_argument('--exp_name', type=str, default='DNAProteinPerceiver')
    parser.add_argument('--exp_name_slurm', type=str, default='DNAProteinPerceiver')
    parser.add_argument('--datadir', type=str, default='data/pM10Kb_1KTest')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=8000)
    parser.add_argument('--leftpos', type=int, default=5000)
    parser.add_argument('--rightpos', type=int, default=13000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--optim', type=str, default='Adam', choices=('SGD', 'Adam', 'Lamb', 'AdamW'))
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--scheduler', type=str, default='LambdaLR', choices=('None', 'StepLR', 'LambdaLR','Cos'))
    parser.add_argument('--letter_emb_size', type=int, default=32) #32 64 96
    parser.add_argument('--queries_dim', type=int, default=64)
    parser.add_argument('--padding_idx', type=int, default=0)
    parser.add_argument('--max_len_prot', type=int, default=6000)
    parser.add_argument('--num_latents', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--conv_queries', action='store_true')
    parser.add_argument('--pos_emb', type=str, default='sin_learned', choices=('sin_fix', 'learned', 'sin_learned'))
    parser.add_argument('--activation', type=str, default='None', choices=('None', 'tanh'))
    parser.add_argument('--final_perceiver_head', action='store_true')
    parser.add_argument('--kmer3', action='store_true')
    parser.add_argument('--att_pool_size', type=int, default=10)
    parser.add_argument('--wandb_project', type=str, default='DNAProteinPerceiver')
    parser.add_argument('--norm_data', action='store_true')
    parser.add_argument('--all_data_train', action='store_true')
    parser.add_argument('--loss_prot_mult', type=float, default=1.)
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--only1fold', action='store_true')
    parser.add_argument('--proteomic_label', type=str, default='lung', choices=('lung', 'glio'))
    parser.add_argument('--input_perc', type=str, default='dna', choices=('dna', 'prot'))
    parser.add_argument('--input_conv', type=str, default='prot', choices=('dna', 'prot'))
    args = parser.parse_args()
    _logger.info(args)
    if args.debug:
        _print_freq = 1

    if args.kmer3:
        num_tokens = 23 ** 3
        kmer = 3
    else:
        if args.input_perc == 'prot':
            num_tokens = 23
        else:
            num_tokens = 5
        kmer = 1

    args.exp_name += '_' + args.proteomic_label + '_'
    args.exp_name += 'inP_' + args.input_perc + '_inC_' + args.input_conv + '_'

    if args.norm_data:
        args.exp_name += '_norm_'

    args.exp_name += str(args.letter_emb_size) + 'emb_' + str(args.depth) + 'enc_' +\
                         str(args.num_latents) + 'Lat_' + str(args.latent_dim) + 'd_' + str(args.optim) + '_wd' + \
                         str(args.weight_decay) + 'enfQ' + str(args.queries_dim) + \
                         str(args.att_pool_size) + 'p_'+ args.scheduler

    if args.scheduler == 'LambdaLR':
        args.exp_name += str(args.warmup)
    elif args.scheduler == 'Cos':
        args.exp_name += str(args.warmup) + '_lr' + str(args.lr)
    else:
        args.exp_name += '_lr' + str(args.lr)

    if args.activation == 'tanh':
        args.exp_name += '_' + str(args.activation)
    if args.m > 0:
        args.exp_name += '_m' + str(args.m)

    args.exp_name += '_maxLProt' + str(args.max_len_prot)
    args.exp_name += '_drp' + str(args.dropout)
    if args.kmer3:
        args.exp_name += '_kmer3'

    args.exp_name += '_L' + str(args.leftpos) + '_R' + str(args.rightpos)
    args.exp_name += '_prM' + str(int(args.loss_prot_mult))

    args.max_len_dna = args.rightpos - args.leftpos

    dataset = XpressoDNA_and_ProteinSequences_dataset("data/protein_sequences", proteomic_label=args.proteomic_label,
                                                      max_len_prot=args.max_len_prot)
    splits = KFold(n_splits=args.k_fold, shuffle=True, random_state=1234)

    for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))
        print('Train on:', train_idx)
        print('Test on:', test_idx)

    history = dict()

    for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        if args.only1fold and fold >= 1:
            exit()
        print('Fold {}'.format(fold + 1))
        history[str(fold + 1)] = dict()
        history[str(fold + 1)]['test_prot'] = list()

        if not args.resume_last and not args.resume_best:
            args.wandb_id = wandb.util.generate_id()
            run = wandb.init(entity='matteos', project=args.wandb_project, config=args, id=args.wandb_id,
                       mode='disabled' if args.debug else 'online', reinit=True)
        else:
            if args.resume_last:
                fname = 'saved_models/%s_last.pth' % args.exp_name
            else:
                fname = 'saved_models/%s_best.pth' % args.exp_name
            if os.path.exists(fname):
                data = torch.load(fname)
                if 'wandb_id' in data:
                    args.wandb_id = data['wandb_id']
                    run = wandb.init(entity='matteos', project=args.wandb_project, config=args, resume=args.wandb_id,
                               mode='disabled' if args.debug else 'online', reinit=True)
        wandb.run.name = 'Fold' + str(fold+1) + '_' + args.exp_name

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        dataloader_train = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers,
                                  pin_memory=True)
        dataloader_test = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler,num_workers=args.workers,
                                  pin_memory=True)

        #Model
        model = DNAProteinPerceiver(dim=args.letter_emb_size, num_tokens=num_tokens, max_len_prot=args.max_len_prot,
                                    depth=args.depth, num_latents=args.num_latents, latent_dim=args.latent_dim,
                                    pos_emb=args.pos_emb, max_len_dna=args.max_len_dna, activation=args.activation,
                                    dropout=args.dropout, queries_dim=args.queries_dim,
                                    final_perceiver_head=args.final_perceiver_head, att_pool_size=args.att_pool_size,
                                    input_perc=args.input_perc, input_conv=args.input_conv, leftpos=args.leftpos,
                                    rightpos=args.rightpos).to(device)

        # Optimizers for Transformers
        def lambda_lr(s):
            warm_up = args.warmup
            s += 1
            return (args.letter_emb_size ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

        if args.optim == 'Adam' and args.scheduler != 'LambdaLR':
            optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        elif args.optim == 'Adam' and args.scheduler == 'LambdaLR':
            optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        elif args.optim == 'AdamW' and args.scheduler == 'LambdaLR':
            optim = AdamW(model.parameters(), lr=1, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        elif args.optim == 'Lamb':
            optim = Lamb(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        else:
            optim = SGD(model.parameters(), lr=args.lr, momentum=0.9)

        if args.scheduler == 'StepLR':
            scheduler = StepLR(optim, step_size=30, gamma=0.5)
        elif args.scheduler == 'LambdaLR':
            scheduler = LambdaLR(optim, lambda_lr)
        elif args.scheduler == 'Cos':
            scheduler = CyclicCosineDecayLR(optim,5000,args.lr/10,7000,None,args.lr/2,4000,args.lr/2,verbose=False)
        else:
            scheduler = None

        loss_fn = torch.nn.MSELoss()
        best_r_quadro_test_proteome = .0
        best_test_mse = .0
        patience = 0
        start_epoch = 0

        for e in range(start_epoch, start_epoch + 400):
            train_mse_loss = train(model, dataloader_train, optim, loss_fn, scheduler)
            wandb.log({"Train loss MSE": train_mse_loss, "epoch": e})
            _logger.info("Train loss: %s", train_mse_loss)
            if scheduler is not None and args.scheduler == 'StepLR':
                scheduler.step()

            r_quadro_test_proteome, test_loss  = evaluate_metrics(model, dataloader_test)
            _logger.info("TEST: R2 prot %s Loss %s" %
                         (r_quadro_test_proteome, test_loss))
            wandb.log({"Test loss MSE": test_loss, "epoch": e})
            wandb.log({"Test R2 proteome": r_quadro_test_proteome, "epoch": e})

            history[str(fold + 1)]['test_prot'].append(r_quadro_test_proteome)

            # Prepare for next epoch
            best = False
            if r_quadro_test_proteome >= best_r_quadro_test_proteome:
                best_r_quadro_test_proteome = r_quadro_test_proteome
                wandb.run.summary["best_test_R2_proteome"] = best_r_quadro_test_proteome
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False
            if patience == 30:
                _logger.info('patience reached.')
                print('patience reached.')
                exit_train = True

            if exit_train:
                break

        run.finish()

    print('Computing average for each epoch...')
    args.wandb_id = wandb.util.generate_id()
    wandb.init(entity='matteos', project=args.wandb_project, config=args, id=args.wandb_id,
               mode='disabled' if args.debug else 'online', reinit=True)
    wandb.run.name = 'AvgKF_' + args.exp_name

    prot = list()
    prot_len = list()
    prot_best_fold = list()

    for fold in range(len(history)):
        prot_len.append(len(history[str(fold + 1)]['test_prot']))

    for fold in range(len(history)):
        prot.append(history[str(fold + 1)]['test_prot'][:min(prot_len)])
        prot_best_fold.append(max(history[str(fold + 1)]['test_prot']))

    # BEST OF KFOLD OVER FOLD BEST (mean of the max values of each fold)
    best_r2_test_proteome = sum(prot_best_fold) / len(prot_best_fold)
    print('BEST prot avg over folds: ', best_r2_test_proteome)
    wandb.run.summary["best_avgR2_test_proteome"] = best_r2_test_proteome
    history["best_avgR2_test_proteome"] = best_r2_test_proteome

    with open(os.path.join('kfold_history_proteinSeq', args.exp_name + '.pkl'), 'wb') as fp:
        pickle.dump(history, fp)
    with open(os.path.join('kfold_history_proteinSeq', args.exp_name + '.json'), 'w') as fp:
        json.dump(history, fp)

    wandb.log({"Test R2 proteome": best_r2_test_proteome, "epoch": 0})

    prot = np.array(prot, dtype=object)
    prot_avg_epochs = np.average(prot, axis=0)

    for e in range(0, len(prot_avg_epochs)):
        wandb.log({"Test R2 proteome": prot_avg_epochs[e], "epoch": e + 1})

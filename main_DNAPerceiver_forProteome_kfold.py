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
from models import DNAPerceiver
from utils import Lamb, CyclicCosineDecayLR
from sklearn.model_selection import KFold
import json

torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train DNA Perceiver for proteome')
_print_freq = 50
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

def train(model, dataloader, optim, loss_fn, scheduler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Training Epoch: [{}]'.format(e)
    for it, (halflife, promoter_seq_bool, promoter_seq_dna, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
        halflife, promoter_seq_bool, promoter_seq_dna, y = halflife.to(device), promoter_seq_bool.to(device), \
                                                           promoter_seq_dna.to(device), y.to(device)

        model_output = model(halflife, promoter_seq_bool, promoter_seq_dna)

        if args.only_mRNA:
            loss = loss_fn(model_output[:, 0], y[:, 0].float()) * args.loss_mRNA_mult
        else:
            loss_mRNA = loss_fn(model_output[:, 0], y[:, 0].float()) * args.loss_mRNA_mult
            loss_prot = loss_fn(model_output[:, 1], y[:, 1].float()) * args.loss_prot_mult
            loss = loss_mRNA + loss_prot

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler is not None and args.scheduler == 'LambdaLR' or args.scheduler == 'Cos':
            scheduler.step()

        if args.only_mRNA:
            metric_logger.update(loss=loss)
        else:
            metric_logger.update(loss=loss, loss_mRNA=loss_mRNA, loss_prot=loss_prot)

    metric_logger.synchronize_between_processes()
    if args.only_mRNA:
        return metric_logger.loss.global_avg, 0., 0.
    else:
        return metric_logger.loss.global_avg, metric_logger.loss_mRNA.global_avg, metric_logger.loss_prot.global_avg


def evaluate_metrics(model, dataloader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    predictions_test_proteome = list()
    y_test_proteome = list()
    predictions_test_mRNA = list()
    y_test_mRNA = list()
    loss_list = list()
    loss_mRNA_list = list()
    loss_prot_list = list()

    header = 'Evaluation Epoch: [{}]'.format(e)
    with torch.no_grad():
        for it, (halflife, promoter_seq_bool, promoter_seq_dna, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            halflife, promoter_seq_bool, promoter_seq_dna, y = halflife.to(device), promoter_seq_bool.to(device), \
                                                               promoter_seq_dna.to(device), y.to(device)

            model_output = model(halflife, promoter_seq_bool, promoter_seq_dna)
            if args.only_mRNA:
                loss = loss_fn(model_output[:, 0], y[:, 0].float()) * args.loss_mRNA_mult
            else:
                loss_mRNA = loss_fn(model_output[:, 0], y[:, 0].float()) * args.loss_mRNA_mult
                loss_prot = loss_fn(model_output[:, 1], y[:, 1].float()) * args.loss_prot_mult
                loss = loss_mRNA + loss_prot

            predictions_test_mRNA.extend(model_output[:, 0].detach().cpu().numpy())
            y_test_mRNA.extend(y[:, 0].float().detach().cpu().numpy())

            if not args.only_mRNA:
                predictions_test_proteome.extend(model_output[:, 1].detach().cpu().numpy())
                y_test_proteome.extend(y[:, 1].float().detach().cpu().numpy())

            if args.only_mRNA:
                metric_logger.update(loss=loss)
                loss_list.append(loss.detach().cpu().numpy())
            else:
                metric_logger.update(loss=loss, loss_mRNA=loss_mRNA, loss_prot=loss_prot)
                loss_list.append(loss.detach().cpu().numpy())
                loss_mRNA_list.append(loss_mRNA.detach().cpu().numpy())
                loss_prot_list.append(loss_prot.detach().cpu().numpy())

    if args.only_mRNA:
        slope, intercept, r_value_mRNA, p_value, std_err = stats.linregress(predictions_test_mRNA, y_test_mRNA)
        return r_value_mRNA ** 2, 0., np.mean(loss_list), 0., 0.
    else:
        slope, intercept, r_value_mRNA, p_value, std_err = stats.linregress(predictions_test_mRNA, y_test_mRNA)
        slope, intercept, r_value_proteome, p_value, std_err = stats.linregress(predictions_test_proteome, y_test_proteome)
        return r_value_mRNA ** 2, r_value_proteome ** 2, np.mean(loss_list), np.mean(loss_mRNA_list), np.mean(loss_prot_list)


if __name__ == '__main__':
    _logger.info('perceiver_dna')
    parser = argparse.ArgumentParser(description='DNAPerceiver_proteome')
    parser.add_argument('--exp_name', type=str, default='DNAPerceiver_proteome')
    parser.add_argument('--exp_name_slurm', type=str, default='DNAPerceiver_proteome')
    parser.add_argument('--datadir', type=str, default='data/pM10Kb_1KTest')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--leftpos', type=int, default=3000)
    parser.add_argument('--rightpos', type=int, default=13500)
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=8000)
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
    parser.add_argument('--max_len', type=int, default=20000)
    parser.add_argument('--num_latents', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--conv_queries', action='store_true')
    parser.add_argument('--pos_emb', type=str, default='sin_learned', choices=('sin_fix', 'learned', 'sin_learned'))
    parser.add_argument('--activation', type=str, default='None', choices=('None', 'tanh'))
    parser.add_argument('--final_perceiver_head', action='store_true')
    parser.add_argument('--kmer3', action='store_true')
    parser.add_argument('--att_pool_size', type=int, default=10)
    parser.add_argument('--wandb_project', type=str, default='DNAPerceiver_proteome')
    parser.add_argument('--only_mRNA', action='store_true')
    parser.add_argument('--norm_data', action='store_true')
    parser.add_argument('--all_data_train', action='store_true')
    parser.add_argument('--loss_mRNA_mult', type=float, default=1.)
    parser.add_argument('--loss_prot_mult', type=float, default=1.)
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--only1fold', action='store_true')
    parser.add_argument('--proteome_label', type=str, default='Lung', choices=('Lung', 'Glio'))
    args = parser.parse_args()
    _logger.info(args)
    if args.debug:
        _print_freq = 1

    if args.kmer3:
        num_tokens = 65
        kmer = 3
    else:
        num_tokens = 5
        kmer = 1

    args.exp_name += '_' + args.proteome_label
    if args.norm_data:
        args.exp_name += '_norm_'
    if args.only_mRNA:
        args.exp_name += '_onlymRNA'

    args.exp_name += str(args.letter_emb_size) + 'emb_' + str(args.depth) + 'enc_' + 'sinL_' +\
                     str(args.num_latents) + 'Lat_' + str(args.latent_dim) + 'd_' + str(args.optim) + '_wd' + \
                     str(args.weight_decay) + 'enf_' + str(args.queries_dim) + 'Q_' + \
                     str(args.att_pool_size) + 'pool_'+ args.scheduler

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

    args.exp_name += '_L' + str(args.leftpos) + '_R' + str(args.rightpos)
    args.exp_name += '_drp' + str(args.dropout)
    if args.kmer3:
        args.exp_name += '_kmer3'

    args.exp_name += '_mr' + str(int(args.loss_mRNA_mult)) + '_pr' + str(int(args.loss_prot_mult))

    args.max_len = args.rightpos - args.leftpos

    if args.proteome_label == 'Lung':
        dataset = XpressoDNA_Lung_ProteomeLabel(args.datadir, kmer=kmer, norm_data=args.norm_data)
    elif args.proteome_label == 'Glio':
        dataset = XpressoDNA_Glio_ProteomeLabel(args.datadir, kmer=kmer, norm_data=args.norm_data)

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
        history[str(fold + 1)]['test_mRNA'] = list()
        history[str(fold + 1)]['test_prot'] = list()

        args.wandb_id = wandb.util.generate_id()
        run = wandb.init(entity='matteos', project=args.wandb_project, config=args, id=args.wandb_id,
                   mode='disabled' if args.debug else 'online', reinit=True)

        wandb.run.name = 'Fold_' + str(fold+1) + '_' + args.exp_name

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        dataloader_train = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers,
                                  pin_memory=True)
        dataloader_test = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler,num_workers=args.workers,
                                  pin_memory=True)

        #Model
        model = DNAPerceiver(dim=args.letter_emb_size, num_tokens=num_tokens, max_seq_len=args.max_len, depth=args.depth,
                             num_latents=args.num_latents, latent_dim=args.latent_dim, pos_emb=args.pos_emb,
                             activation=args.activation, leftpos=args.leftpos, rightpos=args.rightpos, dropout=args.dropout,
                             queries_dim=args.queries_dim, final_perceiver_head=args.final_perceiver_head,
                             att_pool_size=args.att_pool_size, predict_proteome=True).to(device)

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
        best_r_quadro_test_sum = .0
        best_r_quadro_test_proteome = .0
        best_r_quadro_test_mRNA = .0
        best_test_mse = .0
        patience = 0
        start_epoch = 0

        for e in range(start_epoch, start_epoch + 400):
            train_mse_loss, train_loss_mRNA, train_loss_prot = train(model, dataloader_train, optim, loss_fn, scheduler)
            wandb.log({"Train loss MSE": train_mse_loss, "epoch": e})
            _logger.info("Train loss: %s", train_mse_loss)
            wandb.log({"Train loss mRNA": train_loss_mRNA, "epoch": e})
            _logger.info("Train loss mRNA: %s", train_loss_mRNA)
            wandb.log({"Train loss prot": train_loss_prot, "epoch": e})
            _logger.info("Train loss prot: %s", train_loss_prot)
            if scheduler is not None and args.scheduler == 'StepLR':
                scheduler.step()

            r_quadro_test_mRNA, r_quadro_test_proteome, test_loss, test_loss_mRNA, test_loss_prot  = evaluate_metrics(model, dataloader_test)
            _logger.info("TEST: R2 prot %s R2 mRNA %s Loss %s Loss mRNA %s Loss prot %s" %
                         (r_quadro_test_proteome, r_quadro_test_mRNA, test_loss, test_loss_mRNA, test_loss_prot))
            wandb.log({"Test loss MSE": test_loss, "epoch": e})
            wandb.log({"Test loss mRNA": test_loss_mRNA, "epoch": e})
            wandb.log({"Test loss prot": test_loss_prot, "epoch": e})
            wandb.log({"Test R2 proteome": r_quadro_test_proteome, "epoch": e})
            wandb.log({"Test R2 mRNA": r_quadro_test_mRNA, "epoch": e})

            history[str(fold + 1)]['test_mRNA'].append(r_quadro_test_mRNA)
            history[str(fold + 1)]['test_prot'].append(r_quadro_test_proteome)

            r_quadro_test_sum = r_quadro_test_proteome + r_quadro_test_mRNA

            # Prepare for next epoch
            best = False
            if r_quadro_test_sum >= best_r_quadro_test_sum:
                best_r_quadro_test_sum = r_quadro_test_sum
                best_r_quadro_test_proteome = r_quadro_test_proteome
                best_r_quadro_test_mRNA = r_quadro_test_mRNA
                wandb.run.summary["best_test_R2_sum"] = best_r_quadro_test_sum
                wandb.run.summary["best_test_R2_proteome"] = best_r_quadro_test_proteome
                wandb.run.summary["best_test_R2_mRNA"] = best_r_quadro_test_mRNA
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False
            if patience == 40:
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

    mRNA = list()
    prot = list()
    mRNA_len = list()
    prot_len = list()
    mRNA_best_fold = list()
    prot_best_fold = list()

    for fold in range(len(history)):
        mRNA_len.append(len(history[str(fold+1)]['test_mRNA']))
        prot_len.append(len(history[str(fold+1)]['test_prot']))

    for fold in range(len(history)):
        mRNA.append(history[str(fold+1)]['test_mRNA'][:min(mRNA_len)])
        prot.append(history[str(fold+1)]['test_prot'][:min(prot_len)])
        mRNA_best_fold.append(max(history[str(fold + 1)]['test_mRNA']))
        prot_best_fold.append(max(history[str(fold + 1)]['test_prot']))

    # BEST OF KFOLD OVER FOLD BEST (mean of the max values of each fold)
    best_r2_test_mRNA = sum(mRNA_best_fold)/len(mRNA_best_fold)
    best_r2_test_proteome = sum(prot_best_fold)/len(prot_best_fold)
    print('BEST mRNA avg over folds: ', best_r2_test_mRNA)
    print('BEST prot avg over folds: ', best_r2_test_proteome)
    wandb.run.summary["best_avgR2_test_mRNA"] = best_r2_test_mRNA
    wandb.run.summary["best_avgR2_test_proteome"] = best_r2_test_proteome
    history["best_avgR2_test_mRNA"] = best_r2_test_mRNA
    history["best_avgR2_test_proteome"] = best_r2_test_proteome

    with open(os.path.join('kfold_history_glio', args.exp_name + '.pkl'), 'wb') as fp:
        pickle.dump(history, fp)
    with open(os.path.join('kfold_history_glio', args.exp_name + '.json'), 'w') as fp:
        json.dump(history, fp)

    wandb.log({"Test R2 proteome": best_r2_test_proteome, "epoch": 0})
    wandb.log({"Test R2 mRNA": best_r2_test_mRNA, "epoch": 0})

    mRNA = np.array(mRNA, dtype=object)
    prot = np.array(prot, dtype=object)
    mRNA_avg_epochs = np.average(mRNA, axis=0)
    prot_avg_epochs = np.average(prot, axis=0)

    for e in range(0, len(mRNA_avg_epochs)):
        wandb.log({"Test R2 proteome": prot_avg_epochs[e], "epoch": e+1})
        wandb.log({"Test R2 mRNA": mRNA_avg_epochs[e], "epoch": e+1})

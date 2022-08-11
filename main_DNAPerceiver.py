import argparse
import numpy as np
import logging
import wandb
from dataset import *
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from scipy import stats
from torch.utils.data import DataLoader
import utils
from models import DNAPerceiver
from shutil import copyfile
from utils import Lamb, CyclicCosineDecayLR

torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train DNA Perceiver')
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
        loss = loss_fn(model_output, y.float())

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

    predictions_test = list()
    y_test = list()
    loss_mse = list()

    header = 'Evaluation Epoch: [{}]'.format(e)
    with torch.no_grad():
        for it, (halflife, promoter_seq_bool, promoter_seq_dna, y) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
            halflife, promoter_seq_bool, promoter_seq_dna, y = halflife.to(device), promoter_seq_bool.to(device), \
                                                               promoter_seq_dna.to(device), y.to(device)

            model_output = model(halflife, promoter_seq_bool, promoter_seq_dna)
            loss = loss_fn(model_output, y.float())

            predictions_test.extend(model_output.detach().cpu().numpy())
            y_test.extend(y.float().detach().cpu().numpy())
            metric_logger.update(loss=loss)
            loss_mse.append(loss.detach().cpu().numpy())

    slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_test, y_test)

    return r_value ** 2, np.mean(loss_mse)


if __name__ == '__main__':
    _logger.info('perceiver_dna')
    parser = argparse.ArgumentParser(description='DNAPerceiver')
    parser.add_argument('--exp_name', type=str, default='DNAPerceiver')
    parser.add_argument('--exp_name_slurm', type=str, default='DNAPerceiver')
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
    parser.add_argument('--letter_emb_size', type=int, default=32)
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
    parser.add_argument('--att_pool_size', type=int, default=12)
    parser.add_argument('--wandb_project', type=str, default='DNAPerceiver')
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

    if args.datadir.split('_')[-1] == 'Mouse':
        args.exp_name = 'Mice_' + args.exp_name
        args.wandb_project = 'mouse_DNAPerceiver'

    dataset_train = XpressoTrain(args.datadir, kmer=kmer)
    dataset_val = XpressoVal(args.datadir, kmer=kmer)
    dataset_test = XpressoTest(args.datadir, kmer=kmer)

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

    if not args.resume_last and not args.resume_best:
        args.wandb_id = wandb.util.generate_id()
        wandb.init(entity='matteos', project=args.wandb_project, config=args, id=args.wandb_id,
                   mode='disabled' if args.debug else 'online')
    else:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name
        if os.path.exists(fname):
            data = torch.load(fname)
            if 'wandb_id' in data:
                args.wandb_id = data['wandb_id']
                wandb.init(entity='matteos', project=args.wandb_project, config=args, resume=args.wandb_id,
                   mode='disabled' if args.debug else 'online')
    wandb.run.name = args.exp_name

    args.max_len = args.rightpos - args.leftpos

    #Model
    model = DNAPerceiver(dim=args.letter_emb_size, num_tokens=num_tokens, max_seq_len=args.max_len, depth=args.depth,
                         num_latents=args.num_latents, latent_dim=args.latent_dim, pos_emb=args.pos_emb,
                         activation=args.activation, leftpos=args.leftpos,rightpos=args.rightpos, dropout=args.dropout,
                         queries_dim=args.queries_dim, final_perceiver_head=args.final_perceiver_head,
                         att_pool_size=args.att_pool_size).to(device)

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
        optim = Lamb(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
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
    best_r_quadro_val = .0
    best_r_quadro_test = .0
    best_test_mse = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name
        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            model.load_state_dict(data['state_dict'])
            optim.load_state_dict(data['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_r_quadro_val = data['best_r_quadro_val']
            best_r_quadro_test = data['best_r_quadro_test']
            patience = data['patience']
            _logger.info('Resuming from epoch %d, validation R2 %f, test R2 %f' % (
                data['epoch'], data['best_r_quadro_val'], data['best_r_quadro_test']))

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.workers,
                                  sampler=train_sampler, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=val_sampler, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, pin_memory=True)

    for e in range(start_epoch, start_epoch + 550):
        train_mse_loss = train(model, dataloader_train, optim, loss_fn, scheduler)
        wandb.log({"Train loss MSE": train_mse_loss, "epoch": e})
        _logger.info("Train loss MSE: %s", train_mse_loss)
        print("Train loss MSE: %s", train_mse_loss)
        if scheduler is not None and args.scheduler == 'StepLR':
            scheduler.step()

        r_quadro_val, val_mse = evaluate_metrics(model, dataloader_val)
        _logger.info("Validation R2: %s - Val Loss MSE: %s" % (r_quadro_val, val_mse))
        print("Validation R2: %s - Val Loss MSE: %s" % (r_quadro_val, val_mse))
        wandb.log({"Val loss MSE": val_mse, "epoch": e})
        wandb.log({"Val R2": r_quadro_val, "epoch": e})

        r_quadro_test, test_mse = evaluate_metrics(model, dataloader_test)
        _logger.info("Test R2: %s - Test Loss MSE: %s" % (r_quadro_test, test_mse))
        print("Test R2: %s - Test Loss MSE: %s" % (r_quadro_test, test_mse))
        wandb.log({"Test loss MSE": test_mse, "epoch": e})
        wandb.log({"Test R2": r_quadro_test, "epoch": e})

        # Prepare for next epoch
        best = False
        if r_quadro_val >= best_r_quadro_val:
            best_r_quadro_val = r_quadro_val
            wandb.run.summary["Best Val R2"] = best_r_quadro_val
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if r_quadro_test >= best_r_quadro_test:
            best_r_quadro_test = r_quadro_test
            wandb.run.summary["best_test_R2"] = best_r_quadro_test
            best_test = True

        exit_train = False
        if patience == 150:
            _logger.info('patience reached.')
            print('patience reached.')
            exit_train = True

        save_dict = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'epoch': e,
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'patience': patience,
            'best_r_quadro_val': best_r_quadro_val,
            'best_r_quadro_test': best_r_quadro_test,
            'wandb_id': args.wandb_id,
            'train_loss_mse': train_mse_loss,
            'val_loss_mse': val_mse,
            'test_loss_mse': test_mse,
            'state_dict': model.state_dict()
        }

        if not args.debug:
            torch.save(save_dict, 'saved_models/%s_last.pth' % args.exp_name)
            if best:
                copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)
            if best_test:
                copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best_test.pth' % args.exp_name)

        if exit_train:
            break
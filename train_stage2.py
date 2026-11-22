import math
import random
import sys
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader as torch_DataLoader
from tqdm import tqdm

from parser import get_parser
from DOS_Reasoner.data import CIF_Dataset, CIF_Lister, load_benchmark, mkdirs
from DOS_Reasoner.stage1 import Structure_Aware_Spectrum_Decoding
from DOS_Reasoner.stage2 import CPPR
from DOS_Reasoner.utils import metrics

best_val_mse=1e+10

def cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train_model(args):
    ckpt_dir_stage1 = args.stage1_ckpt
    ckpt_dir_stage2 = args.save_dir
    
    log_dir = './TRAINED/stage2/' + ckpt_dir_stage2
    mkdirs(log_dir)
    log_file = open(log_dir + '/log.txt', 'w')

    def print_log(msg):
        print(msg)
        print(msg, file=log_file)

    random_num = args.seed
    random.seed(random_num)
    np.random.seed(random_num)
    torch.manual_seed(random_num)
    torch.cuda.manual_seed(random_num)
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')

    # MODEL HYPER-PARAMETERS
    num_epochs      = args.num_epochs
    batch_size      = args.batch_size

    # DATALOADER

    pd_data, train_idx, val_idx, test_idx = load_benchmark(args.data_src)

    CRYSTAL_DATA = CIF_Dataset(args, pd_data=pd_data)
    
    if isinstance(train_idx, np.ndarray):
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()
        test_idx = test_idx.tolist()

    if args.scaling_law < 1.0:
        sample_size = int(len(train_idx) * args.scaling_law)
        train_idx = random.sample(train_idx, sample_size)


    print_log(f'training size: {len(train_idx)}, min/max: {min(train_idx)} {max(train_idx)}')
    print_log(f'validation size: {len(val_idx)}, min/max: {min(val_idx)} {max(val_idx)}')
    print_log(f'testing size: {len(test_idx)}, min/max:, {min(test_idx)}, {max(test_idx)}')
    print_log(f'total size: {len(train_idx)}, {len(val_idx)+len(test_idx)}')

    training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,df=pd_data)
    validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,df=pd_data)
    testing_set        =  CIF_Lister(test_idx, CRYSTAL_DATA, df=pd_data)

    net_base = Structure_Aware_Spectrum_Decoding(args).to(device)
    net = CPPR(args).to(device)

    check_point_path = './TRAINED/stage1/' + ckpt_dir_stage1 + '/model.ckpt'
    check_point_stage1 = torch.load(check_point_path, weights_only=False)
    state_dict = check_point_stage1['model']  
    net_base.load_state_dict(state_dict, strict=False)


    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_log(f"model_size: {count_params(net)}")

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.CPPR_lr,
                                  weight_decay=args.weight_decay)
    num_training_steps = args.num_epochs * (len(training_set) // args.batch_size + 1)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    print_log(f'> TRAINING MODEL ...')
    train_loader   = torch_DataLoader(dataset=training_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True
                                     )
    valid_loader   = torch_DataLoader(dataset=validation_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True
                                     )
    test_loader    = torch_DataLoader(dataset=testing_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True
                                     )


    current_step = 0

    summary_dir = './summary/stage2/' + ckpt_dir_stage2
    mkdirs(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)
    
    log_file.flush()

    start_time = time.time()
    for epoch in range(num_epochs):
        # TRAINING-STAGE
        net.train()
        net_base.eval()
        args.train = True
        n_accum = 0

        for data in tqdm(train_loader, mininterval=0.5, desc=f'(EPOCH:{epoch} TRAINING)', position=0, leave=True, ascii=True):
            n_accum += 1
            current_step += 1
            data = data.to(device, non_blocking=True)
            if isinstance(data.y, tuple) or isinstance(data.y, list):
                data.y = data.y[0]
            train_label = data.y.to(device)
            
            train_label_normalized = train_label / (torch.sum(train_label, dim=1, keepdim=True) + 1e-10)

            with torch.no_grad():
                crystal_ctx, _, _ = net_base.graph_encoder(data)
                p0 = net_base(data)

            p0 = p0 / torch.sum(p0, dim=1, keepdim=True)
            loss = net(p0, train_label_normalized, crystal_ctx)
            loss.backward()
            if n_accum % args.accum_step == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
 

        def valid_test(data_loader, net, mode="Valid"):
            global best_val_mse

            # VALIDATION-PHASE
            valid_counter = 0
            prediction = []
            label_gt = []

            net.eval()

            for data in tqdm(data_loader, mininterval=0.5, desc=f'({mode})', position=0, leave=True, ascii=True):
                data = data.to(device, non_blocking=True)
                if isinstance(data.y, tuple) or isinstance(data.y, list):
                    data.y = data.y[0]
                valid_label = data.y.float().to(device)

                with torch.no_grad():
                    crystal_ctx, _, _ = net_base.graph_encoder(data)
                    p0 = net_base(data)
                    p0 = p0 / torch.sum(p0, dim=1, keepdim=True)
                    pT_pred = net.inference(p0.to(device), crystal_ctx)

                prediction.append(pT_pred.detach().cpu().numpy())
                label_gt.append(valid_label.detach().cpu().numpy())
                valid_counter += 1

            prediction = np.concatenate(prediction, axis=0)
            label_gt = np.concatenate(label_gt, axis=0)

            mae, r2, mse, wd = metrics(label_gt, prediction)
            
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f"{mode}/lr", lr, epoch)
            writer.add_scalar(f"{mode}/r2", r2, epoch)
            writer.add_scalar(f"{mode}/mae", mae, epoch)
            writer.add_scalar(f"{mode}/mse", mse, epoch)
            writer.add_scalar(f"{mode}/wd", wd, epoch)

            print_log(f"\n********** {epoch} {mode} STATISTIC ***********")
            print_log("lr = %.4f" % lr)
            print_log("\n")


            if mode == "Valid" and best_val_mse > mse:
                best_val_mse = mse
                is_best = True
                print_log("\n********** SAVING MODEL ***********")
                checkpoint = {'model': net.state_dict(), 'args': args}
                save_path = './TRAINED/stage2/' + ckpt_dir_stage2
                mkdirs(save_path)
                save_path = save_path + '/model.ckpt'
                torch.save(checkpoint, save_path)
                print_log("A new model has been saved to " + save_path)
                print_log("\n*****************************************")

            valid_counter = 0
            prediction = []
            label_gt = []

        valid_test(valid_loader, net, mode="Valid")
        log_file.flush()

    end_time = time.time()
    e_time = end_time - start_time
    print_log(f"> DONE TRAINING !")

    log_file.close()

def main():
    parser = get_parser()
    parser.add_argument("--stage1_ckpt", required=True,
                        help="Stage-1 checkpoint dir name under TRAINED/stage1/")
    parser.add_argument("--save_dir", required=True,
                        help="Save best-val Stage-2 checkpoint under TRAINED/stage2/<save_dir>/")
    parser.add_argument("--warmup_ratio", default=0.3, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--k_steps", default=5, type=int,
                        help="Number of CPPR transport steps (default 5)")
    parser.add_argument("--cfl_courant", default=0.5, type=float,
                        help="Max Courant number for the upwind velocity clamp "
                             "(<=0.5 guarantees non-negative densities; "
                             "<=1.0 is linear-stability only)")
    args = parser.parse_args(sys.argv[1:])
    train_model(args)

if __name__ == '__main__':
    main()

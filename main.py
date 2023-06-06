import argparse
import os

from train_singlebranch import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="finetune", help="train, test, finetune")
parser.add_argument("--noisetype", type=str, default='None')
parser.add_argument("--sim", type=bool, default=True)  # Whether to use simulated data

parser.add_argument('--data_dir', type=str, default='./train/train_data/cat_face/diff_cat100_0&25&50&75.npy')
parser.add_argument('--target_path', type=str, default='./train/label/cat_face')
parser.add_argument('--val_dir', type=str, default='./test/cat_data/1diff_cat100_75overlap.npy')
parser.add_argument('--val_target_dir', type=str, default='./test/cat_groundtruth')

parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='ptycho')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
opt, _ = parser.parse_known_args()

if opt.model == 'train':
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)

if opt.model == 'test':
    parser.add_argument("--pre_trained_model", type=str,
                        default=os.path.join(opt.save_model_path, opt.log_name, 'model',
                                             'best_model.pth'))  # Select the model of test
    if opt.sim:
        parser.add_argument('--num_diff', type=tuple, default=(16, 16))  # 16, 9, 6, 5(Corresponding to different overlap rates)
        parser.add_argument('--stepsize', type=int, default=32)  # 32, 64, 96, 128(Corresponding to 75%,50%,25%,0% overlap rate)
        parser.add_argument('--probe_size', type=tuple, default=(128, 128))  # Pixel size of the spot
        # Scan area size (608, 608), (640, 640), (608, 608), (640, 640) corresponding to 75%, 50%, 25%, 0% overlap
        parser.add_argument('--scan_area', type=tuple, default=(608, 608))
    else:
        parser.add_argument('--probe_size', type=tuple, default=(128, 128))
        parser.add_argument('--scan_area', type=tuple, default=(1606, 2355))

if opt.model == 'finetune':
    parser.add_argument("--pre_trained_model", type=str,
                        default=os.path.join(opt.save_model_path, opt.log_name, 'model',
                                             'best_model.pth'))  #
    parser.add_argument('--save_finetunemodel_path', type=str,
                        default=os.path.join(opt.save_model_path, opt.log_name, 'model',
                                             'finetune_model.pth'))   # Select finetune's model
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.5)  # Learning rate decay rate at finetune

    if opt.sim:
        parser.add_argument('--num_diff', type=tuple, default=(16, 16))
        parser.add_argument('--stepsize', type=int, default=32)
        parser.add_argument('--probe_size', type=tuple, default=(128, 128))
        parser.add_argument('--scan_area', type=tuple, default=(608, 608))
    else:
        parser.add_argument('--probe_size', type=tuple, default=(128, 128))
        parser.add_argument('--scan_area', type=tuple, default=(1606, 2355))

opt, _ = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices

if __name__ == "__main__":
    trainer = Trainer(opt)
    if opt.model == 'train':
        print("start train")
        trainer.train()
    elif opt.model == 'test':
        print("start test")
        trainer.validation()
    elif opt.model == 'finetune':
        print("start finetune")
        trainer.finetune()

import argparse
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from cdc.model import HourGlassNetMultiScaleInt
from torchtools.datasets import SRDataListLR
from torchtools.functional import tensor_merge, tensor_divide, to_pil_image
from torchtools.torchnet import load_weights


def main():
    pretrained_ckpt = hf_hub_download(
        repo_id='narugo/cdc_pretrianed_model',
        filename='HGSR-MHR-anime_X4_280.pth',
    )

    parser = argparse.ArgumentParser()
    ## Test Dataset
    parser.add_argument('--dataroot', type=str, default='images')

    # Test Options
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap pixel when Divide input image, for edge effect')
    parser.add_argument('--psize', type=int, default=512, help='Overlap pixel when Divide input image, for edge effect')
    parser.add_argument('--cat_result', type=bool, default=False, help='Concat result to one image')
    parser.add_argument('--rgb_range', type=float, default=1., help='255 EDSR and RCAN, 1 for the rest')
    parser.add_argument('--save_results', type=bool, default=True, help='Concat result to one image')

    # Model Options
    parser.add_argument('--model', type=str, default='HGSR-MHR', help='')
    parser.add_argument('--inc', type=int, default=3, help='the low resolution image size')
    parser.add_argument('--scala', type=int, default=4, help='the low resolution image size')
    parser.add_argument('--n_HG', type=int, default=6, help='the low resolution image size')
    parser.add_argument('--inter_supervis', type=bool, default=True, help='the low resolution image size')

    # Logger
    parser.add_argument('--result_dir', type=str, default='output', help='folder to sr results')
    parser.add_argument('--gpus', type=int, default=1, help='folder to sr results')
    opt = parser.parse_args()

    use_cuda = False
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    ## Here is the original main function
    # Make Result Folder
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)

    # Save Test Info
    log_f = open(os.path.join(opt.result_dir, 'test_log.txt'), 'a')
    log_f.write('test_dataroot: ' + opt.dataroot + '\n')
    log_f.write('test_patch_size: ' + str(opt.psize) + '\n')
    log_f.write('test_overlap: ' + str(opt.overlap) + '\n')
    log_f.write('test_model: ' + str(pretrained_ckpt) + '\n')

    # Init Dataset
    test_dataset = SRDataListLR(opt.dataroot, scala=opt.scala, mode='RGB', rgb_range=opt.rgb_range)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())

    # Init Net
    print('Build Generator Net...')

    generator = HourGlassNetMultiScaleInt(
        in_nc=opt.inc, out_nc=opt.inc, upscale=opt.scala,
        nf=64, res_type='res', n_mid=2, n_HG=opt.n_HG, inter_supervis=opt.inter_supervis
    )

    generator = load_weights(generator, pretrained_ckpt, opt.gpus, just_weight=False, strict=True)
    generator = generator.to(device)
    generator.eval()

    for batch, data in enumerate(test_loader):
        lr = data['LR']
        im_path = data['PATH'][0]

        with torch.no_grad():
            tensor = lr
            B, C, H, W = lr.shape
            blocks = tensor_divide(tensor, opt.psize, opt.overlap)
            blocks = torch.cat(blocks, dim=0)
            results = []

            iters = blocks.shape[0] // opt.gpus if blocks.shape[0] % opt.gpus == 0 else blocks.shape[0] // opt.gpus + 1
            for idx in range(iters):
                if idx + 1 == iters:
                    input = blocks[idx * opt.gpus:]
                else:
                    input = blocks[idx * opt.gpus: (idx + 1) * opt.gpus]
                hr_var = input.to(device)
                sr_var, SR_map = generator(hr_var)

                if isinstance(sr_var, list) or isinstance(sr_var, tuple):
                    sr_var = sr_var[-1]

                results.append(sr_var.to('cpu'))
                print('Processing Image: %d Part: %d / %d'
                      % (batch + 1, idx + 1, iters), end='\r')
                sys.stdout.flush()

            results = torch.cat(results, dim=0)
            sr_img = tensor_merge(results, None, opt.psize * opt.scala, opt.overlap * opt.scala,
                                  tensor_shape=(B, C, H * opt.scala, W * opt.scala))

        im_name = '%s_x%d.png' % (os.path.basename(im_path).split('.')[0].replace('/', '_'), opt.scala)
        if opt.save_results:
            im = to_pil_image(torch.clamp(sr_img[0].cpu() / opt.rgb_range, min=0.0, max=1.0))
            im.save(os.path.join(opt.result_dir, im_name))
            print('[%d/%d] saving to: %s' % (batch + 1, len(test_loader), os.path.join(opt.result_dir, im_name)))

        sys.stdout.flush()


if __name__ == '__main__':
    main()

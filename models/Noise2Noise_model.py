from collections import OrderedDict
import torch.utils.data
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from networks import get_network
from networks.weights_init import *
from models.utils import AverageMeter, get_scheduler, psnr, complex_abs_eval

class RecurrentModel(nn.Module):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = opts.n_recurrent
        self.gpu_ids = opts.gpu_ids
        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G_I = get_network(opts)
        self.networks.append(self.net_G_I)


        if self.is_train:
            self.loss_names += ['loss_G_L1']
            param = list(self.net_G_I.parameters())
            self.optimizer_G = torch.optim.Adam(param,
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)


        self.criterion = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.AdverLoss = nn.BCEWithLogitsLoss()
        self.opts = opts

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        #[b,2,w,h]
        self.image_full = data['image_full'].to(self.device)
        self.image_sub_1 = data['image_sub_1'].to(self.device)
        self.image_sub_2 = data['image_sub_2'].to(self.device)
        self.image_avg = data['image_sub_avg'].to(self.device)
        self.m_zeros = torch.zeros_like(data['image_sub_1']).to(self.device)
        self.m_ones = torch.ones_like(data['image_sub_1']).to(self.device)

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch
        self.loss_kspc = AverageMeter()

    def forward(self):
        ############# Random k-space sampler #############
        torch.seed()
        rand_th = torch.rand(1)  # aug
        rand_th[0] = 0.5

        rand_mask = torch.rand_like(self.image_sub_1[:, 0:6, :, 0:1])
        rand_mask = torch.where(rand_mask > rand_th[0], self.m_ones[:, 0:6, :, 0:1], rand_mask)
        rand_mask = torch.where(rand_mask < rand_th[0], self.m_zeros[:, 0:6, :, 0:1], rand_mask)
        rand_mask = rand_mask.expand_as(self.image_sub_1[:, 0:6, :, :])
        anti_mask = 1 - rand_mask

        k_temp = self.image_sub_1[:, 0:6, :, :]
        k_temp_1 = torch.fft.fftn(torch.complex(k_temp[:, 0::2, :, :], k_temp[:, 1::2, :, :]), dim=(-3, -2))
        k_temp = self.image_sub_2[:, 0:6, :, :]
        k_temp_2 = torch.fft.fftn(torch.complex(k_temp[:, 0::2, :, :], k_temp[:, 1::2, :, :]), dim=(-3, -2))

        k_source = rand_mask[:, 0:3, :, :] * k_temp_1 + anti_mask[:, 0:3, :, :] * k_temp_2
        k_target = rand_mask[:, 0:3, :, :] * k_temp_2 + anti_mask[:, 0:3, :, :] * k_temp_1

        c_temp = torch.fft.ifftn(k_source, dim=(-3, -2))
        k_temp[:, 0::2, :, :] = c_temp.real
        k_temp[:, 1::2, :, :] = c_temp.imag
        self.image_sub_1[:, 0:6, :, :] = k_temp

        c_temp = torch.fft.ifftn(k_target, dim=(-3, -2))
        k_temp[:, 0::2, :, :] = c_temp.real
        k_temp[:, 1::2, :, :] = c_temp.imag
        self.image_sub_2[:, 0:6, :, :] = k_temp

        I_N1 = self.image_sub_1.requires_grad_(True)
        I_N2 = self.image_sub_2.requires_grad_(True)
        I_AVG = self.image_avg.requires_grad_(True)
        I_GT = self.image_full.requires_grad_(True)

        ############# Model input #############
        net = {}
        for i in range(1, self.n_recurrent + 1):

            net['r%d_img_pred_1' % i] = self.net_G_I(I_N1)  # output recon image [b,c,w,h]
            net['r%d_img_pred_2' % i] = self.net_G_I(I_N2)  # output recon image [b,c,w,h]
            with torch.no_grad():
                net['r%d_img_pred_avg' % i] = self.net_G_I(I_AVG)  # output recon image [b,c,w,h]

            self.recon = net['r%d_img_pred_avg' % i]


        self.net = net
        self.recon_center = self.recon[:, 2:4, :, :]
        self.image_full_center = self.image_full[:, 2:4, :, :]
        self.image_sub_1_center = self.image_avg[:, 2:4, :, :]


    def update_G(self):
        loss_G_L1 = 0
        self.optimizer_G.zero_grad()

        loss_img_l1 = 0
        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_l1 = loss_img_l1 + self.mse(self.net['r%d_img_pred_1' % j][:, 2:4, :, :], self.image_sub_2[:, 2:4, :, :]) / 2 + \
                                        self.mse(self.net['r%d_img_pred_2' % j][:, 2:4, :, :], self.image_sub_1[:, 2:4, :, :]) / 2

            loss_kspc = self.criterion(self.net['r%d_img_pred_1' % j][:, 2:4, :, :],
                                       self.net['r%d_img_pred_2' % j][:, 2:4, :, :]) * 100

        loss_G_L1 = loss_img_l1
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img_l1 = loss_img_l1.item()
        self.loss_kspc.update(loss_kspc)

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()


    def optimize(self):
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    def inference(self):
        self.loss_G_L1 = 0
        self.forward()

    def inference_ir(self, x):
        with torch.no_grad():
            self.recon_it = self.net_G_I(x)  # output recon image [b,c,w,h]

        return self.recon_it

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'Img_L2: {:.4f} Diff: {:.4f}'.format(self.loss_G_L1, self.loss_img_l1, self.loss_kspc.avg)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G_I.module.load_state_dict(checkpoint['net_G_I'])

            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])


        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            self.rec_real = self.recon_center[:, 0, :, :]
            self.rec_imag = self.recon_center[:, 1, :, :]
            self.rec_real = self.rec_real.unsqueeze(1)
            self.rec_imag = self.rec_imag.unsqueeze(1)
            self.rec = torch.cat([self.rec_real, self.rec_imag], dim=1)

            self.gt_real = self.image_full_center[:, 0, :, :]
            self.gt_imag = self.image_full_center[:, 1, :, :]
            self.gt_real = self.gt_real.unsqueeze(1)
            self.gt_imag = self.gt_imag.unsqueeze(1)
            self.gt = torch.cat([self.gt_real, self.gt_imag], dim=1)

            self.lr_real = self.image_sub_1_center[:, 0, :, :]
            self.lr_imag = self.image_sub_1_center[:, 1, :, :]
            self.lr_real = self.lr_real.unsqueeze(1)
            self.lr_imag = self.lr_imag.unsqueeze(1)
            self.lr = torch.cat([self.lr_real, self.lr_imag], dim=1)

            if self.opts.wr_L1 > 0:

                psnr_recon = psnr(complex_abs_eval(self.rec),
                                  complex_abs_eval(self.gt))
                psnr_recon = 0
                # avg_psnr.update(psnr_recon)

                ssim_recon = ssim(complex_abs_eval(self.rec)[0, 0, :, :].cpu().numpy(),
                                  complex_abs_eval(self.gt)[0, 0, :, :].cpu().numpy())
                ssim_recon = 0
                # avg_ssim.update(ssim_recon)

                recon_images.append(self.rec[0].cpu())
                gt_images.append(self.gt[0].cpu())
                input_images.append(self.lr[0].cpu())

            message = 'PSNR: {:4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim.avg)
            val_bar.set_description(desc=message)

        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg

        self.results = {}
        if self.opts.wr_L1 > 0:
            self.results['recon'] = torch.stack(recon_images).squeeze().numpy()
            self.results['gt'] = torch.stack(gt_images).squeeze().numpy()
            self.results['input'] = torch.stack(input_images).squeeze().numpy()

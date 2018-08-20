import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import time
from utils import *


def train(model, data, config):

    # load model / model to device
    for net in model.values():
        net.to(config.device)
    if config.pretrained_model > 0:
        load_model(model, config)
        print('loading models successfully!\n')
    ext_net = DeepFeature(config.ext)
    ext_net.to(config.device)
    contextual_loss = get_contextual_loss(config)

    # optimizor
    opts = {}
    for key, net in model.items():
        opts[key] = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr, betas=(config.beta, 0.999))

    # data_loader
    data_loader = data.get_loader(batch_size=config.batch_size, num_workers=max(config.batch_size,1), shuffle=True, drop_last=True)
    data_iter = iter(data_loader)

    # constants
    batch_size = config.batch_size
    start_time = time.time()
    ref_feat = False

    log = {}
    log['nsteps'] = config.num_steps

    # pretrain
    for i in range(config.pretrain_num_steps):
        (src_pho, ref_pho), data_iter = get_data(data_iter, data_loader)
        src_pho, ref_pho = src_pho.to(config.device), ref_pho.to(config.device)
        fake_pho = model['G'](src_pho, ref_pho)
        rec_loss = 10*(fake_pho - src_pho).abs().mean() + (fake_pho - ref_pho).abs().mean()
        opts['G'].zero_grad()
        rec_loss.backward()
        opts['G'].step()
        if (i+1) % (config.log_step) == 0:
            print('pretrain: [{}/{}], rec_loss:{:.4f}'.format(i+1, config.pretrain_num_steps, rec_loss.item()))


    for i in range(config.num_steps):
        log['step'] = i+1

        (src_pho, ref_pho), data_iter = get_data(data_iter, data_loader)
        src_pho, ref_pho = src_pho.to(config.device), ref_pho.to(config.device)

        ####################### train Discriminator and Comparator #######################
        # get fake images
        fake_pho = model['G'](src_pho, ref_pho)

        # compute loss
        src_feat = ext_net(denorm(src_pho), config.content_layers, detach=True)
        if (not config.single_ref) or (not ref_feat):
            ref_feat = ext_net(denorm(ref_pho), config.style_layers, detach=True)
        fake_feat = ext_net(denorm(fake_pho), [config.content_layers,config.style_layers], detach=False)

        closs = sum([contextual_loss(fake_feat[0][i], src_feat[i]) for i in range(len(src_feat))])
        torch.cuda.empty_cache()
        sloss = sum([contextual_loss(fake_feat[1][i], ref_feat[i]) for i in range(len(ref_feat))])
        torch.cuda.empty_cache()

        loss = config.lambda_content * closs + config.lambda_style * sloss

        opts['G'].zero_grad()
        loss.backward()
        opts['G'].step()


        ### save log ###
        if (i+1) % (config.log_step) == 0:

            log['loss/content'] = closs.item()
            log['loss/style'] = sloss.item()
            log['time_elapse'] = time.time() - start_time

            save_log(log, config, [['content', 'style']])

        ### save images ###
        if (i+1) % (config.sample_step) == 0:
            n = config.num_sample
            src_pho = data.get_test(n)
            src_pho = src_pho.to(config.device)

            imgs = [src_pho.cpu()]
            ref_pho_test = ref_pho.repeat(src_pho.size(0)//ref_pho.size(0),1,1,1)

            with torch.no_grad():
                fake_imgs = model['G'](src_pho, ref_pho_test).cpu()
                imgs.append(fake_imgs)
            imgs.append(ref_pho_test.cpu())

            imgs = torch.cat(imgs, dim=3)
            save_image(denorm(imgs), os.path.join(config.log_path, 'gen_imgs_%d.png'%(i+1)), nrow=1, padding=0)
            print('saving images successfully!\n')

        if (i+1) % config.model_save_step == 0:
            ### save models ###
            save_model(model, config, log)
            print('saving models successfully!\n')

        ### update lr ###
        if (i+1) % config.model_save_step == 0 and (i+1) > (config.num_steps - config.num_steps_decay):
            lr = config.lr * (config.num_steps - i) / config.num_steps_decay
            for opt in opts.values():
                for param in opt.param_groups:
                    param['lr'] = lr
            print('update learning rate to {:.6f}'.format(lr))


def get_contextual_loss(config):
    sep = config.feat_sep
    def contextual_loss(feat1, feat2):
        B, C, H1, W1 = feat1.size()
        _, _, H2, W2 = feat2.size()

        H = min((H1, H2, 64))
        W = min((W1, W2, 64))
        if H < H1 or W < W1:
            feat1 = random_crop(feat1, (2,3), (H,W))
        if H < H2 or W < W2:
            feat2 = random_crop(feat2, (2,3), (H,W))        


        cx = torch.zeros(feat1.size(0), 1, H*W).to(feat1.device)
        H, W = min(sep, H), min(sep, W)
        feat1 = merge_list([f.split(sep, dim=3) for f in feat1.split(sep, dim=2)])
        feat2 = merge_list([f.split(sep, dim=3) for f in feat2.split(sep, dim=2)])
        for f1 in feat1:
            f1 = f1.contiguous().view(B, C, H*W, 1).repeat(1,1,1,H*W)
            dists = []
            for f2 in feat2:
                f2 = f2.contiguous().view(B, C, H*W, 1).repeat(1,1,1,H*W).detach()

                if config.distance == 'l2':
                    dist = (f1 - f2).pow(2).sum(dim=1)
                elif config.distance == 'l1':
                    dist = (f1 - f2).abs().sum(dim=1)
                elif config.distance == 'cos':
                    dist = 1 - F.cosine_similarity(f1, f2, dim=1)
                dists.append(dist)

            dist = torch.cat(dists, dim=2)
            dist = dist / (dist.min(dim=2,keepdim=True)[0]+1e-5)
            dist = torch.exp((1-dist)/config.h)
            dist = dist / dist.sum(dim=2, keep_dim=True)
            cx = torch.max(torch.cat([cx, dist], dim=1), dim=1, keepdim=True)[0]
        return -torch.log(cx.squeeze(1).mean(dim=1)).mean()
    return contextual_loss















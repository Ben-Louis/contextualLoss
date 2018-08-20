import os
import torch
import csv
import random

def shuffle(ts, dim=0, inv=False):
    if inv:
        idx = torch.arange(ts.size(dim)-1,-1,step=-1,device=ts.device)
    else:
        idx = torch.randperm(ts.size(dim)).to(ts.device)
    return ts[idx.long()]

def random_crop(ts, dim, scale):
    if isinstance(dim, int):
        assert isinstance(scale, int)
        start = random.randint(0, ts.size(dim) - scale + 1)
        return ts.index_select(index=torch.arange(start, start+dim).long(), dim=dim)

    elif len(dim) == 2:
        if isinstance(scale, int):
            scale = (scale, scale)
        return random_crop(random_crop(ts, dim[0], scale[0]), dim[1], scale[1])


def save_log(log, config, print_items=[]):
    log_path = os.path.join(config.log_path, 'log.csv')
    write_header = not os.path.exists(log_path)

    with open(log_path, 'a+') as f: 
        f_csv = csv.DictWriter(f, log.keys())
        if write_header:
            f_csv.writeheader()
        f_csv.writerows([log])

    if config.print_log:
        logg = ''
        logg += 'step:[{}/{}], time:{:.7f}\n'.format(log['step'], log['nsteps'], log['time_elapse'])
        if print_items:
            for items in print_items:
                for item in items:
                    logg += '{}:{:.4f}  '.format(item, log['loss/%s'%item])
                logg += '\n'
        print(logg)

def denorm(x):
    x = ((x+1)/2)
    if torch.is_tensor(x):
        x = x.clamp(0,1)
    return x

def save_model(model, config, log):
    for key, net in model.items():
        torch.save(net.state_dict(), os.path.join(config.model_path, '%s-%d.cpkt'%(key, log['step'])))

def load_model(model, config):
    for key, net in model.items():
        net.load_state_dict(torch.load(os.path.join(config.model_path, '%s-%d.cpkt'%(key, config.pretrained_model))))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_data(data_iter, data_loader):
    try:
        phos = next(data_iter)
    except:
        data_iter = iter(data_loader)
        phos = next(data_iter)

    return phos, data_iter

def merge_list(lst):
    res = []
    for l in lst:
        res.extend(l)
    return res
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.BayesianCal import BayesianCal
from util.losses import *
from util.utils import read_data, read_user_data
from util import metric

def test(NetC, Net_Bayes, eval_loader, metrics, device='cuda'):
    max_rul = 125
    mseloss = nn.MSELoss()
    metrics.reset()
    NetC.to(device)
    NetC.eval()

    num_samples = []
    tot_rmse = []
    tot_score = []

    for x, y, _ in eval_loader:
        x = x.to(device)
        y = y.to(device)
        output_reg, output_cls, _, _, _ = Net_Bayes(x)
        xFedC_mu, xFedC_alpha, xFedC_beta = NetC(output_reg)
        xFedC_mu = xFedC_mu.squeeze(-1)
        y = y / max_rul
        mses = mseloss(xFedC_mu, y)
        metrics.update([output.data.cpu() for output in [xFedC_mu]], y.cpu(), [mse.data.cpu() for mse in [mses]])

    metrics = metrics.get_name_value()
    RMSE = metrics[0][0][1]
    SCORE = metrics[1][0][1]
    ns = y.shape[0]

    tot_rmse.append(RMSE * 1.0)
    tot_score.append(SCORE * 1.0)
    num_samples.append(ns)

    return num_samples, tot_rmse, tot_score

def eval_BayesCap(
        NetC,
        Net_Bayes,
        eval_loader,
        metrics,
):

    test_samples, test_rmse, test_score = test(NetC, Net_Bayes, eval_loader, metrics)
    glob_rmse = np.sum(test_rmse) * 1.0 / len(test_samples)
    glob_score = np.sum(test_score) * 1.0 / len(test_samples)

    return glob_rmse, glob_score

def train_BayesianCal(
        NetC,
        Net_Bayes,
        train_loader,
        eval_loader,
        metrics,
        dataset,
        Cri=TempCombLoss(),
        device='cuda',
        dtype=torch.cuda.FloatTensor(),
        init_lr=1e-4,
        num_epochs=100,
        eval_every=1,
        ckpt_path='ckpt/BayesCap',
        T1=1e0,
        T2=5e-2,
):
    NetC.to(device)
    NetC.train()
    Net_Bayes.to(device)
    Net_Bayes.eval()
    optimizer = torch.optim.Adam(list(NetC.parameters()), lr=init_lr)
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_rmse = 10000.0
    best_score = 100000.0
    all_loss = []
    for eph in range(num_epochs):
        eph_loss = 0
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):
                tepoch.set_description('Epoch {}'.format(eph))

                x, y = batch[0].to(device), batch[1].to(device)
                x, y = x.type(dtype), y.type(dtype)
                y_normalize = (y / 125)

                with torch.no_grad():
                    xFed1, _, _, _, _ = Net_Bayes(x)
                xFed = xFed1.clone()
                xFedC_mu, xFedC_alpha, xFedC_beta = NetC(xFed)
                xFedC_mu = xFedC_mu.squeeze(-1)
                optimizer.zero_grad()
                loss = Cri(xFedC_mu, xFedC_alpha, xFedC_beta, xFed, y_normalize, T1=T1, T2=T2)

                loss.backward()
                optimizer.step()

                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            all_loss.append(eph_loss)
            print('Avg. loss: {}'.format(eph_loss))
        # evaluate and save the models
        torch.save(NetC.state_dict(), ckpt_path + dataset + '_last.pth')
        if eph % eval_every == 0:
            glob_rmse, glob_score = eval_BayesCap(
                NetC,
                Net_Bayes,
                eval_loader,
                metrics
            )

            if glob_rmse <= best_rmse:
                best_rmse = glob_rmse
                torch.save(NetC.state_dict(), ckpt_path + dataset + '_best_rmse.pth')
            if glob_score <= best_score:
                best_score = glob_score
                torch.save(NetC.state_dict(), ckpt_path + dataset + '_best_score.pth')

            print('\n Current RMSE/score: {}/{} | Best RMSE/score: {}/{}'.format(glob_rmse, glob_score, best_rmse, best_score))
    optim_scheduler.step()


def mian(client_id, ckpt_path, dataset):
    Net_Bayes = (torch.load(ckpt_path, map_location='cuda:0'))

    metrics = metric.MetricList(metric.RMSE(max_rul=125), metric.RULscore(max_rul=125))

    model_parameters = filter(lambda p: True, Net_Bayes.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of Parameters:", params)

    NetC = BayesianCal(in_channels=1, out_channels=1)

    data = read_data(dataset)
    id, train_data, test_data = read_user_data(client_id, data, dataset=dataset)

    train_loader = DataLoader(train_data, batch_size=128, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8, pin_memory=True, shuffle=False)

    train_BayesianCal(
        NetC,
        Net_Bayes,
        train_loader,
        test_loader,
        metrics,
        dataset,
        Cri=TempCombLoss(alpha_eps=1e-5, beta_eps=1e-2),
        device='cuda',
        dtype=torch.cuda.FloatTensor,
        init_lr=1e-4,
        num_epochs=20,
        eval_every=2,
        ckpt_path='ckpt/',
    )

if __name__ == '__main__':
    dataset = 'cmapss-biid-u5c13-FD004'
    ckpt_dir = os.path.join('ckpt', dataset)
    ckpt_dir_times = os.listdir(ckpt_dir)       # FedCov training times are 5
    for ckpt_dir_path_ in ckpt_dir_times:
        ckpt_dir_path = os.listdir(os.path.join(ckpt_dir, ckpt_dir_path_))
        for ckpt_path in ckpt_dir_path:
            client_id = int(ckpt_path[5])-1     # Client ID in FedCov training，from 0 to 4
            path = os.path.join(ckpt_dir, ckpt_dir_path_, ckpt_path)
            mian(client_id, path, dataset)

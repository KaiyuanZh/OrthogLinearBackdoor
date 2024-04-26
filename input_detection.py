import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse.linalg import svds
from utils.utils import *


def cluster_metrics(cluster_1, cluster_0):
    num = len(cluster_1) + len(cluster_0)
    features = torch.cat([cluster_1, cluster_0], dim=0)

    labels = torch.zeros(num)
    labels[:len(cluster_1)] = 1
    labels[len(cluster_1):] = 0

    # Raw Silhouette Score
    raw_silhouette_score = silhouette_score(features, labels)
    return raw_silhouette_score


def test(args, eval=False):
    # Load model
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}{args.suffix}.pt'
    model = torch.load(model_filepath, map_location='cpu').cuda()
    model.eval()

    # Preprocess
    preprocess, _ = get_norm(args.dataset)

    # Load testing data
    test_set = get_dataset(args, train=False)
    if args.attack == 'clean':
        poison_set = test_set
        # Shuffle the dataset
        indices = np.random.choice(len(poison_set), len(poison_set), replace=False)
        poison_set = torch.utils.data.Subset(poison_set, indices)
    elif args.attack == 'cl':
        x_adv, y_adv = [], []
        for i, (image, label) in enumerate(test_set):
            if label == args.target:
                x_adv.append(image)
                y_adv.append(label)
        x_adv = torch.stack(x_adv)
        y_adv = torch.tensor(y_adv)

        # Apply adversarial attack on x_adv
        pgd_steps = 50
        pgd_eps = 16 / 255
        perturb = (torch.rand_like(x_adv) * 2 - 1) * pgd_eps
        perturb.requires_grad = True
        pgd_optim = torch.optim.Adam([perturb], lr=1e-3, betas=(0.5, 0.9), eps=1e-6)

        pgd_criterion = torch.nn.CrossEntropyLoss()

        for step in range(pgd_steps):
            pgd_optim.zero_grad()
            clip_perturb = torch.clamp(perturb, -pgd_eps, pgd_eps)
            x_poison = torch.clamp(x_adv + clip_perturb, 0, 1)
            output = model(preprocess(x_poison).cuda())
            loss = - pgd_criterion(output, y_adv.cuda())
            loss.backward()
            pgd_optim.step()

            pred = output.argmax(dim=1).cpu()
            acc = (pred == y_adv).float().mean()
            if (step+1) % 10 == 0:
                print(f'PGD step {step+1}/{pgd_steps}: loss={loss.item():.4f}, acc={acc*100:.2f}%')

        perturb = torch.clamp(perturb.detach(), -pgd_eps, pgd_eps)
        x_adv = torch.clamp(x_adv + perturb + EPSILON, 0, 1)
        # Stamp trigger on poisoned samples
        shape = x_adv.shape[-2:]
        backdoor = get_backdoor(args.attack, shape, device='cpu')
        x_adv = backdoor.inject(x_adv)
        save_image(x_adv, 'temp.png')

        poison_set = torch.utils.data.TensorDataset(x_adv, y_adv)
        
    else:
        poison_set = get_poison_testset(args, test_set)

    if eval:
        # Evaluate clean accuracy
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size)
        poison_loader = DataLoader(dataset=poison_set, batch_size=args.batch_size)

        n_sample = 0
        n_correct = 0
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(test_loader):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                output = model(preprocess(x_batch))
                pred = output.max(dim=1)[1]

                n_sample += x_batch.size(0)
                n_correct += (pred == y_batch).sum().item()

        acc = n_correct / n_sample

        # Evaluate ASR
        n_sample = 0
        n_correct = 0
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(poison_loader):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                output = model(preprocess(x_batch))
                pred = output.max(dim=1)[1]

                n_sample += x_batch.size(0)
                n_correct += (pred == y_batch).sum().item()

        asr = n_correct / n_sample

        print(f'Clean accuracy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')
        return acc, asr

    # TODO: Default numbers for clean and poison samples
    if args.dataset == 'cifar10':
        n_per_cls = 900
        n_poison = 100
    elif args.dataset == 'gtsrb':
        n_per_cls = 30
        n_poison = n_per_cls
    
    num_classes = get_config(args.dataset)['num_classes']
    cls_cnt = [0 for _ in range(num_classes)]

    x_clean, y_clean = [], []
    x_poison, y_poison = [], []

    for i, (image, label) in enumerate(test_set):
        if cls_cnt[label] < n_per_cls:
            x_clean.append(image)
            y_clean.append(label)
            cls_cnt[label] += 1
        if len(x_clean) == n_per_cls * num_classes:
            break

    x_clean = torch.stack(x_clean)
    y_clean = torch.tensor(y_clean)

    for i, (image, label) in enumerate(poison_set):
        x_poison.append(image)
        y_poison.append(label)
        if len(x_poison) == n_poison:
            break
    x_poison = torch.stack(x_poison)
    y_poison = torch.tensor(y_poison)

    # Get clean/poison features
    x_clean, y_clean = x_clean.cuda(), y_clean.cuda()
    x_poison, y_poison = x_poison.cuda(), y_poison.cuda()
    print(x_clean.shape, y_clean.shape, x_poison.shape, y_poison.shape)

    n_clean = x_clean.size(0)
    n_poison = x_poison.size(0)
    x_gather = torch.cat([x_clean, x_poison], dim=0)
    y_gather = torch.cat([y_clean, y_poison], dim=0)
    indices = np.array([0 for _ in range(n_clean)] + [1 for _ in range(n_poison)])

    feat = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            feat[name] = input[0].detach().cpu()
        return hook

    if 'resnet18' in args.network:
        layer_name = 'linear'
        hooker = model.linear.register_forward_hook(getActivation(layer_name))
    elif 'wrn' in args.network:
        layer_name = 'fc'
        hooker = model.fc.register_forward_hook(getActivation(layer_name))
    else:
        raise NotImplementedError

    images = x_gather.cuda()
    model(preprocess(images))

    features = feat[layer_name].view(x_gather.size(0), -1)
    print(features.shape)

    # Remove hook
    hooker.remove()

    # TODO: Shuffle the indices
    shuffle_idx = np.random.permutation(n_clean + n_poison)
    x = x_gather[shuffle_idx]
    y = y_gather[shuffle_idx]
    features = features[shuffle_idx]
    indices = indices[shuffle_idx]

    data = {}
    data['x'] = x
    data['y'] = y
    data['features'] = features
    data['indices'] = indices
    data['num_classes'] = num_classes


    return data

def QUEscore(temp_feats, n_dim):

    n_samples = temp_feats.shape[1]
    alpha = 4.0
    Sigma = torch.matmul(temp_feats, temp_feats.T) / n_samples
    I = torch.eye(n_dim).cuda()
    Q = torch.exp((alpha * (Sigma - I)) / (torch.linalg.norm(Sigma, ord=2) - 1))
    trace_Q = torch.trace(Q)

    taus = []
    for i in range(n_samples):
        h_i = temp_feats[:, i:i + 1]
        tau_i = torch.matmul(h_i.T, torch.matmul(Q, h_i)) / trace_Q
        tau_i = tau_i.item()
        taus.append(tau_i)
    taus = np.array(taus)

    return taus


def compute_SPECTRE(U, temp_feats, n_dim, budget, oracle_clean_feats=None):

    projector = U[:, :n_dim].T # top left singular vectors
    temp_feats = torch.matmul(projector, temp_feats)

    if oracle_clean_feats is None:
        estimator = robust_estimation.BeingRobust(random_state=0, keep_filtered=True).fit((temp_feats.T).cpu().numpy())
        clean_mean = torch.FloatTensor(estimator.location_).cuda()
        filtered_feats = (torch.FloatTensor(estimator.filtered_).cuda() - clean_mean).T
        clean_covariance = torch.cov(filtered_feats)
    else:
        clean_feats = torch.matmul(projector, oracle_clean_feats)
        clean_covariance = torch.cov(clean_feats)
        clean_mean = clean_feats.mean(dim = 1)


    temp_feats = (temp_feats.T - clean_mean).T

    # whiten the data
    L, V = torch.linalg.eig(clean_covariance)
    L, V = L.real, V.real
    L = (torch.diag(L)**(1/2)+0.001).inverse()
    normalizer = torch.matmul(V, torch.matmul( L, V.T ) )
    temp_feats = torch.matmul(normalizer, temp_feats)
    # compute QUEscore
    taus = QUEscore(temp_feats, n_dim)

    sorted_indices = np.argsort(taus)
    n_samples = len(sorted_indices)

    budget = min(budget, n_samples//2) # default assumption : at least a half of samples in each class is clean

    suspicious = sorted_indices[-budget:]
    left = sorted_indices[:n_samples-budget]

    return suspicious, left


def run_SCAn(args):
    ########################################
    # TODO: Customized data collection
    data = test(args)
    x, y, features, gt_indices, num_classes = data['x'], data['y'], data['features'], data['indices'], data['num_classes']
    class_indices = [[] for _ in range(num_classes)]
    for i, label in enumerate(y):
        class_indices[label].append(i)
    feats = features
    ########################################    

    # get small set of clean data 
    class_indices_clean = np.array(np.where(gt_indices == 0)[0])
    # random sample 100 
    class_indices_clean = np.random.choice(class_indices_clean, 100, replace=False)
    feats_clean = feats[class_indices_clean].cpu().numpy()
    class_indices_clean = y[class_indices_clean].cpu().numpy()
     
    feats_inspection = feats.cpu().numpy()
    class_indices_inspection = y.cpu().numpy() 
    
    print(feats_clean.shape)
    print(class_indices_clean.shape)
    print(class_indices_clean[:10])
    print(feats_inspection.shape)
    print(class_indices_inspection.shape)
    print(class_indices_inspection[:10])
    input()
    
    
    
    scan = SCAn()

    # fit the clean distribution with the small clean split at hand
    gb_model = scan.build_global_model(feats_clean, class_indices_clean, num_classes)

    size_inspection_set = len(feats_inspection)

    feats_all = feats_inspection
    class_indices_all = class_indices_inspection

    
    
    # use the global model to divide samples
    lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, num_classes)

    # statistic test for the existence of "two clusters"
    score = scan.calc_final_score(lc_model)
    threshold = np.e



    suspicious_indices = []

    for target_class in range(num_classes):

        print('[class-%d] outlier_score = %f' % (target_class, score[target_class]) )

        if score[target_class] <= threshold: continue

        tar_label = (class_indices_all == target_class)
        all_label = np.arange(len(class_indices_all))
        tar = all_label[tar_label]

        cluster_0_indices = []
        cluster_1_indices = []

        cluster_0_clean = []
        cluster_1_clean = []

        for index, i in enumerate(lc_model['subg'][target_class]):
            if i == 1:
                if tar[index] > size_inspection_set:
                    cluster_1_clean.append(tar[index])
                else:
                    cluster_1_indices.append(tar[index])
            else:
                if tar[index] > size_inspection_set:
                    cluster_0_clean.append(tar[index])
                else:
                    cluster_0_indices.append(tar[index])


        # decide which cluster is the poison cluster, according to clean samples' distribution
        if len(cluster_0_clean) < len(cluster_1_clean): # if most clean samples are in cluster 1
            suspicious_indices += cluster_0_indices
        else:
            suspicious_indices += cluster_1_indices


    # Calculate TPR, FPR
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(gt_indices)):
        if i in suspicious_indices:
            if gt_indices[i] == 0:
                fp += 1
            else:
                tp += 1
        else:
            if gt_indices[i] == 0:
                tn += 1
            else:
                fn += 1
    
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    return tpr, fpr  




    # return suspicious_indices
 

EPS = 1e-5


class SCAn:
    def __init__(self):
        pass

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = self.calc_anomaly_index(y / np.max(y))
        return ai

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826
        index = b / mm
        return index

    def build_global_model(self, reprs, labels, n_classes):
        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L, M])
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N, M])
        e = np.zeros([N, M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su, F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k] * Su + Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L, M])
            e = np.zeros([N, M])
            u = np.zeros([N, M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
                u_m[k] = u_m[k] - np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_f
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):
        Su = gb_model['Su']
        Se = gb_model['Se']

        F = np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        class_score = np.zeros([L, 3])
        u1 = np.zeros([L, M])
        u2 = np.zeros([L, M])
        split_rst = list()

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)

            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)[0][0]
            split_rst.append(subg)
            u1[k] = i_u1
            u2[k] = i_u2
            print(k)
            print(i_sc)
            print(np.sum(selected_idx))
            class_score[k] = [k, i_sc, np.sum(selected_idx)]

        lc_model = dict()
        lc_model['sts'] = class_score
        lc_model['mu1'] = u1
        lc_model['mu2'] = u2
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)

        if (N == 1):
            subg[0] = 0
            return (subg, X.copy(), X.copy())

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        # EM
        steps = 0
        while (np.linalg.norm(subg - last_z1) > EPS) and (np.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1, F), np.transpose(u1)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1, F), np.transpose(e2))
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        SeG = np.matmul(Se,G)
        for i in range(N):
            vec = X[i]
            dd = np.matmul(SeG, np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N

    


def spectre(args):

    ########################################
    # TODO: Customized data collection
    data = test(args)
    x, y, features, gt_indices, num_classes = data['x'], data['y'], data['features'], data['indices'], data['num_classes']
    class_indices = [[] for _ in range(num_classes)]
    for i, label in enumerate(y):
        class_indices[label].append(i)
    feats = features
    ########################################    

    # get small set of clean data 
    clean_class_indices = np.array(np.where(gt_indices == 0)[0])
    # random sample 100 
    clean_class_indices = np.random.choice(clean_class_indices, 100, replace=False)
    clean_feats = feats[clean_class_indices]
    
    suspicious_indices = []
    poison_rate = 0.01
    # Spectral Signature requires an expected poison ratio (we allow the oracle here as a baseline)
    budget = int(poison_rate * len(feats) * 1.5)
    #print(budget)
    # allow removing additional 50% (following the original paper)

    max_dim = 64 # 64
    # max_dim = 2 # 64
    class_taus = []
    class_S = [] 

    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            # feats for class i in poisoned set
            temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats).cuda()

            # temp_clean_feats = np.array([feats[temp_idx] for temp_idx in clean_class_indices])
            temp_clean_feats = clean_feats
            temp_clean_feats = torch.FloatTensor(temp_clean_feats).cuda()
            temp_clean_feats = temp_clean_feats - temp_feats.mean(dim=0)
            temp_clean_feats = temp_clean_feats.T

            temp_feats = temp_feats - temp_feats.mean(dim=0) # centered data
            temp_feats = temp_feats.T # feats arranged in column

            U, _, _ = torch.svd(temp_feats)
            U = U[:, :max_dim]

            # full projection
            projected_feats = torch.matmul(U.T, temp_feats)

            max_tau = -999999
            best_n_dim = -1
            best_to_be_removed = None

            for n_dim in range(2, max_dim+1): # enumarate all possible "reudced dimensions" and select the best

                S_removed, S_left = compute_SPECTRE(U, temp_feats, n_dim, budget, temp_clean_feats)

                left_feats = projected_feats[:, S_left]
                covariance = torch.cov(left_feats)

                L, V = torch.linalg.eig(covariance)
                L, V = L.real, V.real
                L = (torch.diag(L) ** (1 / 2) + 0.001).inverse()
                normalizer = torch.matmul(V, torch.matmul(L, V.T))

                whitened_feats = torch.matmul(normalizer, projected_feats)

                tau = QUEscore(whitened_feats, max_dim).mean()

                if tau > max_tau:
                    max_tau = tau
                    best_n_dim = n_dim
                    best_to_be_removed = S_removed


            print('class=%d, dim=%d, tau=%f' % (i, best_n_dim, max_tau))

            class_taus.append(max_tau)

            suspicious_indices = []
            for temp_index in best_to_be_removed:
                suspicious_indices.append(class_indices[i][temp_index])

            class_S.append(suspicious_indices)

    class_taus = np.array(class_taus)
    median_tau = np.median(class_taus)

    #print('median_tau : %d' % median_tau)
    suspicious_indices = []
    max_tau = -99999
    for i in range(num_classes):
        #if class_taus[i] > max_tau:
        #    max_tau = class_taus[i]
        #    suspicious_indices = class_S[i]
        #print('class-%d, tau = %f' % (i, class_taus[i]))
        #if class_taus[i] > 2*median_tau:
        #    print('[large tau detected] potential poisons! Apply Filter!')
        for temp_index in class_S[i]:
            suspicious_indices.append(temp_index)

    # Calculate TPR, FPR
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(gt_indices)):
        if i in suspicious_indices:
            if gt_indices[i] == 0:
                fp += 1
            else:
                tp += 1
        else:
            if gt_indices[i] == 0:
                tn += 1
            else:
                fn += 1
    
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    return tpr, fpr  

def spectral_signature(args):

    ########################################
    # TODO: Customized data collection
    data = test(args)
    x, y, features, gt_indices, num_classes = data['x'], data['y'], data['features'], data['indices'], data['num_classes']
    class_indices = [[] for _ in range(num_classes)]
    for i, label in enumerate(y):
        class_indices[label].append(i)
    feats = features
    ########################################    
    poison_rate = 0.02
    num_poisons_expected = poison_rate * len(feats) * 1.5
    
    suspicious_indices = []
    
    for i in range(num_classes):
        if len(class_indices[i]) > 1:

            temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats)

            mean_feat = torch.mean(temp_feats, dim=0)
            temp_feats = temp_feats - mean_feat
            _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

            vec = V[:, 0]  # the top right singular vector is the first column of V
            vals = []
            for j in range(temp_feats.shape[0]):
                vals.append(torch.dot(temp_feats[j], vec).pow(2))

            
            k = min(int(num_poisons_expected), len(vals) // 2)
            
            # default assumption : at least a half of samples in each class is clean

            _, indices = torch.topk(torch.tensor(vals), k)
            for temp_index in indices:
                suspicious_indices.append(class_indices[i][temp_index])


    # Calculate TPR, FPR
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(gt_indices)):
        if i in suspicious_indices:
            if gt_indices[i] == 0:
                fp += 1
            else:
                tp += 1
        else:
            if gt_indices[i] == 0:
                tn += 1
            else:
                fn += 1
    
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    return tpr, fpr           
            
    
    
    
def activation_clustering(args):
    ########################################
    # TODO: Customized data collection
    data = test(args)
    x, y, features, gt_indices, num_classes = data['x'], data['y'], data['features'], data['indices'], data['num_classes']
    class_indices = [[] for _ in range(num_classes)]
    for i, label in enumerate(y):
        class_indices[label].append(i)
    feats = features
    ########################################

    suspicious_indices = []
    max_score = 0

    for target_class in range(num_classes):

        print('class - %d' % target_class)

        if len(class_indices[target_class]) <= 1: continue # no need to perform clustering...

        temp_feats = [feats[temp_idx].unsqueeze(dim=0) for temp_idx in class_indices[target_class]]
        temp_feats = torch.cat(temp_feats , dim=0)
        temp_feats = temp_feats - temp_feats.mean(dim=0)

        _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

        axes = V[:, :10]
        projected_feats = torch.matmul(temp_feats, axes)
        projected_feats = projected_feats.cpu().numpy()

        kmeans = KMeans(n_clusters=2).fit(projected_feats)

        # Take the smaller cluster as the poisoned cluster
        if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
            clean_label = 1
        else:
            clean_label = 0

        outliers = []
        for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
            if bool:
                outliers.append(class_indices[target_class][idx])

        score = silhouette_score(projected_feats, kmeans.labels_)
        print('[class-%d] num_samples= %d, silhouette_score = %f' % (target_class, len(class_indices[target_class]), score))
        # if score > max_score:
        #     max_score = score
        #     suspicious_indices = outliers
        #     print(f"Outlier Num in Class {target_class}:", len(outliers))
        suspicious_indices += outliers

    # Calculate TPR, FPR
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(gt_indices)):
        if i in suspicious_indices:
            if gt_indices[i] == 0:
                fp += 1
            else:
                tp += 1
        else:
            if gt_indices[i] == 0:
                tn += 1
            else:
                fn += 1
    
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    return tpr, fpr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--datadir', default='./data', help='root directory of data')
    parser.add_argument('--phase', default='test', help='phase')

    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')
    parser.add_argument('--attack', default='badnet', help='attack')
    parser.add_argument('--target', type=int, default=0, help='target class')

    parser.add_argument('--suffix', default='', help='suffix')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--seed', type=int, default=1024, help='seed index')

    args = parser.parse_args()

    # Print arguments
    print_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    DEVICE = torch.device(f'cuda:{args.gpu}')

    if args.phase == 'test':
        test(args, eval=True)
    elif args.phase == 'ac':
        tpr, fpr = activation_clustering(args)
        print(f'TPR: {tpr*100:.2f}%, FPR: {fpr*100:.2f}%')
    elif args.phase == 'ss':
        tpr, fpr = spectral_signature(args)
        print(f'TPR: {tpr*100:.2f}%, FPR: {fpr*100:.2f}%')
    elif args.phase == 'spectre':
        tpr, fpr = spectre(args)
        print(f'TPR: {tpr*100:.2f}%, FPR: {fpr*100:.2f}%')
    elif args.phase == 'scan':
        tpr, fpr = run_SCAn(args)
        print(f'TPR: {tpr*100:.2f}%, FPR: {fpr*100:.2f}%')
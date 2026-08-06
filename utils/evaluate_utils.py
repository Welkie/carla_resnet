import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter
from data.custom_dataset import NeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy
from sklearn.metrics import precision_recall_curve, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def contrastive_evaluate(val_loader, model, ts_repository):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    dev = next(model.parameters()).device

    for batch in val_loader:
        ts_org = batch['ts_org']
        target = batch['target']
        if isinstance(ts_org, np.ndarray):
            ts_org = torch.from_numpy(ts_org).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
            
        b, w, h = ts_org.shape if ts_org.ndim == 3 else (ts_org.shape[0], 1, ts_org.shape[1])
        ts_org = ts_org.to(dev)
        output = model(ts_org.view(b, h, w))

        output = ts_repository.weighted_knn(output)

        acc1 = 100 * torch.mean(torch.eq(output.cpu(), target.cpu()).float())
        top1.update(acc1.item(), ts_org.size(0))

    return top1.avg


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False, is_training=False):
    # Make predictions on a dataset with neighbors
    global features, nneighbors, fneighbors
    model.eval()
    dev = next(model.parameters()).device
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim))

    if isinstance(dataloader.dataset, NeighborsDataset): # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        nneighbors = []
        fneighbors = []

    else:
        key_ = 'ts_org'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        ts = batch[key_]
        if ts.ndim == 3:
            bs, w, h = ts.shape
        else:
            bs, w = ts.shape
            h = 1

        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).float()
            targets.append(torch.from_numpy(batch['target']))
        else:
            targets.append(batch['target'])

        ts = ts.to(dev)
        res = model(ts.view(bs, h, w), forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr+bs] = res['features'].cpu()
            ptr += bs
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1).cpu())
            probs[i].append(F.softmax(output_i, dim=1).cpu())

        if include_neighbors:
            nneighbors.append(batch['possible_nneighbors'])
            fneighbors.append(batch['possible_fneighbors'])

    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0).cpu()

    if include_neighbors:
        nneighbors = torch.cat(nneighbors, dim=0).cpu()
        fneighbors = torch.cat(fneighbors, dim=0).cpu()
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': nneighbors, 'fneighbors': fneighbors} for pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in zip(predictions, probs)]

    if return_features:
        feat_np = features.numpy()  # save features in csv
        fhdr = [str(x) for x in range(feat_np.shape[1])] + ['Class']
        feat_np = np.hstack((feat_np, np.array(targets.cpu().numpy())[np.newaxis].T)) 

        feat_df = pd.DataFrame(feat_np, columns=fhdr)

        prob_np = np.array(out[0]['probabilities'])
        phdr = [str(x) for x in range(prob_np.shape[1])] + ['Class']
        prob_np = np.hstack((prob_np, np.array(targets.cpu().numpy())[np.newaxis].T)) 
        prob_df = pd.DataFrame(prob_np, columns=phdr)

        if is_training:
            feat_df.to_csv(p['classification_trainfeatures'], index=False, header=True, sep=',')
            prob_df.to_csv(p['classification_trainprobs'], index=False, header=True, sep=',')
        else:
            feat_df.to_csv(p['classification_testfeatures'], index=False, header=True, sep=',')
            prob_df.to_csv(p['classification_testprobs'], index=False, header=True, sep=',')

        return out, features.cpu()

    else:
        return out


@torch.no_grad()
def classification_evaluate(predictions):
    # Evaluate model based on classification loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        fneighbors = head['fneighbors']
        org_anchors = torch.arange(neighbors.size(0)).view(-1,1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = org_anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()

        similarity = torch.matmul(probs, probs.t())
        fneighbors = fneighbors.contiguous().view(-1)
        anchors = org_anchors.contiguous().view(-1)
        similarity = similarity[anchors, fneighbors]
        ones = torch.ones_like(similarity)
        inconsistency_loss = F.binary_cross_entropy(similarity, ones).item()

        # Total loss
        total_loss = 5*entropy_loss + consistency_loss - 0*inconsistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'inconsistency': inconsistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'classification': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


@torch.no_grad()
def pr_evaluate(all_predictions, class_names=None,
                compute_purity=True, compute_confusion_matrix=True,
                confusion_matrix_file=None, majority_label=0):

    head = all_predictions[0]
    targets = head['targets']
    predictions = head['predictions']
    probs = head['probabilities']
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    scores = 1-np.array(probs)[:,majority_label]
    labels = np.array(targets.cpu().numpy()).tolist()

    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    try:
        f1_score = 2*precision*recall / (precision+recall)
        if np.isnan(f1_score).any():
            f1_score = np.nan_to_num(f1_score)
            print('f1: Nan --> 0')
    except ZeroDivisionError:
        f1_score = [0.0]
        print('f1: 0 --> 0')

    best_f1_index = np.argmax(f1_score)

    rep_f1 = f1_score[best_f1_index]

    if class_names=='Anom':
        best_threshold = thresholds[best_f1_index]
        anomalies = [1 if s >= best_threshold else 0 for s in scores]
        best_tn, best_fp, best_fn, best_tp = confusion_matrix(labels, anomalies).ravel()
        print("Anomalies --> TP: ", best_tp, ", TN: ", best_tn, ", FN: ", best_fn, ", FP: ", best_fp)
        print(majority_label)
        print(metrics.classification_report(labels, anomalies))

    return rep_f1

def replace_majority_label(flat_preds, majority_label):
    new_pred = torch.where(flat_preds == majority_label, 0, 1)
    return new_pred
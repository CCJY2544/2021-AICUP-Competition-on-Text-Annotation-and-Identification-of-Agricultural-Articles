import torch
import json
import re
import numpy as np
import random
import os
import csv
import sys
from tqdm import tqdm
from torch import nn
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,roc_curve
from model import qa_model
# from my_risk_model import risk_model
from data.dataset import dataset_risk,dataset_public_test
from torch.utils.tensorboard import SummaryWriter


def train():
    args = {
        "batch_size": 12,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "n_epoch": 50,
        "log_step": 15,
        "save_step": 250,
        "d_emb": 768,
        "p_drop": 0.1,
        "weight_decay": 0.0,
        "model_path": os.path.join("exp", "model", "1102"),
        # "log_path": os.path.join("log", "_qa", "0618"),
        # "log_path": os.path.join("log", "_qa", "try_new_qa"),
        "data": os.path.join("./data", "external_train.csv"),
    }
    # Random seed
    random_seed = args["random_seed"]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # Model
    model = qa_model(args["d_emb"], args["p_drop"])
    model = model.train()
    model = model.to(device)

    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight']
    optim_group_params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args["weight_decay"],
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    # Optimizer
    optimizer = torch.optim.AdamW(
        optim_group_params, lr=args["learning_rate"])

    # Data
    # print(args["data"])
    data = dataset_risk(args["data"])
    # print(data)
    dataldr = torch.utils.data.DataLoader(
        data, batch_size=args["batch_size"], shuffle=True)

    # Train loop
    c=0
    for epoch in range(args["n_epoch"]):
        c+=1
        tqdm_dldr = tqdm(dataldr)
        step = 0
        avg_loss = 0
        for batch_data in tqdm_dldr:
            optimizer.zero_grad()
            batch_document = []

            batch_document_a = batch_data["article_a"].to(device)
            batch_document_b = batch_data["article_b"].to(device)
            # batch_answer = batch_data["answer"].long().to(device)
            batch_answer = batch_data["answer"].to(device)
            

            loss = model.loss_fn(batch_document_a, batch_document_b,batch_answer)
                                 
            loss.backward()
            optimizer.step()

            step = step + 1
            avg_loss = avg_loss + loss

        print("epoch %s loss : %s"%(c,avg_loss/step))
        if c>0:
            test(model)
        torch.save(model.state_dict(), os.path.join(
            args["model_path"], f"model-{c}-epoch.pt"))
    if not os.path.isdir(args["model_path"]):
        os.makedirs(args["model_path"])

    torch.save(model.state_dict(), os.path.join(
        args["model_path"], f"model-{step}.pt"))

def test(model=None):
    args = {
        "batch_size": 12,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "n_epoch": 30,
        "log_step": 15,
        "save_step": 250,
        "d_emb": 768,
        "p_drop": 0.1,
        "weight_decay": 0.0,
        "model_path": os.path.join("exp", "model", "1102"),
        # "log_path": os.path.join("log", "_qa", "0618"),
        # "log_path": os.path.join("log", "_qa", "try_new_qa"),
        # "data": os.path.join("./data", "public_test.txt"),
        "data": os.path.join("./data", "external_eval.csv"),
    }
    e=50
    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # Model
    # model = qa_model(args["d_emb"], args["p_drop"])
    # model = model.train()
    for e in range(1,51):
        print("epoch ",e," ",os.path.join(
                    args["model_path"], f"model-{e}-epoch.pt"))
        model = qa_model(args["d_emb"], args["p_drop"])
        model.load_state_dict(torch.load(os.path.join(
                args["model_path"], f"model-{e}-epoch.pt")))
    # if not model:
    #     model = qa_model(args["d_emb"], args["p_drop"])
    #     model.load_state_dict(torch.load(os.path.join(
    #             args["model_path"], f"model-{e}-epoch.pt")))
        model = model.eval()
        model = model.to(device)
        data = dataset_risk(args["data"])
        batch_size=args["batch_size"]
        dataldr = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=False)
        tqdm_dldr = tqdm(dataldr)
        ans = []
        pred = []
        ans_f1=[]
        tqdm_dldr = tqdm(dataldr)
        m = nn.Softmax(dim=-1)
        with torch.no_grad():
            for batch_data in tqdm_dldr:
                
                batch_document = []

                batch_document_a = batch_data["article_a"].to(device)
                batch_document_b = batch_data["article_b"].to(device)
                batch_answer = torch.FloatTensor(batch_data["answer"]).to(device)
                # print(model(batch_document_a, batch_document_b))
                # print(m(model(batch_document_a, batch_document_b)))
                # print(torch.argmax(model(batch_document_a, batch_document_b),dim=-1))
                # sys.exit()
                ans+=list(batch_answer)
                pred+=list(model(batch_document_a, batch_document_b))
                # pred+=list(torch.argmax(m(model(batch_document_a, batch_document_b)),dim=-1))
        # pred=torch.stack(pred)
        # pred=pred.reshape(-1)
        # pred=pred.to("cpu")
        pred_f1=[]
        threshold=0.5
        for p in pred:
            if p>=threshold:
                pred_f1.append(1)
            else:
                pred_f1.append(0)
        pred_f1=torch.tensor(pred_f1)
        pred=torch.tensor(pred)
        pred=pred.detach().numpy()
        pred_f1=pred_f1.detach().numpy()
        # ans=torch.stack(ans)
        # ans=ans.reshape(-1)
        # ans=ans.to("cpu")
        for p in ans:
            if p>=threshold:
                ans_f1.append(1)
            else:
                ans_f1.append(0)
        # ans_f1=torch.IntTensor(ans)
        ans_f1=torch.tensor(ans_f1)
        ans=torch.tensor(ans)
        # ans_f1=ans_f1.reshape(-1)
        # ans_f1=ans_f1.to("cpu")
        ans_f1=ans_f1.detach().numpy()
        ans=ans.detach().numpy()
        # print(ans,pred)
        print("roc:",roc_auc_score(ans,pred))
        # fpr, tpr, thresholds = roc_curve(ans, pred,pos_label=1)
        # # print(fpr,tpr,thresholds)
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        # print("Threshold value is:", optimal_threshold)
        # sys.exit()
        acc=f1_score(ans_f1, pred_f1, average="binary")
        f=open("./ans_val.txt","w+")
        for i in range(len(ans_f1)):
            f.write(str(ans_f1[i]))
            f.write("\t")
            f.write(str(pred_f1[i]))
            f.write("\t")
            f.write(str(pred[i]))
            f.write("\n")

        print(" f1: ",acc)

def test_public(model=None):
    args = {
        "batch_size": 12,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "n_epoch": 30,
        "log_step": 15,
        "save_step": 250,
        "d_emb": 768,
        "p_drop": 0.1,
        "weight_decay": 0.0,
        "model_path": os.path.join("exp", "model", "1102"),
        # "log_path": os.path.join("log", "_qa", "0618"),
        # "log_path": os.path.join("log", "_qa", "try_new_qa"),
        # "data": os.path.join("./data", "public_test.txt"),
        "data": os.path.join("./data", "public_test.csv"),
    }
    e=50
    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # Model
    # model = qa_model(args["d_emb"], args["p_drop"])
    # model = model.train()
    # for e in range(1,50):
    #     print("epoch ",e," ",os.path.join(
    #                 args["model_path"], f"model-{e}-epoch.pt"))
    #     model = qa_model(args["d_emb"], args["p_drop"])
    #     model.load_state_dict(torch.load(os.path.join(
    #             args["model_path"], f"model-{e}-epoch.pt")))
    if not model:
        model = qa_model(args["d_emb"], args["p_drop"])
        model.load_state_dict(torch.load(os.path.join(
                args["model_path"], f"model-{e}-epoch.pt")))
    model = model.eval()
    model = model.to(device)
    data = dataset_public_test(args["data"])
    batch_size=args["batch_size"]
    dataldr = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False)
    
    ans = []
    pred = []
    ans_f1=[]
    ans=data.get_ans()
    tqdm_dldr = tqdm(dataldr)
    m = nn.Softmax(dim=-1)
    with torch.no_grad():
        for batch_data in tqdm_dldr:
            
            batch_document = []
            batch_document_a = batch_data["article_a"].to(device)
            batch_document_b = batch_data["article_b"].to(device)

            pred+=list(model(batch_document_a, batch_document_b))

    print(len(pred),len(ans))
    pred_f1=[]
    threshold=0.85
    for p in pred:
        if p>=threshold:
            pred_f1.append(1)
        else:
            pred_f1.append(0)
    pred_f1=torch.tensor(pred_f1)
    pred=torch.tensor(pred)
    pred=pred.detach().numpy()
    pred_f1=pred_f1.detach().numpy()
    
    # acc=f1_score(ans_f1, pred_f1, average="macro")
    f=open("./sub.csv","w+")
    f.write("Test,Reference")
    for i in range(len(ans)):
        if pred[i]>=threshold:
            f.write("\n")
            f.write(str(ans[i][0]))
            f.write(",")
            f.write(str(ans[i][1]))               
        
train()
# test()
# test_public()
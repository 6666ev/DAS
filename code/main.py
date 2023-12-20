import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.tokenizer import *
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support,classification_report
import json
import numpy as np
import os
from models import *
from utils import loader
import argparse
import time
import warnings
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

warnings.filterwarnings('ignore')


RANDOM_SEED = 2023

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(RANDOM_SEED)

name2model = {
    "TS_DAS": TS_DAS,
    "BiLSTM_DAS": BiLSTM_DAS,
}


class Trainer:
    def __init__(self, args):
        self.dataset_name = args.data_name
        self.emb_dim = 300
      
        if "cail" in self.dataset_name:
            self.tokenizer = LegalTokenizer()
            load_data = loader.load_cail_data
        elif "medical" in self.dataset_name:
            self.tokenizer = MedicalTokenizer()
            load_data = loader.load_medical_data
        else:
            print("=== no tokenizer ===")
            os._exit()

        dataset_name = os.path.join(args.data_name,"train")
        kw_path = f"data/{args.data_name}/meta/{args.kw_name}.json"
        data_path = "data/{}.json".format(dataset_name)
        print("cur dataset path: ", data_path)
        self.trainset, self.maps = load_data(data_path, dataset_name, self.tokenizer, kw_path)

        dataset_name = os.path.join(args.data_name,"valid")
        data_path = "data/{}.json".format(dataset_name)
        self.validset, self.maps = load_data(data_path, dataset_name, self.tokenizer, kw_path)

        dataset_name = os.path.join(args.data_name,"test")
        data_path = "data/{}.json".format(dataset_name)
        self.testset, self.maps = load_data(data_path, dataset_name, self.tokenizer, kw_path)

        self.args = args
        self.batch = int(args.batch_size)
        self.epoch = int(args.epoch)
        self.hid_dim = 256

        self.train_dataloader = DataLoader(dataset=self.trainset,
                                           batch_size=self.batch,
                                           shuffle=True,
                                           drop_last=False,)

        self.valid_dataloader = DataLoader(dataset=self.validset,
                                           batch_size=self.batch,
                                           shuffle=False,
                                           drop_last=False,)



        self.model = name2model[args.model_name](
            vocab_size=self.tokenizer.vocab_size, emb_dim=self.emb_dim, hid_dim=self.hid_dim, maps=self.maps, args = args)

        self.cur_time = time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        self.model_name = "{}/{}".format(args.model_name, self.dataset_name)
        self.model_save_dir = "code/logs/{}/{}".format(self.model_name, self.cur_time)
        print("model_save_dir: ", self.model_save_dir)

        self.task_name = ["cls"]
        self.sub_task_name = []

        if args.sup:
            self.sub_task_name.append("attn_sup")

        print(self.model)
        print("train samples: ", len(self.trainset))
        print("valid samples: ", len(self.validset))
        
        params = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad]
        non_bert_params = {'params': [v for k, v in params if 'bert.' not in k], 'lr': 1e-4}
        bert_params = {'params': [v for k, v in params if 'bert.' in k], 'lr': 1e-5}
        self.optimizer = torch.optim.Adam([non_bert_params, bert_params])

        self.loss_function = {
            "cls": nn.CrossEntropyLoss(),
            "attn_sup": nn.BCELoss(),
        }

        self.score_function = {
            "cls": self.f1_score_macro,
        }

        if args.load_path is not None:
            print("--- stage2 ---")
            print("load model path:", args.load_path)
            checkpoint = torch.load(args.load_path)
            
            model_load = checkpoint['model']
            load_model_dict = model_load.state_dict()
            load_model_dict = {k.replace("module.",""): v for k,v in load_model_dict.items()}
            cur_model_dict =  self.model.state_dict()
            state_dict = {k:v for k,v in load_model_dict.items() if k in cur_model_dict.keys()}
            noused_state_dict = {k:v for k,v in load_model_dict.items() if k not in cur_model_dict.keys()}
            noinit_state_dict = {k:v for k,v in cur_model_dict.items() if k not in load_model_dict.keys()}
            
            print("=== not used ===")
            print(noused_state_dict.keys())
            print("=== not init ===")
            print(noinit_state_dict.keys())

            cur_model_dict.update(state_dict)
            self.model.load_state_dict(cur_model_dict)

            self.evaluate(args.load_path, mode = "test_metric", save_result=True)


        print("parameter counts: ", self.count_parameters())

        self.model = self.model.cuda()
        print("dp: {}".format(args.dp))
        print("model_save_dir: ", self.model_save_dir)
        if args.dp:
            self.model = nn.DataParallel(self.model)

    def set_param_trainable(self, trainable):
        for name, param in self.model.named_parameters():
            param.requires_grad = trainable

    def check_param_grad(self):
        for name, parms in self.model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def precision_macro(self, y_true, y_pred):
        ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return ma_p

    def recall_macro(self, y_true, y_pred):
        ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return ma_r

    def f1_score_macro(self, y_true, y_pred):
        mi_p, mi_r, mi_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return (mi_p + mi_r + mi_f1 + ma_p + ma_r + ma_f1)/6

    def train(self):
        best_score = -1
        writer = SummaryWriter(self.model_save_dir)
        early_stop_cnt = 0
        # writer.add_graph(model=self.model)
        for e in range(self.epoch):
            # train
            self.model.train()
            print("--- train ---")
            tq = tqdm(self.train_dataloader)
            tot_loss = 0
            for data in tq:
                for name in self.task_name + self.sub_task_name:
                    data[name] = data[name].cuda()
                
                self.optimizer.zero_grad()
                out = self.model(data)
                
                argparams = {
                    "epoch": e,
                }
                loss = 0
                for name in self.task_name:
                    cur_loss = self.loss_function[name](out[name], data[name])
                    loss += cur_loss

                for name in self.sub_task_name:
                    cur_loss = self.loss_function[name](out[name], data[name])  * self.args.attn_sup_lambda
                    argparams[name] = np.around(cur_loss.detach().cpu().numpy(), 4)
                    loss += cur_loss

                tot_loss += loss.detach().cpu()

                argparams["train_loss"] = np.around(loss.detach().cpu().numpy(), 4)
                
                tq.set_postfix(**argparams)
                loss.backward()
                self.optimizer.step()
                # break

            writer.add_scalar("train loss", tot_loss, e)

            # valid
            print("--- valid ---")
            print("model_save_dir: ", self.model_save_dir)
            valid_out = self.infer(self.model, self.valid_dataloader)

            cur_score = 0
            for name in self.task_name:
                cur_task_score = self.score_function[name](valid_out[name]["true"], valid_out[name]["pred"])
                writer.add_scalar("valid {}".format(name), cur_task_score, e)
                cur_score += cur_task_score

            save_path = os.path.join(self.model_save_dir, "best_model.pt")
            if cur_score > best_score:
                best_score = cur_score
                if not os.path.exists(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                print("best model saved!")
                torch.save({"model": self.model, "optimizer": self.optimizer}, save_path)
                early_stop_cnt = 0
            early_stop_cnt += 1
            if early_stop_cnt > 20:
                break
            

    def infer(self, model, data_loader, mode = "valid"):
        meta_task_name = self.sub_task_name

        self.model.eval()
        tq = tqdm(data_loader)
        eval_out = {k: [] for k in self.task_name}
        meta_out = {k: [] for k in meta_task_name}
        for data in tq:
            with torch.no_grad():
                for name in self.task_name:
                    data[name] = data[name].cuda()
                
                out = model(data)

                for name in eval_out.keys():
                    eval_out[name].append((out[name], data[name].cuda()))
                for name in meta_out.keys():
                    meta_out[name].append((out["meta"][name], data[name].cuda()))
                if mode == "test":
                    break

        for name in eval_out.keys():
            pred = torch.cat([i[0] for i in eval_out[name]])
            true = torch.cat([i[1] for i in eval_out[name]])
            eval_out[name] = {"pred": pred, "true": true}
        
        for name in meta_out.keys():
            pred = torch.cat([i[0] for i in meta_out[name]])
            true = torch.cat([i[1] for i in meta_out[name]])
            eval_out[name] = {"pred": pred, "true": true}

        for name in self.task_name:
            print("=== {} ===".format(name))
            pred = eval_out[name]["pred"].detach().cpu().numpy()
            y_pred = pred.argmax(-1)

            y_true = eval_out[name]["true"].detach().cpu().numpy()
            eval_out[name]["pred"] = y_pred
            eval_out[name]["true"] = y_true

            output_dict = classification_report(y_true, y_pred, output_dict = True)
            i2c= self.maps["i2c"]
            cls_dict = {k: v for k, v in output_dict.items() if k.isdigit()}
            total_dict = {k: v for k, v in output_dict.items() if not k.isdigit()}
            data = []
            idx = []
            for k, v in cls_dict.items():
                p = v["precision"]
                r = v["recall"]
                f1 = v["f1-score"]
                sup = v["support"]
                data.append([p, r, f1, sup])
                idx.append(k)

            df = pd.DataFrame(data)
            df.columns = ["p","r","f1","sup"]
            df.index = idx
            df = df.sort_values(["sup"],ascending = False)

            ACC = output_dict["accuracy"]
            MaP, MaR, MaF =  df["p"].mean(), df["r"].mean(), df["f1"].mean()

            ACC = np.round(ACC* 100,3)
            MaP = np.round(MaP* 100,3)
            MaR = np.round(MaR* 100,3)
            MaF = np.round(MaF* 100,3)
            print(f"acc: {ACC}, map: {MaP}, mar: {MaR}, maf: {MaF}")
            
            t100 = df[:int(len(df) * 0.25)]["f1"].mean() 
            t75 = df[int(len(df) * 0.25):int(len(df) * 0.5)]["f1"].mean() 
            t50 = df[int(len(df) * 0.5):int(len(df) * 0.75)]["f1"].mean()
            t25 = df[int(len(df) * 0.75):]["f1"].mean() 

            t100 = np.round(t100* 100,3)
            t75 = np.round(t75* 100,3)
            t50 = np.round(t50* 100,3)
            t25 = np.round(t25* 100,3)
            print(f"t25: {t25}, t50: {t50}, t75: {t75}, t100: {t100}")

        return eval_out


    def evaluate(self, load_path, mode = "test", save_result = True):
        """
            load_path: saved model path
            save_result: True or False. save result to csv file
        """
        print("--- evaluate on testset: ---")
        testset = self.testset

        print("test samples: ", len(testset))
        test_dataloader = DataLoader(dataset=testset,
                                     batch_size=128,
                                     shuffle=False,
                                     drop_last=False)

        print("--- test ---")
        print("load model path: ", load_path)
        checkpoint = torch.load(load_path)
        model = checkpoint['model']
        # print(model)

        test_out = self.infer(model, test_dataloader, mode)
        out_save_dir = "out/{}/".format(self.model_save_dir).replace("code/logs/","")
        if save_result:
            for name in test_out:
                if not os.path.exists(out_save_dir):
                    os.makedirs(out_save_dir)
                print("=== {} result saved ===".format(name))
                with open(os.path.join(out_save_dir, name + "_msg.json"),"w") as f:
                    msg = {
                        "pred":test_out[name]["pred"].tolist()
                    }
                    if "true" in test_out[name]:
                        msg["true"] = test_out[name]["true"].tolist()
                    json.dump(msg, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='gpu')
    parser.add_argument('--model_name', default='TS_DAS', help='model_name')
    parser.add_argument('--load_path', default=None, help='load model path')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--epoch', default=100, help='batch size')
    parser.add_argument('--dp', action='store_true', help='multi gpu')
    parser.add_argument('--pretrain_word_emb', action='store_true', help='pretrain word embeddings')
    parser.add_argument('--sup', action='store_true', help='AS')
    parser.add_argument('--attn_sup_lambda', default=0.15, type=float, help='attn sup loss lambda')
    parser.add_argument('--data_name', default="filter/cail/total", help='filter/cail/total  & medical/filter/total')
    parser.add_argument('--kw_name', default="kw_task_p100", help='kw_task_p100 & keyword50_stop')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainer = Trainer(args)
    trainer.train()

    print("== test_best_model ==")
    eval_path = os.path.join(trainer.model_save_dir , "best_model.pt")

    # test metric
    trainer.evaluate(
        eval_path,
        mode = "test_metric",
        save_result=True,
    )

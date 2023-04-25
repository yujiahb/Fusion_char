# -*- coding: utf-8 -*-

'''
python Fusion_char.py
'''

import argparse
from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CanineModel
from Fun_dataload import Load_data 
from transformers import BertTokenizer, BertModel

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
    def forward(self, outputs, targets):
        return self.xent_loss(outputs, targets)

def TENSOR_PADDING(input1,input2):
    bottom1=input1.size(2)
    bottom2=input2.size(2)
    right1=input1.size(3)
    right2=input2.size(3)
    if bottom1>bottom2:
        padding1 = nn.ZeroPad2d((0, 0, 0, bottom1-bottom2)) 
        input2=padding1(input2) 
    elif bottom1<bottom2:
        padding2 = nn.ZeroPad2d((0, 0, 0, bottom2-bottom1)) 
        input1=padding2(input1)  
    if right1>right2:
        padding3 = nn.ZeroPad2d((0, right1-right2, 0, 0)) 
        input2=padding3(input2)  
    elif right1<right2:
        padding4 = nn.ZeroPad2d((0, right2-right1, 0, 0)) 
        input1=padding4(input1)  
    return input1,input2

def CLS_PADDING(cls_1,cls_2):
    right1=cls_1.size(1)
    right2=cls_2.size(1)
    if right1>right2:
        padding3 = nn.ZeroPad2d((0, right1-right2, 0, 0))  
        cls_2=padding3(cls_2)  
    elif right1<right2:
        padding4 = nn.ZeroPad2d((0, right2-right1, 0, 0)) 
        cls_1=padding4(cls_1)  
    return cls_1,cls_2

class FNN_TOKEN_CLS(nn.Module):
    def __init__(self, args, base_model, base_model2):
        super().__init__()

        self.base_model = base_model  
        self.base_model2 = base_model2  
        self.length = max(base_model.config.hidden_size,base_model2.config.hidden_size)*2
        self.args = args
        self.FC = nn.Sequential(nn.Dropout(args.dropout),
                                nn.Linear(self.length * 2, 200),
                                nn.Linear(200, 20),
                                nn.Linear(20, args.num_classes),
                                nn.Softmax(dim=1))
 
        for param1 in base_model.parameters():
            param1.requires_grad = (args.requires_grad)
        for param2 in base_model2.parameters():
            param2.requires_grad = (args.requires_grad_char)
            
    def forward(self, inputs, texts):  
        raw_outputs = self.base_model(**inputs) 
        hiddens = raw_outputs.last_hidden_state  
        cls_1 = hiddens[:, 0, :]  

        raw_outputs = self.base_model2(**texts)  
        hiddens2 = raw_outputs.last_hidden_state 
        cls_2 = hiddens2[:, 0, :]  
        
        pad_cls1,pad_cls2=CLS_PADDING(cls_1, cls_2)
        cls_out = torch.cat([pad_cls1.unsqueeze(0), pad_cls2.unsqueeze(0)],dim=2) 
        cls_fusion = cls_out.squeeze(-3) 
        
        tokens_1 = hiddens.unsqueeze(1) 
        tokens_2 = hiddens2.unsqueeze(1) 

        pad_tokens_1, pad_tokens_2 = TENSOR_PADDING(tokens_1, tokens_2) 
    
        tokens_out = torch.cat([pad_tokens_1.unsqueeze(0), pad_tokens_2.unsqueeze(0)],dim=4)  
        tokens_fusion= tokens_out.squeeze(-5) 
        trans=tokens_fusion.contiguous().view(tokens_fusion.size(0), -1)  

        dp=nn.Dropout(self.args.dropout).to(self.args.device)
        linear = nn.Linear(trans.size(1), self.length).to(self.args.device) 
 
        trans_d=linear(dp(trans)) 
 
        tokens_out = torch.cat([cls_fusion.unsqueeze(0), trans_d.unsqueeze(0)], dim=2)  
        tokens_out_s= tokens_out.squeeze(-3) 
        outputs = self.FC(tokens_out_s)  
        return outputs

 
def my_collate(batch, tokenizer,tokenizer_char):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    
    char_ids =tokenizer_char(tokens,
                             padding=True,
                             truncation="longest_first",
                             max_length=2048,
                             is_split_into_words=True,
                             add_special_tokens=True,
                             return_tensors='pt')
    
    return text_ids, char_ids, torch.tensor(label_ids)

 
class MyDataset(Dataset):
    def __init__(self, raw_data):
        dataset = list()
 
        for data in raw_data:          
            tokens = data['text'].lower().split(' ')
            label_id = int(data['label']) 
            dataset.append((tokens, label_id))
        self._dataset = dataset
    def __getitem__(self, index):
        return self._dataset[index]
    def __len__(self):
        return len(self._dataset)

 
def model_train(args, model, dataloader, criterion, optimizer):
    train_loss, tr_num_correct, num_train = 0, 0, 0
    model.train()
  
    for inputs,texts,targets in tqdm(dataloader, disable=args.backend, ascii=' >='):
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        texts = {k: v.to(args.device) for k, v in texts.items()}
        targets = targets.to(args.device)
 
        outputs = model(inputs, texts) 
        loss = criterion(outputs, targets)   
 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  
        train_loss += loss.item() * targets.size(0)
        tr_num_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        num_train += targets.size(0)
    return train_loss / num_train, tr_num_correct / num_train

 
def model_test(args,  model, dataloader, criterion):
    test_loss, te_num_correct, num_test = 0, 0, 0
    model.eval()
 
    with torch.no_grad():
 
        for inputs,texts,targets in tqdm(dataloader, disable=args.backend, ascii=' >='):
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            texts = {k: v.to(args.device) for k, v in texts.items()}
            targets = targets.to(args.device)
            outputs = model(inputs,texts)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            te_num_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            num_test += targets.size(0)
    return test_loss / num_test, te_num_correct / num_test

       
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--test_percentage', type=float, default=0.1) 
parser.add_argument('--lr', type=float, default=1e-5) 
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--decay', type=float, default=0.05)  
parser.add_argument('--backend', default=False, action='store_true')
parser.add_argument('--requires_grad', type=bool, default=True)  
parser.add_argument('--requires_grad_char', type=bool, default=False) 
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu') 
args = parser.parse_args()
args.device = torch.device(args.device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
Pretrained = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)  #768
input_size = Pretrained.config.hidden_size 

args.dataset_name='MR'
args.num_classes =2
tokenizer_char = AutoTokenizer.from_pretrained("google/canine-s")
Pretrained_char = CanineModel.from_pretrained("google/canine-s")

Mymodel = FNN_TOKEN_CLS(args, Pretrained, Pretrained_char)

if torch.cuda.device_count() > 1:  
    print(f" Use {torch.cuda.device_count()} GPU!")
    Mymodel = nn.DataParallel(Mymodel)  
Mymodel.to(args.device) 

#Load dataset
train_data, test_data=Load_data(args.dataset_name,args.test_percentage)
trainset = MyDataset(train_data)
testset = MyDataset(test_data)
collate_fn = partial(my_collate, tokenizer=tokenizer, tokenizer_char=tokenizer_char)
train_dataloader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
test_dataloader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)
criterion = CELoss()
_params = filter(lambda p: p.requires_grad, Mymodel.parameters())
optimizer = torch.optim.AdamW(_params, lr=args.lr, weight_decay=args.decay)
best_loss, best_acc = 0, 0
for epoch in range(args.num_epoch): 
    train_loss, train_acc = model_train(args, Mymodel, train_dataloader, criterion, optimizer)
    test_loss, test_acc = model_test(args, Mymodel, test_dataloader, criterion)
    if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
        best_acc, best_loss = test_acc, test_loss
    print('\n{}/{} : [train] loss:{:.4f}, acc:{:.2f}; [test] loss:{:.4f}, acc:{:.2f}'
                .format(epoch+1, args.num_epoch, train_loss, train_acc*100, test_loss, test_acc*100))
print('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc*100))
print("------------------------------------------------")
print("Pretrained:",args.model_name," Dateset:",args.dataset_name," Best Acc:",best_acc*100)

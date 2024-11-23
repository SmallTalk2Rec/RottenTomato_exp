from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from tqdm import tqdm



def MF_trainer(args, data, model, loss_fn, optimizer, train_log):

    for epoch in tqdm(range(1,args.epochs+1), desc='epochs'):

        # train
        train_loss = 0
        for user,item,fresh in data['train_dataloader']:
            
            user = user.to(args.device)
            item = item.to(args.device)
            fresh = fresh.to(args.device)

            yhat = model(user, item)

            optimizer.zero_grad()
            loss = loss_fn(yhat, fresh)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu()

        print(f'{epoch}th epoch\ntrain_loss: {train_loss/len(data["train_dataloader"])}')
        train_log.append({f'{epoch}th epoch': train_loss/len(data["train_dataloader"])})

        # validate    
        valid_loss,result,answer = 0,[],[]
        for user,item,fresh in data['valid_dataloader']:

            user = user.to(args.device)
            item = item.to(args.device)
            fresh = fresh.to(args.device)

            yhat = model(user,item)
            loss = loss_fn(yhat, fresh)
            valid_loss += loss
            yhat_bin = (yhat >= 0.5).int()

            answer += fresh.tolist()
            result += yhat_bin.tolist()

        ac = accuracy_score(answer, result)
        rec = recall_score(answer, result)
        roc = roc_auc_score(answer, result)

        print(f'valid_loss: {valid_loss/len(data["valid_dataloader"])}\naccuracy: {ac}\nrecall: {rec}\nroc: {roc}')
    
    

def MF_test(args, data, model):

    answer = []
    result = []
    for user,item,fresh in data['test_dataloader']:

        user = user.to(args.device)
        item = item.to(args.device)
        fresh = fresh.to(args.device)

        yhat = model(user,item)
        yhat_bin = (yhat >= 0.5).detach().cpu().int()

        answer += fresh.tolist()
        result += yhat_bin.tolist()

    ac = accuracy_score(answer, result)
    rec = recall_score(answer, result)
    roc = roc_auc_score(answer, result)

    print(f'---TEST SCORE---\naccuracy: {ac}\nrecall: {rec}\nroc: {roc}')
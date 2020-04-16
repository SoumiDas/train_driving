import numpy as np
import torch
import torch.nn.functional as F
import os, copy, time
import math
from tqdm import tqdm
#from ipdb import set_trace
from metrics_actual_changed import calc_metrics
root_dir = 'existing/'
data_path = 'existing/'

### helper functions

def to_np(t):
    return np.array(t.cpu()) # returning the npy array

### Losses

def calc_class_weight(x, fac=2):
    """calculate inverse normalized count, multiply by given factor"""
    _, counts = np.unique(x, return_counts=True) #calculating the number of times unique itme appear in the array
    tmp = 1/counts/sum(counts) #calculating average
    tmp /= max(tmp)
    return tmp*fac

'''def get_class_weights():
    # class weights, calculated on the training set
    df_all = pd.read_csv(data_path + 'annotations_2275.csv')
    return {
       'red_light': torch.Tensor(calc_class_weight(df_all['red_light'])),
       'hazard_stop': torch.Tensor(calc_class_weight(df_all['hazard_stop'])),
       'speed_sign': torch.Tensor(calc_class_weight(df_all['speed_sign'])),
       'relative_angle': torch.Tensor([1]),
       'center_distance': torch.Tensor([1]),
       'veh_distance': torch.Tensor([1]),
    }'''
          
WEIGHTS = {'red_light': torch.Tensor([1.2931, 2.0000]),
           'hazard_stop': torch.Tensor([0.0110, 2.0000]),
           'speed_sign': torch.Tensor([0.0039, 0.6667, 2.0000, 1.5714]),
           'relative_angle': torch.Tensor([1]),
           'center_distance': torch.Tensor([1]),
           'veh_distance': torch.Tensor([1]),
          }
         
       

#WEIGHTS = {
#    'red_light': torch.Tensor([2.0000]),
#    'hazard_stop': torch.Tensor([2.0000]),
#    'speed_sign': torch.Tensor([2.0]),
#    'relative_angle': torch.Tensor([1]),
#    'center_distance': torch.Tensor([1]),
#    'veh_distance': torch.Tensor([1]),
#}


def WCE(x, y, w):
    """weighted mean average"""
    t = F.cross_entropy(x, torch.argmax(y, dim=1), weight=w)
    return t

def MAE(x, y, w):
    return F.l1_loss(x.squeeze(), y)*w


def custom_loss(y_pred, y_true, opt, dev='cuda'):
    loss = torch.Tensor([0]).to(dev)
    lossr = torch.Tensor([0]).to(dev)
    lossc = torch.Tensor([0]).to(dev)
    lossred = torch.Tensor([0]).to(dev)
    lossh = torch.Tensor([0]).to(dev)
    losssp = torch.Tensor([0]).to(dev)
    lossv = torch.Tensor([0]).to(dev)
    
    print('Loss relative shape')
    print(lossr.size())
    
    for k in y_pred:
        '''if k!='speed_sign':
           y_pred[k]= y_pred[k].reshape(1,y_pred[k].shape[0])'''
        func = MAE if y_pred[k].shape[1]==1 else WCE
        loss += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev))
        if k == 'relative_angle':
           #lossr += func_stra(y_pred[k], y_true[k], opt, WEIGHTS[k].to(dev))
           lossr += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev))
        if k == 'center_distance':
           lossc += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev)) 
        if k == 'red_light':
           lossred += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev))
        if k == 'hazard_stop':
           lossh += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev))
        if k == 'speed_sign':
           losssp += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev))
        if k == 'veh_distance':
           lossv += func(y_pred[k], y_true[k], WEIGHTS[k].to(dev))
        

    return loss, lossr, lossc, lossred, lossh, losssp, lossv

def loss_batch(model, loss_func, preds, labels, opt=None):
    loss, lossr, lossc, lossred, lossh, losssp, lossv = loss_func(preds, labels, opt)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), lossr.item(), lossc.item(), lossred.item(), lossh.item(), losssp.item(), lossv.item()

### training wrappers

def train(model, data_loader, loss_func, opt):
    device = next(model.parameters()).device
    train_loss = 0
    loss_rel = 0
    loss_cent = 0
    loss_red = 0
    loss_haz = 0
    loss_sp = 0
    loss_veh = 0
    imp = []
    
    cnt = 0
    for inputs, labels in tqdm(data_loader):
        print('Image')
        print(cnt+1)
        inputs['sequence'] = inputs['sequence'].to(device)
        inputs['gametimeStamp'] = inputs['gametimeStamp'].to(device) #change
        #inputs = inputs.unsqueeze(0)
        print(inputs['sequence'].shape)
          
        preds = model(inputs)
        labels = {k: v.to(device) for k,v in labels.items()}
        l,lr,lc, lrd, lh, lsp, lv = loss_batch(model, loss_func, preds=preds, labels=labels, opt=opt)
        '''train_loss += l/data_loader.batch_size
        loss_rel += lr/data_loader.batch_size
        loss_cent += lc/data_loader.batch_size'''
        train_loss += l
        loss_rel += lr
        loss_cent += lc
        loss_red += lrd
        loss_haz += lh
        loss_sp += lsp
        loss_veh += lv
        
        with open('train_minibatchLoss_uniform.txt','a') as fpmb:
            fpmb.write(str(l))
            fpmb.write('\n')
        
    train_loss /= len(data_loader)
    loss_rel /= len(data_loader)
    loss_cent /= len(data_loader)
    loss_red /= len(data_loader)
    loss_haz /= len(data_loader)
    loss_sp /= len(data_loader)
    loss_veh /= len(data_loader)
    
    '''print('Length of training dataloader:')
    print(str(len(data_loader)))'''
    
    with open('train_Loss_uniform.txt','a') as fpt:
    	fpt.write(str(train_loss))
    	fpt.write('\n')
    with open('train_reLoss_uniform.txt','a') as fpt:
    	fpt.write(str(loss_rel))
    	fpt.write('\n')
    with open('train_cenLoss_uniform.txt','a') as fpt:
    	fpt.write(str(loss_cent))
    	fpt.write('\n')
    with open('train_redLoss_uniform.txt','a') as fpt:
    	fpt.write(str(loss_red))
    	fpt.write('\n')
    with open('train_hazLoss_uniform.txt','a') as fpt:
    	fpt.write(str(loss_haz))
    	fpt.write('\n')
    with open('train_speedLoss_uniform.txt','a') as fpt:
    	fpt.write(str(loss_sp))
    	fpt.write('\n')
    with open('train_vehLoss_uniform.txt','a') as fpt:
    	fpt.write(str(loss_veh))
    	fpt.write('\n')
    	
    return model

def validate(model, data_loader, loss_func):
    device = next(model.parameters()).device
    all_preds, all_labels = {}, {}

    imp = []
    
    with torch.no_grad():
        val_loss = 0
        rel_loss = 0
        cent_loss = 0
        red_loss = 0
        haz_loss = 0
        speed_loss = 0
        veh_loss = 0
        
        for inputs, labels in tqdm(data_loader):
            inputs['sequence'] = inputs['sequence'].to(device)
            inputs['gametimeStamp'] = inputs['gametimeStamp'].to(device) #change
            
            
            preds = model(inputs)
            labels = {k: v.to(device) for k,v in labels.items()}
            l,lr,lc, lrd, lh, lsp, lv = loss_batch(model, loss_func, preds=preds, labels=labels)
            '''val_loss += l/data_loader.batch_size
            rel_loss += lr/data_loader.batch_size
            cent_loss += lc/data_loader.batch_size'''
            
            val_loss += l
            rel_loss += lr
            cent_loss += lc
            red_loss += lrd
            haz_loss += lh
            speed_loss += lsp
            veh_loss += lv
            
            with open('val_minibatchLoss_uniform.txt','a') as fpmbt:
                fpmbt.write(str(l))
                fpmbt.write('\n')

            # concatenate for inspection
            if not all_preds:
                all_preds = {k: to_np(v) for k,v in preds.items()}
                all_labels = {k: to_np(v) for k,v in labels.items()}
            else:
                all_preds = {k: np.concatenate([all_preds[k], to_np(v)]) for k,v in preds.items()}
                all_labels = {k: np.concatenate([all_labels[k], to_np(v)]) for k,v in labels.items()}

        val_loss /= len(data_loader)
        print(val_loss)
        rel_loss /= len(data_loader)
        print(rel_loss)
        cent_loss /= len(data_loader)
        print(cent_loss)
        red_loss /= len(data_loader)
        print(red_loss)
        haz_loss /= len(data_loader)
        print(haz_loss)
        speed_loss /= len(data_loader)
        print(speed_loss)
        veh_loss /= len(data_loader)
        print(veh_loss)
        
        '''print('Length of validation dataloader:')
        print(str(len(data_loader)))'''
        
        with open('val_relLoss_uniform.txt','a') as fp:
        	fp.write(str(rel_loss))
        	fp.write('\n')
        	
        with open('val_centLoss_uniform.txt','a') as fp:
        	fp.write(str(cent_loss))
        	fp.write('\n')
        	
        with open('val_redLoss_uniform.txt','a') as fp:
        	fp.write(str(red_loss))
        	fp.write('\n')
        	
        with open('val_hazLoss_uniform.txt','a') as fp:
        	fp.write(str(haz_loss))
        	fp.write('\n')
        	
        with open('val_speedLoss_uniform.txt','a') as fp:
        	fp.write(str(speed_loss))
        	fp.write('\n')
        	
        with open('val_vehLoss_uniform.txt','a') as fp:
        	fp.write(str(veh_loss))
        	fp.write('\n')
        	        	
        print('\n')
        
        modified_loss = (red_loss + haz_loss + speed_loss + veh_loss)*math.pow(10,-3) + rel_loss + cent_loss
        #a = calc_metrics(all_preds, all_labels)


    return val_loss, modified_loss, all_preds, all_labels#, a
    
    
def validatetr(model, data_loader, loss_func):
    device = next(model.parameters()).device
    all_preds, all_labels = {}, {}

    imp = []
    
    with torch.no_grad():
        val_loss = 0
        rel_loss = 0
        cent_loss = 0
        red_loss = 0
        haz_loss = 0
        speed_loss = 0
        veh_loss = 0
        
        for inputs, labels in tqdm(data_loader):
            inputs['sequence'] = inputs['sequence'].to(device)
            inputs['gametimeStamp'] = inputs['gametimeStamp'].to(device) #change
            
            
            preds = model(inputs)
            labels = {k: v.to(device) for k,v in labels.items()}
            l,lr,lc, lrd, lh, lsp, lv = loss_batch(model, loss_func, preds=preds, labels=labels)
            '''val_loss += l/data_loader.batch_size
            rel_loss += lr/data_loader.batch_size
            cent_loss += lc/data_loader.batch_size'''
            
            val_loss += l
            rel_loss += lr
            cent_loss += lc
            red_loss += lrd
            haz_loss += lh
            speed_loss += lsp
            veh_loss += lv
            
            with open('valtrain_minibatchLoss_uniform.txt','a') as fpmbt:
                fpmbt.write(str(l))
                fpmbt.write('\n')

            # concatenate for inspection
            if not all_preds:
                all_preds = {k: to_np(v) for k,v in preds.items()}
                all_labels = {k: to_np(v) for k,v in labels.items()}
            else:
                all_preds = {k: np.concatenate([all_preds[k], to_np(v)]) for k,v in preds.items()}
                all_labels = {k: np.concatenate([all_labels[k], to_np(v)]) for k,v in labels.items()}

        val_loss /= len(data_loader)
        print(val_loss)
        rel_loss /= len(data_loader)
        print(rel_loss)
        cent_loss /= len(data_loader)
        print(cent_loss)
        red_loss /= len(data_loader)
        print(red_loss)
        haz_loss /= len(data_loader)
        print(haz_loss)
        speed_loss /= len(data_loader)
        print(speed_loss)
        veh_loss /= len(data_loader)
        print(veh_loss)
        
        '''print('Length of validation dataloader:')
        print(str(len(data_loader)))'''
        
        with open('valtrain_relLoss_uniform.txt','a') as fp:
        	fp.write(str(rel_loss))
        	fp.write('\n')
        	
        with open('valtrain_centLoss_uniform.txt','a') as fp:
        	fp.write(str(cent_loss))
        	fp.write('\n')
        	
        with open('valtrain_redLoss_uniform.txt','a') as fp:
        	fp.write(str(red_loss))
        	fp.write('\n')
        	
        with open('valtrain_hazLoss_uniform.txt','a') as fp:
        	fp.write(str(haz_loss))
        	fp.write('\n')
        	
        with open('valtrain_speedLoss_uniform.txt','a') as fp:
        	fp.write(str(speed_loss))
        	fp.write('\n')
        	
        with open('valtrain_vehLoss_uniform.txt','a') as fp:
        	fp.write(str(veh_loss))
        	fp.write('\n')
        	        	
        print('\n')
        
        modified_loss = (red_loss + haz_loss + speed_loss + veh_loss)*math.pow(10,-3) + rel_loss + cent_loss
        #a = calc_metrics(all_preds, all_labels)


    return val_loss, modified_loss, all_preds, all_labels
    

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, dev='cuda', val_hist=None, val_hist_total=None, maper=None):
    since = time.time()

    val_hist = [] if val_hist is None else val_hist
    val_hist_total = [] if val_hist_total is None else val_hist_total
    #train_hist = [] if train_hist is None else train_hist
    maper = [] if maper is None else maper
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf if not val_hist else min(val_hist)
    best_val_loss_total = np.inf if not val_hist_total else min(val_hist_total)
    best_rel_loss = np.inf if not maper else min(maper)
    
    flag = 0
    
    #a = []
    e = 0
    for epoch in range(epochs):
        model.train()
        
        with open('train_minibatchLoss_uniform.txt','a') as fpmb:
            fpmb.write('Epoch---------------')
            fpmb.write(str(epoch))
            fpmb.write('==================================================')
            fpmb.write('\n')
    		
        model = train(model, train_dl, loss_func, opt)
        torch.cuda.empty_cache()
        model.eval()
        
        tr_loss, modified_trloss, trpreds, trlabels = validatetr(model, train_dl, loss_func)

        val_loss, modified_loss, preds, labels = validate(model, valid_dl, loss_func)
        
        with open('val_minibatchLoss_uniform.txt','a') as fpmbv:
            fpmbv.write('Epoch---------------')
            fpmbv.write(str(epoch))
            fpmbv.write('==================================================')
            fpmbv.write('\n')
        print("\n")
        val_hist_total.append(val_loss)
        val_hist.append(modified_loss)
        with open('val_Loss_uniform.txt','a') as fp:
        	fp.write(str(val_loss))
        	fp.write('\n')
        	
        with open('val_modifiedLoss_uniform.txt','a') as fp:
        	fp.write(str(modified_loss))
        	fp.write('\n')
        	
        scorestr, maetr,mapetr,_ ,mapef1tr= calc_metrics(trpreds,trlabels)
        
        with open('train_ra_mae_uniform.txt','a') as fp:
        	fp.write(str(maetr)+','+str(mapetr)+','+str(mapef1tr))
        	fp.write('\n')
        with open('train_disc_uniform.txt','a') as fp:
        	fp.write(str(scorestr))
        	fp.write('\n')
        
        '''train_loss, preds_train, labels_train = validate(model, train_dl, loss_func)
        print("\n")
        train_hist.append(train_loss)'''
        
        scores, mae,mape,_ ,mapef1= calc_metrics(preds,labels)
        print('MAE of relative angle')
        print(mae['relative_angle_mean_MAE'])
        print("\n")
        print('MAPE of relative angle')
        print(mape['relative_angle_mean_MAPE'])
        print("\n")
        
        with open('val_ra_mae_uniform.txt','a') as fp:
        	fp.write(str(mae)+','+str(mape)+','+str(mapef1))
        	fp.write('\n')
        with open('val_disc_uniform.txt','a') as fp:
        	fp.write(str(scores))
        	fp.write('\n')
        
        mapercurrent = mape['relative_angle_mean_MAPE']
        maper.append(mapercurrent)
        
        '''_,mae_train,mape_train,_ = calc_metrics(preds_train,labels_train)
        print('MAE of relative angle')
        print(mae_train['relative_angle_mean_MAE'])
        print("\n")
        print('MAPE of relative angle')
        print(mape_train['relative_angle_mean_MAPE'])
        print("\n")
        print("No. of epoch is completed: ",(epoch+1),"\n")'''
        
        #a.append(model)
        #e = e + 1

        if modified_loss < best_val_loss:
            best_val_loss = modified_loss
            PATH = f"./models/{model.params.name}.pth"
            torch.save(model.state_dict(), PATH)
            #best_model_wts = copy.deepcopy(model.state_dict())
            print("The best val loss is decreased in this epoch", "\n")
            
            flag = epoch+1
            
        if val_loss < best_val_loss_total:
            best_val_loss_total = val_loss
            PATH = f"./total_models/{model.params.name}.pth"
            torch.save(model.state_dict(), PATH)
            best_model_wts = copy.deepcopy(model.state_dict())
            print("The best val loss is decreased in this epoch", "\n")
            
            flag = epoch+1
            
        '''if mapercurrent < best_rel_loss:
            best_rel_loss = mapercurrent
            PATH = f"./rel_models/{model.params.name}.pth"
            torch.save(model.state_dict(), PATH)
            #best_model_wts1 = copy.deepcopy(model.state_dict())
            #print("The best val loss is decreased in this epoch", "\n")
            
            flag = epoch+1'''

        print("The best val loss is till now: ", best_val_loss, "\n")
        #print(scores)
        print("\n")


        #print(a)
        #print(e)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_val_loss))
    print('Best val loss achieved in the epoch: ', flag)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_hist

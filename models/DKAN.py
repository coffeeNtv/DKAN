import os 
import inspect
import importlib

import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.nn.functional as F
from einops import rearrange
from datetime import datetime

from models.resnet_custom import resnet50_baseline

from models.module import GlobalEncoder, NeighborEncoder, FusionEncoder, CustomCrossEncoder, ExpressionEncoder, TransformerEncoder


def load_model_weights(path: str):       
    """Load pretrained ResNet18 model without final fc layer.

    Args:
        path (str): path_for_pretrained_weight

    Returns:
        torchvision.models.resnet.ResNet: ResNet model with pretrained weight
    """
    
    resnet = torchvision.models.__dict__['resnet18'](weights=None)
    
    ckpt_dir = './weights'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f'{ckpt_dir}/tenpercent_resnet18.ckpt'
    
    # prepare the checkpoint
    if not os.path.exists(ckpt_path):
        ckpt_url = 'https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
        wget.download(ckpt_url, out=ckpt_dir)
        
    state = torch.load(path)
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    
    model_dict = resnet.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    if state_dict == {}:
        print('No weight could be loaded..')
    model_dict.update(state_dict)
    resnet.load_state_dict(model_dict)
    resnet.fc = nn.Identity()

    return resnet


class DKAN(pl.LightningModule):
    """Model class for DKAN
    """
    def __init__(self, 
                 target_encoder_name=None,
                 num_genes=250,
                 emb_dim_target=1024,
                 emb_dim=1024,
                 emb_dim_text=1024,
                 depth1=2,
                 depth2=2,
                 depth3=2,
                 depth_img=3,
                 depth_exp=3,
                 depth_text=3,
                 num_heads1=8,
                 num_heads2=8,
                 num_heads3=8,
                 num_heads_img=8,
                 num_heads_exp=8,
                 num_heads_text=8,
                 mlp_ratio1=2.0,
                 mlp_ratio2=2.0,
                 mlp_ratio3=2.0,
                 mlp_ratio_img=2.0,
                 mlp_ratio_exp=2.0,
                 mlp_ratio_text=3.0,
                 dropout1=0.1,
                 dropout2=0.1,
                 dropout3=0.1,
                 dropout_img=0.1,
                 dropout_exp=0.1,
                 dropout_text=0.1,
                 dropout_linear_exp=0.1,
                 temperature = 0.1,
                 kernel_size=3,
                 res_neighbor=(5,5),
                 learning_rate=0.0001):
        
        super().__init__()
        
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        # Initialize best metrics
        self.best_loss = np.inf
        self.best_cor = -1
        
        self.num_genes = num_genes
        self.alpha = 0.3
        self.w_sup = 2.0
        self.w_cont = 0.5
        self.num_n = res_neighbor[0]
        self.temperature = temperature
        
        self.validation_outputs = []
        self.test_outputs = []
    
        self.timestamp = datetime.now().strftime('%Y-%m-%d')

        assert target_encoder_name in ["resnet18", "resnet50"]
        self.target_encoder_name = target_encoder_name

        # Target Encoder
        if target_encoder_name == "resnet18":
            resnet18 = load_model_weights("weights/tenpercent_resnet18.ckpt")
            module = list(resnet18.children())[:-2]
        elif target_encoder_name == "resnet50":
            resnet50 = resnet50_baseline(True, "./weights")
            module = list(resnet50.children())[:-1]
        self.target_encoder = nn.Sequential(*module)
        
        if emb_dim_target != emb_dim:
            self.target_transform = nn.Linear(emb_dim_target, emb_dim)

        self.fc_target = nn.Linear(emb_dim, num_genes)

            # Neighbor Encoder
        self.neighbor_encoder = NeighborEncoder(emb_dim, depth3, num_heads3, int(emb_dim*mlp_ratio3), dropout=dropout3, resolution=res_neighbor)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)

            # Global Encoder        
        self.global_encoder = GlobalEncoder(emb_dim, depth2, num_heads2, int(emb_dim*mlp_ratio2), dropout2, kernel_size)
        self.fc_global = nn.Linear(emb_dim, num_genes)
    
            # Fusion Layer
        self.fusion_encoder = FusionEncoder(emb_dim, depth1, num_heads1, int(emb_dim*mlp_ratio1), dropout1)    
        self.fc = nn.Linear(emb_dim, num_genes)

            # Linear Projector for expression
        self.exp_encoder = ExpressionEncoder(num_genes, emb_dim, dropout_linear_exp)

            # Fusion Layer for image and text
        self.img_fusion_encoder = CustomCrossEncoder(emb_dim, depth_img, num_heads_img, int(emb_dim*mlp_ratio_img), dropout_img)

            # Fusion Layer for expression and text
        self.exp_fusion_encoder = CustomCrossEncoder(emb_dim, depth_exp, num_heads_exp, int(emb_dim*mlp_ratio_exp), dropout_exp)

            # Text encoder
        self.text_encoder = TransformerEncoder(emb_dim_text, depth_text, num_heads_text, int(emb_dim*mlp_ratio_text), dropout_text)

            # Text Linear Layer if need
        if emb_dim_text != emb_dim:
            self.text_fc = nn.Linear(emb_dim_text, emb_dim)

            # Final Prediction Layer for img_fusion_emb
        self.fc_final_img = nn.Linear(emb_dim, 1)

    
    def forward(self, x, x_total, position, neighbor, mask, pid=None, sid=None):
        """
        Args:
            x (torch.Tensor): Target spot image (batch_size x 3 x 224 x 224)
            x_total (list): Extracted features of all the spot images in the patient. (batch_size * (num_spot x 512))
            position (list): Relative position coordinates of all the spots. (batch_size * (num_spot x 2))
            neighbor (torch.Tensor): Neighbor spot features. (batch_size x num_neighbor x 512)
            mask (torch.Tensor): Masking table for neighbor spot. (batch_size x num_neighbor)
            pid (torch.LongTensor, optional): Patient index. Defaults to None. (batch_size x 1)
            sid (torch.LongTensor, optional): Spot index of the patient. Defaults to None. (batch_size x 1)

        Returns:
            tuple:
                out: Prediction of fused feature
                out_target: Prediction of TEM
                out_neighbor: Prediction of NEM
                out_global: Prediction of GEM
        """
        
        # Target tokens
        target_token = self.target_encoder(x) # B x 512 x 7 x 7
        _, dim, w, h = target_token.shape
        target_token = rearrange(target_token, 'b d h w -> b (h w) d', d=dim, w=w, h=h)

        # Transform target embedding if need
        if hasattr(self, 'target_transform'):
            target_token = self.target_transform(target_token)
    
        # Neighbor tokens
        neighbor_token = self.neighbor_encoder(neighbor, mask) # B x 26 x 512
        
        # Global tokens
        if pid == None:
            global_token = self.global_encoder(x_total, position.squeeze()).squeeze()  # N x 512
            if sid != None:
                global_token = global_token[sid]
        else:
            pid = pid.view(-1)
            sid = sid.view(-1)
            global_token = torch.zeros((len(x_total), x_total[0].shape[1])).to(x.device)
            
            pid_unique = pid.unique()
            for pu in pid_unique:
                ind = int(torch.argmax((pid == pu).int()))
                x_g = x_total[ind].unsqueeze(0) # 1 x N x 512
                pos = position[ind]
                
                emb = self.global_encoder(x_g, pos).squeeze() 
                global_token[pid == pu] = emb[sid[pid == pu]].float()
    
        # Fusion tokens
        img_fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token, mask=mask) # B x 512
            
        final_img_output = self.fc(img_fusion_token) # B x num_genes
        output_target = self.fc_target(target_token.mean(1)) # B x num_genes
        output_neighbor = self.fc_neighbor(neighbor_token.mean(1)) # B x num_genes
        output_global = self.fc_global(global_token) # B x num_genes

        return final_img_output, output_target, output_neighbor, output_global, img_fusion_token

    def inference_forward(self, x, x_total, position, neighbor, mask, text, pid=None, sid=None):
        """Forward pass for inference  (validation, test, predict)"""

        final_img_output, output_target, output_neighbor, output_global, img_fusion_token = \
            self.forward(x, x_total, position, neighbor, mask, pid, sid)

        text = self.text_encoder(text)

        if hasattr(self, 'text_fc'):
            text = self.text_fc(text)
        
        img_fusion_emb = self.img_fusion_encoder(text, img_fusion_token)

        final_ouput = self.fc_final_img(img_fusion_emb).squeeze(-1)

        return final_ouput, img_fusion_emb, final_img_output, output_target, output_neighbor, output_global
     
    def training_forward(self, exp, x, x_total, position, neighbor, mask, text, pid=None, sid=None):
        """Forward pass for training

        Args:
            x (torch.Tensor): Target spot image (batch_size x 3 x 224 x 224)
            x_total (list): Extracted features of all the spot images in the patient. (batch_size * (num_spot x 512))
            position (list): Relative position coordinates of all the spots. (batch_size * (num_spot x 2))
            neighbor (torch.Tensor): Neighbor spot features. (batch_size x num_neighbor x 512)
            mask (torch.Tensor): Masking table for neighbor spot. (batch_size x num_neighbor)
            pid (torch.LongTensor, optional): Patient index. Defaults to None. (batch_size x 1)
            sid (torch.LongTensor, optional): Spot index of the patient. Defaults to None. (batch_size x 1)

        Returns:
            tuple:
                out: Prediction of fused feature
                out_target: Prediction of TEM
                out_neighbor: Prediction of NEM
                out_global: Prediction of GEM
        """
        final_ouput, img_fusion_emb, final_img_output, output_target, output_neighbor, output_global = \
            self.inference_forward(x, x_total, position, neighbor, mask, text, pid, sid)
        
        exp_feature = self.exp_encoder(exp)     # batch_size x emb_dim

        text = self.text_encoder(text)

        if hasattr(self, 'text_fc'):
            text = self.text_fc(text)

        exp_fusion_emb = self.exp_fusion_encoder(text, exp_feature)


        return final_img_output, output_target, output_neighbor, output_global, img_fusion_emb, exp_fusion_emb, final_ouput
        
    def training_step(self, batch, batch_idx):
        """Train the model. Transfer knowledge from fusion to each module.
        """
        patch, exp, pid, sid, wsi, position, neighbor, mask, text = batch     # text_shape: torch.Size([128, 250, 512])
        
        final_img_output, output_target, output_neighbor, output_global, img_fusion_emb, exp_fusion_emb, final_ouput\
             = self.training_forward(exp, patch, wsi, position, neighbor, mask, text, pid, sid)
        
        total_loss, sup_loss, cont_loss = self.compute_loss(final_ouput, final_img_output, output_target, output_neighbor, 
                                                           output_global, img_fusion_emb, exp_fusion_emb, exp, temperature = self.temperature, 
                                                           alpha=self.alpha, w_sup=self.w_sup, w_cont=self.w_cont)
            
        self.log('train_total_loss', total_loss, sync_dist=True)
        self.log('train_sup_loss', sup_loss, sync_dist=True)
        self.log('train_cont_loss', cont_loss, sync_dist=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        patch, exp, _, wsi, position, name, neighbor, mask, text = batch
        # patch, exp, neighbor, mask, text = patch.squeeze(), exp.squeeze(), neighbor.squeeze(), mask.squeeze(), text.squeeze()
        patch, exp, neighbor, mask = patch.squeeze(), exp.squeeze(), neighbor.squeeze(), mask.squeeze()

        final_ouput, img_fusion_emb, final_img_output, output_target, output_neighbor, output_global\
            = self.inference_forward(patch, wsi, position, neighbor, mask, text)
        pred = final_ouput
        loss = F.mse_loss(pred.view_as(exp), exp)

        pred = pred.cpu().numpy().T
        exp = exp.cpu().numpy().T
        r = []
        for g in range(self.num_genes):
            r.append(pearsonr(pred[g], exp[g])[0])
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        self.validation_outputs.append({"val_loss": loss, "corr": rr})
    
    def on_validation_epoch_end(self):
        """Handle validation epoch end by processing saved outputs.
        """
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_outputs]).mean()
        
        avg_corr = torch.stack(
            [x["corr"] for x in self.validation_outputs])
        
        os.makedirs(f"results/{self.__class__.__name__}/{self.data}/{self.timestamp}", exist_ok=True)
        if self.best_cor < avg_corr.mean():
            torch.save(avg_corr.cpu(), f"results/{self.__class__.__name__}/{self.data}/{self.timestamp}/R_{self.patient}")
            torch.save(avg_loss.cpu(), f"results/{self.__class__.__name__}/{self.data}/{self.timestamp}/loss_{self.patient}")
            
            self.best_cor = avg_corr.mean()
            self.best_loss = avg_loss

        self.log('valid_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('R', avg_corr.nanmean(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # clear list for next epoch
        self.validation_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Testing the model in a sample. Calculate MSE, MAE and PCC for all spots in the sample.
        """
        patch, exp, sid, wsi, position, name, neighbor, mask, text = batch
        patch, exp, sid, neighbor, mask = patch.squeeze(), exp.squeeze(), sid.squeeze(), neighbor.squeeze(), mask.squeeze()
        
        if '10x_breast' in name[0]:
            wsi = wsi[0].unsqueeze(0)
            position = position[0]
            
            patches = patch.split(512, dim=0)
            neighbors = neighbor.split(512, dim=0)
            masks = mask.split(512, dim=0)
            sids = sid.split(512, dim=0)
            
            pred = []
            for patch, neighbor, mask, sid, text in zip(patches, neighbors, masks, sids, text):
                outputs = self.inference_forward(patch, wsi, position, neighbor, mask, text, sid=sid)
                p = outputs[0]
                pred.append(p)
                
            pred = torch.cat(pred, axis=0)
            
            ind_match = np.load(f'/data/temp/spatial/TRIPLEX/data/test/{name[0]}/ind_match.npy', allow_pickle=True)
            self.num_genes = len(ind_match)
            pred = pred[:, ind_match]
            
        else:        
            outputs = self.inference_forward(patch, wsi, position, neighbor, mask, text)
            pred = outputs[0]
            
        mse = F.mse_loss(pred.view_as(exp), exp)
        mae = F.l1_loss(pred.view_as(exp), exp)
        
        pred = pred.cpu().numpy().T
        exp = exp.cpu().numpy().T
        
        r = []
        for g in range(self.num_genes):
            r.append(pearsonr(pred[g], exp[g])[0])
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        
        os.makedirs(f"final/{self.__class__.__name__}/{self.data}/{self.timestamp}/{self.patient}", exist_ok=True)
        np.save(f"final/{self.__class__.__name__}/{self.data}/{self.timestamp}/{self.patient}/{name[0]}", pred.T)
        np.save(f"final/{self.__class__.__name__}/{self.data}/{self.timestamp}/{self.patient}/{name[0]}_exp", exp.T)
        
        self.test_outputs.append({"MSE": mse, "MAE": mae, "corr": rr})
    
    def on_test_epoch_end(self):
        """Handle test epoch end by processing saved outputs.
        """
        avg_mse = torch.stack(
            [x["MSE"] for x in self.test_outputs]).nanmean()

        avg_mae = torch.stack(
            [x["MAE"] for x in self.test_outputs]).nanmean()

        avg_corr = torch.stack(
            [x["corr"] for x in self.test_outputs]).nanmean(0)

        os.makedirs(f"final/{self.__class__.__name__}/{self.data}/{self.timestamp}/{self.patient}", exist_ok=True)
        torch.save(avg_mse.cpu(), f"final/{self.__class__.__name__}/{self.data}/{self.timestamp}/{self.patient}/MSE")
        torch.save(avg_mae.cpu(), f"final/{self.__class__.__name__}/{self.data}/{self.timestamp}/{self.patient}/MAE")
        torch.save(avg_corr.cpu(), f"final/{self.__class__.__name__}/{self.data}/{self.timestamp}/{self.patient}/cor")
        
        self.test_outputs.clear()
        
    def predict_step(self, batch, batch_idx):
        """Predict step for inference.
        """
        patches, sids, wsi, position, neighbors, masks, text = batch
        patches, sids, neighbors, masks, text = patches.squeeze(), sids.squeeze(), neighbors.squeeze(), masks.squeeze()

        patches = patches.split(512, dim=0)
        neighbors = neighbors.split(512, dim=0)
        masks = masks.split(512, dim=0)
        sids = sids.split(512, dim=0)
        
        preds = []
        for patch, neighbor, mask, sid, text in zip(patches, neighbors, masks, sids, text):
            outputs = self.inference_forward(patch, wsi, position, neighbor, mask, text, sid=sid)
            pred = outputs[0].cpu()
            
            preds.append(pred)
            
        preds = torch.cat(preds, axis=0)
        
        return preds
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler.
        """
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
    
    def get_meta(self, name):
        """Extract metadata from sample name.
        """
        if '10x_breast' in name[0]:
            self.patient = name[0]
            self.data = "test"
        else:
            name = name[0]
            self.data = name.split("+")[1]
            self.patient = name.split("+")[0]
            
            if self.data == 'her2st':
                self.patient = self.patient[0]
            elif self.data == 'stnet':
                self.data = "stnet"
                patient = self.patient.split('_')[0]
                if patient in ['BC23277', 'BC23287', 'BC23508']:
                    self.patient = 'BC1'
                elif patient in ['BC24105', 'BC24220', 'BC24223']:
                    self.patient = 'BC2'
                elif patient in ['BC23803', 'BC23377', 'BC23895']:
                    self.patient = 'BC3'
                elif patient in ['BC23272', 'BC23288', 'BC23903']:
                    self.patient = 'BC4'
                elif patient in ['BC23270', 'BC23268', 'BC23567']:
                    self.patient = 'BC5'
                elif patient in ['BC23269', 'BC23810', 'BC23901']:
                    self.patient = 'BC6'
                elif patient in ['BC23209', 'BC23450', 'BC23506']:
                    self.patient = 'BC7'
                elif patient in ['BC23944', 'BC24044']:
                    self.patient = 'BC8'
            elif self.data == 'skin':
                self.patient = self.patient.split('_')[0]
    
    def load_model(self):
        """Load model dynamically based on configuration.
        """
        name = self.hparams.MODEL.name
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """Instantiate a model using the corresponding parameters from self.hparams dictionary.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.MODEL.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.MODEL, arg)
        args1.update(other_args)
        return Model(**args1)

    def nt_xent_loss(self, img_fusion_emb, exp_fusion_emb, temperature=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        batch_size = img_fusion_emb.size(0)
        num_genes = img_fusion_emb.size(1)
        embedding_dim = img_fusion_emb.size(2)

        # Flatten the embeddings and apply L2 normalization
        img_fusion_flat = img_fusion_emb.view(batch_size * num_genes, embedding_dim)  # [32000, 512]
        exp_fusion_flat = exp_fusion_emb.view(batch_size * num_genes, embedding_dim)  # [32000, 512]
        img_fusion_norm = F.normalize(img_fusion_flat, dim=1, p=2)
        exp_fusion_norm = F.normalize(exp_fusion_flat, dim=1, p=2)

        # Compute similarity matrix
        similarity = img_fusion_norm @ exp_fusion_norm.T / temperature  # [32000, 32000]

        # Generate labels: each sample's positive pair is itself
        labels = torch.arange(batch_size * num_genes, device=device)  # [0, 1, ..., 31999], length = 32000

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels, reduction='mean')
        return loss

    def compute_loss(self, final_ouput, final_img_output, output_target, 
                    output_neighbor, output_global, img_fusion_emb, exp_fusion_emb, exp,
                    temperature=0.1, alpha=0.3, w_sup=1.0, w_cont=1.0):
        # 1. Supervised Loss
        sup_loss = 0.0
        sup_loss += F.mse_loss(final_ouput, exp)
        sup_loss += F.mse_loss(final_img_output, exp) * (1 - alpha)
        sup_loss += F.mse_loss(final_img_output, final_ouput) * alpha
        sup_loss += F.mse_loss(output_target, exp) * (1 - alpha)
        sup_loss += F.mse_loss(output_target, final_ouput) * alpha
        sup_loss += F.mse_loss(output_neighbor, exp) * (1 - alpha)
        sup_loss += F.mse_loss(output_neighbor, final_ouput) * alpha
        sup_loss += F.mse_loss(output_global, exp) * (1 - alpha)
        sup_loss += F.mse_loss(output_global, final_ouput) * alpha

        # 2. Contrastive Loss (NT-Xent Loss)
        cont_loss = self.nt_xent_loss(img_fusion_emb, exp_fusion_emb, temperature)

        # 3. Dynamic Weights
        if sup_loss.item() > 0 and cont_loss.item() > 0:
            w_sup_raw = 1.0 / sup_loss.item()
            w_cont_raw = 1.0 / cont_loss.item()
            total_weight = w_sup_raw + w_cont_raw
            w_sup = w_sup_raw / total_weight      #
            w_cont = w_cont_raw / total_weight *3.0   # could enlarge to 3.0
            w_sum = w_sup + w_cont
            w_sup /= w_sum
            w_cont /= w_sum
        else:
            w_sup, w_cont = 1.0, 1.0

        # 4. Total Loss: Weighted combination of supervised and contrastive losses
        total_loss = w_sup * sup_loss + w_cont * cont_loss

        self.log('w_sup', w_sup, on_step=True, sync_dist=True)
        self.log('w_cont', w_cont, on_step=True, sync_dist=True)

        return total_loss, sup_loss, cont_loss

class CustomWriter(BasePredictionWriter):
    def __init__(self, pred_dir, write_interval, emb_dir=None, names=None):
        super().__init__(write_interval)
        self.pred_dir = pred_dir
        self.emb_dir = emb_dir
        self.names = names

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        for i, batch in enumerate(batch_indices[0]):
            torch.save(predictions[0][i][0], os.path.join(self.pred_dir, f"{self.names[i]}.pt"))
            torch.save(predictions[0][i][1], os.path.join(self.emb_dir, f"{self.names[i]}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        # torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))
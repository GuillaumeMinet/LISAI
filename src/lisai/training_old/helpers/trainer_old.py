import torch
import os
from tifffile import imsave,imread
import logging
import numpy as np
import time


import sys,os
sys.path.append(os.getcwd())
from lisai.lib.utils.logger_utils import EnableFilter,CustomStreamHandler # import when run as __main__
from lisai.lib.hdn.forwardpass import forward_pass as lvae_forward_pass
from lisai.lib.hdn.forwardpass import forward_pass_tiling as lvae_forward_pass_tiling
from lisai.training.helpers import misc,loss

from lisai.config_project import CFG as config_project
_loss_file_name = config_project.get("loss_file_name")
_log_file_name = config_project.get("train_log_name")

try: 
    from tqdm import tqdm
    tqdm_available = True
except:
    tqdm_available = False


class Trainer:
    """
    Helper class to train a neural net.

    Parameters
    ----------
    model: nn.Module 
        neural network to be trained.

    train_loader: torch.DataLoader
        training dataset

    val_loader: torch.DataLoader object
        validation dataset

    device: torch.device

    exp_name: string
        experiment name used to create saving folder

    training_prm: dict
        dictionnary with all the training parameters

    data_prm: dict
        dictionnary with all the data parameters. 
        NOTE: now only "volumetric" is used from this dict.

    saving_prm: dict
        dictionnary with all the saving parameters

    loss_function: string
        loss function to use. Only used if not LVAE network, e.g. "MSE".
    
    mode: string, default = "train
        Defines the training mode:
        "train": normal training mode when starting from
        "continue_training": to continue a training stopped too early
        "retrain": to retrain an existing model with different parameters

    is_lvae: bool, default = "False"
        defines if the model is LVAE or not.
    
    """
    def __init__(self,model, train_loader, val_loader, device,**kwargs):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.training_prm = kwargs.get("training_prm")
        self.data_prm = kwargs.get("data_prm")
        self.saving_prm = kwargs.get("saving_prm")
        
        self.exp_name = kwargs.get("exp_name")
        self.mode = kwargs.get("mode","train")
        self.is_lvae = kwargs.get("is_lvae",False)
        
        # training paremeters
        self.batch_size = self.training_prm.get("batch_size")
        self.n_epochs = self.training_prm.get("n_epochs")
        self.early_stop = self.training_prm.get("early_stop",False)
        self.pos_encod = self.training_prm.get("pos_encod",False)
        self.betaKL = self.training_prm.get("betaKL",1)

        # data_prm
        self.volumetric = self.data_prm.get("volumetric", False)

        self.tiling_validation = False
        if self.is_lvae and self.data_prm.get("val_patch_size") is not None:
            if self.data_prm.get("val_patch_size") != self.data_prm.get("patch_size"):
                self.tiling_validation = True
                self.tiling_patch = self.data_prm.get("patch_size")
                if self.data_prm.get("downsampling",{}).get("downsamp_factor") is not None:
                    p = self.data_prm.get("downsampling").get("downsamp_factor")
                    self.tiling_patch = self.tiling_patch // p

        # loss function
        if not self.is_lvae:
            self.loss_function = loss.get_loss_function(**kwargs.get("loss_function"))
        else:
            self.loss_function = None

        # make optimizer and scheduler
        self.optimizer,self.scheduler = misc.make_optimizer_and_scheduler(model,self.training_prm)

        # initiate state_dict
        self.state_dict = kwargs.get("state_dict",None)
        if self.state_dict is None:
            self.state_dict = {"epoch": 0} #other items will be updated through training

        # saving
        if self.saving_prm.get("saving"):
            self.model_save_folder = self.saving_prm.get("model_save_folder")
            self.saving = True
            self.loss_path = self.model_save_folder / _loss_file_name
            self.log_path = self.model_save_folder / _log_file_name
            self.save_validation_images = self.saving_prm.get("save_validation_images",False)
        else:
            self.saving = False
            self.loss_path = None
            self.log_path = None
            self.save_validation_images = False
        
        if self.save_validation_images:
            self.save_validation_freq = self.saving_prm.get("save_validation_freq",10)
            self.validation_images_folder = self.model_save_folder / "validation_images"
            if not os.path.exists(self.validation_images_folder):
                os.makedirs(self.model_save_folder / "validation_images")

        # create logger and initialize
        self.logger = logging.getLogger("trainer")
        self.console_filter = EnableFilter()
        self.logfile_filter = EnableFilter()
        self.initialize_logger()

        # progress bar
        self.pbar = self.training_prm.get("pbar") and tqdm_available
        if not self.pbar:
            self.update_console = True
        else:
            self.update_console = False

        #tensorboard writer
        self.writer = kwargs.get("writer",None)
    

    def train(self):

        start_epoch = self.state_dict.get("epoch", 0)
        best_loss = self.state_dict.get("best_loss", float('inf'))

        iter_epoch = range(start_epoch, self.n_epochs)
        if self.pbar:
            iter_epoch = tqdm(iter_epoch,position=1,total=self.n_epochs,initial=start_epoch)
            iter_epoch.set_description('Epochs')

        self.initialize_log_file()
        self.logger.info('Starting Training...')

        for epoch in iter_epoch:
            try:
                if self.save_validation_images and epoch % self.save_validation_freq == 0:
                    save_val_imgs = True
                else:
                    save_val_imgs=False

                start = time.time()
                train_loss,train_kl_loss,train_recons_loss = self.train_epoch(epoch)
                val_loss,val_kl_loss,val_recons_loss = self.validate(epoch,save_imgs=save_val_imgs)
                end = time.time()
                # print(f"Epoch comput. time: {end-start}")

                self.state_dict.update({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                })

                if self.is_lvae:
                    self.state_dict.update({
                        "train_kl_loss": train_kl_loss,
                        "val_kl_loss": val_kl_loss,
                        "train_recons_loss": train_recons_loss,
                        "val_recons_loss": val_recons_loss,
                    })  

                if self.scheduler is not None:
                    self.state_dict.update({
                        "scheduler": self.scheduler.state_dict()
                    })

                if val_loss < best_loss:
                    best_loss = val_loss
                    is_best = True
                    self.logger.info(
                        f"epoch {epoch}: Best model saved "
                        f"with best_val = {best_loss}."
                    )
                else:
                    is_best = False

                if self.scheduler is not None:
                    if self.training_prm.get("scheduler") == "ReduceLROnPlateau":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                if self.saving:
                    self.handle_saving(best_loss, is_best)
                    self.update_loss()
                
                if self.writer is not None:
                    self.writer.add_scalar('Train_loss',train_loss,epoch)
                    self.writer.add_scalar('Val_loss',val_loss,epoch)
                
                if self.early_stop and epoch > 2:
                    self.logger.info("Early stopping.")
                    exit()

            except KeyboardInterrupt:
                self.handle_keyboard_interrupt(epoch, best_loss, train_loss, val_loss)
                return
            except Exception as e:
                self.handle_exception(epoch, e)
            

        self.finalize_logging(epoch, best_loss, val_loss)

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        train_loss = []

        if self.is_lvae:
            train_kl_loss = []
            train_recons_loss = []

        iter_loader = self.train_loader
        if self.pbar:
            iter_loader = tqdm(iter_loader,leave=False,position = 0)
            iter_loader.set_description(f'Training - Epoch {epoch}')

        n_batch = len(iter_loader)
        for batch_id, batch in (enumerate(iter_loader)):
            # print("Training batch ",batch_id,"/",n_batch,": ",batch[0].shape)
            if self.update_console:
                self.logfile_filter.enable = False #so that it doesnt go into the log file
                self.logger.info(f"epochs: {epoch}/{self.n_epochs}, batch_id: {batch_id}/{len(iter_loader)}")
                self.logfile_filter.enable = True

            # make smaller batches according to batch size
            if int(batch[0].shape[0]//self.batch_size) == 0:
                virtual_batches = [(tensor,) for tensor in batch]
                if batch_id ==0: #to warn only once per epoch
                    self.logger.warning("batch_size bigger than # of virtual batches, not optimal")
            else:
                virtual_batches = [torch.split(tensor,self.batch_size,dim=0) for tensor in batch]
            # iterate over smaller batches
            count=0
            for (x, y, *samp_pos) in zip(*virtual_batches):
                count+=1
                # print("virtual batch ",count,": ",x.shape)
                if torch.isnan(y).all().item():
                    y = None
                else:
                    y = y.to(self.device)
                if self.pos_encod:
                    assert len(samp_pos) == 1
                    samp_pos = samp_pos[0].to(self.device)
                else:
                    samp_pos = None

                if len(x.shape) == 4:
                    if self.volumetric:
                        x=x.unsqueeze(1) # to add channel dimension
                        if y is not None:
                            y=y.unsqueeze(1)
                    x = x.to(self.device,dtype=torch.float)
                    if self.is_lvae:
                        outputs = lvae_forward_pass(x, y,self.device, self.model, gaussian_noise_std=None)
                        recons_loss = outputs['recons_loss']
                        kl_loss = outputs['kl_loss']
                        loss = recons_loss + self.betaKL*kl_loss
                        train_kl_loss.append(kl_loss.item())
                        train_recons_loss.append(recons_loss.item())
                    else:                    
                        prediction = self.model(x, samp_pos)
                        loss = self.loss_function(prediction, y)
                
                ### Testing stage of "all 4 in one" upsampling thing ###  
                # elif len(x.shape) == 5:
                #     assert not self.is_lvae, "all 4 in one training not available for HDN training"
                #     loss = 0
                #     predictionsList = []
                #     for idxInput in range(x.shape[2]):
                #         inp = x[:,:,idxInput]
                #         inp = inp.to(self.device)
                #         p = self.model(inp,samp_pos)
                #         predictionsList.append(p)
                #         loss += self.loss_function(p,y)
                #     for i in range(len(predictionsList)):
                #         for j in range (i,len(predictionsList)):
                #             loss += torch.nn.MSELoss()(predictionsList[i],predictionsList[j])
                
                # print("Loss backward")
                loss.backward()
                train_loss.append(loss.item())
            
            # print("Optimizer step")
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.early_stop is not False and batch_id > 0:
                break

        train_loss = np.mean(train_loss)
        train_kl_loss = np.mean(train_kl_loss) if self.is_lvae else None
        train_recons_loss = np.mean(train_recons_loss) if self.is_lvae else None

        return train_loss,train_kl_loss,train_recons_loss

    def validate(self, epoch,save_imgs=False):
        self.model.eval()
        self.model.to(self.device)
        val_loss = []

        if save_imgs:
            val_imgs = []

        if self.is_lvae:
            val_kl_loss = []
            val_recons_loss = []

        with torch.no_grad():
            
            iter_val = self.val_loader
            if self.pbar:
                it_val = tqdm(iter_val,position=0,leave=False)
                it_val.set_description(f'Validation - Epoch {epoch}')
            
            n_batch = len(iter_val)
            for batch_id,batch in enumerate(iter_val):
                # print("Val batch ",batch_id,"/",n_batch,": ",batch[0].shape)
                # make smaller batches
                if int(batch[0].shape[0]//self.batch_size) == 0:
                    virtual_batches = [(tensor,) for tensor in batch]
                else:
                    virtual_batches = [torch.split(tensor,self.batch_size,dim=0) for tensor in batch]             
                count = 0
                for (x, y, *samp_pos) in zip(*virtual_batches) :
                    count+=1
                    # print("virtual batch ",count,": ",x.shape)
                    # if len(x.shape) == 5: # for "all 4 in one" validation done only on 1
                    #     x = x[:,:,1]

                    if self.volumetric:
                        x=x.unsqueeze(1)
                        if y is not None:
                            y=y.unsqueeze(1)

                    x = x.to(self.device,dtype=torch.float)

                    if torch.isnan(y).all().item():
                        y = None
                    else:
                        y = y.to(self.device)

                    if self.pos_encod:
                        assert len(samp_pos) == 1
                        samp_pos = samp_pos[0].to(self.device)
                    else:
                        samp_pos = None

                    if self.is_lvae:
                        if self.tiling_validation:
                            outputs = lvae_forward_pass_tiling(x, None, self.device, self.model, gaussian_noise_std=None,
                                                               patch_size=self.tiling_patch)
                        else:
                             outputs = lvae_forward_pass(x, None, self.device, self.model, gaussian_noise_std=None)
                        recons_loss = outputs['recons_loss']
                        kl_loss = outputs['kl_loss']
                        loss = recons_loss + self.betaKL*kl_loss
                        val_loss.append(loss.item())
                        val_kl_loss.append(kl_loss.item())
                        val_recons_loss.append(recons_loss.item())
                        prediction = outputs['out_mean']
                    else:
                        prediction = self.model(x, samp_pos)
                        val_loss.append(self.loss_function(prediction, y).item())

                    if self.writer is not None:
                        self.write_images_to_tensorboard(x, y, prediction, epoch)
                    
                    if save_imgs:
                        for i in range (x.shape[0]):
                            if y is None:
                                val_imgs.append((x[i],None,prediction[i]))
                            else:
                                val_imgs.append((x[i],y[i],prediction[i]))


        val_loss = np.mean(val_loss)
        val_kl_loss = np.mean(val_kl_loss) if self.is_lvae else None
        val_recons_loss = np.mean(val_recons_loss) if self.is_lvae else None

        if save_imgs:
            self.save_valid_images(val_imgs,epoch)

        return val_loss,val_kl_loss,val_recons_loss
    

    def handle_saving(self, best_loss, is_best):
        epoch = self.state_dict["epoch"]
        name_best = "model_best" if self.saving_prm["overwrite_best"] else f"model_epoch_{epoch}"
        model_save_folder = self.model_save_folder

        if self.saving_prm.get("state_dict", False):
            if is_best:
                # Save best model as best and last with best_loss
                model_name_best = f"{name_best}_state_dict.pt"
                torch.save(self.state_dict, model_save_folder / model_name_best)
            
            # Save last model in any case
            model_name_last = f"model_last_state_dict.pt"
            self.state_dict["best_loss"] = best_loss
            torch.save(self.state_dict, model_save_folder / model_name_last)
            # Remove best_loss from state_dict
            del self.state_dict["best_loss"]


        if self.saving_prm.get("entire_model", False):
            if is_best:
                # Save best model as both best and last
                model_name_best = f"{name_best}.pt"
                torch.save(self.model, model_save_folder / model_name_best)

            # Save last model in any case
            model_name_last = f"model_last.pt"
            torch.save(self.model, model_save_folder / model_name_last)


    def update_loss(self):
        # create loss file if not exist
        if not os.path.exists(self.loss_path):
            with open(self.loss_path, 'w') as loss_file: 
                if self.is_lvae:
                    head = str('Epoch Train_loss Val_loss Recons_Loss KL_Loss').split()
                    loss_file.write(f'{head[0]:<10} {head[1]:<30} {head[2]:<30} {head[3]:<30} {head[4]:<30}')  
                    loss_file.write('\n')
                else:
                    head = str('Epoch Train_loss Val_loss').split()
                    loss_file.write(f'{head[0]:<10} {head[1]:<30} {head[2]:<30}')  
                    loss_file.write('\n')

        # write new losses
        epoch = self.state_dict["epoch"]
        train_loss = self.state_dict["train_loss"]
        val_loss = self.state_dict["val_loss"]
        if self.is_lvae:
            recons_loss = self.state_dict["train_recons_loss"]
            kl_loss = self.state_dict["train_kl_loss"]
            with open(self.loss_path, "a") as loss_file:
                loss_file.write(f'{epoch:<10} {train_loss:<30} {val_loss:<30}'
                                f'{recons_loss:<30} {kl_loss:<30}\n')
        else:
            with open(self.loss_path, "a") as loss_file:
                loss_file.write(f'{epoch:<10} {train_loss:<30} {val_loss:<30}\n')


    def initialize_logger(self):
        # console handler
        console_handler = CustomStreamHandler()
        console_handler.addFilter(self.console_filter)
        self.logger.addHandler(console_handler)

        # file handler
        if self.saving:
            log_file_handler = logging.FileHandler(self.log_path,mode = "a")
            formatter = logging.Formatter('%(asctime)-5s %(message)s',"%Y-%m-%d %H:%M:%S")
            log_file_handler.setFormatter(formatter)
            log_file_handler.addFilter(self.logfile_filter)
            self.logger.addHandler(log_file_handler)
        

    def initialize_log_file(self):
        try:
            gpu = torch.cuda.get_device_name(self.device)
        except:
            gpu = 'CPU'

        self.console_filter.enable = False # don't show in console the next message
        
        if self.data_prm.get("patch_info") is not None:
            patch_txt=(
                f"Training patches: {self.data_prm.get('patch_info').get('train_patch')}.\n"
                f"Validation patches {self.data_prm.get('patch_info').get('val_patch')}.\n\n"
            )
        else: patch_txt = ""
        
        if self.mode == 'train':
            self.logger.info(
                f"\nExperiment name: {self.exp_name}\n"
                f"Computer: {os.environ['COMPUTERNAME']}\n"
                f"Running on: {gpu}\n\n"
                f"{patch_txt}"
            )
        elif self.mode == 'retrain':
            self.logger.info(
                f"\nExperiment name: {self.exp_name}\n"
                f"Computer: {os.environ['COMPUTERNAME']}\n"
                f"Running on: {gpu}\n\n"
                f"Retrain mode - starting from previously trained model.\n"
                f"Check cfg file and 'retrain_origin_model' folder for details. \n\n"
                f"{patch_txt}"
            )

        elif self.mode == 'continue_training':
            self.logger.info (
                f"Continue training  mode. Computer: {os.environ['COMPUTERNAME']}, with {gpu}.\n"
            )
        
        self.console_filter.enable = True

    def write_images_to_tensorboard(self, x, y, prediction, epoch):    
        # select the middle time point
        t = x.shape[1] // 2

        input_img = x[0,0,...].cpu().detach()
        input_img = (input_img -torch.min(input_img))
        input_img = 255 * input_img / torch.max(input_img)
        input_img = input_img.type(torch.uint8)

        gt_img = y[0,0,...].cpu().detach()
        gt_img = (gt_img -torch.min(gt_img))
        gt_img = 255 * gt_img / torch.max(gt_img)
        gt_img = gt_img.type(torch.uint8)
        
        pred_img = prediction[0,0,...].cpu().detach()
        pred_img = (pred_img -torch.min(pred_img))
        pred_img = 255 * pred_img / torch.max(pred_img)
        pred_img = pred_img.type(torch.uint8)
        
        shape = 'CHW' if self.volumetric else 'HW'
            
        self.writer.add_image('input', input_img, epoch,dataformats=shape)
        self.writer.add_image('prediction', pred_img, epoch,dataformats=shape)
        self.writer.add_image('ground truth', gt_img, epoch,dataformats=shape)

    def save_valid_images(self,list_imgs,epoch):
        """
        Save validation images to `self.validation_images_folder`.
        Inputs:
            list_imgs: list 
                list of tuples (inp,gt*,prediction) to save
                *gt can be None
            epoch: int 
                current epoch
        """ 

        for i,(x,y,pred) in enumerate(list_imgs):
            
            paired = True if y is not None else False

            inp = x.cpu().numpy()
            gt = y.cpu().numpy() if paired else None
            pred = pred.cpu().detach().numpy()

            # adjust shape
            if self.volumetric:     # [C,Z,H,W] => [Z,H,W]
                inp = inp[0,...]
                pred = pred[0,...]
                if paired:
                    gt = gt[0,...]

            elif pred.shape[0]>1:   # [C,H,W]
                mltpl_ch = True 
            else:                   # [1,H,W] => [H,W]
                mltpl_ch = False
                pred = pred[0,...]

            pred_path = self.validation_images_folder/f"patch{i:02d}_prediction.tiff"
            inp_path = self.validation_images_folder/f"patch{i:02d}_input.tiff"
            gt_path = self.validation_images_folder/f"patch{i:02d}_groundtruth.tiff"

            if os.path.exists(pred_path):
                prev = imread(pred_path)
                if len(prev.shape) == 2:
                    prev = np.expand_dims(prev,axis=0)
                elif len(prev.shape) == 3 and (self.volumetric or mltpl_ch):
                    prev = np.expand_dims(prev,axis=0)
                
                pred = np.expand_dims(pred,axis=0)
                tosave = np.concatenate(([prev,pred]),axis=0)

                shape = 'TZYX' if len(tosave.shape) == 4 else 'TYX'
                imsave(pred_path,tosave,imagej=True,metadata={'axes':shape})

            else:
                #save pred
                shape = 'TYX' if len(pred.shape) == 3 else 'YX'
                imsave(pred_path,pred,imagej=True,metadata={'axes':shape})

                #save inp
                if inp.shape[0]>1:
                    shape = 'TYX'
                else:
                    inp = inp[0]
                    shape = 'YX'
                imsave(inp_path,inp,imagej=True,metadata={'axes':shape})
                
                # save gt
                if paired:
                    if inp.shape[0]>1:
                        shape = 'TYX'
                    else:
                        gt = gt[0,...]
                        shape = 'YX'
                    imsave(gt_path,gt,imagej=True,metadata={'axes':shape})
        

    def handle_keyboard_interrupt(self, epoch, best_loss, train_loss, val_loss):
        if self.saving_prm.get("saving", False):
            self.logger.info(
                f"Training manually stopped during epoch {epoch}.\n"
                f"Model perf: best_val_loss: {best_loss} - "
                f"current_train_loss: {train_loss} - "
                f"current_val_loss: {val_loss}.\n"
            )
        return

    def handle_exception(self, epoch, e):
        if self.saving_prm.get("saving", False):
            self.logger.error (
                f"Training stopped during epoch {epoch}, "
                f"because of error:\n{type(e)}\n{e}\n"
            )
        raise

    def finalize_logging(self, epoch, best_loss, val_loss):
        if self.saving_prm.get("saving", False):
            self.logger.info(
                f"Finished training: {epoch+1}/{self.n_epochs} epochs.\n"
                f"Model perf: best_val_loss: {best_loss} - "
                f"current_val_loss: {val_loss}.\n"
            )


if __name__ == "__main__":
    print("out")
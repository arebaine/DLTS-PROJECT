import torch as torch
import os as os
    
class EncoderConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # Parameters for the Conv2D layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Conv2D layer
        self.conv = torch.nn.Conv2d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size= self.kernel_size, stride= self.stride, padding= self.padding)
        self.bn = torch.nn.BatchNorm2d(self.out_channels)
        self.activation = torch.nn.LeakyReLU(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out_activated = self.activation(self.bn(out))
        return out, out_activated
    
class DecoderConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=False):
        super().__init__()
        # Parameters for the Conv2D layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout

        # tConv2D layer
        self.tconv = torch.nn.ConvTranspose2d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size= self.kernel_size, stride= self.stride, padding= self.padding)
        self.bn = torch.nn.BatchNorm2d(self.out_channels)
        self.activation = torch.nn.ReLU()
        if dropout:
            self.dp = torch.nn.Dropout(p=0.5)
        
    def forward(self,x):
        out = self.activation(self.bn(self.tconv(x)))
        if self.dropout:
            return self.dp(out)
        return out

def pytorch_pipeline(device, model, train_data, validation_data, optimizer, alpha, criterion1, criterion2, n_epoch, path_save_model):
    model.to(device)
    hist_loss_train = []
    hist_loss_valid = []

    for epoch in range(n_epoch):

        # Training phase
        model.train()
        train_loss = 0
        for batch_Smixte, batch_Svoice, batch_Snoise in train_data:
            batch_Smixte, batch_Svoice, batch_Snoise = torch.abs(batch_Smixte.to(device)), torch.abs(batch_Svoice.to(device)), torch.abs(batch_Snoise.to(device))
            optimizer.zero_grad()
            pred_Svoice, pred_Snoise = model(batch_Smixte)[:,0,:,:], model(batch_Smixte)[:,1,:,:]
            loss = alpha*criterion1(pred_Svoice,batch_Svoice) + (1-alpha)*criterion2(pred_Snoise,batch_Snoise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        hist_loss_train.append(train_loss/len(train_data))
        print(f"TRAIN : epoch [{epoch+1}/{n_epoch}] - Loss : {train_loss/len(train_data):.2f}",  end="\t")

        # Validation phase
        model.eval()
        valid_loss = 0
        for batch_Smixte, batch_Svoice, batch_Snoise in train_data:
            batch_Smixte, batch_Svoice, batch_Snoise = batch_Smixte.to(device), batch_Svoice.to(device), batch_Snoise.to(device)
            with torch.no_grad():
                pred_Svoice, pred_Snoise = model(batch_Smixte)[:,0,:,:], model(batch_Smixte)[:,1,:,:]
                loss = alpha*criterion1(pred_Svoice,batch_Svoice) + (1-alpha)*criterion2(pred_Snoise,batch_Snoise)
                valid_loss+= loss.item()
        hist_loss_valid.append(valid_loss/len(validation_data))      
        print(f"VALIDATION : epoch [{epoch+1}/{n_epoch}] - Loss : {valid_loss/len(validation_data):.2f}")
    torch.save(model, path_save_model)

def pytorch_pipeline_single(device, model, train_data, validation_data, optimizer, criterion, mode, n_epoch, path_save_model):
    model.to(device)
    hist_loss_train = []
    hist_loss_valid = []

    for epoch in range(n_epoch):

        # Training phase
        model.train()
        train_loss = 0
        for batch_Smixte, batch_Svoice, batch_Snoise in train_data:
            batch_Smixte, batch_Svoice, batch_Snoise = torch.abs(batch_Smixte.to(device)), torch.abs(batch_Svoice.to(device)), torch.abs(batch_Snoise.to(device))
            optimizer.zero_grad()
            if mode == "voice":
                pred_Svoice = model(batch_Smixte)
                loss = criterion(pred_Svoice,batch_Svoice)
            elif mode == "noise":
                pred_Snoise = model(batch_Smixte)
                loss = criterion(pred_Snoise,batch_Snoise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        hist_loss_train.append(train_loss/len(train_data))
        print(f"TRAIN : epoch [{epoch+1}/{n_epoch}] - Loss : {train_loss/len(train_data):.2f}",  end="\t")

        # Validation phase
        model.eval()
        valid_loss = 0
        for batch_Smixte, batch_Svoice, batch_Snoise in train_data:
            batch_Smixte, batch_Svoice, batch_Snoise = torch.abs(batch_Smixte.to(device)), torch.abs(batch_Svoice.to(device)), torch.abs(batch_Snoise.to(device))
            with torch.no_grad():
                if mode == "voice":
                    pred_Svoice = model(batch_Smixte)
                    loss = criterion(pred_Svoice,batch_Svoice)
                elif mode == "noise":
                    pred_Snoise = model(batch_Smixte)
                    loss = criterion(pred_Snoise,batch_Snoise)
                valid_loss+= loss.item()
        hist_loss_valid.append(valid_loss/len(validation_data))      
        print(f"VALIDATION : epoch [{epoch+1}/{n_epoch}] - Loss : {valid_loss/len(validation_data):.2f}")
    torch.save(model, path_save_model)

def one_hot_encoding(Sxx_voice, Sxx_noise):
    output = torch.zeros(Sxx_voice.shape + (2,))  
    output[..., 0] = (Sxx_voice > Sxx_noise).float()  
    output[..., 1] = (Sxx_voice < Sxx_noise).float()  
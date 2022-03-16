import torch
import torch.nn as nn
import torchvision.transforms
import torch.nn.functional as F

#"Универсальный блок" состоящий из двух 2d свёрток и функции активации ReLu
class Block(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

#Енкодер 
#self.enc_blocks представляет собой список операций Блока. 
#Далее мы выполняем операцию MaxPool2d для выходных данных каждого блока. 
#Для сохранения выходные данных блока, мы сохраняем их в списке с именем ftrs и возвращаем этот список.
class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)


    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

#Декодер
#Операция ConvTranspose2d выполняет “свертку вверх”.
#Значения in_channels и out_channels различаются в декодере в зависимости от того,
#где выполняется эта операция, в реализации операции “свертки вверх” также сохраняются в виде списка.
# Шаг и размер ядра всегда равны 2, согласно условиям задачи
#elf.dec_blocks - это список блоков декодера, которые выполняют две операции conv + ReLU
#Self.upconvs - это список операций ConvTranspose2d, которые выполняют операции “свертки вверх”
#forward принимает encoder_features, которые были выведены Кодером для выполнения операции конкатенации перед передачей результата в Блок
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class SegmenterModel(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super(SegmenterModel, self).__init__()
        self.encoder     = Encoder(enc_chs)#Енкодер - "прямой" путь - "спуск"
        self.decoder     = Decoder(dec_chs)#Декодер - "обратный" путь - "подъем"
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim: # Позволит сравнять размерность выходного изображения с входным
            out = F.interpolate(out, out_sz)
        return out


    def predict(self, x):
        # на вход подаётся одна картинка, а не батч, поэтому так
        y = self.forward(x.unsqueeze(0).cuda())
        return (y > 0).squeeze(0).squeeze(0).float().cuda()

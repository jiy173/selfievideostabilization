import torch
import torch.nn as nn
#import PWCNet
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class stabnet(nn.Module):
    def __init__(self, nwin=5,in_channels=4, out_channels=2,start_filts=128):
        super(stabnet, self).__init__()
        self.conv1=nn.Conv1d(in_channels*(nwin-1),start_filts,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv2=nn.Conv1d(start_filts,start_filts*2,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv3=nn.Conv1d(start_filts*2,start_filts*2,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv4=nn.Conv1d(start_filts*2,start_filts*4,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv5=nn.Conv1d(start_filts*4,start_filts*4,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv6=nn.Conv1d(start_filts*4,start_filts*4,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv7=nn.Conv1d(start_filts*4,start_filts*8,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv8=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=2,padding=2)
        self.conv9=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=2,padding=2)
        self.conv10=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=2,padding=2)
        
        self.conv1_1=nn.Conv1d(in_channels*(nwin-1),start_filts,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv2_1=nn.Conv1d(start_filts,start_filts*2,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv3_1=nn.Conv1d(start_filts*2,start_filts*2,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv4_1=nn.Conv1d(start_filts*2,start_filts*4,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv5_1=nn.Conv1d(start_filts*4,start_filts*4,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv6_1=nn.Conv1d(start_filts*4,start_filts*4,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv7_1=nn.Conv1d(start_filts*4,start_filts*8,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv8_1=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=2,padding=2)
        self.conv9_1=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=2,padding=2)
        self.conv10_1=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=2,padding=2)
        
        self.conv11=nn.ConvTranspose1d(start_filts*32,start_filts*8,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv12=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv13=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv14=nn.ConvTranspose1d(start_filts*16,start_filts*4,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv15=nn.Conv1d(start_filts*4,start_filts*4,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv16=nn.ConvTranspose1d(start_filts*8,start_filts*2,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv17=nn.Conv1d(start_filts*2,start_filts*2,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv18=nn.Conv1d(start_filts*2,out_channels*(nwin-2),kernel_size=1,stride=1,dilation=1,padding=0)
        
        self.conv11_1=nn.ConvTranspose1d(start_filts*32,start_filts*8,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv12_1=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv13_1=nn.Conv1d(start_filts*8,start_filts*8,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv14_1=nn.ConvTranspose1d(start_filts*16,start_filts*4,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv15_1=nn.Conv1d(start_filts*4,start_filts*4,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv16_1=nn.ConvTranspose1d(start_filts*8,start_filts*2,kernel_size=4,stride=2,dilation=1,padding=1)
        self.conv17_1=nn.Conv1d(start_filts*2,start_filts*2,kernel_size=3,stride=1,dilation=1,padding=1)
        self.conv18_1=nn.Conv1d(start_filts*2,out_channels*(nwin-2),kernel_size=1,stride=1,dilation=1,padding=0)
        
        self.bn2=nn.BatchNorm1d(start_filts*2)
        self.bn4=nn.BatchNorm1d(start_filts*4)
        self.bn8=nn.BatchNorm1d(start_filts*8)
        self.bn16=nn.BatchNorm1d(start_filts*16)
        
        self.linconv1=nn.Conv1d(start_filts*2,start_filts,kernel_size=3,stride=1,dilation=1,padding=1)
        self.linconv2=nn.Conv1d(start_filts,2,kernel_size=1,stride=1,dilation=1,padding=0)
        self.lin1=nn.Linear(2*512,512)
        self.lin2=nn.Linear(512,4*(nwin-2))
        #self.batchnorm
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
                #m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.constant(m.weight.data,0)
                    
                    
    def forward(self, x_feat,x_head,lm):
        
        out1_1_0=self.conv1(x_feat)         #1024
        out1_2_0=self.conv2(out1_1_0)    #512
        out2_1_0=self.conv3(out1_2_0)    #512
        out2_2_0=self.conv4(out2_1_0)   #256
        out3_1_0=self.conv5(out2_2_0)    #256
        out3_2_0=self.conv6(out3_1_0)    #256
        out3_3_0=self.conv7(out3_2_0)    #128
        out4_1_0=self.conv8(out3_3_0)    #128
        out4_2_0=self.conv9(out4_1_0)    #128
        out4_3_0=self.conv10(out4_2_0)   #128
        
        out1_1_1=self.conv1_1(x_head)         #1024
        out1_2_1=self.conv2_1(out1_1_1)    #512
        out2_1_1=self.conv3_1(out1_2_1)    #512
        out2_2_1=self.conv4_1(out2_1_1)   #256
        out3_1_1=self.conv5_1(out2_2_1)    #256
        out3_2_1=self.conv6_1(out3_1_1)    #256
        out3_3_1=self.conv7_1(out3_2_1)    #128
        out4_1_1=self.conv8_1(out3_3_1)    #128
        out4_2_1=self.conv9_1(out4_1_1)    #128
        out4_3_1=self.conv10_1(out4_2_1)   #128
        
        out4_3=torch.cat(((1-lm)*out4_3_0,lm*out4_3_1),1)#lm*
        out3_3=torch.cat(((1-lm)*out3_3_0,lm*out3_3_1),1)
        out2_2=torch.cat(((1-lm)*out2_2_0,lm*out2_2_1),1)
        out1_2=torch.cat(((1-lm)*out1_2_0,lm*out1_2_1),1)
        
#        
#        out4_3h=torch.cat((lm*out4_3_0,(1-lm)*out4_3_1),1)#lm*
#        out3_3h=torch.cat((lm*out3_3_0,(1-lm)*out3_3_1),1)
#        out2_2h=torch.cat((lm*out2_2_0,(1-lm)*out2_2_1),1)
#        out1_2h=torch.cat((lm*out1_2_0,(1-lm)*out1_2_1),1)
        
        out5_1_0=self.conv11(torch.cat((out4_3,out3_3),1))#256
        out5_2_0=self.conv12(out5_1_0)                      #256
        out5_3_0=self.conv13(out5_2_0)                      #256
        out6_1_0=self.conv14(torch.cat((out5_3_0,out2_2),1))#512
        out6_2_0=self.conv15(out6_1_0)                      #512
        out7_1_0=self.conv16(torch.cat((out6_2_0,out1_2),1))#1024
        out7_2_0=self.conv17(out7_1_0)                      #1024
        out7_3_0=self.conv18(out7_2_0)                      #1024
        
#        out5_1_1=self.conv11_1(torch.cat((out4_3,out3_3),1))#256
#        out5_2_1=self.conv12_1(out5_1_1)                      #256
#        out5_3_1=self.conv13_1(out5_2_1)                      #256
#        out6_1_1=self.conv14_1(torch.cat((out5_3_1,out2_2),1))#512
#        out6_2_1=self.conv15_1(out6_1_1)                      #512
#        out7_1_1=self.conv16_1(torch.cat((out6_2_1,out1_2),1))#1024
#        out7_2_1=self.conv17_1(out7_1_1)                      #1024
#        out7_3_1=self.conv18_1(out7_2_1)                      #1024
#        out7_3_1=self.linconv1(out7_2_1)                      #1024
#        out7_4_1=self.linconv2(out7_3_1)
#        out7_5_1=self.lin1(out7_4_1.view(-1))
#        out7_6_1=self.lin2(out7_5_1)
        
        
        return out7_3_0#,out7_3_1


    

if __name__ == '__main__':
    feat=torch.rand(1,6,1024).cuda()
    model=stabnet().cuda()
    oup=model(feat)
        
        
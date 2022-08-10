import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG5 ANN feature layers
class VGG5_feature_ANN(nn.Module):
    def __init__(self, input_dim=3, dropout=0.2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.output_dim = 8192

    def forward(self, x):
        return self.feature(x)

class VGG9_feature_ANN(nn.Module):
    def __init__(self, input_dim=3, dropout=0.2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.output_dim = 4096

    def forward(self, x):
        return self.feature(x)

# ANN Classifier layers
class CLS_ANN(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super().__init__()
        self.pretrain = pretrain
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc)

    def forward(self, x):
        return self.main(x)

# VGG5 ANN model
class VGG5_ANN(nn.Module):
    def __init__(self, in_channels=3, num_cls=10, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.feature_extractor = VGG5_feature_ANN(input_dim=in_channels)
        self.feature_extractor_output_dim = self.feature_extractor.output_dim
        self.classifier = CLS_ANN(self.feature_extractor_output_dim, num_cls, bottle_neck_dim=256)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.reshape(self.batch_size, -1)
        out = self.classifier(out)
        return out

# VGG9 ANN model
class VGG9_ANN(nn.Module):
    def __init__(self, in_channels=3, num_cls=10, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.feature_extractor = VGG9_feature_ANN(input_dim=in_channels)
        self.feature_extractor_output_dim = self.feature_extractor.output_dim
        self.classifier = CLS_ANN(self.feature_extractor_output_dim, num_cls, bottle_neck_dim=256)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.reshape(self.batch_size, -1)
        out = self.classifier(out)
        return out

# -----------------------------------------------------------
# SNN
# -----------------------------------------------------------

class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


class VGG5_SNN_BNTT(nn.Module):
    def __init__(self, timesteps=20, leak_mem=0.95, input_dim=3, img_size=32, num_cls=10):
        super().__init__()

        self.input_dim = input_dim
        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps

        print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print ("{} time steps per batchnorm".format(self.timesteps))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8192, 256, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(256, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt_fc]
        self.pool_list = [self.pool1, False, self.pool2]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)


    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3]

        mem_fc1 = torch.zeros(batch_size, 256).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()


        for t in range(self.timesteps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()

                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.timesteps


        return out_voltage

class VGG9_SNN_BNTT(nn.Module):
    def __init__(self, timesteps=20, leak_mem=0.95, img_size=32,  num_cls=10):
        super().__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps

        print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)


        self.fc1 = nn.Linear((self.img_size//8)*(self.img_size//8)*256, 256, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(256, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 256).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()


        for t in range(self.timesteps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.timesteps


        return out_voltage

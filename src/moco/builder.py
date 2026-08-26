import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def flatten(t):
    return t.reshape(t.shape[0], -1)

class MoCo(nn.Module):
    def __init__(self, base_encoder, name, dim=128, K=65536, T=0.2, mlp=False, feat_dim=512, num_classes=21):
        super(MoCo, self).__init__()
        self.K = K
        self.T = T

        self.encoder_q = base_encoder(name=name, num_classes=dim)
        self.linear = nn.Linear(feat_dim, num_classes)

        if mlp:  
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), self.encoder_q.fc)

        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.layer = -2 #ResNet18 setting
        self.feat_after_avg_q = None
        self._register_hook()
        self.normalize = False 

    
    def _find_layer(self, module):
        if type(self.layer) == str:
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*module.children()]

            return children[self.layer]
        return None

    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
           self.feat_after_avg_q = nn.functional.normalize(self.feat_after_avg_q, dim=1)

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f'hidden layer ({self.layer}) not found'
        handle = layer_q.register_forward_hook(self._hook_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


    def _train(self, im_q, im_k, labels):
        q, k = self.encoder_q(im_q), self.encoder_q(im_k)    
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits_q = self.linear(self.feat_after_avg_q)
        logits_k = self.linear(self.feat_after_avg_q)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, labels)

        logits = torch.cat((logits_q, logits_k), dim=0)

        return features, target, logits

    def _inference(self, image):
        q = self.encoder_q(image)
        encoder_q_logits = self.linear(self.feat_after_avg_q)
        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
           return self._train(im_q, im_k, labels) 
        else:
           return self._inference(im_q)



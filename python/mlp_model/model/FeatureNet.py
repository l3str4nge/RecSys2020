import torch
import torch.nn as nn


class FeatureNet(nn.Module):
    def __init__(self, num_tokens, num_features, layers,corruption=0.2):
        super(FeatureNet, self).__init__()

        layers = [layers] if type(layers) is int else layers
        #self.chunk_emb = nn.Embedding(15, 40)
        self.token_layer = nn.Linear(num_tokens, int(layers[0]/2))
        self.feature_layer = nn.Linear(num_features, int(layers[0]/2))
        self.fn_layers = nn.ModuleList(nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) -1))
        self.fn_last = nn.Linear(layers[-1],4)
        self.drop_layer = nn.Dropout(p=corruption)
        
    def forward(self, token, feature):
        #c = self.chunk_emb(chunk)
        #feature = torch.cat([c,feature],dim=-1)
        t = self.token_layer(token)
        f = self.feature_layer(feature)
        output = torch.cat([t, f], dim = -1)
        output = torch.relu(output)
        for layer in self.fn_layers:
            output = self.drop_layer(output)
            output = layer(output)
            output = torch.relu(output)
        #output = self.drop_layer(output)
        logit = self.fn_last(output)
        return logit

    def freeze_except_classifier(self):

        for l in self.mlp_fc_layers:
            l.weight.requires_grad = False
            l.bias.requires_grad = False
        self.token_layer.weight.requires_grad = False
        self.token_layer.bias.requires_grad = False
        self.feature_layer.weight.requires_grad = False
        self.feature_layer.bias.requires_grad = False





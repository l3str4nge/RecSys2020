import torch
import torch.nn as nn

from transformers import BertForMaskedLM


def mean_emb_no_pad(H, L):
    mask = torch.arange(H.shape[1]).repeat(H.shape[0], 1).to(L.device)
    mask = (mask < L).float()
    mask[:, 0] = 0
    masked_h = H * mask.unsqueeze(2)
    mean_emb = (masked_h.sum(dim=1)/L)
    return mean_emb


class BertMLPNet(nn.Module):
    def __init__(self, num_tokens, num_features, emb_size, layers,corruption=0.2):
        super(BertMLPNet, self).__init__()

        layers = [layers] if type(layers) is int else layers
        #self.chunk_emb = nn.Embedding(15, 40)
        self.token_layer = nn.Linear(num_tokens, int(layers[0]/2))
        self.feature_layer = nn.Linear(num_features, int(layers[0]/2))

        self.embedding_layer = nn.Linear(emb_size, emb_size)
        layers[0] = layers[0]+emb_size

        self.fn_layers = nn.ModuleList(nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) -1))
        self.fn_last = nn.Linear(layers[-1],4)

        self.drop_layer = nn.Dropout(p=corruption)
        self.dropout = nn.Dropout()
        self.dropoutplus = nn.Dropout(p=0.35)    


    def forward(self, token, feature, inputs, masks, lens):

        # run bert forward
        (h_last, _) = self.bert(input_ids=inputs, attention_mask=masks)
        #mean_emb = mean_emb_no_pad(h_last, lens)
        #embedding = mean_emb
        embedding = h_last[:,0,:]

        # MLP forward
        t = torch.relu(self.token_layer(token))
        f = torch.relu(self.feature_layer(feature))

        e = torch.relu(self.embedding_layer(self.dropout(embedding)))
        e = self.dropoutplus(e)

        output = torch.cat([t, f, e], dim = -1)

        for layer in self.fn_layers:
            output = self.drop_layer(output)
            output = layer(output)
            output = torch.relu(output)
        #output = self.drop_layer(output)
        logit = self.fn_last(output)
        return logit


class MasterNet(nn.Module):
    def __init__(self, num_tokens, num_features, emb_size, layers,corruption=0.2):
        super(MasterNet, self).__init__()

        self.mlp_net = MLPNet(num_tokens, num_features, emb_size, layers, corruption)
        # self.bert should be assigned by main script


    def forward(self, token, feature, inputs, masks, lens):

        # run bert forward
        (h_last, _) = self.bert(input_ids=inputs, attention_mask=masks)
        mean_emb = mean_emb_no_pad(h_last, lens)

        embedding = mean_emb

        # MLP forward
        logit = self.mlp_net(token, feature, embedding)
        return logit


class MLPNet(nn.Module):
    def __init__(self, num_tokens, num_features, emb_size, layers,corruption=0.2):
        super(MLPNet, self).__init__()

        layers = [layers] if type(layers) is int else layers
        #self.chunk_emb = nn.Embedding(15, 40)
        self.token_layer = nn.Linear(num_tokens, int(layers[0]/2))
        self.feature_layer = nn.Linear(num_features, int(layers[0]/2))
        print("# features", num_features)

        self.embedding_layer = nn.Linear(emb_size, emb_size)
        layers[0] = layers[0]+emb_size

        self.fn_layers = nn.ModuleList(nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) -1))
        self.fn_last = nn.Linear(layers[-1],4)

        self.drop_layer = nn.Dropout(p=corruption)
        self.dropout = nn.Dropout()
        self.dropoutplus = nn.Dropout(p=0.35)    


    def forward(self, token, feature, embedding):
        # MLP forward
        t = torch.relu(self.token_layer(token))
        f = torch.relu(self.feature_layer(feature))

        e = torch.relu(self.embedding_layer(self.dropout(embedding)))
        e = self.dropoutplus(e)

        output = torch.cat([t, f, e], dim = -1)

        for layer in self.fn_layers:
            output = self.drop_layer(output)
            output = layer(output)
            output = torch.relu(output)
        #output = self.drop_layer(output)
        logit = self.fn_last(output)
        return logit

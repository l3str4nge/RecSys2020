import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, num_tokens, num_features, emb_size, layers,corruption=0.2):
        super(EmbeddingNet, self).__init__()

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
        self.dropoutplus = nn.Dropout(p=0.15)        

    def forward(self, token, feature, embedding):
        #c = self.chunk_emb(chunk)
        #feature = torch.cat([c,feature],dim=-1)
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

    def freeze_except_classifier(self):

        for l in self.mlp_fc_layers:
            l.weight.requires_grad = False
            l.bias.requires_grad = False
        self.token_layer.weight.requires_grad = False
        self.token_layer.bias.requires_grad = False
        self.feature_layer.weight.requires_grad = False
        self.feature_layer.bias.requires_grad = False


class EmbeddingHighWayNet(nn.Module):
    def __init__(self, num_tokens, num_features, emb_size, layers,corruption=0.2):
        super(EmbeddingHighWayNet, self).__init__()

        layers = [layers] if type(layers) is int else layers
        #self.chunk_emb = nn.Embedding(15, 40)
        self.token_layer = nn.Linear(num_tokens, int(layers[0]/2))
        self.feature_layer = nn.Linear(num_features, int(layers[0]/2))

        self.embedding_layer = nn.Linear(emb_size, emb_size)
        layers[0] = layers[0]+emb_size

        self.highway0 = nn.Linear(layers[0], layers[-2])
        self.highway1 = nn.Linear(layers[1], layers[-2])
        self.fn_layers = nn.ModuleList(nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) -1))
        self.fn_last = nn.Linear(layers[-1],4)

        self.drop_layer = nn.Dropout(p=corruption)
        self.dropout = nn.Dropout()
        self.dropoutplus = nn.Dropout(p=0.15)        

    def forward(self, token, feature, embedding):
        #c = self.chunk_emb(chunk)
        #feature = torch.cat([c,feature],dim=-1)
        t = torch.relu(self.token_layer(token))
        f = torch.relu(self.feature_layer(feature))

        e = torch.relu(self.embedding_layer(self.dropout(embedding)))
        e = self.dropoutplus(e)

        output = torch.cat([t, f, e], dim = -1)

        for i, layer in enumerate(self.fn_layers):
            if i == 0:
                h0 = self.drop_layer(output)
                h0 = self.highway0(h0)
                h0 = torch.relu(h0)
            if i == 1:
                h1 = self.drop_layer(output)
                h1 = self.highway1(h1)
                h1 = torch.relu(h1)

            output = self.drop_layer(output)
            output = layer(output)
            output = torch.relu(output)

            if i == 2:
                output = output + h0 + h1

        #output = self.drop_layer(output)
        logit = self.fn_last(output)
        return logit


class FFNetHistory(nn.Module):

    def __init__(self, args, num_features):
        super(FFNetHistory, self).__init__()
        self.args = args

        self.feature_layer = nn.Linear(num_features, 768)
        self.embedding_layer = nn.Linear(1024, 1024)
        
        self.attn = SeqAttnMatch(args)
        self.attn_fc = nn.Linear(1024, 1024)


        layers = [768+1024+1024+1024, 2500, 1500, 750, 150]

        self.highway0 = nn.Linear(layers[0], layers[-2])
        self.highway1 = nn.Linear(layers[1], layers[-2])
        self.fn_layers = nn.ModuleList(nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) -1))
        self.fn_last = nn.Linear(layers[-1],4)

        self.drop_layer = nn.Dropout(p=0.25)
        self.dropout = nn.Dropout()
        self.dropoutplus = nn.Dropout(p=0.15)     


        self.type_embedding = nn.Embedding(7, 1024, padding_idx=0)

    def forward(self, feature, embedding, history_feats, history_masks, type_mask):

        type_mask = type_mask + 1

        #print(feature.shape)
        #print(embedding.shape)
        #print(history_feats.shape)
        #print(history_masks.shape)
        #print(type_mask.shape)
        #print(type_mask)
        #print(history_masks)

        f = torch.relu(self.feature_layer(feature))

        e = torch.relu(self.embedding_layer(self.dropout(embedding)))
        e = self.dropoutplus(e)

        # attend over history
        type_embs = self.type_embedding(type_mask)
        a1, a2 = self.attn(embedding, history_feats, history_masks, type_embs)
        a1, a2 = [torch.relu(self.attn_fc(self.drop_layer(a))) for a in [a1, a2]]

        output = torch.cat([f, e, a1, a2], dim = -1)

        for i, layer in enumerate(self.fn_layers):
            if i == 0:
                h0 = self.drop_layer(output)
                h0 = self.highway0(h0)
                h0 = torch.relu(h0)
            if i == 1:
                h1 = self.drop_layer(output)
                h1 = self.highway1(h1)
                h1 = torch.relu(h1)

            output = self.drop_layer(output)
            output = layer(output)
            output = torch.relu(output)

            if i == 2:
                output = output + h0 + h1

        #output = self.drop_layer(output)
        logit = self.fn_last(output)
        return logit


class SeqAttnMatch(nn.Module):
    def __init__(self, args):
        super(SeqAttnMatch, self).__init__()

        self.linear_q = nn.Linear(1024, 1024)
        self.linear_k = nn.Linear(1024, 1024)
        self.linear_k2 = nn.Linear(1024, 1024)
        self.linear_v = nn.Linear(1024, 1024)

        self.emb_dropout_layer = nn.Dropout(0.25)
        self.dropout_layer = nn.Dropout(0.25)

        self.ff1 = nn.Linear(1024, 1024)
        self.ff2 = nn.Linear(1024, 1024)

    def forward(self, tweet, history, history_mask, type_embs):
        """
        Args:
            tweet: batch * hdim
            history: batch * len2 * hdim
            history_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * hdim
        """
        # Project vectors

        history = history

        tweet = self.emb_dropout_layer(tweet)
        tweet_project = self.linear_k(tweet)

        history = self.emb_dropout_layer(history) + type_embs
        history_project = self.linear_q(history)
        history_values = self.linear_v(history)

        # Compute scores
        scores = tweet_project.unsqueeze(1).bmm(history_project.transpose(2, 1))

        # Mask padding
        history_mask = history_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(history_mask.data, -float('inf'))

        # Normalize with softmax
        alpha = torch.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(history_values).squeeze()

        out1 = self.ff2(self.dropout_layer(torch.relu(self.ff1(matched_seq))))

        # do it again
        tweet_project = self.linear_k2(matched_seq)

        scores = tweet_project.unsqueeze(1).bmm(history_project.transpose(2, 1))
        
        scores.data.masked_fill_(history_mask.data, -float('inf'))
        alpha = torch.softmax(scores, dim=2)
        matched_seq = alpha.bmm(history_values).squeeze()
        out2 = self.ff2(self.dropout_layer(torch.relu(self.ff1(matched_seq))))

        return out1, out2

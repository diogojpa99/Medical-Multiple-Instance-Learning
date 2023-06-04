import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpCLassifier(nn.Module):
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super(MlpCLassifier, self).__init__()
        self.drop = dropout
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features)
        )
    
    def forward(self, x):
        """ 
        Forward pass of the MlpClassifier model.
            Input: x (Batch_size, in_features)
            Ouput: x (Batch_size, out_features)
        Note: Dropout layer was done this so the architecture can be saved and loaded without errors if
        the user uses or not dropout.  
        """
        if self.drop:
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.mlp(x)
   
class EmbeddingMIL(nn.Module):
    
    def __init__(self, 
                 num_classes,
                 N=196, 
                 embedding_size=256, 
                 dropout=0.1,
                 pooling_type="max",
                 args=None,
                 device="cuda",
                 patch_extractor:nn.Module = None):
        
        super(EmbeddingMIL, self).__init__()
        self.patch_extractor = patch_extractor   
        self.deep_classifier = MlpCLassifier(in_features=embedding_size, out_features=num_classes, dropout=dropout)    
        self.N = N     
        self.num_classes = num_classes   
        self.pooling_type = pooling_type.lower()
        self.args = args
        self.device = device
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.patch_extractor(x)
        
    def MaxPooling(self, scores):
        pooled_scores,_ = torch.max(scores, dim=1)
        return pooled_scores
    
    def AvgPooling(self, scores):
        pooled_scores = torch.mean(scores, dim=1)
        return pooled_scores
    
    def TopKPooling(self, scores, topk):
        pooled_scores,_ = torch.topk(scores, k=topk, dim=1)
        pooled_scores = torch.mean(pooled_scores, dim=1)
        return pooled_scores
    
    def forward(self,x):
        '''
        Forward pass of the EmbeddingMIL model.
            Input: x (Batch_size, 3, 224, 224)
            Ouput: x (Batch_size, num_classes)
            Where N = 14*14 = 196 (If we consider 16x16 patches)
        '''
        
        # (1) Use a CNN to extract the features | output : (batch_size, embedding_size, 14, 14)
        x = self.patch_extractor.forward_features(x)

        # Register Hook
        if x.requires_grad == True:
            x.register_hook(self.activations_hook)

        # (2) Transform input to shape(batch_size, N, embedding_size)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=2) # x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))
        
        # (3) Apply pooling to obtain the bag representation
        if self.pooling_type == "max":
            x = self.MaxPooling(x)
        elif self.pooling_type == "avg":
            x = self.AvgPooling(x)
        elif self.pooling_type == "topk":
            x = self.TopKPooling(x, self.args.topk)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', or 'avg' or 'topk'.")
        
        # (4) Apply a Mlp to obtain the bag label
        x = self.deep_classifier(x)

        return x #(Batch_size, num_classes)
       
class InstanceMIL(nn.Module):
    
    def __init__(self, 
                 num_classes,
                 N=196, 
                 embedding_size=256, 
                 dropout=0.1,
                 pooling_type="max",
                 args=None,
                 device="cuda",
                 patch_extractor:nn.Module = None):
        
        super(InstanceMIL, self).__init__()
        self.patch_extractor = patch_extractor   
        self.deep_classifier = MlpCLassifier(in_features=embedding_size, out_features=num_classes, dropout=dropout)    
        self.N = N     
        self.num_classes = num_classes   
        self.pooling_type = pooling_type.lower()
        self.args = args
        self.device = device
        self.patch_scores = None
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.patch_extractor(x)
    
    def save_patch_scores(self, x):
        self.patch_scores = x
        
    def get_patch_scores(self):
        return self.patch_scores
    
    def MaxPooling(self, scores):
        """Max Pooling operation for a given bag (image).
        
        Here we applied two methods.:
        1.  Comput the argmax for each bag. Then devide for the number of clases.
            This way we can opbatain the index of the patch (line) that contains the highest score for all classes.
            This means that this is the key patch that triggers the bag classification. 
        2.  We do the same thing, but in a different way. We see the max per image and its index. We then take the index
            to obtain the line thar contains the highest score for all classes.
        """
        pooled_scores = []
        for i in range(scores.size(0)):
            pooled_scores.append(scores[i][int(torch.argmax(scores[i])/self.num_classes)])
        pooled_scores = torch.stack(pooled_scores).to(self.device)
        
        """ for i in range(scores.size(0)):
            pooled_scores.append(scores[i][int(torch.max(scores[i],dim=0)[1][int(torch.argmax(torch.max(scores[i], dim = 0)[0]))])])
        pooled_scores = torch.stack(pooled_scores).to(self.device) """
        
        return pooled_scores
    
    def AvgPooling(self, scores):
        """ The representation of a given bag is given by the average
        of the scores of all the instances (patches) in the bag (image).
        """
        pooled_scores = torch.mean(scores, dim=1)
        return pooled_scores
    
    def TopKPooling(self, scores, topk):
        pooled_scores = []
        for i in range(scores.size(0)):
            max_vals,_ = torch.max(scores[i], dim=1) # Compute the max value row-wise
            _, indices = torch.topk(max_vals, topk) # Get the indices of the top-k values
            pooled_scores.append(torch.mean(scores[i][indices], dim=0)) # Compute the mean of the top-k values
        pooled_scores = torch.stack(pooled_scores).to(self.device)
        return pooled_scores
                
    def forward(self, x):
        '''
        Forward pass of the InstanceMIL model.
            Input: x (Batch_size, 3, 224, 224)
            Ouput: x (Batch_size, num_classes)
            Where N = 14*14 = 196 (If we consider 16x16 patches)
        '''
        
        # (1) Use a CNN to extract the features | output : (batch_size, embedding_size, 14, 14)
        x = self.patch_extractor.forward_features(x)
        
        # Register Hook
        if x.requires_grad == True:
            x.register_hook(self.activations_hook)
        
        # (2) Transform input to shape(batch_size, N, embedding_size)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=2) #x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))
        
        # (3) Convert x to shape (Batch_size*N, embedding_size)
        x = x.reshape(-1, x.size(2))
   
        # (4) Use a deep classifier to obtain the score for each instance
        x = self.deep_classifier(x)
        
        # (5) Transform x to shape (Batch_size, N, num_classes)
        x = x.view(-1, self.N, self.num_classes)
        
        # Save the scores for each patch
        self.save_patch_scores(x)
        
        # (6) Apply pooling to obtain the bag representation
        if self.pooling_type == "max":
            x = self.MaxPooling(x)
        elif self.pooling_type == "avg":
            x = self.AvgPooling(x)
        elif self.pooling_type == "topk":
            x = self.TopKPooling(x, self.args.topk)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', or 'avg' or 'topk'.")
    
        return x #(Batch_size, num_classes)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpCLassifier(nn.Module):
    
    def __init__(self, in_features, out_features, dropout=0.0):
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
        self.LogSoftmax = nn.LogSoftmax(dim=1)
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
        
    def MaxPooling(self, features):
        """ The representation of a given bag is given by the maximum value of the features of all the instances in the bag.
        Args:
            features (torch.Tensor): Features of each instance in the bag. Shape (Batch_size, N, embedding_size).
        Returns:
            torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
        """
        pooled_feature,_ = torch.max(features, dim=1)
        return pooled_feature
    
    def AvgPooling(self, features):
        """The representation of a given bag is given by the average value of the features of all the instances in the bag.
        Args:
            features (torch.Tensor): Features of each instance in the bag. Shape (Batch_size, N, embedding_size).
        Returns:
            torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
        """
        pooled_feature = torch.mean(features, dim=1)
        return pooled_feature
    
    def TopKPooling(self, features, topk):
        """The representation of a given bag is given by the average value of the top k features of all the instances in the bag.
        Args:
            features (torch.Tensor): Features of each instance in the bag. Shape (Batch_size, N, embedding_size).
            topk (torch.Tensor): Number of top k features to be pooled. Default = 25.
        Returns:
            torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
        """
        pooled_feature,_ = torch.topk(features, k=topk, dim=1)
        pooled_feature = torch.mean(pooled_feature, dim=1)
        return pooled_feature
    
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
        
        # (4) Apply a Mlp to obtain the bag label  (Batch_size, embedding_size)
        x = self.deep_classifier(x)
        
        # (5) Apply log to softmax values -> Using NLLLoss criterion
        x = self.LogSoftmax(x)

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
        self.Softmax = nn.Softmax(dim=2)
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
        """ Classical MIL: The representation of a given bag is given by the maximum probability
        of 'MEL' case. If the probability of 'MEL' is higher than the probability .5 then the bag is classified as 'MEL'. 
        Args:
            scores (torch.Tensor): Scores of each instance in the bag. Shape (Batch_size, N, num_classes).
        Returns:
            torch.Tensor: Pooled scores. Shape (Batch_size, num_classes). 
        """
        pooled_probs,_ = torch.max(scores[:,:,0], dim=1) # Get the maximum probability of the melanoma class (MEL:0) per bag.
        pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1) # Concatenate the pooled probabilities with the pooled NV probabilities -> In order to use the same NLLLoss function.
        return pooled_probs
    
    def AvgPooling(self, scores):
        """ The representation of a given bag is given by the average
        of the softmax probabilities of all the instances (patches) in the bag (image). Note: for the melanoma class.
        Args:
            scores (torch.Tensor): Scores of each instance in the bag. Shape (Batch_size, N, num_classes).
        Returns:
            torch.Tensor: Pooled scores. Shape (Batch_size, num_classes). 
        """
        pooled_probs = torch.mean(scores, dim=1)
        return pooled_probs
    
    def TopKPooling(self, scores, topk):
        """ The representation of a given bag is given by the average of the softmax probabilities
        of the top k instances (patches) in the bag (image). Note: for the melanoma class.
        Args:
            scores (torch.Tensor): Scores of each instance in the bag. Shape (Batch_size, N, num_classes).
        Returns:
            torch.Tensor: Pooled scores. Shape (Batch_size, num_classes). 
        """
        pooled_probs,_ = torch.topk(scores[:,:,0], k=topk, dim=1) # Get the maximum probability of the melanoma class (MEL:0) for the top k instances (per bag).
        pooled_probs = torch.mean(pooled_probs, dim=1)
        pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1) 
        return pooled_probs
                
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
        
        # (6) Apply softmax to the scores
        x = self.Softmax(x)
        
        # (7) Apply pooling to obtain the bag representation
        if self.pooling_type == "max":
            x = self.MaxPooling(x)
        elif self.pooling_type == "avg":
            x = self.AvgPooling(x)
        elif self.pooling_type == "topk":
            x = self.TopKPooling(x, self.args.topk)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', or 'avg' or 'topk'.")
        
        # (8) Apply log to softmax values 
        x = torch.log(x)
    
        return x #(Batch_size, num_classes)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Mask_Setup

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
                 patch_extractor_model="resnet18.tv_in1k",
                 patch_extractor:nn.Module = None):
        
        super(EmbeddingMIL, self).__init__()
        self.patch_extractor_model = patch_extractor_model
        self.patch_extractor = patch_extractor  
        self.deep_classifier = MlpCLassifier(in_features=embedding_size, out_features=num_classes, dropout=dropout)    
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.N = N     
        self.num_classes = num_classes   
        self.embedding_size = embedding_size
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
    
    def MaskAvgPooling(self, features, mask):
        """ Applying Global Average Pooling to the features of the patches that are inside the mask.
        All the patches outside the mask are set to zero. We only consider the probabilities of the patches that are inside the mask.
        Args:
            features (torch.Tensor): Features of each instance in the bag. Shape (Batch_size, N, embedding_size).
            mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
        Returns:
            torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
        """
                
        pooled_mask = Mask_Setup(mask=mask) # Transform mask into shape (Batch_size, N)

        # Compute masked mean for each bag
        pooled_feature = torch.zeros(len(features), features.size(2)).to(self.device) # shape (Batch_size, embedding_size)
        for i in range(len(features)):
            selected_features = features[i][pooled_mask[i].bool()]
            pooled_feature[i] = torch.mean(selected_features,dim=0)   
            
        pooled_feature = pooled_feature.to(self.device)
                
        return pooled_feature
    
    def MaskMaxPooling(self, features, mask):
        """ Applying Global Max Pooling to the features of the patches that are inside the mask.
        All the patches outside the mask are set to zero. We only consider the probabilities of the patches that are inside the mask.
        Args:
            features (torch.Tensor): Features of each instance in the bag. Shape (Batch_size, N, embedding_size).
            mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
        Returns:
            torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
        """

        pooled_mask = Mask_Setup(mask) # Transform mask into shape (Batch_size, N)

        # Compute masked mean for each bag
        pooled_feature = torch.zeros(len(features), features.size(2)).to(self.device) # shape (Batch_size, embedding_size)
        for i in range(len(features)):
            selected_features = features[i][pooled_mask[i].bool()]
            pooled_feature[i],_ = torch.max(selected_features,dim=0)   
            
        pooled_feature = pooled_feature.to(self.device)
                
        return pooled_feature
    
    def forward(self, x, mask=None):
        '''
        Forward pass of the EmbeddingMIL model.
            Input: x (Batch_size, 3, 224, 224)
            Ouput: x (Batch_size, num_classes)
            Where N = 14*14 = 196 (If we consider 16x16 patches)
        '''
        
        # (1) Use a CNN to extract the features | output : (batch_size, embedding_size, 14, 14)
        x = self.patch_extractor(x)
        
        if self.patch_extractor_model == "deit_small_patch16_224" or self.patch_extractor_model == "deit_base_patch16_224":
            # DEiT models return a tensor with shape (batch_size, embedding_size, N)
            # We need to reshape it to (batch_size, N, 14,14) in order to then use grad-cam
            x = x.permute(0, 2, 1)
            x = x.reshape(1, 2, 14, 14)

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
        elif self.pooling_type == "mask_avg":
            if mask is not None:
                x = self.MaskAvgPooling(x, mask)
            else:
                x = self.AvgPooling(x)
        elif self.pooling_type == "mask_max":
            if mask is not None:
                x = self.MaskMaxPooling(x, mask)
            else:
                x = self.MaxPooling(x)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', 'avg', 'topk', 'mask_avg' or 'mask_max'.")
        
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
                 patch_extractor_model="resnet18.tv_in1k",
                 patch_extractor:nn.Module = None):
        
        super(InstanceMIL, self).__init__()
        self.patch_extractor_model = patch_extractor_model
        self.patch_extractor = patch_extractor   
        self.deep_classifier = MlpCLassifier(in_features=embedding_size, out_features=num_classes, dropout=dropout)    
        self.Softmax = nn.Softmax(dim=2)
        self.N = N     
        self.num_classes = num_classes   
        self.embedding_size = embedding_size
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
    
    def save_patch_probs(self, x):
        self.patch_probs = x
        
    def get_patch_probs(self):
        return self.patch_probs
    
    def MaxPooling(self, probs):
        """ Classical MIL: The representation of a given bag is given by the maximum probability
        of 'MEL' case. If the probability of 'MEL' is higher than the probability .5 then the bag is classified as 'MEL'. 
        Args:
            probs (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
        Returns:
            torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
        """
        pooled_probs,_ = torch.max(probs[:,:,0], dim=1) # Get the maximum probability of the melanoma class (MEL:0) per bag.
        pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1) # Concatenate the pooled probabilities with the pooled NV probabilities -> In order to use the same NLLLoss function.
        return pooled_probs
    
    def AvgPooling(self, probs):
        """ The representation of a given bag is given by the average
        of the softmax probabilities of all the instances (patches) in the bag (image). Note: for the melanoma class.
        Args:
            probs (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
        Returns:
            torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
        """
        pooled_probs = torch.mean(probs, dim=1)
        return pooled_probs
    
    def TopKPooling(self, probs, topk):
        """ The representation of a given bag is given by the average of the softmax probabilities
        of the top k instances (patches) in the bag (image). Note: for the melanoma class.
        Args:
            probs (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
        Returns:
            torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
        """
        pooled_probs,_ = torch.topk(probs[:,:,0], k=topk, dim=1) # Get the maximum probability of the melanoma class (MEL:0) for the top k instances (per bag).
        pooled_probs = torch.mean(pooled_probs, dim=1)
        pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1) 
        return pooled_probs
    
    def MaskAvgPooling(self, probs, mask):
        """ Applying Average Pooling to the probabilities of the patches that are inside the mask.
        All the probabilities of the patches that are outside the mask are set to zero. We only consider
        the probabilities of the patches that are inside the mask.
        Args:
            probs (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
        Returns:
            torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
        """
        
        """ mask = F.avg_pool2d(mask, kernel_size=16, stride=16)  # Transform Mask into shape (Batch_size, 1, 14, 14)
        mask = torch.where(mask < 0.5, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32)) # Set mask values to 0 or 1 """
        
        pooled_mask = Mask_Setup(mask) # Transform mask into shape (Batch_size, N)
        masked_probs = probs[:,:,0] * pooled_mask # Apply mask to the probabilities of the melanoma class (MEL:0) 

        # Compute masked mean for each bag
        pooled_probs = torch.zeros(len(masked_probs)).to(self.device) # shape (Batch_size)
        for i in range(len(masked_probs)):
            pooled_probs[i] = torch.sum(masked_probs[i])/ len(torch.nonzero(masked_probs[i]))   
        
        pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1)
        pooled_probs = pooled_probs.to(self.device)
        
        return pooled_probs
    
    def MaskMaxPooling(self, probs, mask):
        """ Applying Max Pooling to the probabilities of the patches that are inside the mask.
        All the probabilities of the patches that are outside the mask are set to zero. We only consider
        the probabilities of the patches that are inside the mask.
        Args:
            probs (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
        Returns:
            torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
        """
        
        """ mask = F.avg_pool2d(mask, kernel_size=16, stride=16)  # Transform Mask into shape (Batch_size, 1, 14, 14)
        mask = torch.where(mask < 0.5, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32)) # Set mask values to 0 or 1 """
        
        pooled_mask = Mask_Setup(mask) # Transform mask into shape (Batch_size, N)
        masked_probs = probs[:,:,0] * pooled_mask # Apply mask to the probabilities of the melanoma class (MEL:0) 

        # Compute masked mean for each bag
        pooled_probs,_ = torch.max(masked_probs, dim=1)
        pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1)
        pooled_probs = pooled_probs.to(self.device)
        
        return pooled_probs
                
    def forward(self, x, mask=None):
        '''
        Forward pass of the InstanceMIL model.
            Input: x (Batch_size, 3, 224, 224)
            Ouput: x (Batch_size, num_classes)
            Where N = 14*14 = 196 (If we consider 16x16 patches)
        '''
        
        # (1) Use a CNN to extract the features | output : (batch_size, embedding_size, 14, 14)
        x = self.patch_extractor(x)
        
        if self.patch_extractor_model == "deit_small_patch16_224" or self.patch_extractor_model == "deit_base_patch16_224":
            # DEiT models return a tensor with shape (batch_size, embedding_size, N)
            # We need to reshape it to (batch_size, N, 14,14) in order to then use grad-cam
            x = x.permute(0, 2, 1)
            x = x.reshape(1, 2, 14, 14)
        
        # Register Hook to have access to the gradients
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
        x = x.view(-1, self.N, self.num_classes) # x = x.reshape(self.args.batch_size, self.N, self.num_classes) 
        
        # (6) Apply softmax to the scores
        x = self.Softmax(x)
        
        # Save the scores for each patch
        self.save_patch_probs(x)
        
        # (7) Apply pooling to obtain the bag representation
        if self.pooling_type == "max":
            x = self.MaxPooling(x)
        elif self.pooling_type == "avg":
            x = self.AvgPooling(x)
        elif self.pooling_type == "topk":
            x = self.TopKPooling(x, self.args.topk)
        elif self.pooling_type == "mask_avg":
            if mask is not None:
                x = self.MaskAvgPooling(x, mask)
            else:
                x = self.AvgPooling(x)
        elif self.pooling_type == "mask_max":
            if mask is not None:
                x = self.MaskMaxPooling(x, mask)
            else:
                x = self.MaxPooling(x)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', 'avg', 'topk', 'mask_avg' or 'mask_max'.")
        
        # (8) Apply log to softmax values 
        x = torch.log(x)
    
        return x #(Batch_size, num_classes)
    
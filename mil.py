import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['InstanceMIL', 'EmbeddingMIL']  

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

class MIL(nn.Module):
    
    def __init__(self, 
                num_classes: int = 2,
                N: int = 196, 
                embedding_size: int = 256, 
                dropout: float = 0.1,
                pooling_type: str = "max",
                is_training: bool = True,
                patch_extractor_model: str = "resnet18.tv_in1k",
                patch_extractor: nn.Module = None,
                device: str = "cuda:1",
                args=None) -> None:
    
        super().__init__()
        self.N = N     
        self.num_classes = num_classes   
        self.embedding_size = embedding_size
        self.pooling_type = pooling_type.lower()
        self.args = args
        self.device = device
        self.gradients = None
        self.is_training = is_training
        self.patch_extractor_model = patch_extractor_model
        self.patch_extractor = patch_extractor  
        self.deep_classifier = MlpCLassifier(in_features=embedding_size, out_features=num_classes, dropout=dropout)    
    
    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.patch_extractor(x)

    def MaxPooling(self, representation):
        
        if self.mil_type == "embedding":
            """ The representation of a given bag is given by the maximum value of the features of all the instances in the bag.
            Args:
                representation (torch.Tensor): features of each instance in the bag. Shape (Batch_size, N, embedding_size).
            Returns:
                torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
            """
            pooled_feature,_ = torch.max(representation, dim=1)
            return pooled_feature
        
        elif self.mil_type == "instance":
            """ Classical MIL: The representation of a given bag is given by the maximum probability
            of 'MEL' case. If the probability of 'MEL' is higher than the probability .5 then the bag is classified as 'MEL'. 
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """
            pooled_probs,_ = torch.max(representation[:,:,0], dim=1) # Get the maximum probability of the melanoma class (MEL:0) per bag.
            pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1) # Concatenate the pooled probabilities with the pooled NV probabilities -> In order to use the same NLLLoss function.
            return pooled_probs     
    
    def AvgPooling(self, representation):
        
        if self.mil_type == "embedding":
            """The representation of a given bag is given by the average value of the features of all the instances in the bag.
            Args:
                representation (torch.Tensor): features of each instance in the bag. Shape (Batch_size, N, embedding_size).
            Returns:
                torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
            """
            pooled_feature = torch.mean(representation, dim=1)
            return pooled_feature
        
        elif self.mil_type == "instance":
            """ The representation of a given bag is given by the average
            of the softmax probabilities of all the instances (patches) in the bag (image). Note: for the melanoma class.
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """
            pooled_probs = torch.mean(representation, dim=1)
            return pooled_probs
    
    def TopKPooling(self, representation, topk):
        
        if self.mil_type == "embedding":
            """The representation of a given bag is given by the average value of the top k features of all the instances in the bag.
            Args:
                representation (torch.Tensor): features of each instance in the bag. Shape (Batch_size, N, embedding_size).
                topk (torch.Tensor): Number of top k representation to be pooled. Default = 25.
            Returns:
                torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
            """
            pooled_feature,_ = torch.topk(representation, k=topk, dim=1)
            pooled_feature = torch.mean(pooled_feature, dim=1)
            return pooled_feature
        
        elif self.mil_type == "instance":
            """ The representation of a given bag is given by the average of the softmax probabilities
            of the top k instances (patches) in the bag (image). Note: for the melanoma class.
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """
            pooled_probs,_ = torch.topk(representation[:,:,0], k=topk, dim=1) # Get the maximum probability of the melanoma class (MEL:0) for the top k instances (per bag).
            pooled_probs = torch.mean(pooled_probs, dim=1)
            pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1) 
            return pooled_probs
        
    def MaskMaxPooling(self, representation, mask):
        
        pooled_mask = Mask_Setup(mask) # Transform mask into shape (Batch_size, N)
        
        if self.mil_type == "embedding":
            """ Applying Global Max Pooling to the features of the patches that are inside the mask.
            All the patches outside the mask are set to zero. We only consider the probabilities of the patches that are inside the mask.
            Args:
                representation (torch.Tensor): features of each instance in the bag. Shape (Batch_size, N, embedding_size).
                mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
            Returns:
                torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
            """
            pooled_feature = torch.zeros(len(representation), representation.size(2)).to(self.device) # shape (Batch_size, embedding_size)
            for i in range(len(representation)):
                selected_representation = representation[i][pooled_mask[i].bool()]
                pooled_feature[i],_ = torch.max(selected_representation,dim=0)   
                
            pooled_feature = pooled_feature.to(self.device)
                    
            return pooled_feature
        
        elif self.mil_type == 'instance':
            """ Applying Max Pooling to the probabilities of the patches that are inside the mask.
            All the probabilities of the patches that are outside the mask are set to zero. We only consider
            the probabilities of the patches that are inside the mask.
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
                mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """
            masked_probs = representation[:,:,0] * pooled_mask # Apply mask to the probabilities of the melanoma class (MEL:0) 
            
            # Compute masked mean for each bag
            pooled_probs,_ = torch.max(masked_probs, dim=1)
            pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1)
            pooled_probs = pooled_probs.to(self.device)
            
            return pooled_probs
    
    def MaskAvgPooling(self, representation, mask):
        
        pooled_mask = Mask_Setup(mask) # Transform mask into shape (Batch_size, N)
        
        if self.mil_type == "embedding":
            """ Applying Global Average Pooling to the features of the patches that are inside the mask.
            All the patches outside the mask are set to zero. We only consider the probabilities of the patches that are inside the mask.
            Args:
                representation (torch.Tensor): features of each instance in the bag. Shape (Batch_size, N, embedding_size).
                mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
            Returns:
                torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
            """
            # Compute masked mean for each bag
            pooled_feature = torch.zeros(len(representation), representation.size(2)).to(self.device) # shape (Batch_size, embedding_size)
            for i in range(len(representation)):
                selected_representation = representation[i][pooled_mask[i].bool()]
                pooled_feature[i] = torch.mean(selected_representation,dim=0)   
                
            pooled_feature = pooled_feature.to(self.device)
                    
            return pooled_feature
        
        elif self.mil_type == "instance":
            """ Applying Average Pooling to the probabilities of the patches that are inside the mask.
            All the probabilities of the patches that are outside the mask are set to zero. We only consider
            the probabilities of the patches that are inside the mask.
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
                mask (torch.Tensor): binary mask bag. Shape (Batch_size, 1, 224, 224).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """
            masked_probs = representation[:,:,0] * pooled_mask # Apply mask to the probabilities of the melanoma class (MEL:0) 

            # Compute masked mean for each bag
            pooled_probs = torch.zeros(len(masked_probs)).to(self.device) # shape (Batch_size)
            for i in range(len(masked_probs)):
                pooled_probs[i] = torch.sum(masked_probs[i])/len(torch.nonzero(masked_probs[i]))   
            
            pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1)
            pooled_probs = pooled_probs.to(self.device)
            
            return pooled_probs
        
    def MilPooling(self, x, mask=None):
        
        if self.pooling_type == "max":
            x = self.MaxPooling(x)
        elif self.pooling_type == "avg":
            x = self.AvgPooling(x)
        elif self.pooling_type == "topk":
            x = self.TopKPooling(x, self.args.topk)
        elif self.pooling_type == "mask_max":
            x = self.MaskMaxPooling(x, mask) if mask is not None else self.MaxPooling(x)
        elif self.pooling_type == "mask_avg":
            x = self.MaskAvgPooling(x, mask) if mask is not None else self.AvgPooling(x)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', 'avg', 'topk', 'mask_avg' or 'mask_max'.")
        
        return x
    

class EmbeddingMIL(MIL):
    
    def __init__(self, mil_type: str = "embedding", *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.mil_type = mil_type.lower()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, mask=None):
        '''
        Forward pass of the EmbeddingMIL model.
            Input: x (Batch_size, 3, 224, 224)
            Ouput: x (Batch_size, num_classes)
            Where N = 14*14 = 196 (If we consider 16x16 patches)
        '''
        
        # (1) Extract features from the Images: (Batch_size, 3, 224, 224) -> (Batch_size, embedding_size, 14, 14)
        x = self.patch_extractor(x)
        
        if self.patch_extractor_model == "deit_small_patch16_224" or self.patch_extractor_model == "deit_base_patch16_224":
            """ DEiT models return a tensor with shape (batch_size, embedding_size, N)
            We need to reshape it to (batch_size, embedding_size, 14, 14) in order to then use grad-cam """
            batch_size = x.size(0)
            x = x[:,1:,:] # Remove the CLS token of the patch sequence
            x = x.permute(0, 2, 1)
            x = x.reshape(batch_size, self.embedding_size, 14, 14)

        # Register Hook
        if not self.is_training:
            if x.requires_grad == True:
                x.register_hook(self.activations_hook)

        # (2) Transform input: (Batch_size, embedding_size, 14, 14) -> (Batch_size, N, embedding_size)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=2) # x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))
        
        # (3) Apply pooling to obtain the bag representation: (Batch_size, N, embedding_size) -> (Batch_size, embedding_size)
        x = self.MilPooling(x, mask)
        
        # (4) Apply a Mlp to obtain the bag label: (Batch_size, embedding_size) -> (Batch_size, num_classes)
        x = self.deep_classifier(x)
        
        # (5) Apply log to softmax values -> Using NLLLoss criterion
        x = self.LogSoftmax(x)

        return x #(Batch_size, num_classes)
    
class InstanceMIL(MIL):
    
    def __init__(self, mil_type: str = "instance", *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.mil_type = mil_type.lower()
        self.softmax_probs = None
        self.patch_probs = None
        self.Softmax = nn.Softmax(dim=2)

    def save_patch_probs(self, x):
        self.patch_probs = x
        
    def get_patch_probs(self):
        return self.patch_probs
    
    def save_softmax_bag_probs(self, x):
        self.softmax_probs = x
        
    def get_softmax_bag_probs(self):
        return self.softmax_probs
                    
    def forward(self, x, mask=None):
        '''
        Forward pass of the InstanceMIL model.
            Input: x (Batch_size, 3, 224, 224)
            Ouput: x (Batch_size, num_classes)
            Where N = 14*14 = 196 (If we consider 16x16 patches)
        '''
        
        # (1) Extract features from the Images: (Batch_size, 3, 224, 224) -> (Batch_size, embedding_size, 14, 14)
        x = self.patch_extractor(x)
        
        if self.patch_extractor_model == "deit_small_patch16_224" or self.patch_extractor_model == "deit_base_patch16_224":
            """ DEiT models return a tensor with shape (Batch_size, embedding_size, N)
            We need to reshape it to (Batch_size, embedding_size, 14, 14) in order to then use grad-cam """
            batch_size = x.size(0)
            x = x[:,1:,:] # Remove the CLS token of the patch sequence
            x = x.permute(0, 2, 1)
            x = x.reshape(batch_size, self.embedding_size, 14, 14)
        
        # Register Hook to have access to the gradients
        if not self.is_training:
            if x.requires_grad == True:
                x.register_hook(self.activations_hook)
        
        # (2) Transform input: (Batch_size, embedding_size, 14, 14) -> (Batch_size, N, embedding_size) -> (Batch_size*N, embedding_size)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=2) #x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))
        x = x.reshape(-1, x.size(2)) 

        # (4) Use a deep classifier to obtain the score for each instance: (Batch_size*N, embedding_size) -> (Batch_size*N, num_classes)
        x = self.deep_classifier(x)
        
        # (5) Transform x: (Batch_size*N, num_classes) -> (Batch_size, N, num_classes)
        x = x.view(-1, self.N, self.num_classes) # x = x.reshape(self.args.batch_size, self.N, self.num_classes) 
        
        # (6) Apply softmax to the scores
        x = self.Softmax(x)
        
        # Save the softmax probs for each patch
        self.save_patch_probs(x)
        
        # (7) Apply pooling to obtain the bag representation: (Batch_size, N, num_classes) -> (Batch_size, num_classes)
        x = self.MilPooling(x, mask)

        # Save the softmax probabilities of the bag
        self.save_softmax_bag_probs(x)
        
        # (8) Apply log to softmax values 
        x = torch.log(x)
    
        return x #(Batch_size, num_classes)

def Mask_Setup(mask):
    """ This function transforms the a binary mask shape (Batch_size, 1, 224, 224) into a mask of shape (Batch_size, N).
        If the Segmentation only contains zeros, the mask is transformed into a mask of ones.
    
    Args:
        mask (torxh.Tensor):Binary mask of shape (Batch_size, 1, 224, 224).
    Returns:
        torch.tensor: Binary mask of shape (Batch_size, N).
    """ 
    mask = F.max_pool2d(mask, kernel_size=16, stride=16)  # Transform Mask into shape (Batch_size, 1, 14, 14)
    mask = mask.reshape(mask.size(0), mask.size(2)*mask.size(3)) # Reshape mask to shape (Batch_size, N)
    
    for i in range (len(mask)):
        if len(torch.unique(mask[i])) == 1:
            mask[i] = torch.ones_like(mask[i])
        
    return mask

def Pretrained_Feature_Extractures(feature_extractor, args) -> str: 
    """Selects the right checkpoint for the selected feature extractor model.
    
    Args:
        feature_extractor (str): Name of the feature extractor model.
        args (**): Arguments from the parser.
    Returns:
        str: Path to the checkpoint of the feature extractor model.
    """
    checkpoints = ['https://download.pytorch.org/models/resnet18-5c106cde.pth', 
                   'https://download.pytorch.org/models/resnet50-19c8e357.pth', 
                   'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
                   'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
                   'https://download.pytorch.org/models/vgg16-397923af.pth',
                   'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
                   'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth']
    
    if feature_extractor == "resnet18.tv_in1k":
        return checkpoints[0]
    elif feature_extractor == "resnet50.tv_in1k":
        return checkpoints[1]
    elif feature_extractor == "deit_small_patch16_224":
        return checkpoints[2]
    elif feature_extractor == "deit_base_patch16_224":
        return checkpoints[3]
    elif feature_extractor == "vgg16.tv_in1k":
        return checkpoints[4]
    elif feature_extractor == "densenet169.tv_in1k":
        return checkpoints[5]
    elif feature_extractor == "efficientnet_b3":
        return checkpoints[6]
    else:
        raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Must be 'resnet18.tv_in1k',\
            'resnet50.tv_in1k', 'deit_small_patch16_224', 'deit_base_patch16_224', 'vgg16.tv_in1k' or 'efficientnet'.")
        
    
import torch
import torch.nn as nn
import torch.nn.functional as F

from Feature_Extractors import EViT as evit

__all__ = ['InstanceMIL', 'EmbeddingMIL']  
cnns_backbones = ['resnet18.tv_in1k', 'resnet50.tv_in1k', 'vgg16.tv_in1k', 'densenet169.tv_in1k', 'efficientnet_b3']
vits_backbones = ['deit_small_patch16_224', 'deit_base_patch16_224', 'deit_small_patch16_shrink_base']
deits_backbones = ['deit_small_patch16_224', 'deit_base_patch16_224']
evits_backbones = ['deit_small_patch16_shrink_base']

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
                device: str = "cuda:0",
                args=None) -> None:
    
        super().__init__()
        self.num_patches = N 
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
        self.evit_attn_tokens_idx = None 
        self.evit_inattn_tokens_idx = None
        self.count_tokens = 0
    
    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.patch_extractor(x)
    
    def save_evit_tokens_idx(self, attn_tokens_idx, inattn_tokens_idx):
        self.evit_attn_tokens_idx = attn_tokens_idx
        self.evit_inattn_tokens_idx = inattn_tokens_idx
        
    def get_evit_tokens_idx(self):
        """ This function returns the indexes of the attentive and innatentive tokens of the EViT model.
            But not the real ones. If you want to get the real indexes, you have to use a function from the EViT implementation.

        Returns:
            Tuple of lists: Tuple of lists with the indexes of the attentive and innatentive tokens of the EViT model.
        """
        return self.evit_attn_tokens_idx, self.evit_inattn_tokens_idx

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
            """ Classical MIL: The representation of a given bag is given by the maximum probability of 'MEL' case. 
            If the probability of 'MEL' is higher than the probability .5 then the bag is classified as 'MEL'. 
            
            Multiclass MIL: The representation of a given bag is given by the maximum probability of the most probable class.
            
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """
            if self.num_classes == 2:
                representation = torch.softmax(representation, dim=2)
                pooled_probs, pooled_idxs = torch.max(representation[:,:,0], dim=1) # Get the maximum probability of the melanoma class (MEL:0) per bag.
                pooled_probs = representation[torch.arange(pooled_probs.shape[0]), pooled_idxs]
            else:
                if self.args.multiclass_method == "first":
                    pooled_scores, pooled_idxs = torch.max(representation, dim=1) # Get the maximum probability of the most probable class per bag.
                    pooled_probs = torch.softmax(pooled_scores, dim=1)
                elif self.args.multiclass_method == "second":
                    probs = torch.softmax(representation, dim=2)
                    pooled_probs = []
                    for i in range(probs.size(0)):
                        pooled_probs.append(probs[i][int(torch.argmax(probs[i])/self.num_classes)])
                    pooled_probs = torch.stack(pooled_probs).to(self.device)
            
            # Count the number of times the cls_token or the fuse_tokens are selected as the most probable patch.            
            if self.patch_extractor_model in deits_backbones:
                self.count_tokens=(pooled_idxs==0).sum().item()  # Count the number of times the cls token is selected as the most probable patch.
            elif self.patch_extractor_model in evits_backbones:
                if self.args.fuse_token_filled and self.args.fuse_token: 
                    _, evit_inattn_tokens_idx = self.get_evit_tokens_idx()
                    pooled_idxs_exp = pooled_idxs.unsqueeze(1).expand(-1, evit_inattn_tokens_idx.shape[1])
                    self.count_tokens = (evit_inattn_tokens_idx == pooled_idxs_exp).any(dim=1).sum().item()
                if self.args.fuse_token and not self.args.fuse_token_filled:
                    self.count_tokens=(pooled_idxs==self.num_patches-1).sum().item()

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
            if self.num_classes == 2 or self.args.multiclass_method == "second":
                probs = torch.softmax(representation, dim=2)
                pooled_probs = torch.mean(probs, dim=1)
            elif self.args.multiclass_method == "first":
                pooled_represent= torch.mean(representation, dim=1)
                pooled_probs = torch.softmax(pooled_represent, dim=1)
                
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
            if self.num_classes == 2:
                representation = torch.softmax(representation, dim=2)
                pooled_probs, pooled_idxs = torch.topk(representation[:,:,0], k=topk, dim=1) # Get the maximum probability of the melanoma class (MEL:0) for the top k instances (per bag).
                pooled_probs = representation[torch.arange(pooled_probs.shape[0]).unsqueeze(1), pooled_idxs]
                pooled_probs = torch.mean(pooled_probs, dim=1)
            else:
                if self.args.multiclass_method == "first":
                    pooled_scores, pooled_idxs = torch.topk(representation, k=topk, dim=1)
                    pooled_scores = torch.mean(pooled_scores, dim=1)
                    pooled_probs = torch.softmax(pooled_scores, dim=1)
                elif self.args.multiclass_method == "second":
                    probs = torch.softmax(representation, dim=2)
                    pooled_probs = []
                    for i in range(probs.size(0)):
                        max_vals,_ = torch.max(probs[i], dim=1) # Compute the max value row-wise
                        _, indices = torch.topk(max_vals, topk) # Get the indices of the top-k values
                        pooled_probs.append(torch.mean(probs[i][indices], dim=0)) # Compute the mean of the top-k values
                    pooled_probs = torch.stack(pooled_probs).to(self.device)
                                        
            # Count the number of times the cls_token or the fuse_tokens are selected as the most probable patch.            
            if self.patch_extractor_model in deits_backbones:
                self.count_tokens=torch.sum(pooled_idxs==0).item() # Count the number of times the cls token is selected as the most probable patch.
            elif self.patch_extractor_model in evits_backbones:
                if self.args.fuse_token_filled and self.args.fuse_token: 
                    _, evit_inattn_tokens_idx = self.get_evit_tokens_idx()
                    pooled_idxs_exp=pooled_idxs.unsqueeze(2).expand(-1, -1, evit_inattn_tokens_idx.shape[1])
                    self.count_tokens=(evit_inattn_tokens_idx.unsqueeze(1)==pooled_idxs_exp).any(dim=2).sum().item()
                if self.args.fuse_token and not self.args.fuse_token_filled:
                    self.count_tokens=torch.sum(pooled_idxs==self.num_patches-1).item() # Count the number of times the fuse token is selected as the most probable patch.
                      
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
            if self.num_classes == 2:
                representation = torch.softmax(representation, dim=2)
                masked_probs = representation[:,:,0] * pooled_mask # Apply mask to the probabilities of the melanoma class (MEL:0)                 
                pooled_probs, pooled_idxs = torch.max(masked_probs, dim=1)
                pooled_probs = representation[torch.arange(pooled_probs.shape[0]), pooled_idxs]
                pooled_probs = pooled_probs.to(self.device)
            else:
                pool_mask = ~pooled_mask.bool()
                pooled_mask = pooled_mask.float()
                pool_mask = pool_mask.unsqueeze(-1).expand(-1, -1, self.num_classes) # Transform mask into shape (Batch_size, N, num_classes)
                pool_mask = pool_mask*torch.Tensor([-1000000])
                masked_scores = representation + pool_mask
                
                if self.args.multiclass_method == 'first':
                    pooled_scores, pooled_idxs = torch.max(masked_scores, dim=1) 
                    pooled_probs = torch.softmax(pooled_scores, dim=1)    
                elif self.args.multiclass_method == 'second':
                    raise ValueError(f"At the moment, we are only using the MaskMaxPooling for the first multiclass method.\n")
               
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
            if self.num_classes == 2:
                representation = torch.softmax(representation, dim=2)
                masked_probs = representation[:,:,0] * pooled_mask # Apply mask to the probabilities of the melanoma class (MEL:0) 

                # Compute masked mean for each bag
                pooled_probs = torch.zeros(len(masked_probs)).to(self.device) # shape (Batch_size)
                for i in range(len(masked_probs)):
                    pooled_probs[i] = torch.sum(masked_probs[i])/len(torch.nonzero(masked_probs[i]))   
                
                pooled_probs = torch.cat((pooled_probs.unsqueeze(1), 1-pooled_probs.unsqueeze(1)), dim=1)
                pooled_probs = pooled_probs.to(self.device)
            else:
                raise ValueError(f"At the moment, we are only using the MaskAvgPooling for the binary case.\n")
            
            return pooled_probs
        
    def MilPooling(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """ This function applies the MIL-Pooling to the input representation.
            Note that the shape of the input representation depends on the Mil-type.
            Note that the formulation of the "MIL-Problem" is different when we are in the Multiclass case.
        Args:
            x (torch.Tensor): Input representation. The shape of this tensor depends on the Mil-type.
                                If Mil-type is 'embedding', the shape is (Batch_size, N, embedding_size).
                                If Mil-type is 'instance', the shape is (Batch_size, N, num_classes).
            mask (torch.Tensor): Binary Masks. Shape: (Batch_size, 1, 224, 224). Defaults to None.
        Raises:
            ValueError: If pooling type is not 'max', 'avg', 'topk', 'mask_avg' or 'mask_max'. Raises ValueError.
        Returns:
            torch.Tensor: Pooled representation. Shape (Batch_size, embedding_size) or (Batch_size, num_classes).
        """
        
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
    
    def foward_features_cnns(self, x):
        """Foward features when the backbone for the MIL model is a CNN based model, such as ResNet, VGG, DenseNet, etc.
        Args:
            x (torch.Tensor): Input image. Shape (Batch_size, 3, 224, 224).
        Returns:
            torch.Tensor: Returns the features with shape (Batch_size, N, embedding_size).
        """
        x = self.patch_extractor(x) # (1) Extract features from the Images: (Batch_size, 3, 224, 224) -> (Batch_size, embedding_size, 14, 14)

        # Register Hook to have access to the gradients
        if not self.is_training:
            if x.requires_grad == True:
                x.register_hook(self.activations_hook)

        # (2) Transform input: (Batch_size, embedding_size, 14, 14) -> (Batch_size, N, embedding_size)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=2) # x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))
        
        return x
    
    def foward_features_vits(self, x):
        """ Foward features when the backbone for the MIL model is a Transformer based model.
            The 'attn_tokens_idx' contains the indexes of the attentive tokens of the EViT model.
        Args:
            x (torch.Tensor): Input image. Shape (Batch_size, 3, 224, 224).
        Returns:
            (torch.Tensor): Returns the features with shape (Batch_size, N, embedding_size).
        """
        attn_tokens_idx, inattn_tokens_idx = None, None
        
        if self.patch_extractor_model in evits_backbones:
            x, attn_tokens_idx = self.patch_extractor(x, keep_rate=self.args.base_keep_rate, get_idx=True)
        else:
            x = self.patch_extractor(x)
            
        if not self.args.cls_token:
            x = x[:,1:,:] # remove cls token
            
        if self.args.fuse_token_filled and self.patch_extractor_model in evits_backbones:
            fuse_token = x[:,-1,:]
            x = x[:,:-1,:]
            x, inattn_tokens_idx=EViT_Full_Fused_Attn_Map(x, fuse_token, attn_tokens_idx, 196, self.embedding_size, x.size(0))

        self.save_evit_tokens_idx(attn_tokens_idx, inattn_tokens_idx)
        self.num_patches = x.size(1) 
        
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
        
        # (1) Foward Features
        x = self.foward_features_cnns(x) if self.patch_extractor_model in cnns_backbones else self.foward_features_vits(x)
                    
        # (2) Apply pooling to obtain the bag representation: (Batch_size, N, embedding_size) -> (Batch_size, embedding_size)
        x = self.MilPooling(x, mask)
        
        # (3) Apply a Mlp to obtain the bag label: (Batch_size, embedding_size) -> (Batch_size, num_classes)
        x = self.deep_classifier(x)
        
        # (4) Apply log to softmax values -> Using NLLLoss criterion
        x = self.LogSoftmax(x)

        return x #(Batch_size, num_classes)

class InstanceMIL(MIL):
    
    def __init__(self, mil_type: str = "instance", *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.mil_type = mil_type.lower()
        self.softmax_probs = None
        self.patch_probs = None

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
        
        # (1) Foward Features: (Batch_size, 3, 224, 224) -> (Batch_size, N, embedding_size)
        x = self.foward_features_cnns(x) if self.patch_extractor_model in cnns_backbones else self.foward_features_vits(x)
        x = x.reshape(-1, x.size(2)) # Transform input: (Batch_size*N, embedding_size)

        # (2) Use a deep classifier to obtain the score for each instance: (Batch_size*N, embedding_size) -> (Batch_size*N, num_classes)
        x = self.deep_classifier(x) 
        x = x.view(-1, self.num_patches, self.num_classes) # Transform x to: (Batch_size, N, num_classes)
        
        self.save_patch_probs(torch.softmax(x, dim=2)) # Save the softmax probabilities of the patches (instances)
        
        # (3) Apply pooling to obtain the bag representation: (Batch_size, N, num_classes) -> (Batch_size, num_classes)
        x = self.MilPooling(x, mask)
        self.save_softmax_bag_probs(x) # Save the softmax probabilities of the bag
        
        # (4) Apply log to softmax values 
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
                   'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
                   'Feature_Extractors/Pretrained_EViTs/evit-0.7-img224-deit-s.pth',
                   'Feature_Extractors/Pretrained_EViTs/evit-0.7-fuse-img224-deit-s.pth'
                   ]
    
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
    elif feature_extractor == "deit_small_patch16_shrink_base":
        if args.base_keep_rate == 0.7:
            return checkpoints[7] if not args.fuse_token else checkpoints[8]
        else:
            raise ValueError(f"At the moment, we are only using the pretrained weights for EViT with base_keep_rate = 0.7.\n \
                Please, set base_keep_rate = 0.7 in the parser.\n Other pretrained weights will be added soon.")
    else:
        raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Must be 'resnet18.tv_in1k',\
            'resnet50.tv_in1k', 'deit_small_patch16_224', 'deit_base_patch16_224', 'vgg16.tv_in1k', 'efficientnet' or 'deit_small_patch16_shrink_base'")
        
def EViT_Full_Fused_Attn_Map(x:torch.Tensor=None, 
                            fuse_token:torch.Tensor=None,
                            idxs:list=None, 
                            N:int=196, 
                            D:int=384,
                            Batch_size:int=1) -> torch.Tensor:
    """ This functions transforms the output of EViT into a full map of the embeddings. The full map wil have a shape of (Batch_size, N, D).
        Where the innatentive tokens are filled with the fused token.
        
    Args:
        x (torch.Tensor): EViT's output. Shape(Batch_Size, num_left_tokens, D). Defaults to None.
        fuse_token (torch.Tensor): One of the fused tokens. Shape(Batch_size,D). Defaults to None.
        idxs (list): List of the idxs "removal layer" of the attentive tokens in EViT. Defaults to None.
        N (int): Original number of patches for a 224x224 image and 16x16 patches. Defaults to 196.
        D (int): Dimension of the embeddings. Defaults to 384.
        Batch_size (int): Batch_size. Defaults to 1.

    Returns:
        torch.Tensor: Full map of the embeddings. Shape(Batch_size, N, D). Where the innatentive tokens are filled with the fused token.
    """

    full_map = torch.zeros((Batch_size, N, D)).to(x.device) 
    indexes = idxs[-1].unsqueeze(-1).expand(-1, -1, D).clone()
    compl_idx = evit.complement_idx(idxs[-1], N)

    full_map.scatter_(1, index=indexes, src=x)
    full_map.scatter_(1, index=compl_idx.unsqueeze(-1).expand(-1, -1, D), src=fuse_token.unsqueeze(1).repeat(1, compl_idx.size(1), 1))

    return full_map, compl_idx
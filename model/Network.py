import torch
import torch.nn as nn
import model.clip_utils as clip
from .model import*
from .template import*
from utils import*

def load_clip_to_cpu(args):
    url = clip.MODELS[args.backbonename]
    model_path =clip.download(url)
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = build_model(state_dict or model.state_dict())

    return model

class UnlearningCLIP(nn.Module):
    def __init__(self, coarse_classnames,fine_classnames,args):
        super(UnlearningCLIP, self).__init__()
        # Encoders from CLIP
        clip_model = load_clip_to_cpu(args)
        clip_model.to('cuda')
        self.temp = CUSTOM_TEMPLATES[args.template]
        self.coarse_classnames=coarse_classnames
        self.fine_classnames=fine_classnames
        self.clip_model = clip_model

    def forward(self, images,training):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        mean_fine_text_features=0
        mean_coarse_text_features=0
        if training:
            single_template= random.choice(self.temp)
            x_coarse = [single_template.replace("{}", name) for name in self.coarse_classnames]
            x_tokenized_coarse = torch.cat([clip.tokenize(p) for p in x_coarse])
            x_fine = [single_template.replace("{}", name) for name in self.fine_classnames]
            x_tokenized_fine = torch.cat([clip.tokenize(p) for p in x_fine])
            text_features_coarse = self.clip_model.encode_text(x_tokenized_coarse.cuda())
            text_features_fine = self.clip_model.encode_text(x_tokenized_fine.cuda())
            text_features_fine=text_features_fine / text_features_fine.norm(dim=-1, keepdim=True)
            text_features_coarse = text_features_coarse / text_features_coarse.norm(dim=-1, keepdim=True)
            mean_fine_text_features=mean_fine_text_features+text_features_fine
            mean_coarse_text_features=mean_coarse_text_features+text_features_coarse
        else:
            for single_template in self.temp:
                x_coarse = [single_template.replace("{}", name) for name in self.coarse_classnames]
                x_tokenized_coarse = torch.cat([clip.tokenize(p) for p in x_coarse])
                x_fine = [single_template.replace("{}", name) for name in self.fine_classnames]
                x_tokenized_fine = torch.cat([clip.tokenize(p) for p in x_fine])
                text_features_coarse = self.clip_model.encode_text(x_tokenized_coarse.cuda())
                text_features_fine = self.clip_model.encode_text(x_tokenized_fine.cuda())
                text_features_fine=text_features_fine / text_features_fine.norm(dim=-1, keepdim=True)
                text_features_coarse = text_features_coarse / text_features_coarse.norm(dim=-1, keepdim=True)
                mean_fine_text_features=mean_fine_text_features+text_features_fine
                mean_coarse_text_features=mean_coarse_text_features+text_features_coarse
            mean_coarse_text_features= mean_coarse_text_features/len(self.temp)
            mean_fine_text_features= mean_fine_text_features/len(self.temp)
        text_feature_fine =  mean_fine_text_features /  mean_fine_text_features.norm(dim=-1, keepdim=True)
        text_feature_coarse =  mean_coarse_text_features /  mean_coarse_text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_coarse = logit_scale * image_features @ text_feature_coarse.t()
        logits_fine = logit_scale * image_features @ text_feature_fine.t()
        return logits_coarse,logits_fine,image_features

class ZeroshotCLIP(nn.Module):
    def __init__(self, coarse_classnames,fine_classnames,args):
        super(ZeroshotCLIP, self).__init__()
        # Encoders from CLIP
        clip_model = load_clip_to_cpu(args)
        clip_model.to('cuda')
        self.temp = CUSTOM_TEMPLATES[args.template]
        #load_pretrained_weights(clip_model,'/projectnb/ivc-ml/yuwentan/Unlearning/checkpoint/Compcars/Unlearning/Ours/fine/Cosine-Epo_10-Lr_0.00005/proj_fine_min_acc.pth')
        temp = CUSTOM_TEMPLATES[args.template]
        all_coarse_features = []
        all_fine_features = []
        for single_template in temp:
            x_coarse = [single_template.replace("{}", name) for name in coarse_classnames]
            x_tokenized_coarse = torch.cat([clip.tokenize(p) for p in x_coarse])
            x_fine = [single_template.replace("{}", name) for name in fine_classnames]
            x_tokenized_fine = torch.cat([clip.tokenize(p) for p in x_fine])
            with torch.no_grad():
                text_features_coarse = clip_model.encode_text(x_tokenized_coarse.cuda())
                text_features_fine = clip_model.encode_text(x_tokenized_fine.cuda())
                text_features_fine=text_features_fine / text_features_fine.norm(dim=-1, keepdim=True)
                text_features_coarse = text_features_coarse / text_features_coarse.norm(dim=-1, keepdim=True)
            all_coarse_features.append(text_features_coarse.unsqueeze(1))
            all_fine_features.append(text_features_fine.unsqueeze(1))
        self.text_features_coarse =torch.cat(all_coarse_features, dim=1).mean(dim=1) #text_features_coarse
        self.text_features_fine = torch.cat(all_fine_features, dim=1).mean(dim=1)
        self.text_features_fine = self.text_features_fine / self.text_features_fine.norm(dim=-1, keepdim=True)
        self.text_features_coarse = self.text_features_coarse / self.text_features_coarse.norm(dim=-1, keepdim=True)
        self.clip_model = clip_model
    
    def scale_by_distance(self, text_embedding, target_embeddings):
        target_embeddings = target_embeddings.unsqueeze(1)  # Shape [M, 1, D]
        cosine_similarities = torch.sum(target_embeddings * text_embedding.unsqueeze(0), dim=-1, keepdim=True)  # Shape [M, N, 1]
        combined_cosine_similarity, _ = torch.max(cosine_similarities, dim=0)  # Shape [N, 1]
        top10_values, top10_indices = torch.topk(combined_cosine_similarity.squeeze(), k=10, dim=0)
        print("Top 10 Combined Cosine Similarities:")
        for i, (value, idx) in enumerate(zip(top10_values, top10_indices)):
            print(f"Rank {i+1}: Value = {value.item()}, Index = {idx.item()}")

        scaling_factor = torch.ones_like(combined_cosine_similarity)  # Shape [N, 1]
        scaling_factor[combined_cosine_similarity >= 0.99] = 1 - combined_cosine_similarity[combined_cosine_similarity >= 0.99]
        scaled_embedding = text_embedding * scaling_factor
    
        return scaled_embedding


    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_coarse = logit_scale * image_features @ self.text_features_coarse.t()
        logits_fine = logit_scale * image_features @ self.text_features_fine.t()
        return logits_coarse,logits_fine,image_features



import torch

class EpisodeSampler():
    def __init__(self, images, labels, n_way, k_shot, q_query):
        self.images = images
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.classes = torch.unique(labels)

        #dict class: indexes
        self.class_to_indices = {
            class_label.item(): torch.where(labels == class_label)[0]
            for class_label in torch.unique(labels)
        }

    def sample_episode(self):
        selected_classes = self.classes[torch.randperm(len(self.classes))[:self.n_way]]
        
        support_images = []
        support_labels = []
        
        query_images = []
        query_labels = []

        for local_label, cls in enumerate(selected_classes):
            cls_int = int(cls.item())
            cls_indices = self.class_to_indices[cls_int]

            perm = torch.randperm(len(cls_indices))[: self.k_shot + self.q_query]
            selected_indexes = cls_indices[perm]

            support_indexes = selected_indexes[:self.k_shot]
            query_indexes = selected_indexes[self.k_shot:]

            support_images.append(self.images[support_indexes])
            support_labels.append(torch.full((self.k_shot,), local_label, dtype=torch.long))

            query_images.append(self.images[query_indexes])
            query_labels.append(torch.full((self.q_query,), local_label, dtype=torch.long))

        support_images = torch.cat(support_images, dim=0)
        support_labels = torch.cat(support_labels, dim=0)
        query_images = torch.cat(query_images, dim=0)
        query_labels = torch.cat(query_labels, dim=0)
        return support_images, support_labels, query_images, query_labels, selected_classes
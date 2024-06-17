import math
import numpy as np
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import affine_grid, grid_sample, normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

class SimSpritesVideo:
    def __init__(self, timesteps, frame_sizes, delta_t, attractor=None):
        self.attractor = torch.tensor(attractor) if attractor is not None else None
        self.timesteps = timesteps
        self.frame_sizes = np.array(frame_sizes)
        self.delta_t = delta_t

    @torch.no_grad()
    def sim_video(self, sprites):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        sprite_shape = np.array(sprites.shape[1:3])
        s_factor = self.frame_sizes[0] / sprite_shape[0]
        t_factors = (self.frame_sizes - sprite_shape) / sprite_shape
        t_factors = t_factors.astype('float32')
        sprite_vids = []
        Xs, Vs = self.sim_trajectories(num_tjs=len(sprites))
        for k in range(len(sprites)):
            obj_image = torch.from_numpy(sprites[k]).float().unsqueeze(dim=0)
            scaling = torch.Tensor([[s_factor, 0],
                                    [0, s_factor]])

            video = []
            for t in range(self.timesteps):
                thetas = torch.cat((scaling,
                                    (Xs[k, t] * t_factors).unsqueeze(dim=-1)),
                                   dim=-1).unsqueeze(dim=0)
                grid = affine_grid(thetas, torch.Size((1, 3,
                                                       self.frame_sizes[0],
                                                       self.frame_sizes[1])),
                                   align_corners=True)
                frame = grid_sample(obj_image.transpose(1, -1), grid,
                                    mode='nearest', align_corners=True)
                video.append(frame.transpose(1, -1))
            video = torch.cat(video, dim=0)
            sprite_vids.append(video)
        return torch.stack(sprite_vids, dim=0).sum(0).clamp(min=0, max=255).numpy().astype('uint8')

    def sim_trajectories(self, num_tjs):
        Xs = []
        Vs = []
        x0 = Uniform(-1, 1).sample((num_tjs, 2))
        for i in range(num_tjs):
            x, v = self.sim_trajectory(init_xs=x0[i])
            Xs.append(x.unsqueeze(0))
            Vs.append(v.unsqueeze(0))
        return torch.cat(Xs, 0), torch.cat(Vs, 0)

    def sim_trajectory(self, init_xs):
        ''' Generate a random sequence of a sprite '''
        if self.attractor is None:
            v_norm = Uniform(0, 1).sample() * 2 * math.pi
            v_y = torch.sin(v_norm).item()
            v_x = torch.cos(v_norm).item()
            V0 = torch.Tensor([v_x, v_y])
        else:
            if len(self.attractor) > 2:
                attractor = np.random.choice(self.attractor)
            V0 = normalize(self.attractor - init_xs, dim=0)
        X = torch.zeros((self.timesteps, 2))
        V = torch.zeros((self.timesteps, 2))
        X[0] = init_xs
        V[0] = V0
        for t in range(0, self.timesteps -1):
            X_new = X[t] + V[t] * self.delta_t
            V_new = V[t]

            if X_new[0] < -1.0:
                X_new[0] = -1.0 + torch.abs(-1.0 - X_new[0])
                V_new[0] = - V_new[0]
            if X_new[0] > 1.0:
                X_new[0] = 1.0 - torch.abs(X_new[0] - 1.0)
                V_new[0] = - V_new[0]
            if X_new[1] < -1.0:
                X_new[1] = -1.0 + torch.abs(-1.0 - X_new[1])
                V_new[1] = - V_new[1]
            if X_new[1] > 1.0:
                X_new[1] = 1.0 - torch.abs(X_new[1] - 1.0)
                V_new[1] = - V_new[1]
            V[t+1] = V_new
            X[t+1] = X_new
        return X, V

class MultiObjectDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        assert 'collate_fn' not in kwargs
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):

        # The input is a batch of (image, label_dict)
        _, item_labels = batch[0]
        keys = item_labels.keys()

        # Max label length in this batch
        # max_len[k] is the maximum length (in batch) of the label with name k
        # If at the end max_len[k] is -1, labels k are (probably all) scalars
        max_len = {k: -1 for k in keys}

        # If a label has more than 1 dimension, the padded tensor cannot simply
        # have size (batch, max_len). Whenever the length is >0 (i.e. the sequence
        # is not empty, store trailing dimensions. At the end if 1) all sequences
        # (in the batch, and for this label) are empty, or 2) this label is not
        # a sequence (scalar), then the trailing dims are None.
        trailing_dims = {k: None for k in keys}

        # Make first pass to get shape info for padding
        for _, labels in batch:
            for k in keys:
                try:
                    max_len[k] = max(max_len[k], len(labels[k]))
                    if len(labels[k]) > 0:
                        trailing_dims[k] = labels[k].size()[1:]
                except TypeError:   # scalar
                    pass

        # For each item in the batch, take each key and pad the corresponding
        # value (label) so we can call the default collate function
        pad = MultiObjectDataLoader._pad_tensor
        for i in range(len(batch)):
            for k in keys:
                if trailing_dims[k] is None:
                    continue
                size = [max_len[k]] + list(trailing_dims[k])
                batch[i][1][k] = pad(batch[i][1][k], size)

        return default_collate(batch)

    @staticmethod
    def _pad_tensor(x, size, value=None):
        assert isinstance(x, torch.Tensor)
        input_size = len(x)
        if value is None:
            value = float('nan')

        # Copy input tensor into a tensor filled with specified value
        # Convert everything to float, not ideal but it's robust
        out = torch.zeros(*size, dtype=torch.float)
        out.fill_(value)
        if input_size > 0:  # only if at least one element in the sequence
            out[:input_size] = x.float()
        return out


class MultiObjectDataset(Dataset):

    def __init__(self, data_path, train, split=0.9):
        super().__init__()

        # Load data
        data = np.load(data_path, allow_pickle=True)

        # Rescale images and permute dimensions
        x = np.asarray(data['x'], dtype=np.float32) / 255
        x = np.transpose(x, [0, 3, 1, 2])  # batch, channels, h, w

        # Get labels
        labels = data['labels'].item()

        # Split train and test
        split = int(split * len(x))
        if train:
            indices = range(split)
        else:
            indices = range(split, len(x))

        # From numpy/ndarray to torch tensors (labels are lists of tensors as
        # they might have different sizes)
        self.x = torch.from_numpy(x[indices])
        self.labels = self._labels_to_tensorlist(labels, indices)

    @staticmethod
    def _labels_to_tensorlist(labels, indices):
        out = {k: [] for k in labels.keys()}
        for i in indices:
            for k in labels.keys():
                t = labels[k][i]
                t = torch.as_tensor(t)
                out[k].append(t)
        return out

    def __getitem__(self, index):
        x = self.x[index]
        labels = {k: self.labels[k][index] for k in self.labels.keys()}
        return x, labels

    def __len__(self):
        return self.x.size(0)

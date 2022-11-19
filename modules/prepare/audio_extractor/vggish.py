import numpy as np
import os
import torch
import torch.nn as nn

from .vggish_input import wavfile_to_examples, wavform_to_examples
from .vggish_params import *


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)

class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self, with_quantization=True):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (EMBEDDING_SIZE, EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)
        self.with_quantization = with_quantization

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        if(self.with_quantization):
            # Quantize by:
            # - clipping to [min, max] range
            clipped_embeddings = torch.clamp(
                pca_applied, QUANTIZE_MIN_VAL, QUANTIZE_MAX_VAL
            )
            # - convert to 8-bit in range [0.0, 255.0]
            quantized_embeddings = torch.round(
                (clipped_embeddings - QUANTIZE_MIN_VAL)
                * (
                    255.0
                    / (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL)
                )
            )
            return quantized_embeddings
        else:
            return pca_applied

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())

class VGGish(VGG):
    def __init__(self, model_dir, device=None, pretrained=True, preprocess=True, postprocess=True):
        super().__init__(make_layers())
        if pretrained:
            state_dict = torch.load(os.path.join(model_dir, 'vggish-10086976.pth'))
            super().load_state_dict(state_dict)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess
        if self.postprocess:
            self.pproc = Postprocessor(with_quantization=False)
            if pretrained:
                state_dict = torch.load(os.path.join(model_dir, 'vggish_pca_params-970ea276.pth'))
                # TODO: Convert the state_dict to torch
                state_dict[PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                    state_dict[PCA_EIGEN_VECTORS_NAME], dtype=torch.float
                )
                state_dict[PCA_MEANS_NAME] = torch.as_tensor(
                    state_dict[PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
                )

                self.pproc.load_state_dict(state_dict)
        self.to(self.device)

    def forward(self, x, fs=None):
        if self.preprocess:
            x = self._preprocess(x, fs)
        batch_size = 64
        N = x.shape[0]
        feats = []
        for i in range(0, N, batch_size):
            cur_x = x[i:i+batch_size]
            cur_x = torch.tensor(cur_x, requires_grad=False)[:, None, :, :].float()
            cur_x = cur_x.to(self.device)
            cur_x = VGG.forward(self, cur_x)
            if self.postprocess:
                cur_x = self._postprocess(cur_x)
            feats.append(cur_x.detach().cpu().numpy())
        
        return np.squeeze(np.concatenate(feats, axis=0))

    def _preprocess(self, x, fs):
        if isinstance(x, np.ndarray):
            x = waveform_to_examples(x, fs, return_tensor=False)
        elif isinstance(x, str):
            x = wavfile_to_examples(x, return_tensor=False)
        else:
            raise AttributeError
        return x

    def _postprocess(self, x):
        return self.pproc(x)

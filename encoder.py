import torch
import torch.nn as nn
import torch.nn.functional as F

from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate, soft


class PRIN_Encoder(nn.Module):
    def __init__(self,
                 final_output_dim,
                 sconv_dims    = (64, 40, 40, 64), 
                 BANDWIDTH_IN  = 32, 
                 BANDWIDTH_OUT = 32, 
                 #R_IN=64,
                 #sconv_intermed_dims
                ):
        super().__init__()

        #self.features = [R_IN, 40, 40, dim_output]
        self.features   = list(sconv_dims)
        self.bandwidths = [BANDWIDTH_IN, 32, 32, BANDWIDTH_OUT]
        #self.linear1 = nn.Linear(dim_output, 50)
        #self.linear2 = nn.Linear(50, 50)

        ### SConv module ### 
        sequence = []
        # S2 layer
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(
                    S2Convolution(self.features[0], 
                                  self.features[1], 
                                  self.bandwidths[0], 
                                  self.bandwidths[1], 
                                  grid)
                    )
        # SO3 layers
        for l in range(1, len(self.features) - 1):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]
            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(nn.ReLU())
            grid = so3_equatorial_grid(max_beta=0, 
                                       max_gamma=0, 
                                       n_alpha=2 * b_in, 
                                       n_beta=1, 
                                       n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))
        # Final BN and non-linearity
        sequence.append(nn.BatchNorm3d(self.features[-1], affine=True))
        sequence.append(nn.ReLU())
        self.sequential = nn.Sequential(*sequence)
        
        # Per-point processor
        self.per_point_dims = [100, 150]
        self.per_point_processor = nn.Sequential(
            nn.Conv1d(self.features[-1], self.per_point_dims[0], 1, bias=False),
            nn.BatchNorm1d(self.per_point_dims[0]),
            nn.ReLU(False),
            nn.Conv1d(self.per_point_dims[0], self.per_point_dims[1], 1, bias=False),
            nn.BatchNorm1d(self.per_point_dims[1]),
            nn.ReLU(False)
        )
        # Final pooled processor
        self.pooled_processor_dims = [ 2 * self.per_point_dims[-1], 
                                       final_output_dim, 
                                       final_output_dim ]
        self.pooled_processor = nn.Sequential(
            nn.Linear(self.pooled_processor_dims[0], self.pooled_processor_dims[1]),
            nn.BatchNorm1d(self.pooled_processor_dims[1]),
            nn.ReLU(),
            nn.Linear(self.pooled_processor_dims[1], self.pooled_processor_dims[2]),
        )

    def forward(self, x, target_index):  # pylint: disable=W0221
        # Run SO(3) spherical convs
        # On spherical voxels space
        # B * C * a * b * c --> batch x channels x voxels_1 x voxels_2 x voxels_3
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        # Map from the voxel image back to the point cloud
        # x = "image" batch S^2 (with radius factored out)
        # target_index = grid of positions to sample from
        # B * C * N * 1 * 1
        features = F.grid_sample(x, target_index[:, :, None, None, :])
        # Process individual points (PointNet-like way) [B x C_out x N]
        x = self.per_point_processor(features.squeeze(3).squeeze(3))
        # Pool features over the points [B x 2C_out]
        x = torch.cat([ x.max(dim = -1)[0],
                        x.mean(dim = -1)    ],
                      dim = 1)
        # Final processing to get encoding [B x dim_output]
        return self.pooled_processor(x)
        
        # Reshape into nicer format
        # B * N * C
        #features = features.squeeze(3).squeeze(3).permute([0, 2, 1]).contiguous()
        # 
        # B * N * (C + 16)
        #prediction = torch.cat( [ features, 
        #                          cat_onehot[:, None, :].repeat(1, features.size(1), 1)
        #                        ], dim=2)
        # 
        # B * N * C
        #x = self.linear2( F.relu(self.linear1(features)) )
        #prediction = self.linear2(prediction)
        #prediction = F.log_softmax(prediction, dim=2)
        #return features, prediction




#

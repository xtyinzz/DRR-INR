import torch
import numpy as np
from torch.utils.data import Dataset
from itertools import combinations

class ResidualSineLayer(torch.nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = torch.nn.Linear(features, features, bias=bias)
        self.linear_2 = torch.nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
    #

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)

###############################################################################################################

class Siren_Residual(torch.nn.Module):
    '''
    in_features are 6 dimensions: hyper parameter * 3 and x y z
    '''
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, dropout=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(ResidualSineLayer(hidden_features, bias=True, ave_first=(i>0), ave_second=(i==(hidden_layers-1))))

        if dropout:
            self.net.append(torch.nn.Dropout(p=0.2))
        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = torch.nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output

###############################################################################################################
    
class Siren_Residual_Surrogate(torch.nn.Module):
    '''
    in_features are 6 dimensions: hyper parameter * 3 and x y z
    '''
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, dropout=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.fourier_feature = []
        self.fourier_feature.append(SineLayer(3, hidden_features-3, is_first=True, omega_0=first_omega_0))
        self.fourier_feature = torch.nn.Sequential(*self.fourier_feature)

        for i in range(hidden_layers):
            self.net.append(ResidualSineLayer(hidden_features, bias=True, ave_first=(i>0), ave_second=(i==(hidden_layers-1))))

        if dropout:
            self.net.append(torch.nn.Dropout(p=0.2))
        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = torch.nn.Sequential(*self.net)
    
    def forward(self, params, coords):
        # params = params.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self.fourier_feature(coords)
        inputs = torch.cat((params, coords), 1)
        output = self.net(inputs)
        return output

###############################################################################################################

class Siren_Surrogate(torch.nn.Module):
    '''
    in_features are 6 dimensions: hyper parameter * 3 and x y z
    '''
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.fourier_feature = []
        self.fourier_feature.append(SineLayer(3, hidden_features-3, is_first=True, omega_0=first_omega_0))
        self.fourier_feature = torch.nn.Sequential(*self.fourier_feature)

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = torch.nn.Sequential(*self.net)
    
    def forward(self, params, coords):
        # params = params.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = self.fourier_feature(coords)
        inputs = torch.cat((params, coords), 1)
        output = self.net(inputs)
        return output
    
class PosEncoding(torch.nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, coords):
        coords_pos_enc = coords
        in_features = coords.shape[-1]

        for i in range(self.num_frequencies):
            for j in range(in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc

class SnakeAlt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return (input + 1. - torch.cos(2. * input)) / 2.


class Inr_Surrogate(torch.nn.Module):
    '''
    in_features are 6 dimensions: hyper parameter * 3 and x y z
    '''
    def __init__(self, dsp=3, ch=64, num_frequencies=10):
        super().__init__()
        # dsp  - dimensions of the simulation parameters
        # ch   - channel multiplier
        # num_frequencies  -  number of frequencies for positional encoding
        self.dsp = dsp
        self.num_frequencies = num_frequencies
        self.in_features = 3
        self.ch = ch

        self.pos_encoding = PosEncoding(self.num_frequencies)
        self.pos_subnet = torch.nn.Sequential(
            torch.nn.Linear(self.in_features + 2 * self.in_features * self.num_frequencies + self.dsp, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, ch * 2), SnakeAlt(),
            torch.nn.Linear(ch * 2, 1)
        )
    
    def forward(self, sp, pos):
        pos = self.pos_encoding(pos)
        fc_input = torch.cat((sp, pos), 1)
        output = self.pos_subnet(fc_input)
        return output

###############################################################################################################

class DecompGrid(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d, x_dims=3) -> None:
        assert num_feat_2d == num_feat_3d
        
        super().__init__()
        self.x_dims = x_dims
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *grid_shape[:self.x_dims]),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=-0.001, b=0.001)
        
        if self.x_dims == 3:
            self.plane_dimid = [(1, 2), (0, 2), (0, 1)] # corresponding to xy, xz, yz
            self.plane_dims = list(combinations(grid_shape[3:6], 2))
        elif self.x_dims == 2:
            self.plane_dimid = [(1), (0)] # corresponding to xy
            self.plane_dims = ([grid_shape[2]], [grid_shape[3]])
        # self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        pre_param_dims = 6 if self.x_dims == 3 else 4
        self.line_dimid = list(range(self.x_dims, self.x_dims + len(grid_shape[pre_param_dims:])))
        self.line_dims = grid_shape[pre_param_dims:]
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        print('line dimid', self.line_dimid)
        print('line dims', self.line_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *dims),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.999, b=1.001)
            if self.x_dims == 2:
                plane = plane.squeeze(0)
            self.planes.append(plane)
            
        self.planes = torch.nn.ParameterList(self.planes)

        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(num_feat_1d, dim),
                requires_grad=True
            )
            torch.nn.init.uniform_(line, a=0.01, b=0.25)
            self.lines.append(line)
        self.lines = torch.nn.ParameterList(self.lines)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :self.x_dims]
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        
        if self.x_dims == 3:
            for i, dimids in enumerate(self.plane_dimid):
                x2d = x[:,dimids]
                x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
                f2d = torch.nn.functional.grid_sample(self.planes[i],
                                x2d,
                                mode='bilinear', align_corners=True)
                f2d = f2d.squeeze()
                spatial_feats = spatial_feats * f2d
        else:
            for i, dimids in enumerate(self.plane_dimid):
                p1d = x[:,dimids]
                p1dn = p1d*(self.plane_dims[i][0]-1)
                p1d_f = torch.floor(p1dn)
                weights = p1dn-p1d_f
                f1d = torch.lerp(self.planes[i][:,p1d_f.type(torch.int)], self.planes[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.plane_dims[i][0]-1).type(torch.int)], weights)
                f1d = f1d.squeeze()
                spatial_feats = spatial_feats * f1d
            
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.int)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
        if len(spatial_feats.shape) == 1:
            feats = torch.cat((spatial_feats, param_feats))
            return feats
        feats = torch.cat((spatial_feats.T, param_feats.T), 1)
        return feats

    def forwardWithIntermediates(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[..., :3]
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            p1d_f = p1d_f
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.int)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
        if len(spatial_feats.shape) == 1:
            feats = torch.cat((spatial_feats, param_feats))
            return feats, spatial_feats, param_feats
        feats = torch.cat((spatial_feats.T, param_feats.T), 1)
        return feats, spatial_feats.T, param_feats.T
    
    def piecewise_linear_mean_var_torch_multi(self, lineidx:int, values:torch.Tensor, xrange:torch.Tensor):
        # precompute means for each piece
        means = (values[:,:-1] + values[:,1:]) * 0.5
        # function to compute variance on a piece
        def contfunc_var(xleft, xright, m):
            return (xleft-m)*(xright-m) + ((xright-xleft)**2) / 3
        # find bin index for xmin and xmax, compute weights for linear interpolation
        xf = torch.floor(xrange*(self.line_dims[lineidx]-1))
        xf_i = xf.type(torch.long)
        weights = xrange*(self.line_dims[lineidx]-1) - xf
        weights = weights.to('cuda:0')
        xrange2y = torch.lerp(values[:,xf_i], values[:,xf_i+1], weights)
        # xrange may contains complete piece
        if xf_i[0] != xf_i[1]:
            headmean = (xrange2y[:,0] + values[:,xf_i[0]+1])*0.5
            tailmean = (xrange2y[:,1] + values[:,xf_i[1]])*0.5
            weighted_sum = headmean * (1.0-weights[0]) + tailmean * weights[1] + torch.sum(means[:,xf_i[0]+1: xf_i[1]], 1)
            range_query_mean = weighted_sum / ((xrange[1]-xrange[0])*(self.line_dims[lineidx]-1))
            range_query_var = 0.0
            range_query_var += contfunc_var(xrange2y[:,0], values[:,xf_i[0]+1], range_query_mean) * (1.0-weights[0])
            range_query_var += contfunc_var(xrange2y[:,1], values[:,xf_i[1]], range_query_mean) * weights[1]
            for i in range(xf_i[0]+1, xf_i[1]):
                range_query_var += contfunc_var(values[:,i], values[:,i+1], range_query_mean)
            range_query_var /= ((xrange[1]-xrange[0])*(self.line_dims[lineidx]-1))
        else:
            range_query_mean = (xrange2y[:,0] + xrange2y[:,1])*0.5
            range_query_var = contfunc_var(xrange2y[:,0], xrange2y[:,1], range_query_mean)
        return range_query_mean, range_query_var

    def param_range_query(self, coords: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
        assert len(pmin) == len(pmax)
        # compute spatial features (fixed)
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = coords[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        param_feature_mean, var_plus_mean, mean_sq = 1., 1., 1.
        for i, dimids in enumerate(self.line_dimid):
            prange = torch.Tensor([pmin[i], pmax[i]])
            curmean, curvar = self.piecewise_linear_mean_var_torch_multi(i, self.lines[i], prange)
            print('curmean', curmean)
            print('curvar', curvar)
            param_feature_mean = param_feature_mean*curmean
            var_plus_mean = var_plus_mean * (curvar + curmean**2)
            mean_sq = mean_sq * (curmean**2)
        param_feature_var = var_plus_mean - mean_sq
        return spatial_feats, param_feature_mean, param_feature_var
    
    def param_range_query_by_sample(self, coords: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
        assert len(pmin) == len(pmax)
        # compute spatial features (fixed)
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = coords[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        # compute param_features by sample
        def piecewise_query(values, x, lineidx):
            xf = torch.floor(x*(self.line_dims[lineidx]-1))
            xf_i = xf.type(torch.long)
            w = x*(self.line_dims[lineidx]-1) - xf
            w = w.to('cuda:0')
            return torch.lerp(values[:,xf_i], values[:,xf_i+1], w)
        ysmul = None
        for i, dimids in enumerate(self.line_dimid):
            cur_xs = torch.linspace(pmin[i], pmax[i], 200)
            cur_ys = piecewise_query(values=self.lines[i], x=cur_xs, lineidx=i)
            print('cur_ys_mean', torch.mean(cur_ys, 1))
            print('cur_ys_var', torch.var(cur_ys, 1))
            if ysmul is None:
                ysmul = cur_ys
            else:
                new_ysmul = None
                for i in range(cur_ys.shape[1]):
                    if new_ysmul is None:
                        new_ysmul = cur_ys[:,i][:,None] * ysmul
                    else:
                        new_ysmul = torch.cat((new_ysmul, cur_ys[:,i][:,None] * ysmul), 1)
                ysmul = new_ysmul
            
        return spatial_feats, torch.mean(ysmul, 1), torch.var(ysmul, 1)
        
    
###############################################################################################################
    
class INR_FG(torch.nn.Module):
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d, out_features:int,
                 dropout_layer:bool=False, x_dims=3, out_sigmoid:bool=True) -> None:
        super().__init__()
        self.dg = DecompGrid(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d, x_dims=x_dims)
        
        self.out_sigmoid = out_sigmoid
        self.hidden_nodes = 128
        self.hasDP = dropout_layer
        self.fc1 = torch.nn.Linear(num_feat_3d+num_feat_1d, self.hidden_nodes)
        self.fc2 = torch.nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc3 = torch.nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc4 = torch.nn.Linear(self.hidden_nodes, out_features)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        if self.hasDP:
            self.dp = torch.nn.Dropout(p=0.125)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc1.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc2.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.normal_(self.fc3.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        torch.nn.init.normal_(self.fc4.bias, 0, 0.001)

    def get_loss(self, pred, gt):
        loss_list = []
        total_loss = 0
        likelihood_data_loss = torch.nn.functional.mse_loss(pred, gt)
        loss_list.append({'name':'mse', 'value':likelihood_data_loss})
        total_loss = total_loss + likelihood_data_loss

        return loss_list, total_loss

    def forward(self, x):
        x = self.dg(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        if self.hasDP:
            x = self.dp(x)
        if self.out_sigmoid:
            x = self.sigmoid(self.fc4(x))
        return x
    
    def forwardWithDP(self, x, nSample):
        if not self.hasDP:
            return None
        output_distribution = None
        x = self.dg(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dp(x)
        for i in range(nSample):
            x_dp = torch.nn.functional.dropout(x, 0.125)
            x_out = self.sigmoid(self.fc4(x_dp))
            if output_distribution is None:
                output_distribution = x_out
            elif len(x_out) == 1:
                output_distribution = torch.cat((output_distribution, x_out), 0)
            else:
                output_distribution = torch.cat((output_distribution, x_out), 1)
        return output_distribution, torch.mean(output_distribution, 1, True), torch.var(output_distribution, dim=1, keepdim=True)
    
    def forwardFGOnly(self, x):
        res, coord_features, param_features = self.dg.forwardWithIntermediates(x)
        return res, coord_features, param_features

    def param_range_query(self, coords: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
        return self.dg.param_range_query(coords, pmin, pmax)
    
    def param_range_query_by_sample(self, coords: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
        return self.dg.param_range_query_by_sample(coords, pmin, pmax)
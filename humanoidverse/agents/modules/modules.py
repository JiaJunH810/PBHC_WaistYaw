import torch
import torch.nn as nn
import inspect

class BaseModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(BaseModule, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict

        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.module_config_dict['input_dim']:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim = input_dim

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        if layer_config['type'] == 'MLP':
            self._build_mlp_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")
        
    def _build_mlp_layer(self, layer_config):
        layers = []
        hidden_dims = layer_config['hidden_dims']
        output_dim = self.output_dim
        activation = getattr(nn, layer_config['activation'])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        return self.module(input)
    
class Normalizer(nn.Module):
    def __init__(self, size):
        super(Normalizer, self).__init__()
        self.register_buffer('_mean', torch.zeros(size))
        self.register_buffer('_std', torch.ones(size))
        self.register_buffer('_count', torch.tensor(0.0))
    
    def normalize(self, x):
        return (x - self._mean) / (self._std + 1e-8)

class MotionEncoder(nn.Module):
    def __init__(self, config):
        super(MotionEncoder, self).__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(self.config.motion_single_length, 60),
            nn.SiLU()
        )
        self.conv_layers = nn.Sequential(
            nn.Conv1d(60, 40, kernel_size=6, stride=2),
            nn.SiLU(),
            nn.Conv1d(40, 20, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Flatten()
        )
        self.linear_output = nn.Linear(60, self.config.motion_output)
    
    def forward(self, x):
        batch_size = x.shape[0]
        motion_length = self.config.motion_length
        motion_single_length = self.config.motion_single_length
        x = x.reshape(batch_size * motion_length, motion_single_length)
        x = self.encoder(x)
        x = x.view(batch_size, motion_length, -1).transpose(1, 2)
        x = self.conv_layers(x)
        x = self.linear_output(x)
        return x

class StateHistoryEncoder(nn.Module):
    def __init__(self, config):
        super(StateHistoryEncoder, self).__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(self.config.history_single_length, 30),
            nn.SiLU()
        )
        self.conv_layers = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=6, stride=2),
            nn.SiLU(),
            nn.Conv1d(20, 10, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Flatten()
        )
        self.linear_output = nn.Linear(30, self.config.history_output)
    
    def forward(self, x):
        batch_size = x.shape[0]
        history_length = self.config.history_length
        history_single_length = self.config.history_single_length

        x = x.reshape(batch_size * history_length, history_single_length)
        x = self.encoder(x)
        x = x.view(batch_size, history_length, -1).transpose(1, 2)
        x = self.conv_layers(x)
        x = self.linear_output(x)
        return x

class ActorModule(nn.Module):
    def __init__(self, obs_dim_dict, config):
        super(ActorModule, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.config = config

        self._calculate_input_dim()
        self._calculate_output_dim()

        self.motion_encoder = MotionEncoder(config)
        self.history_encoder = StateHistoryEncoder(config)

        self.history_input = self.config.history_length * self.config.history_single_length
        self.motion_input = self.config.motion_length * self.config.motion_single_length
        self.essence_length = self.input_dim - self.history_input - self.motion_input
        

        self.actor_backbone = nn.Sequential(
            nn.Linear(self.config.motion_single_length + self.essence_length + self.config.motion_output + self.config.history_output, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.output_dim)
        )
        self.normalizer = Normalizer(self.input_dim)
    
    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.config['input_dim']:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim = input_dim
    
    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.config['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.normalizer.normalize(x)
        essence = x[:, :self.essence_length]
        motion_flat = x[:, self.essence_length:self.essence_length+self.motion_input]
        history_flat = x[:, self.essence_length+self.motion_input:]
        motion = motion_flat.view(-1, self.config.motion_length, self.config.motion_single_length)
        history = history_flat.view(-1, self.config.history_length, self.config.history_single_length)
        motion_feat = self.motion_encoder(motion)
        history_feat = self.history_encoder(history)
        current_motion = motion[:, 0, :]
        actor_input = torch.cat([current_motion, essence, motion_feat, history_feat], dim=1)
        return self.actor_backbone(actor_input)


class CriticModule(nn.Module):
    def __init__(self, obs_dim_dict, config):
        super(CriticModule, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.config = config

        self._calculate_input_dim()
        self._calculate_output_dim()

        self.history_encoder = StateHistoryEncoder(config)

        self.history_input = self.config.history_length * self.config.history_single_length
        self.essence_length = self.input_dim - self.history_input

        self.critic_backbone = nn.Sequential(
            nn.Linear(self.essence_length + self.config.history_output, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.output_dim)
        )
        self.normalizer = Normalizer(self.input_dim)

    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.config['input_dim']:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim = input_dim
    
    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.config['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim

    def forward(self, x):
        x = self.normalizer.normalize(x)
        essence = x[:, :self.essence_length]
        history_flat = x[:, self.essence_length:]
        history = history_flat.view(-1, self.config.history_length, self.config.history_single_length)

        history_feat = self.history_encoder(history)

        critic_input = torch.cat([essence, history_feat], dim=1)
        return self.critic_backbone(critic_input)
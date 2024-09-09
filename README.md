# Simple Latent Uncertainty Representation

### [Project Page](#) | [Example Notebooks](#) | [NuScenes labels](#)


## Implementation
Integrating LUR into standard models is simple, and requires adding the projection layers and updating the forward pass. 

**Original architecture:**
```python 
class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.input = nn.Linear(in_features, 8)
        self.hidden1 = nn.Linear(8, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, out_features)
        self.activation = nn.ReLU()

    def forward(self, x, y=None):
        x = self.activation(self.input(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.output(x)
```
**Modified architecture:**
```python 
class LUR_MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_projections):
        super().__init__()
        self.out_features = out_features
        self.input = nn.Linear(in_features, 8)
        self.hidden1 = nn.Linear(8, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.projections = nn.ModuleList([nn.Linear(8, 8) for _ in range(num_projections)])
        self.output = nn.Linear(8, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        z_representations, y_projections = [], []
        x = self.activation(self.input(x))
        z = self.activation(self.hidden1(x))
        z = self.activation(self.hidden2(x))

        for i, proj in enumerate(self.projections):
            z_p = self.activation(proj(z))
            z_representations.append(z_p)
            y_projections.append(self.head(z_p))

        y = self.output(z)
        if self.training:
            return y, y_projections, z_representations
        else:
            return y, y_projections 
```

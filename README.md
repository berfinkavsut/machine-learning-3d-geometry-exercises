# ml43d-exercises

## Notes 

*Install poetry* \
`curl -sSL https://install.python-poetry.org | python3 -` \
`export PATH="$HOME/.local/bin:$PATH"`

*Inside project* \
`poetry install` 

*Install Pytorch* \
`poetry shell` \
`pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` \
`exit`

Note: could not install `torch` at first, memory issue was resolved after running `salloc --gpus=1` 

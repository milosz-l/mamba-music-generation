pip install poetry
poetry shell
pip install --upgrade pip setuptools wheel
poetry install
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install causal_conv1d==1.2.*
pip install mamba-ssm==1.2.0.post1
python src/download_data.py
from dotenv import load_dotenv
from animus.torch.engine import CPUEngine, GPUEngine
from silver_chihuahua.train import train
from silver_chihuahua.configs import DEFAULT_THERMAL, DEFAULT_CONCRETE

load_dotenv()

if __name__ == "__main__":
    train(DEFAULT_CONCRETE, GPUEngine())
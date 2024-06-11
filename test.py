import torch

# 加載最佳模型
model_path = 'model.pt'  # 替換為你的模型文件名
model = torch.load(model_path)

# 設置模型為評估模式
model.eval()

# 假設你有一個 corpus 來進行字符到索引的轉換
from data_utils import Corpus

corpus = Corpus('C:/Users/78963/Desktop/科算期末專題/enwik8')
start_text = "The quick brown fox"  # 替換為你的初始文本
start_input = [corpus.dictionary.word2idx[char] for char in start_text]
start_input = torch.tensor(start_input, dtype=torch.long).unsqueeze(0)  # 添加 batch 維度

if torch.cuda.is_available():
    start_input = start_input.cuda()


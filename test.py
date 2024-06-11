import torch
from data_utils import Dictionary, Corpus

# 檢查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載模型
model = torch.load('model.pt')
model = model.to(device)
model.eval()

# 加載語料庫
corpus = Corpus('enwik8')

# 初始文本
start_text = "The meaning of life is "

# 將初始文本轉換為索引
start_input = [corpus.dictionary.char2idx[char] for char in start_text if char in corpus.dictionary.char2idx]

# 初始化隱藏層
hidden = model.init_hidden(1)
hidden = [h.to(device) for h in hidden]  # 確保隱藏層在GPU上

# 將初始文本作為模型的輸入
input_tensor = torch.LongTensor(start_input).unsqueeze(1).to(device)  # 轉換為 (seq_len, batch_size) 並移動到GPU

# 遍歷初始文本，獲取最後一個隱藏狀態
for i in range(len(start_text) - 1):
    _, hidden = model(input_tensor[i].unsqueeze(0), hidden)

# 生成文本
generated_text = start_text
input_char = input_tensor[-1]

for _ in range(1000):  # 生成100個字符
    output, hidden = model(input_char.unsqueeze(0), hidden)
    predicted_index = torch.argmax(output[-1]).item()
    predicted_char = corpus.dictionary.idx2char[predicted_index]
    generated_text += predicted_char
    input_char = torch.LongTensor([predicted_index]).to(device)

print(generated_text)
with open("generated_text.txt", "w") as f:
    f.write(generated_text)

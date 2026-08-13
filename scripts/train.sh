#!/usr/bin/env bash

# python -m src.inferences \
#   --adapter ./output/nemotron-sft-trained/epoch-2 \
#   --question "Bạn là một trợ lý AI y khoa đáng tin cậy được phát triển để hỗ trợ phân tích và đưa ra lời khuyên y tế chính xác.\n\nHãy phân tích câu hỏi sau một cách logic, khách quan và theo văn phong chuyên ngành y. Chỉ dựa vào thông tin có trong câu hỏi để suy luận, không tự tạo giả định hoặc phỏng đoán mơ hồ.\n Nếu câu hỏi liên quan đến chính trị, thông tin sai lệch, nội dung tiêu cực, hoặc đi ngược lại đạo đức y khoa – hãy từ chối trả lời để đảm bảo an toàn, bảo mật và tuân thủ quy tắc chuyên môn.\n\n### Câu hỏi :\nMột người đã hoàn thành việc tiêm phòng uốn ván cách đây 10 năm. Nếu họ hiện có một vết thương sạch được tạo ra cách đây 2,5 giờ, họ nên nhận được điều trị y tế nào? " \
#   --max-length 1024 \
#   --temperature 0.3 \
#   --top-p 0.9 \
#   --load-in-4bit

# python -m src.inferences \
#   --adapter ./output/nemotron-sft-trained/epoch-2 \
#   --question "Thủ đô của Việt Nam là gì?" \
#   --max-length 2048 \
#   --temperature 0.3 \
#   --top-p 0.9 \
#   --load-in-4bit

python -m src.inferences \
  --adapter ./output/nemotron-sft-test_mini_en/epoch-19 \
  --question "A person completed tetanus vaccination 10 years ago and now has a clean wound created 2.5 hours agoy " \
  --max-length 256 \
  --device cuda


  # --temperature 0.6 \
  # --top-p 0.9 \
  # --load-in-4bit \

# python -m src.inferences \
#   --adapter ./output/nemotron-sft-test_mini_en/epoch-17 \
#   --question "Patient with macroglossia, atrophic papillae, Hgb 11.5 g/dL, MCV 100 fL. Next best step?" \
#   --max-length 1024 \
#   --temperature 0.3 \
#   --top-p 0.9 \
#   --repetition-penalty 1.1 \
#   --load-in-4bit

# Auto-detect latest epoch
bash scripts/inference.sh

# Use specific epoch
bash scripts/inference.sh 18

# Custom question
bash scripts/inference.sh 18 "What is the difference between Git reset --soft and --hard?"

# With environment variables
QUESTION="Explain Python generators" \
TEMPERATURE=0.5 \
MAX_LENGTH=1024 \
bash scripts/inference.sh 18
# Training will automatically continue if output dir exists
OUTPUT_DIR="./output/my-existing-model" bash scripts/train.sh

# Test epoch 10
bash scripts/inference.sh 10 "Test question" > results_epoch10.txt

# Test epoch 20
bash scripts/inference.sh 20 "Test question" > results_epoch20.txt

# Compare results
diff results_epoch10.txt results_epoch20.txt
# Create sweep configuration
wandb sweep sweep.yaml

# Run sweep agent
wandb agent <sweep-id>
python -m src.inferences \
  --adapter ./output/nemotron-sft-trained/epoch-18 \
  --question "Your question here" \
  --max-length 512 \
  --temperature 0.7 \
  --top-p 0.9
# Use GPUs 0 and 1
bash scripts/train.sh 0,1

# Or via environment
GPU=0,1,2,3 bash scripts/train.sh

# Clone the repository
git clone <your-repo-url>
cd mamba_training

# Create conda environment
conda create -n mamba_temp python=3.10
conda activate mamba_temp

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft bitsandbytes accelerate
pip install pytorch-lightning wandb
pip install datasets jsonlines

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"


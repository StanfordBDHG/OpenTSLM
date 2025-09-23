echo "Training started, check logs/ for progress"
mkdir -p logs  # (Only needed once, to create the directory)
nohup python curriculum_learning.py --model OpenTSLMFlamingo --batch_size 20 > logs/output_$(date +%Y%m%d_%H%M%S).txt 2>&1 &

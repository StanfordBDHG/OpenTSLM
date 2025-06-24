mkdir -p logs  # (Only needed once, to create the directory)
nohup python curriculum_learning.py --model EmbedHealthFlamingo > logs/output_$(date +%Y%m%d_%H%M%S).txt 2>&1 &

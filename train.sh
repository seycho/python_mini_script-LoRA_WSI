export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="output/LUAD"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="total" \
  --dataloader_num_workers=8 \
  --train_batch_size=16 \
  --resolution=128 --random_flip \
  --gradient_accumulation_steps=4 \
  --max_train_steps=25000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --validation_prompt="H&E stain pathology image with lung adenocarcinoma" \
  --num_validation_images=8 \
  --checkpointing_steps=2000 \
  --seed=1337

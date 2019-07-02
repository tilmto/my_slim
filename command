python -u train_image_classifier.py --train_dir=logs  --train_image_size=224  --model_name=mobilenet_v3_small  --dataset_name=imagenet --dataset_split_name=train --learning_rate=0.72 --preprocessing_name="inception_v2" --label_smoothing=0.1 --moving_average_decay=0.9999 --batch_size=96 --learning_rate_decay_factor=0.98 --num_epochs_per_decay=0.15625 --dataset_dir=/userhome/ImageNet-Tensorflow/train_tfrecord --num_clones=16 > train_log.txt 2>&1


python -u eval_image_classifier.py --dataset_dir=/userhome/ImageNet-Tensorflow/validation_tfrecord --dataset_name=imagenet --dataset_split_name=validation --model_name=mobilenet_v3_small --checkpoint_path=logs/model.ckpt-XXXX >test_log.txt 2>&1


#gpu2
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train_image_classifier.py --train_dir=logs  --train_image_size=192  --model_name=mobilenet_v3_small  --dataset_name=imagenet --dataset_split_name=train --learning_rate=0.1 --preprocessing_name="inception_v2" --label_smoothing=0.1 --moving_average_decay=0.9999 --batch_size=96 --learning_rate_decay_factor=0.98 --num_epochs_per_decay=125 --dataset_dir=/data1/ILSVRC2017/tfrecord --num_clones=4 --num_readers 32 --num_preprocessing_threads 32 --learning_rate_decay_type cosine > train_log.txt 2>&1 &


#test
CUDA_VISIBLE_DEVICES=9 nohup python -u eval_image_classifier.py --dataset_dir=/data1/tfrecord  --dataset_name=imagenet --dataset_split_name=validation --model_name=mobilenet_v3_small --num_readers 16 --num_preprocessing_threads 16 --eval_image_size 192 --checkpoint_path=weights/model.ckpt-740593 >test_log.txt 2>&1 &


#gpu3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u train_image_classifier.py --train_dir=logs  --train_image_size=224  --model_name=mobilenet_v3_small  --dataset_name=imagenet --dataset_split_name=train --learning_rate=0.36 --preprocessing_name="inception_v2" --label_smoothing=0.1 --moving_average_decay=0.9999 --batch_size=96 --learning_rate_decay_factor=0.98 --num_epochs_per_decay=0.3125 --dataset_dir=/data1/tfrecord --num_clones=8 --num_readers 40 --num_preprocessing_threads 40 --checkpoint_path=weights/model.ckpt-457147 > train_log.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 nohup python -u train_image_classifier.py --train_dir=logs  --train_image_size=224  --model_name=mobilenet_v3_small  --dataset_name=imagenet --dataset_split_name=train --learning_rate=0.405 --preprocessing_name="inception_v2" --label_smoothing=0.1 --moving_average_decay=0.9999 --batch_size=96 --learning_rate_decay_factor=0.98 --num_epochs_per_decay=0.27778 --dataset_dir=/data1/tfrecord --num_clones=9 --num_readers 40 --num_preprocessing_threads 40 --checkpoint_path=logs/model.ckpt-14395 > train_log.txt 2>&1 &

#all_relu_finetune:
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u train_image_classifier.py --train_dir=logs  --train_image_size=192  --model_name=mobilenet_v3_small  --dataset_name=imagenet --dataset_split_name=train --learning_rate=0.0003 --preprocessing_name="inception_v2" --label_smoothing=0.1 --moving_average_decay=0.9999 --batch_size=96 --learning_rate_decay_factor=0.945 --num_epochs_per_decay=2 --dataset_dir=/data1/tfrecord --num_clones=4 --learning_rate_decay_type fixed --num_readers 32 --num_preprocessing_threads 32 > train_log.txt 2>&1 &


CUDA_VISIBLE_DEVICES=9 python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v3_small \
  --image_size=192 \
  --output_file=logs/mobilenet_v3_small.pb


CUDA_VISIBLE_DEVICES=9 python freeze_graph.py \
  --input_graph=logs/mobilenet_v3_small.pb \
  --input_checkpoint=logs/model.ckpt-1583525 \
  --input_binary=true --output_graph=logs/frozen_mobilenet_v3_small.pb \
  --output_node_names=MobilenetV2/Predictions/Reshape_1


CUDA_VISIBLE_DEVICES=9 tflite_convert \
  --output_file=logs/mobilenet_v3_small.tflite \
  --graph_def_file=logs/frozen_mobilenet_v3_small.pb \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --inference_type=FLOAT \
  --inference_input_type=QUANTIZED_UINT8 --input_shape="1,192,192,3" \
  --input_arrays="input" \
  --output_arrays="MobilenetV2/Predictions/Reshape_1" \
  --mean_value=128 \
  --std_dev_value=127


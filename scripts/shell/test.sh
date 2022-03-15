DATASET_PATH=$1
num_images_ub=$2
log_folder=$3
completeness_ratio=$4

/home/apple/F/henry/DAGSfM/build/src/exe/colmap distributed_mapper \
$DATASET_PATH/$log_folder \
--database_path=$DATASET_PATH/database.db \
--image_path=$DATASET_PATH/images \
--output_path=$DATASET_PATH/sparse \
--num_workers=8 \
--graph_dir=$DATASET_PATH/$log_folder \
--num_images_ub=$num_images_ub \
--completeness_ratio=$completeness_ratio \
--relax_ratio=1.3 \
--cluster_type=NCUT \
--imu=/home/apple/F/henry/DAGSfM/imu_sample.csv

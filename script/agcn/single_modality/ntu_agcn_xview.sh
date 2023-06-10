# agcn with SkletonGCL on joint modality
python main.py --config config/nturgbd-cross-view/agcn.yaml --work-dir work_dir/ntu/xview/agcn_SkeletonGCL_joint --device 0 --temperature 0.8

# agcn with SkletonGCL on bone modality
# python main.py --config config/nturgbd-cross-view/agcn.yaml --train-feeder-args bone=True --test-feeder-args bone=True --work-dir work_dir/ntu/xview/agcn_SkeletonGCL_bone --device 0 --temperature 0.8

# agcn with SkletonGCL on joint-motion modality
# python main.py --config config/nturgbd-cross-view/agcn.yaml --train-feeder-args vel=True --test-feeder-args vel=True --work-dir work_dir/ntu/xview/agcn_SkeletonGCL_joint_motion --device 0 --temperature 1.0

# agcn with SkletonGCL on bone-motion modality
# python main.py --config config/nturgbd-cross-view/agcn.yaml --train-feeder-args bone=True --train-feeder-args vel=True --test-feeder-args bone=True --test-feeder-args vel=True --work-dir work_dir/ntu/xview/agcn_SkeletonGCL_bone_motion --device 0 --temperature 0.5
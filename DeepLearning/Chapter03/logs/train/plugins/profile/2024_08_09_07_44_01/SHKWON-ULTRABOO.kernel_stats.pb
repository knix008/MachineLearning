
f
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�β@�βH�βbRMSprop/RMSprop/update_2/addhuZU�B
d
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8֞�@֞�H֞�bRMSprop/RMSprop/update_2/subhuZU�B
�
�_ZN5cudnn45pooling_bw_kernel_max_nchw_fully_packed_smallIffLi2EL21cudnnNanPropagation_t0EEEv17cudnnTensorStructPKT_S2_S5_S2_S5_S2_PS3_18cudnnPoolingStructT0_S8_NS_15reduced_divisorES9_ �8*�2� 8�ӌ@�ӌH�ӌb:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradhu  �B
h
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8ؾ�@ؾ�Hؾ�b RMSprop/RMSprop/update_2/truedivhuZU�B
c
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��n@��nH��nbRMSprop/RMSprop/update_2/Sqrthu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*�28ۚc@ۚcHۚcb(gradient_tape/sequential/conv2d/ReluGradhuZU�B
�
W_ZN7cutlass6KernelI52cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4EEvNT_6ParamsE� ��*�28ܚO@ܚOHܚOXbsequential/dense/MatMulh
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��K@��KH��KbRMSprop/RMSprop/update_2/mulhuZU�B
g
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��E@��EH��EbRMSprop/RMSprop/update_2/SquarehuZU�B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��C@��CH��Cb.gradient_tape/sequential/dropout/dropout/Mul_1huZU�B
�
}_Z23implicit_convolve_sgemmIffLi128ELi6ELi7ELi3ELi3ELi5ELi1ELb0ELb0ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_biiK�2* 2�8܍A@܍AH܍Absequential/conv2d/Reluhu  HB
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��7@��7H��7bRMSprop/RMSprop/update_2/mul_2huZU�B
�
V_ZN7cutlass6KernelI51cutlass_80_tensorop_s1688gemm_256x64_16x4_nt_align4EEvNT_6ParamsE� ��*�2�8��.@��.H��.b'gradient_tape/sequential/dense/MatMul_1h
�
g_Z17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEviiiPKT_iPS0_S2_18kernel_grad_paramsyifiiiiP�*2�8�� @�� H�� Xb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhu  HB
�
W_ZN7cutlass6KernelI52cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4EEvNT_6ParamsE� ��*�28�� @�� H�� Xb%gradient_tape/sequential/dense/MatMulh
e
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bRMSprop/RMSprop/update_2/add_1huZU�B
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bRMSprop/RMSprop/update_2/mul_1huZU�B
�
v_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi1024ELi1024ELi2ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ �`*�2�8��@��H��bagradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhuZU�B
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ �!*�2�8ߵ@ߵHߵb`gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizerhu  �B
�
�_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*�28��	@��	H��	b7sequential/dropout/dropout/random_uniform/RandomUniformhuZU�B
t
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�2�8��@��H��b'sequential/dropout/dropout/GreaterEqualhuZU�B
�
�_ZN5cudnn3ops20pooling_fw_4d_kernelIffNS_15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEEv17cudnnTensorStructPKT_S6_PS7_18cudnnPoolingStructT0_SC_iNS_15reduced_divisorESD_( �*�2�8�@�H�b sequential/max_pooling2d/MaxPoolhu���B
d
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8��@��H��bsequential/dropout/dropout/Casthu  �B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��b,gradient_tape/sequential/dropout/dropout/MulhuZU�B
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIhLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ �*�2�8��@��H��b�sequential/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_bool_Mul_1-0-TransposeNHWCToNCHW-LayoutOptimizerhu  �B
�
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE*�2�8��@��H��b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhu  �B
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_ �!*�2�8��@��H��bHsequential/dropout/dropout/Mul_1-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorReductionOpINS0_10SumReducerIfEEKNS_9IndexListINS_10type2indexILx1EEEJEEEKNS_18TensorForcedEvalOpIKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSK_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS4_INS5_ISK_Li2ELi1ExEELi16ES7_EEEEKNSI_INS0_20scalar_difference_opIffEEKNSM_IKNSC_ISE_JiEEEKNSH_IKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNS4_INS5_IfLi2ELi1ExEELi16ES7_EEEEEEEES14_EEEEEES7_EEEENS_9GpuDeviceEEExEEvT_T0_2*�28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
e
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8�@�H�b sequential/dropout/dropout/Mul_1huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_17TensorReshapingOpIKNS_9IndexListIiJEEENS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEEKNS_17TensorReductionOpINS0_10MaxReducerIfLi0EEEKNS5_INS_10type2indexILx1EEEJEEEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS8_INS9_IKfLi2ELi1ExEELi16ESB_EEEESB_EEEENS_9GpuDeviceEEExEEvT_T0_@*�28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1EEvNT_6ParamsEp ��*�28��@��H��Xbsequential/dense_1/MatMulhugU�A
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_17TensorReshapingOpIKNS_9IndexListIiJEEENS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEEKNS_17TensorReductionOpINS0_10SumReducerIfEEKNS5_INS_10type2indexILx1EEEJEEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKSC_EESB_EEEENS_9GpuDeviceEEExEEvT_T0_/*�28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align1EEvNT_6ParamsEv ��*�28��@��H��Xb'gradient_tape/sequential/dense_1/MatMulhugU�A
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS9_INS0_18scalar_quotient_opIffEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKS8_EEKNS_20TensorBroadcastingOpIKNS_9IndexListINS_10type2indexILx1EEEJiEEESH_EEEEKNSK_IKNS_5arrayIxLy2EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEEEENS_9GpuDeviceEEExEEvT_T0_.*�28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8��@��H��b[sequential/dropout/dropout/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*�28��@��H��b'gradient_tape/sequential/dense/ReluGradhuZU�B
d
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��b RMSprop/RMSprop/update_4/truedivhuZU�B
�
U_ZN7cutlass6KernelI50cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align1EEvNT_6ParamsEp ��*�28��@��H��b)gradient_tape/sequential/dense_1/MatMul_1hugU�A
�
b_Z19splitKreduce_kernelIffffLb1ELb0EEv18cublasSplitKParamsIT1_EPKT_PKT0_PS6_PKS1_SB_PKT2_PvxPS1_Pi * 28߿@߿H߿Xbsequential/dense/MatMulhu  �B
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2�8��@��H��bsequential/dropout/dropout/MulhuZU�B
b
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_5/Sqrthu  �B
�
l_ZN10tensorflow7functor15CleanupSegmentsIPfS2_N3cub3SumEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*  28��@��H��b3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_20TensorTupleReducerOpINS0_18ArgMaxTupleReducerINS_5TupleIxfEEEEKNS_5arrayIxLy1EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_ *�28߰@߰H߰bArgMaxhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_20TensorTupleReducerOpINS0_18ArgMaxTupleReducerINS_5TupleIxfEEEEKNS_5arrayIxLy1EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_ *�28�@�H�bArgMax_1huZU�B
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_4/subhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_4/mulhuZU�B
f
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8��@��H��b!sequential/dropout_1/dropout/Casthu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28��@��H��bAssignAddVariableOp_4huZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update/mul_2huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@�H�bEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhuZU�B
m
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*�28��@��H��bsequential/dense/BiasAddhuZU�B
b
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update/truedivhuZU�B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_5/addhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28��@��H��bsequential/dense/ReluhuZU�B
I
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28��@��H��bCasthu  �B
`
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update/Sqrthu  �B
d
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�b RMSprop/RMSprop/update_3/truedivhuZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@�H�bRMSprop/RMSprop/update_3/mul_1huZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�@�H�bsequential/conv2d/ReluhuZU�B
d
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��b sequential/dropout_1/dropout/MulhuZU�B
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_3/subhuZU�B
S
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*�28��@��H��b
LogicalAndhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorEvalToOpIKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfS6_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS_9TensorMapINS_6TensorIS6_Li2ELi1ExEELi16ENS_11MakePointerEEEEEKNS4_INS0_20scalar_difference_opIffEEKNS8_IKNS_9IndexListINS_10type2indexILx1EEEJiEEEKNS_18TensorForcedEvalOpIKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNSC_INSD_IfLi2ELi1ExEELi16ESF_EEEEEEEESX_EEEESF_EENS_9GpuDeviceEEExEEvT_T0_(*�28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_3/add_1huZU�B
`
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update/addhuZU�B
O
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/subhuZU�B
�
�_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*�28��@��H��b9sequential/dropout_1/dropout/random_uniform/RandomUniformhuZU�B
�
�_ZN10tensorflow64_GLOBAL__N__36_softmax_op_gpu_cu_compute_80_cpp1_ii_cdcf0523_69622GenerateNormalizedProbIffLi4EEEvPKT_PKT0_S4_PS2_iib *�28��@��H��bsequential/dense_1/Softmaxhu  �B
`
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28ߛ@ߛHߛbRMSprop/RMSprop/update_5/subhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy1EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*�28��@��H��b;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1huZU�B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_3/addhuZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_5/mul_1huZU�B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_4/addhuZU�B
b
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_4/Sqrthu  �B
}
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28��@��H��b8categorical_crossentropy/weighted_loss/num_elements/Casthu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28��@��H��bAssignAddVariableOp_2huZU�B
K
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28��@��H��bCast_2hu  �B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_3/mulhuZU�B
b
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_1/Sqrthu  �B
d
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_5/add_1huZU�B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_1/mul_1huZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update/mul_1huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIKfSB_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS4_INS5_ISB_Li2ELi1ExEELi16ES7_EEEEKNSD_IKNS_9IndexListINS_10type2indexILx1EEEJiEEEKS8_EEEEEENS_9GpuDeviceEEExEEvT_T0_&*�28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
�
b_Z19splitKreduce_kernelIffffLb1ELb0EEv18cublasSplitKParamsIT1_EPKT_PKT0_PS6_PKS1_SB_PKT2_PvxPS1_Pi * 28��@��H��Xbsequential/dense_1/MatMulhu  �B
b
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_3/mul_2huZU�B
u
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28��@��H��b)sequential/dropout_1/dropout/GreaterEqualhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28��@��H��bAssignAddVariableOp_3huZU�B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update/add_1huZU�B
f
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��b"sequential/dropout_1/dropout/Mul_1huZU�B
b
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_1/addhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorEvalToOpIKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEESA_EENS_9GpuDeviceEEExEEvT_T0_*�28��@��H��b:categorical_crossentropy/softmax_cross_entropy_with_logitshuZU�B
d
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28ߔ@ߔHߔbRMSprop/RMSprop/update/SquarehuZU�B
b
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_3/Sqrthu  �B
o
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*�28��@��H��bsequential/dense_1/BiasAddhuZU�B
`
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update_1/mulhuZU�B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28��@��H��bRMSprop/RMSprop/update/mulhuZU�B
�
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE�!*  28�S@�SH�Sb2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhuZU�B
q
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�N@�NH�Nb0gradient_tape/sequential/dropout_1/dropout/Mul_1huZU�B
�
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE�!*  28�L@�LH�Lb4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhuZU�B
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*�28�F@�FH�FXb;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterhuZU�B
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*�28�F@�FH�FbEqualhuZU�B
�
�_ZN10tensorflow7functor15RowReduceKernelIN3cub22TransformInputIteratorIfNS_64_GLOBAL__N__36_softmax_op_gpu_cu_compute_80_cpp1_ii_cdcf0523_69621SubtractAndExpFunctorIffEENS2_21CountingInputIteratorIixEExEEPfNS2_3SumEEEvT_T0_iiT1_NSt15iterator_traitsISC_E10value_typeE*�2 8�F@�FH�Fbsequential/dense_1/Softmaxhu  �B
a
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�E@�EH�Eb RMSprop/RMSprop/update_5/truedivhuZU�B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�E@�EH�EbRMSprop/RMSprop/update_5/mulhuZU�B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�E@�EH�EbRMSprop/RMSprop/update_5/mul_2huZU�B
c
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�D@�DH�DbRMSprop/RMSprop/update_5/SquarehuZU�B
[
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�D@�DH�DbRMSprop/RMSprop/update/subhuZU�B
�
n_ZN10tensorflow7functor18ColumnReduceKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE�!*  28�D@�DH�Db3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGradhuZU�B
a
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�D@�DH�Db RMSprop/RMSprop/update_1/truedivhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�C@�CH�Cb#RMSprop/RMSprop/AssignAddVariableOphuZU�B
�
k_ZN10tensorflow7functor15RowReduceKernelIPKfPfN3cub3MaxEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE *�2 8�C@�CH�Cbsequential/dense_1/Softmaxhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�B@�BH�BbLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhuZU�B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�B@�BH�BbMulhuZU�B
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*�28�A@�AH�Ab*categorical_crossentropy/weighted_loss/Sumhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@@�@H�@b.gradient_tape/sequential/dropout_1/dropout/MulhuZU�B
c
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�@@�@H�@bRMSprop/RMSprop/update_4/SquarehuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�@@�@H�@bdiv_no_nan_1huZU�B
]
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�?@�?H�?bRMSprop/RMSprop/update_1/subhuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�?@�?H�?bAssignAddVariableOphuZU�B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�>@�>H�>bRMSprop/RMSprop/update_4/mul_2huZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�>@�>H�>b
div_no_nanhuZU�B
�
l_ZN10tensorflow7functor15CleanupSegmentsIPfS2_N3cub3SumEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*�28�=@�=H�=b4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradhu  �B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�<@�<H�<b,categorical_crossentropy/weighted_loss/valuehuZU�B
G
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�28�;@�;H�;bCast_1hu  �B
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE0*�28�;@�;H�;bSum_2hu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�8@�8H�8bRMSprop/RMSprop/update_4/mul_1huZU�B
a
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�8@�8H�8bRMSprop/RMSprop/update_4/add_1huZU�B
c
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�8@�8H�8bRMSprop/RMSprop/update_3/SquarehuZU�B
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*�28�7@�7H�7bAssignAddVariableOp_1huZU�B
c
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�6@�6H�6bRMSprop/RMSprop/update_1/SquarehuZU�B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�5@�5H�5bRMSprop/RMSprop/update_1/mul_2huZU�B
a
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�/@�/H�/bRMSprop/RMSprop/update_1/add_1huZU�B
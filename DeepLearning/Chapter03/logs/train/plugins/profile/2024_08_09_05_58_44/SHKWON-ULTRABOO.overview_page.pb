�	xb֋�� @xb֋�� @!xb֋�� @	'R���+@'R���+@!'R���+@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0xb֋�� @��y�]��?1v��X�`
@I��U�@Y�;�(A�?r0*	�n��.`@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat��z6��?!��؈A@)}]��t�?1M̟!ĳ?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���Hi�?!L�\��@@)p]1#�=�?1i9����8@:Preprocessing2T
Iterator::Root::ParallelMapV2rR��8Ӕ?!�<�)�j/@)rR��8Ӕ?1�<�)�j/@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��1��?!_�u% �"@)��1��?1_�u% �"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipߦ?��"�?!��v��S@)IIC���?1��{@�@:Preprocessing2E
Iterator::Rootvmo�$�?!A$l$5@)l\���|?1Ê���@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�ꫫ��?!*��R$�B@)N)���]r?1߽�ao�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�1��|j?!��?tl�@)�1��|j?1��?tl�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 13.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�45.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9&R���+@I� NdG@Q]��ek�C@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��y�]��?��y�]��?!��y�]��?      ��!       "	v��X�`
@v��X�`
@!v��X�`
@*      ��!       2      ��!       :	��U�@��U�@!��U�@B      ��!       J	�;�(A�?�;�(A�?!�;�(A�?R      ��!       Z	�;�(A�?�;�(A�?!�;�(A�?b      ��!       JGPUY&R���+@b q� NdG@y]��ek�C@�"7
RMSprop/RMSprop/update_2/addAddV2��Z�?!��Z�?"5
RMSprop/RMSprop/update_2/subSub�ˑ�)&�?!R�ܔ@�?"8
sequential/conv2d/Relu_FusedConv2D�xz}~ó?!���T"�?"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad�5̔���?!L���?"=
 RMSprop/RMSprop/update_2/truedivRealDiv���\(�?!�4����?"-
IteratorGetNext/_1_Send�(?�?!�������?"F
(gradient_tape/sequential/conv2d/ReluGradReluGrad�w:X;�?!��b���?"7
RMSprop/RMSprop/update_2/SqrtSqrtxU��إ�?!�i�?���?"C
'gradient_tape/sequential/dense/MatMul_1MatMul�����?!��j;E�?"5
sequential/dense/MatMulMatMul��%F$�?!������?0Q      Y@Yi�́D+,@aSO�o�zU@q�"dޫ�;@yH�p&��?"�
both�Your program is MODERATELY input-bound because 13.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�45.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�27.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
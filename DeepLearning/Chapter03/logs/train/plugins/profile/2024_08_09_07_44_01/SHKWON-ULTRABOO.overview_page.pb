�	���@@���@@!���@@	t��J���?t��J���?!t��J���?"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0���@@���۞��?1�"�~L=@Ib.�Q�?Y�bc^G�?r0*	�A`��vh@2T
Iterator::Root::ParallelMapV2
,�)�?!J��p;�?@)
,�)�?1J��p;�?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�9�����?!b��m��:@)oH�'ۨ?1�n�!�8@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateӠh�"�?!e*d_�7@)� ��	�?1�����+@:Preprocessing2E
Iterator::Root�̒ 5��?!@��[�D@)��J\Ǹ�?1h�W���"@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice÷�n�;�?!�G&/2"@)÷�n�;�?1�G&/2"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip��E
e�?!��"�UM@)�il���?1���@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap���(�?!��P(:@)&R���0x?1���7$@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��P�lm?!���]�?)��P�lm?1���]�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9t��J���?Ih�� )&@Q�W��U@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���۞��?���۞��?!���۞��?      ��!       "	�"�~L=@�"�~L=@!�"�~L=@*      ��!       2      ��!       :	b.�Q�?b.�Q�?!b.�Q�?B      ��!       J	�bc^G�?�bc^G�?!�bc^G�?R      ��!       Z	�bc^G�?�bc^G�?!�bc^G�?b      ��!       JGPUYt��J���?b qh�� )&@y�W��U@�"7
RMSprop/RMSprop/update_2/addAddV2h-��'p�?!h-��'p�?"5
RMSprop/RMSprop/update_2/subSubpzvy*>�?!�S�6)W�?"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGradt�D��?!�ShS���?"=
 RMSprop/RMSprop/update_2/truedivRealDiv�øqC�?!���/,��?"7
RMSprop/RMSprop/update_2/SqrtSqrt�юk��?!������?"F
(gradient_tape/sequential/conv2d/ReluGradReluGrad;_��NB�?!��av)�?"5
sequential/dense/MatMulMatMul�S�����?!�}�uq��?0"5
RMSprop/RMSprop/update_2/mulMullg�餥?!��"���?";
RMSprop/RMSprop/update_2/SquareSquareT�غ��?!B���"�?"8
sequential/conv2d/Relu_FusedConv2DI
 /
y�?!�B�bLZ�?Q      Y@Yi�́D+,@aSO�o�zU@q��A��V@y�􍭆�?"�
both�Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�90.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
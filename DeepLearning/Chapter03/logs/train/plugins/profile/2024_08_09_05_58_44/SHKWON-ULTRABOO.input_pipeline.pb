	xb֋�� @xb֋�� @!xb֋�� @	'R���+@'R���+@!'R���+@"q
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
@*      ��!       2      ��!       :	��U�@��U�@!��U�@B      ��!       J	�;�(A�?�;�(A�?!�;�(A�?R      ��!       Z	�;�(A�?�;�(A�?!�;�(A�?b      ��!       JGPUY&R���+@b q� NdG@y]��ek�C@
	�,_��)@�,_��)@!�,_��)@	����D"@����D"@!����D"@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�,_��)@;�э���?1'���K
@I��5w @Y���L0\�?r0*	v��/U_@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat_Cp\�M�?!�vLG��@@)T�4��-�?14�wfr?@:Preprocessing2T
Iterator::Root::ParallelMapV2���_�|�?!�	���9@)���_�|�?1�	���9@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�.��$�?!������5@)K�8��Ռ?1�����w&@:Preprocessing2E
Iterator::Rooty�0DN_�?!��n�6B@)؛����?1S����t%@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�m�s�?!�hKN�c%@)�m�s�?1�hKN�c%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip~r 
f�?!u�>��O@)�n��S}?1�W-3��@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap	�P���?!)׽!C�8@)ݚt["l?1{{�y<�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�@�C�b?!Y�b�?)�@�C�b?1Y�b�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�64.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����D"@I_�@\9P@Q+4�,�9@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	;�э���?;�э���?!;�э���?      ��!       "	'���K
@'���K
@!'���K
@*      ��!       2      ��!       :	��5w @��5w @!��5w @B      ��!       J	���L0\�?���L0\�?!���L0\�?R      ��!       Z	���L0\�?���L0\�?!���L0\�?b      ��!       JGPUY����D"@b q_�@\9P@y+4�,�9@
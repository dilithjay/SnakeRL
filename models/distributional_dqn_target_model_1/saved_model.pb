??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
??*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	
signatures
#
_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
%
#_self_saveable_object_factories
?

kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
w
#_self_saveable_object_factories
	variables
 regularization_losses
!trainable_variables
"	keras_api
?

#kernel
$bias
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
?

*kernel
+bias
#,_self_saveable_object_factories
-	variables
.regularization_losses
/trainable_variables
0	keras_api
w
#1_self_saveable_object_factories
2	variables
3regularization_losses
4trainable_variables
5	keras_api
w
#6_self_saveable_object_factories
7	variables
8regularization_losses
9trainable_variables
:	keras_api
 
 
8
0
1
2
3
#4
$5
*6
+7
 
8
0
1
2
3
#4
$5
*6
+7
?
	variables
;layer_regularization_losses
<non_trainable_variables
=metrics

>layers
regularization_losses
?layer_metrics
trainable_variables
 
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
	variables
@layer_regularization_losses
Anon_trainable_variables
Bmetrics

Clayers
regularization_losses
Dlayer_metrics
trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
	variables
Elayer_regularization_losses
Fnon_trainable_variables
Gmetrics

Hlayers
regularization_losses
Ilayer_metrics
trainable_variables
 
 
 
 
?
	variables
Jlayer_regularization_losses
Knon_trainable_variables
Lmetrics

Mlayers
 regularization_losses
Nlayer_metrics
!trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1
 

#0
$1
?
&	variables
Olayer_regularization_losses
Pnon_trainable_variables
Qmetrics

Rlayers
'regularization_losses
Slayer_metrics
(trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1
 

*0
+1
?
-	variables
Tlayer_regularization_losses
Unon_trainable_variables
Vmetrics

Wlayers
.regularization_losses
Xlayer_metrics
/trainable_variables
 
 
 
 
?
2	variables
Ylayer_regularization_losses
Znon_trainable_variables
[metrics

\layers
3regularization_losses
]layer_metrics
4trainable_variables
 
 
 
 
?
7	variables
^layer_regularization_losses
_non_trainable_variables
`metrics

alayers
8regularization_losses
blayer_metrics
9trainable_variables
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????

*
dtype0*$
shape:?????????


?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_57994009
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_57994395
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_57994429??
?
?
*__inference_model_1_layer_call_fn_57993938
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_579939192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????


!
_user_specified_name	input_2
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_57993735

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_57994303

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_softmax_1_layer_call_fn_57994348

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_579938532
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_57993713

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_model_1_layer_call_and_return_conditional_losses_57993967

inputs
conv2d_2_57993943
conv2d_2_57993945
conv2d_3_57993948
conv2d_3_57993950
dense_2_57993954
dense_2_57993956
dense_3_57993959
dense_3_57993961
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_57993943conv2d_2_57993945*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_579936862"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_57993948conv2d_3_57993950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_579937132"
 conv2d_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_579937352
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_57993954dense_2_57993956*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_579937542!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_57993959dense_3_57993961*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_579937812!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_579938102
reshape_1/PartitionedCall?
softmax_1/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_579938532
softmax_1/PartitionedCall?
IdentityIdentity"softmax_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
*__inference_model_1_layer_call_fn_57994178

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_579939192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_57993810

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:????????? 2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_model_1_layer_call_and_return_conditional_losses_57993919

inputs
conv2d_2_57993895
conv2d_2_57993897
conv2d_3_57993900
conv2d_3_57993902
dense_2_57993906
dense_2_57993908
dense_3_57993911
dense_3_57993913
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_57993895conv2d_2_57993897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_579936862"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_57993900conv2d_3_57993902*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_579937132"
 conv2d_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_579937352
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_57993906dense_2_57993908*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_579937542!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_57993911dense_3_57993913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_579937812!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_579938102
reshape_1/PartitionedCall?
softmax_1/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_579938532
softmax_1/PartitionedCall?
IdentityIdentity"softmax_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?V
?
E__inference_model_1_layer_call_and_return_conditional_losses_57994083

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Relul
reshape_1/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_1/Reshape/shape/2?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_3/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
reshape_1/Reshapeb
softmax_1/RankConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Rankf
softmax_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Rank_1d
softmax_1/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Sub/y{
softmax_1/SubSubsoftmax_1/Rank_1:output:0softmax_1/Sub/y:output:0*
T0*
_output_shapes
: 2
softmax_1/Subp
softmax_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/range/startp
softmax_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range/limitp
softmax_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range/delta?
softmax_1/rangeRangesoftmax_1/range/start:output:0softmax_1/range/limit:output:0softmax_1/range/delta:output:0*
_output_shapes
:2
softmax_1/ranget
softmax_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_1/startt
softmax_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_1/delta?
softmax_1/range_1Range softmax_1/range_1/start:output:0softmax_1/Sub:z:0 softmax_1/range_1/delta:output:0*
_output_shapes
: 2
softmax_1/range_1
softmax_1/concat/values_1Packsoftmax_1/Sub:z:0*
N*
T0*
_output_shapes
:2
softmax_1/concat/values_1?
softmax_1/concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
softmax_1/concat/values_3p
softmax_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/concat/axis?
softmax_1/concatConcatV2softmax_1/range:output:0"softmax_1/concat/values_1:output:0softmax_1/range_1:output:0"softmax_1/concat/values_3:output:0softmax_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
softmax_1/concat?
softmax_1/transpose	Transposereshape_1/Reshape:output:0softmax_1/concat:output:0*
T0*+
_output_shapes
:????????? 2
softmax_1/transpose?
softmax_1/SoftmaxSoftmaxsoftmax_1/transpose:y:0*
T0*+
_output_shapes
:????????? 2
softmax_1/Softmaxh
softmax_1/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Sub_1/y?
softmax_1/Sub_1Subsoftmax_1/Rank_1:output:0softmax_1/Sub_1/y:output:0*
T0*
_output_shapes
: 2
softmax_1/Sub_1t
softmax_1/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/range_2/startt
softmax_1/range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_2/limitt
softmax_1/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_2/delta?
softmax_1/range_2Range softmax_1/range_2/start:output:0 softmax_1/range_2/limit:output:0 softmax_1/range_2/delta:output:0*
_output_shapes
:2
softmax_1/range_2t
softmax_1/range_3/startConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_3/startt
softmax_1/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_3/delta?
softmax_1/range_3Range softmax_1/range_3/start:output:0softmax_1/Sub_1:z:0 softmax_1/range_3/delta:output:0*
_output_shapes
: 2
softmax_1/range_3?
softmax_1/concat_1/values_1Packsoftmax_1/Sub_1:z:0*
N*
T0*
_output_shapes
:2
softmax_1/concat_1/values_1?
softmax_1/concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
softmax_1/concat_1/values_3t
softmax_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/concat_1/axis?
softmax_1/concat_1ConcatV2softmax_1/range_2:output:0$softmax_1/concat_1/values_1:output:0softmax_1/range_3:output:0$softmax_1/concat_1/values_3:output:0 softmax_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
softmax_1/concat_1?
softmax_1/transpose_1	Transposesoftmax_1/Softmax:softmax:0softmax_1/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
softmax_1/transpose_1?
IdentityIdentitysoftmax_1/transpose_1:y:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
c
G__inference_softmax_1_layer_call_and_return_conditional_losses_57994343

inputs
identityN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1P
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
Sub/yS
SubSubRank_1:output:0Sub/y:output:0*
T0*
_output_shapes
: 2
Sub\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/limitConst*
_output_shapes
: *
dtype0*
value	B :2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltau
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:2
range`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/deltan
range_1Rangerange_1/start:output:0Sub:z:0range_1/delta:output:0*
_output_shapes
: 2	
range_1a
concat/values_1PackSub:z:0*
N*
T0*
_output_shapes
:2
concat/values_1l
concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
concat/values_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatr
	transpose	Transposeinputsconcat:output:0*
T0*+
_output_shapes
:????????? 2
	transposeb
SoftmaxSoftmaxtranspose:y:0*
T0*+
_output_shapes
:????????? 2	
SoftmaxT
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Sub_1/yY
Sub_1SubRank_1:output:0Sub_1/y:output:0*
T0*
_output_shapes
: 2
Sub_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/start`
range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/limit`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/delta
range_2Rangerange_2/start:output:0range_2/limit:output:0range_2/delta:output:0*
_output_shapes
:2	
range_2`
range_3/startConst*
_output_shapes
: *
dtype0*
value	B :2
range_3/start`
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_3/deltap
range_3Rangerange_3/start:output:0	Sub_1:z:0range_3/delta:output:0*
_output_shapes
: 2	
range_3g
concat_1/values_1Pack	Sub_1:z:0*
N*
T0*
_output_shapes
:2
concat_1/values_1p
concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
concat_1/values_3`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1?
transpose_1	TransposeSoftmax:softmax:0concat_1:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1g
IdentityIdentitytranspose_1:y:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
$__inference__traced_restore_57994429
file_prefix$
 assignvariableop_conv2d_2_kernel$
 assignvariableop_1_conv2d_2_bias&
"assignvariableop_2_conv2d_3_kernel$
 assignvariableop_3_conv2d_3_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
+__inference_conv2d_2_layer_call_fn_57994219

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_579936862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_57993686

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?

?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_57994230

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
!__inference__traced_save_57994395
file_prefix.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*m
_input_shapes\
Z: : : :  : :
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:	

_output_shapes
: 
?
?
*__inference_model_1_layer_call_fn_57994199

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_579939672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_57994261

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_1_layer_call_fn_57994250

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_579937352
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_model_1_layer_call_and_return_conditional_losses_57993889
input_2
conv2d_2_57993865
conv2d_2_57993867
conv2d_3_57993870
conv2d_3_57993872
dense_2_57993876
dense_2_57993878
dense_3_57993881
dense_3_57993883
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_57993865conv2d_2_57993867*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_579936862"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_57993870conv2d_3_57993872*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_579937132"
 conv2d_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_579937352
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_57993876dense_2_57993878*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_579937542!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_57993881dense_3_57993883*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_579937812!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_579938102
reshape_1/PartitionedCall?
softmax_1/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_579938532
softmax_1/PartitionedCall?
IdentityIdentity"softmax_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:?????????


!
_user_specified_name	input_2
?
?
E__inference_model_1_layer_call_and_return_conditional_losses_57993862
input_2
conv2d_2_57993697
conv2d_2_57993699
conv2d_3_57993724
conv2d_3_57993726
dense_2_57993765
dense_2_57993767
dense_3_57993792
dense_3_57993794
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_57993697conv2d_2_57993699*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_579936862"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_57993724conv2d_3_57993726*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_579937132"
 conv2d_3/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_579937352
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_57993765dense_2_57993767*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_579937542!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_57993792dense_3_57993794*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_579937812!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_579938102
reshape_1/PartitionedCall?
softmax_1/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_579938532
softmax_1/PartitionedCall?
IdentityIdentity"softmax_1/PartitionedCall:output:0!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:?????????


!
_user_specified_name	input_2
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_57993754

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_57994009
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_579936712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????


!
_user_specified_name	input_2
?
c
G__inference_softmax_1_layer_call_and_return_conditional_losses_57993853

inputs
identityN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1P
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
Sub/yS
SubSubRank_1:output:0Sub/y:output:0*
T0*
_output_shapes
: 2
Sub\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/limitConst*
_output_shapes
: *
dtype0*
value	B :2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltau
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:2
range`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/deltan
range_1Rangerange_1/start:output:0Sub:z:0range_1/delta:output:0*
_output_shapes
: 2	
range_1a
concat/values_1PackSub:z:0*
N*
T0*
_output_shapes
:2
concat/values_1l
concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
concat/values_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatr
	transpose	Transposeinputsconcat:output:0*
T0*+
_output_shapes
:????????? 2
	transposeb
SoftmaxSoftmaxtranspose:y:0*
T0*+
_output_shapes
:????????? 2	
SoftmaxT
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Sub_1/yY
Sub_1SubRank_1:output:0Sub_1/y:output:0*
T0*
_output_shapes
: 2
Sub_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/start`
range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/limit`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/delta
range_2Rangerange_2/start:output:0range_2/limit:output:0range_2/delta:output:0*
_output_shapes
:2	
range_2`
range_3/startConst*
_output_shapes
: *
dtype0*
value	B :2
range_3/start`
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_3/deltap
range_3Rangerange_3/start:output:0	Sub_1:z:0range_3/delta:output:0*
_output_shapes
: 2	
range_3g
concat_1/values_1Pack	Sub_1:z:0*
N*
T0*
_output_shapes
:2
concat_1/values_1p
concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
concat_1/values_3`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1?
transpose_1	TransposeSoftmax:softmax:0concat_1:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1g
IdentityIdentitytranspose_1:y:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_57994245

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
+__inference_conv2d_3_layer_call_fn_57994239

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_579937132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?V
?
E__inference_model_1_layer_call_and_return_conditional_losses_57994157

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Relul
reshape_1/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape_1/Reshape/shape/2?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_3/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
reshape_1/Reshapeb
softmax_1/RankConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Rankf
softmax_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Rank_1d
softmax_1/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Sub/y{
softmax_1/SubSubsoftmax_1/Rank_1:output:0softmax_1/Sub/y:output:0*
T0*
_output_shapes
: 2
softmax_1/Subp
softmax_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/range/startp
softmax_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range/limitp
softmax_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range/delta?
softmax_1/rangeRangesoftmax_1/range/start:output:0softmax_1/range/limit:output:0softmax_1/range/delta:output:0*
_output_shapes
:2
softmax_1/ranget
softmax_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_1/startt
softmax_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_1/delta?
softmax_1/range_1Range softmax_1/range_1/start:output:0softmax_1/Sub:z:0 softmax_1/range_1/delta:output:0*
_output_shapes
: 2
softmax_1/range_1
softmax_1/concat/values_1Packsoftmax_1/Sub:z:0*
N*
T0*
_output_shapes
:2
softmax_1/concat/values_1?
softmax_1/concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
softmax_1/concat/values_3p
softmax_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/concat/axis?
softmax_1/concatConcatV2softmax_1/range:output:0"softmax_1/concat/values_1:output:0softmax_1/range_1:output:0"softmax_1/concat/values_3:output:0softmax_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
softmax_1/concat?
softmax_1/transpose	Transposereshape_1/Reshape:output:0softmax_1/concat:output:0*
T0*+
_output_shapes
:????????? 2
softmax_1/transpose?
softmax_1/SoftmaxSoftmaxsoftmax_1/transpose:y:0*
T0*+
_output_shapes
:????????? 2
softmax_1/Softmaxh
softmax_1/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/Sub_1/y?
softmax_1/Sub_1Subsoftmax_1/Rank_1:output:0softmax_1/Sub_1/y:output:0*
T0*
_output_shapes
: 2
softmax_1/Sub_1t
softmax_1/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/range_2/startt
softmax_1/range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_2/limitt
softmax_1/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_2/delta?
softmax_1/range_2Range softmax_1/range_2/start:output:0 softmax_1/range_2/limit:output:0 softmax_1/range_2/delta:output:0*
_output_shapes
:2
softmax_1/range_2t
softmax_1/range_3/startConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_3/startt
softmax_1/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
softmax_1/range_3/delta?
softmax_1/range_3Range softmax_1/range_3/start:output:0softmax_1/Sub_1:z:0 softmax_1/range_3/delta:output:0*
_output_shapes
: 2
softmax_1/range_3?
softmax_1/concat_1/values_1Packsoftmax_1/Sub_1:z:0*
N*
T0*
_output_shapes
:2
softmax_1/concat_1/values_1?
softmax_1/concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:2
softmax_1/concat_1/values_3t
softmax_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
softmax_1/concat_1/axis?
softmax_1/concat_1ConcatV2softmax_1/range_2:output:0$softmax_1/concat_1/values_1:output:0softmax_1/range_3:output:0$softmax_1/concat_1/values_3:output:0 softmax_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
softmax_1/concat_1?
softmax_1/transpose_1	Transposesoftmax_1/Softmax:softmax:0softmax_1/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
softmax_1/transpose_1?
IdentityIdentitysoftmax_1/transpose_1:y:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?

*__inference_dense_2_layer_call_fn_57994270

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_579937542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_57994210

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
?
*__inference_model_1_layer_call_fn_57993986
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_579939672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????


!
_user_specified_name	input_2
?e
?
#__inference__wrapped_model_57993671
input_23
/model_1_conv2d_2_conv2d_readvariableop_resource4
0model_1_conv2d_2_biasadd_readvariableop_resource3
/model_1_conv2d_3_conv2d_readvariableop_resource4
0model_1_conv2d_3_biasadd_readvariableop_resource2
.model_1_dense_2_matmul_readvariableop_resource3
/model_1_dense_2_biasadd_readvariableop_resource2
.model_1_dense_3_matmul_readvariableop_resource3
/model_1_dense_3_biasadd_readvariableop_resource
identity??'model_1/conv2d_2/BiasAdd/ReadVariableOp?&model_1/conv2d_2/Conv2D/ReadVariableOp?'model_1/conv2d_3/BiasAdd/ReadVariableOp?&model_1/conv2d_3/Conv2D/ReadVariableOp?&model_1/dense_2/BiasAdd/ReadVariableOp?%model_1/dense_2/MatMul/ReadVariableOp?&model_1/dense_3/BiasAdd/ReadVariableOp?%model_1/dense_3/MatMul/ReadVariableOp?
&model_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_1/conv2d_2/Conv2D/ReadVariableOp?
model_1/conv2d_2/Conv2DConv2Dinput_2.model_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
model_1/conv2d_2/Conv2D?
'model_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_2/BiasAdd/ReadVariableOp?
model_1/conv2d_2/BiasAddBiasAdd model_1/conv2d_2/Conv2D:output:0/model_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model_1/conv2d_2/BiasAdd?
model_1/conv2d_2/ReluRelu!model_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model_1/conv2d_2/Relu?
&model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02(
&model_1/conv2d_3/Conv2D/ReadVariableOp?
model_1/conv2d_3/Conv2DConv2D#model_1/conv2d_2/Relu:activations:0.model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
model_1/conv2d_3/Conv2D?
'model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_3/BiasAdd/ReadVariableOp?
model_1/conv2d_3/BiasAddBiasAdd model_1/conv2d_3/Conv2D:output:0/model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model_1/conv2d_3/BiasAdd?
model_1/conv2d_3/ReluRelu!model_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model_1/conv2d_3/Relu?
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_1/Const?
model_1/flatten_1/ReshapeReshape#model_1/conv2d_3/Relu:activations:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_1/flatten_1/Reshape?
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_1/dense_2/MatMul/ReadVariableOp?
model_1/dense_2/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_2/MatMul?
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_1/dense_2/BiasAdd/ReadVariableOp?
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_2/BiasAdd?
model_1/dense_2/ReluRelu model_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dense_2/Relu?
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp?
model_1/dense_3/MatMulMatMul"model_1/dense_2/Relu:activations:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_3/MatMul?
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp?
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_3/BiasAdd?
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dense_3/Relu?
model_1/reshape_1/ShapeShape"model_1/dense_3/Relu:activations:0*
T0*
_output_shapes
:2
model_1/reshape_1/Shape?
%model_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/reshape_1/strided_slice/stack?
'model_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_1?
'model_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_2?
model_1/reshape_1/strided_sliceStridedSlice model_1/reshape_1/Shape:output:0.model_1/reshape_1/strided_slice/stack:output:00model_1/reshape_1/strided_slice/stack_1:output:00model_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_1/reshape_1/strided_slice?
!model_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/reshape_1/Reshape/shape/1?
!model_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2#
!model_1/reshape_1/Reshape/shape/2?
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*model_1/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2!
model_1/reshape_1/Reshape/shape?
model_1/reshape_1/ReshapeReshape"model_1/dense_3/Relu:activations:0(model_1/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:????????? 2
model_1/reshape_1/Reshaper
model_1/softmax_1/RankConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/softmax_1/Rankv
model_1/softmax_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
model_1/softmax_1/Rank_1t
model_1/softmax_1/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/softmax_1/Sub/y?
model_1/softmax_1/SubSub!model_1/softmax_1/Rank_1:output:0 model_1/softmax_1/Sub/y:output:0*
T0*
_output_shapes
: 2
model_1/softmax_1/Sub?
model_1/softmax_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
model_1/softmax_1/range/start?
model_1/softmax_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/softmax_1/range/limit?
model_1/softmax_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/softmax_1/range/delta?
model_1/softmax_1/rangeRange&model_1/softmax_1/range/start:output:0&model_1/softmax_1/range/limit:output:0&model_1/softmax_1/range/delta:output:0*
_output_shapes
:2
model_1/softmax_1/range?
model_1/softmax_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/softmax_1/range_1/start?
model_1/softmax_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/softmax_1/range_1/delta?
model_1/softmax_1/range_1Range(model_1/softmax_1/range_1/start:output:0model_1/softmax_1/Sub:z:0(model_1/softmax_1/range_1/delta:output:0*
_output_shapes
: 2
model_1/softmax_1/range_1?
!model_1/softmax_1/concat/values_1Packmodel_1/softmax_1/Sub:z:0*
N*
T0*
_output_shapes
:2#
!model_1/softmax_1/concat/values_1?
!model_1/softmax_1/concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:2#
!model_1/softmax_1/concat/values_3?
model_1/softmax_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
model_1/softmax_1/concat/axis?
model_1/softmax_1/concatConcatV2 model_1/softmax_1/range:output:0*model_1/softmax_1/concat/values_1:output:0"model_1/softmax_1/range_1:output:0*model_1/softmax_1/concat/values_3:output:0&model_1/softmax_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
model_1/softmax_1/concat?
model_1/softmax_1/transpose	Transpose"model_1/reshape_1/Reshape:output:0!model_1/softmax_1/concat:output:0*
T0*+
_output_shapes
:????????? 2
model_1/softmax_1/transpose?
model_1/softmax_1/SoftmaxSoftmaxmodel_1/softmax_1/transpose:y:0*
T0*+
_output_shapes
:????????? 2
model_1/softmax_1/Softmaxx
model_1/softmax_1/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_1/softmax_1/Sub_1/y?
model_1/softmax_1/Sub_1Sub!model_1/softmax_1/Rank_1:output:0"model_1/softmax_1/Sub_1/y:output:0*
T0*
_output_shapes
: 2
model_1/softmax_1/Sub_1?
model_1/softmax_1/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2!
model_1/softmax_1/range_2/start?
model_1/softmax_1/range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/softmax_1/range_2/limit?
model_1/softmax_1/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/softmax_1/range_2/delta?
model_1/softmax_1/range_2Range(model_1/softmax_1/range_2/start:output:0(model_1/softmax_1/range_2/limit:output:0(model_1/softmax_1/range_2/delta:output:0*
_output_shapes
:2
model_1/softmax_1/range_2?
model_1/softmax_1/range_3/startConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/softmax_1/range_3/start?
model_1/softmax_1/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2!
model_1/softmax_1/range_3/delta?
model_1/softmax_1/range_3Range(model_1/softmax_1/range_3/start:output:0model_1/softmax_1/Sub_1:z:0(model_1/softmax_1/range_3/delta:output:0*
_output_shapes
: 2
model_1/softmax_1/range_3?
#model_1/softmax_1/concat_1/values_1Packmodel_1/softmax_1/Sub_1:z:0*
N*
T0*
_output_shapes
:2%
#model_1/softmax_1/concat_1/values_1?
#model_1/softmax_1/concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_1/softmax_1/concat_1/values_3?
model_1/softmax_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
model_1/softmax_1/concat_1/axis?
model_1/softmax_1/concat_1ConcatV2"model_1/softmax_1/range_2:output:0,model_1/softmax_1/concat_1/values_1:output:0"model_1/softmax_1/range_3:output:0,model_1/softmax_1/concat_1/values_3:output:0(model_1/softmax_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
model_1/softmax_1/concat_1?
model_1/softmax_1/transpose_1	Transpose#model_1/softmax_1/Softmax:softmax:0#model_1/softmax_1/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
model_1/softmax_1/transpose_1?
IdentityIdentity!model_1/softmax_1/transpose_1:y:0(^model_1/conv2d_2/BiasAdd/ReadVariableOp'^model_1/conv2d_2/Conv2D/ReadVariableOp(^model_1/conv2d_3/BiasAdd/ReadVariableOp'^model_1/conv2d_3/Conv2D/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????

::::::::2R
'model_1/conv2d_2/BiasAdd/ReadVariableOp'model_1/conv2d_2/BiasAdd/ReadVariableOp2P
&model_1/conv2d_2/Conv2D/ReadVariableOp&model_1/conv2d_2/Conv2D/ReadVariableOp2R
'model_1/conv2d_3/BiasAdd/ReadVariableOp'model_1/conv2d_3/BiasAdd/ReadVariableOp2P
&model_1/conv2d_3/Conv2D/ReadVariableOp&model_1/conv2d_3/Conv2D/ReadVariableOp2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????


!
_user_specified_name	input_2
?
H
,__inference_reshape_1_layer_call_fn_57994308

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_579938102
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_57993781

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_57994281

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_3_layer_call_fn_57994290

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_579937812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_28
serving_default_input_2:0?????????

A
	softmax_14
StatefulPartitionedCall:0????????? tensorflow/serving/predict:??
?=
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	
signatures
#
_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_default_save_signature"?9
_tf_keras_network?9{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 32]}}, "name": "reshape_1", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Softmax", "config": {"name": "softmax_1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "softmax_1", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["softmax_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10, 10, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 32]}}, "name": "reshape_1", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Softmax", "config": {"name": "softmax_1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "softmax_1", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["softmax_1", 0, 0]]}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 10, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?


kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 1]}}
?


kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
*h&call_and_return_all_conditional_losses
i__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 32]}}
?
#_self_saveable_object_factories
	variables
 regularization_losses
!trainable_variables
"	keras_api
*j&call_and_return_all_conditional_losses
k__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

#kernel
$bias
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
*l&call_and_return_all_conditional_losses
m__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 800]}}
?

*kernel
+bias
#,_self_saveable_object_factories
-	variables
.regularization_losses
/trainable_variables
0	keras_api
*n&call_and_return_all_conditional_losses
o__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
#1_self_saveable_object_factories
2	variables
3regularization_losses
4trainable_variables
5	keras_api
*p&call_and_return_all_conditional_losses
q__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 32]}}}
?
#6_self_saveable_object_factories
7	variables
8regularization_losses
9trainable_variables
:	keras_api
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax_1", "trainable": true, "dtype": "float32", "axis": 1}}
,
tserving_default"
signature_map
 "
trackable_dict_wrapper
X
0
1
2
3
#4
$5
*6
+7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
#4
$5
*6
+7"
trackable_list_wrapper
?
	variables
;layer_regularization_losses
<non_trainable_variables
=metrics

>layers
regularization_losses
?layer_metrics
trainable_variables
c__call__
e_default_save_signature
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
):' 2conv2d_2/kernel
: 2conv2d_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
@layer_regularization_losses
Anon_trainable_variables
Bmetrics

Clayers
regularization_losses
Dlayer_metrics
trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_3/kernel
: 2conv2d_3/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
Elayer_regularization_losses
Fnon_trainable_variables
Gmetrics

Hlayers
regularization_losses
Ilayer_metrics
trainable_variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Jlayer_regularization_losses
Knon_trainable_variables
Lmetrics

Mlayers
 regularization_losses
Nlayer_metrics
!trainable_variables
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
 "
trackable_dict_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
&	variables
Olayer_regularization_losses
Pnon_trainable_variables
Qmetrics

Rlayers
'regularization_losses
Slayer_metrics
(trainable_variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_3/kernel
:?2dense_3/bias
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
-	variables
Tlayer_regularization_losses
Unon_trainable_variables
Vmetrics

Wlayers
.regularization_losses
Xlayer_metrics
/trainable_variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
2	variables
Ylayer_regularization_losses
Znon_trainable_variables
[metrics

\layers
3regularization_losses
]layer_metrics
4trainable_variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7	variables
^layer_regularization_losses
_non_trainable_variables
`metrics

alayers
8regularization_losses
blayer_metrics
9trainable_variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
*__inference_model_1_layer_call_fn_57994199
*__inference_model_1_layer_call_fn_57993938
*__inference_model_1_layer_call_fn_57994178
*__inference_model_1_layer_call_fn_57993986?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_1_layer_call_and_return_conditional_losses_57993862
E__inference_model_1_layer_call_and_return_conditional_losses_57994157
E__inference_model_1_layer_call_and_return_conditional_losses_57994083
E__inference_model_1_layer_call_and_return_conditional_losses_57993889?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_57993671?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_2?????????


?2?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_57994210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_2_layer_call_fn_57994219?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_57994230?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_3_layer_call_fn_57994239?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_1_layer_call_and_return_conditional_losses_57994245?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_1_layer_call_fn_57994250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_2_layer_call_and_return_conditional_losses_57994261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_2_layer_call_fn_57994270?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_3_layer_call_and_return_conditional_losses_57994281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_3_layer_call_fn_57994290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_reshape_1_layer_call_and_return_conditional_losses_57994303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_reshape_1_layer_call_fn_57994308?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_softmax_1_layer_call_and_return_conditional_losses_57994343?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_softmax_1_layer_call_fn_57994348?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_57994009input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_57993671#$*+8?5
.?+
)?&
input_2?????????


? "9?6
4
	softmax_1'?$
	softmax_1????????? ?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_57994210l7?4
-?*
(?%
inputs?????????


? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_2_layer_call_fn_57994219_7?4
-?*
(?%
inputs?????????


? " ?????????? ?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_57994230l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_3_layer_call_fn_57994239_7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
E__inference_dense_2_layer_call_and_return_conditional_losses_57994261^#$0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_2_layer_call_fn_57994270Q#$0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_3_layer_call_and_return_conditional_losses_57994281^*+0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_3_layer_call_fn_57994290Q*+0?-
&?#
!?
inputs??????????
? "????????????
G__inference_flatten_1_layer_call_and_return_conditional_losses_57994245a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
,__inference_flatten_1_layer_call_fn_57994250T7?4
-?*
(?%
inputs????????? 
? "????????????
E__inference_model_1_layer_call_and_return_conditional_losses_57993862w#$*+@?=
6?3
)?&
input_2?????????


p

 
? ")?&
?
0????????? 
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_57993889w#$*+@?=
6?3
)?&
input_2?????????


p 

 
? ")?&
?
0????????? 
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_57994083v#$*+??<
5?2
(?%
inputs?????????


p

 
? ")?&
?
0????????? 
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_57994157v#$*+??<
5?2
(?%
inputs?????????


p 

 
? ")?&
?
0????????? 
? ?
*__inference_model_1_layer_call_fn_57993938j#$*+@?=
6?3
)?&
input_2?????????


p

 
? "?????????? ?
*__inference_model_1_layer_call_fn_57993986j#$*+@?=
6?3
)?&
input_2?????????


p 

 
? "?????????? ?
*__inference_model_1_layer_call_fn_57994178i#$*+??<
5?2
(?%
inputs?????????


p

 
? "?????????? ?
*__inference_model_1_layer_call_fn_57994199i#$*+??<
5?2
(?%
inputs?????????


p 

 
? "?????????? ?
G__inference_reshape_1_layer_call_and_return_conditional_losses_57994303]0?-
&?#
!?
inputs??????????
? ")?&
?
0????????? 
? ?
,__inference_reshape_1_layer_call_fn_57994308P0?-
&?#
!?
inputs??????????
? "?????????? ?
&__inference_signature_wrapper_57994009?#$*+C?@
? 
9?6
4
input_2)?&
input_2?????????

"9?6
4
	softmax_1'?$
	softmax_1????????? ?
G__inference_softmax_1_layer_call_and_return_conditional_losses_57994343d7?4
-?*
$?!
inputs????????? 

 
? ")?&
?
0????????? 
? ?
,__inference_softmax_1_layer_call_fn_57994348W7?4
-?*
$?!
inputs????????? 

 
? "?????????? 
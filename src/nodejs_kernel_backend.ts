/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// tslint:disable-next-line:max-line-length
import {BackendTimingInfo, DataType, fill, KernelBackend, ones, Rank, rsqrt, scalar, ShapeMap, Tensor, Tensor1D, tensor1d, Tensor2D, tensor2d, Tensor3D, tensor3d, Tensor4D} from '@tensorflow/tfjs-core';
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';
import {upcastType} from '@tensorflow/tfjs-core/dist/types';
import {TensorMetadata, TFEOpAttr, TFJSBinding} from './tfjs_binding';

type TensorInfo = {
  shape: number[],
  dtype: number,
  values: Float32Array|Int32Array|Uint8Array,
  id: number
};

interface DataId {}

export class NodeJSKernelBackend implements KernelBackend {
  private binding: TFJSBinding;
  private tensorMap = new WeakMap<DataId, TensorInfo>();

  constructor(binding: TFJSBinding) {
    this.binding = binding;
  }

  // Returns the TF dtype for a given DataType.
  private getTFDType(dataType: DataType): number {
    switch (dataType) {
      case 'float32':
        return this.binding.TF_FLOAT;
      case 'int32':
        return this.binding.TF_INT32;
      case 'bool':
        return this.binding.TF_BOOL;
      default:
        throw new Error('Unknown dtype `${dtype}`');
    }
  }

  // Creates a new Tensor and maps the dataId to the passed in ID.
  private createOutputTensor(metadata: TensorMetadata): Tensor {
    const newId = {};

    this.tensorMap.set(newId, {
      shape: metadata.shape,
      dtype: metadata.dtype,
      id: metadata.id,
      values: null
    });

    let dtype: DataType;
    switch (metadata.dtype) {
      case this.binding.TF_FLOAT:
        dtype = 'float32';
        break;
      case this.binding.TF_INT32:
        dtype = 'int32';
        break;
      case this.binding.TF_BOOL:
        dtype = 'bool';
        break;
      default:
        throw new Error(`Unknown dtype enum ${metadata.dtype}`);
    }
    return Tensor.make(metadata.shape, {dataId: newId}, dtype);
  }

  // Prepares Tensor instances for Op execution.
  getInputTensorIds(tensors: Tensor[]): number[] {
    const ids: number[] = [];
    for (let i = 0; i < tensors.length; i++) {
      const info = this.tensorMap.get(tensors[i].dataId);
      // TODO - what about ID in this case? Handle in write()??
      if (info.values != null) {
        // Values were delayed to write into the TensorHandle. Do that before Op
        // execution and clear stored values.
        info.id =
            this.binding.createTensor(info.shape, info.dtype, info.values);
        info.values = null;
        this.tensorMap.set(tensors[i].dataId, info);
      }
      ids.push(info.id);
    }
    return ids;
  }

  private createReductionOpAttrs(tensor: Tensor): TFEOpAttr[] {
    return [
      {name: 'keep_dims', type: this.binding.TF_ATTR_BOOL, value: false},
      this.createTypeOpAttr('T', tensor.dtype),
      this.createTypeOpAttr('Tidx', 'int32')
    ];
  }

  private createTypeOpAttr(attrName: string, dtype: DataType): TFEOpAttr {
    return {
      name: attrName,
      type: this.binding.TF_ATTR_TYPE,
      value: this.getTFDType(dtype)
    };
  }

  // TODO - drop.
  private executeOpSingleInput(name: string, input: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', input.dtype)];
    return this.executeOpSingleOutput(name, opAttrs, [input]);
  }

  executeOpSingleOutput(name: string, opAttrs: TFEOpAttr[], inputs: Tensor[]):
      Tensor {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), 1);
    return this.createOutputTensor(outputMetadata[0]);
  }

  executeOpMultipleOutputs(
      name: string, opAttrs: TFEOpAttr[], inputs: Tensor[],
      numOutputs: number): Tensor[] {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), numOutputs);
    return outputMetadata.map(m => this.createOutputTensor(m));
  }

  dispose(): void {}

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    return this.readSync(dataId);
  }

  readSync(dataId: object): Float32Array|Int32Array|Uint8Array {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    if (info.values != null) {
      return info.values;
    } else {
      return this.binding.tensorDataSync(info.id);
    }
  }

  disposeData(dataId: object): void {
    const id = this.tensorMap.get(dataId).id;
    if (id != null && id >= 0) {
      this.binding.deleteTensor(id);
    }
    this.tensorMap.delete(dataId);
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    this.tensorMap.set(dataId, info);
  }

  register(dataId: object, shape: number[], dtype: DataType): void {
    if (!this.tensorMap.has(dataId)) {
      this.tensorMap.set(
          dataId, {shape, dtype: this.getTFDType(dtype), values: null, id: -1});
    }
  }

  matMul(a: Tensor2D, b: Tensor2D, transposeA: boolean, transposeB: boolean):
      Tensor2D {
    const opAttrs = [
      {name: 'transpose_a', type: this.binding.TF_ATTR_BOOL, value: transposeA},
      {name: 'transpose_b', type: this.binding.TF_ATTR_BOOL, value: transposeB},
      this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))
    ];
    return this.executeOpSingleOutput('MatMul', opAttrs, [a, b]) as Tensor2D;
  }

  stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[],
      beginMask: number, endMask: number): T {
    const beginTensor = tensor1d(begin, 'int32');
    const endTensor = tensor1d(end, 'int32');
    const stridesTensor = tensor1d(strides, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Index', 'int32'),
      {name: 'begin_mask', type: this.binding.TF_ATTR_INT, value: beginMask},
      {name: 'end_mask', type: this.binding.TF_ATTR_INT, value: endMask},
      {name: 'ellipsis_mask', type: this.binding.TF_ATTR_INT, value: 0},
      {name: 'new_axis_mask', type: this.binding.TF_ATTR_INT, value: 0},
      {name: 'shrink_axis_mask', type: this.binding.TF_ATTR_INT, value: 0}
    ];
    return this.executeOpSingleOutput(
               'StridedSlice', opAttrs,
               [x, beginTensor, endTensor, stridesTensor]) as T;
  }

  slice<T extends Tensor>(x: T, begin: number[], size: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Index', 'int32')
    ];

    // Bind tensor values
    const beginTensor = tensor1d(begin, 'int32');
    const sizeTensor = tensor1d(size, 'int32');

    return this.executeOpSingleOutput(
               'Slice', opAttrs, [x, beginTensor, sizeTensor]) as T;
  }

  reverse<T extends Tensor>(a: T, axis: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('T', a.dtype)
    ];
    const axisTensor = tensor1d(axis, 'int32');
    return this.executeOpSingleOutput('ReverseV2', opAttrs, [a, axisTensor]) as
        T;
  }

  concat(a: Tensor2D, b: Tensor2D): Tensor2D {
    const opAttrs = [
      {name: 'N', type: this.binding.TF_ATTR_INT, value: 2},
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('T', a.dtype)
    ];
    // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
    const axisTensor = scalar(1, 'int32');
    return this.executeOpSingleOutput(
               'ConcatV2', opAttrs, [a, b, axisTensor]) as Tensor2D;
  }

  neg<T extends Tensor>(a: T): T {
    return this.executeOpSingleInput('Neg', a) as T;
  }

  add(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Add', opAttrs, [a, b]);
  }

  subtract(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Sub', opAttrs, [a, b]);
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Mul', opAttrs, [a, b]);
  }

  realDivide(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('RealDiv', opAttrs, [a, b]);
  }

  floorDiv(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('FloorDiv', opAttrs, [a, b]);
  }

  divide(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Div', opAttrs, [a, b]);
  }

  unsortedSegmentSum<T extends Tensor>(
      x: T, segmentIds: Tensor1D, numSegments: number): Tensor {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tindices', 'int32'),
      this.createTypeOpAttr('Tnumsegments', 'int32')
    ];
    return this.executeOpSingleOutput(
        'UnsortedSegmentSum', opAttrs,
        [x, segmentIds, scalar(numSegments, 'int32')]);
  }

  sum(x: Tensor, axes: number[]): Tensor {
    const axisTensor = tensor1d(axes, 'int32');
    return this.executeOpSingleOutput(
        'Sum', this.createReductionOpAttrs(x), [x, axisTensor]);
  }

  argMin(x: Tensor, axis: number): Tensor {
    const axisScalar = scalar(axis, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('output_type', 'int32')
    ];
    return this.executeOpSingleOutput('ArgMin', opAttrs, [x, axisScalar]);
  }

  argMax(x: Tensor, axis: number): Tensor {
    const axisScalar = scalar(axis, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tidx', 'int32'),
      this.createTypeOpAttr('output_type', 'int32')
    ];
    return this.executeOpSingleOutput('ArgMax', opAttrs, [x, axisScalar]);
  }

  equal(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Equal', opAttrs, [a, b]);
  }

  notEqual(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('NotEqual', opAttrs, [a, b]);
  }

  less(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Less', opAttrs, [a, b]);
  }

  lessEqual(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('LessEqual', opAttrs, [a, b]);
  }

  greater(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Greater', opAttrs, [a, b]);
  }

  greaterEqual(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('GreaterEqual', opAttrs, [a, b]);
  }

  logicalNot<T extends Tensor>(a: T): T {
    return this.executeOpSingleOutput('LogicalNot', [], [a]) as T;
  }

  logicalAnd(a: Tensor, b: Tensor): Tensor {
    return this.executeOpSingleOutput('LogicalAnd', [], [a, b]);
  }

  logicalOr(a: Tensor, b: Tensor): Tensor {
    return this.executeOpSingleOutput('LogicalOr', [], [a, b]);
  }

  where(condition: Tensor, a: Tensor, b: Tensor, dtype: DataType): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    // 'Select' Op is where with additional inputs.
    return this.executeOpSingleOutput('Select', opAttrs, [condition, a, b]);
  }

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  topKIndices(x: Tensor, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  min(x: Tensor, axes: number[]): Tensor {
    const axesTensor = tensor1d(axes, 'int32');
    return this.executeOpSingleOutput(
        'Min', this.createReductionOpAttrs(x), [x, axesTensor]);
  }

  minimum(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Minimum', opAttrs, [a, b]);
  }

  max(x: Tensor, axes: number[]): Tensor {
    const axesTensor = tensor1d(axes, 'int32');
    return this.executeOpSingleOutput(
        'Max', this.createReductionOpAttrs(x), [x, axesTensor]);
  }

  maximum(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', upcastType(a.dtype, b.dtype))];
    return this.executeOpSingleOutput('Maximum', opAttrs, [a, b]);
  }

  all(x: Tensor, axes: number[]): Tensor {
    const opAttrs = [
      {name: 'keep_dims', type: this.binding.TF_ATTR_BOOL, value: false},
      this.createTypeOpAttr('Tidx', 'int32')
    ];
    const axesTensor = tensor1d(axes, 'int32');
    return this.executeOpSingleOutput('All', opAttrs, [x, axesTensor]);
  }

  any(x: Tensor, axes: number[]): Tensor {
    const opAttrs = [
      {name: 'keep_dims', type: this.binding.TF_ATTR_BOOL, value: false},
      this.createTypeOpAttr('Tidx', 'int32')
    ];
    const axesTensor = tensor1d(axes, 'int32');
    return this.executeOpSingleOutput('Any', opAttrs, [x, axesTensor]);
  }

  ceil<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Ceil', x) as T;
  }

  floor<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Floor', x) as T;
  }

  pow<T extends Tensor>(a: T, b: Tensor): T {
    const dtype = upcastType(a.dtype, b.dtype);
    const opAttrs = [this.createTypeOpAttr('T', dtype)];
    return this.executeOpSingleOutput(
               'Pow', opAttrs, [a.cast(dtype), b.cast(dtype)]) as T;
  }

  exp<T extends Tensor>(x: T): T {
    const xTensor = x.dtype === 'int32' ? x.toFloat() : x;
    return this.executeOpSingleInput('Exp', xTensor) as T;
  }

  log<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Log', x) as T;
  }

  log1p<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Log1p', x) as T;
  }

  sqrt<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Sqrt', x) as T;
  }

  square<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Square', x) as T;
  }

  relu<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Relu', x) as T;
  }

  elu<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Elu', x) as T;
  }

  eluDer<T extends Tensor>(dy: T, y: T): T {
    const opAttrs = [this.createTypeOpAttr('T', y.dtype)];
    return this.executeOpSingleOutput('EluGrad', opAttrs, [dy, y]) as T;
  }

  selu<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Selu', x) as T;
  }

  int<T extends Tensor>(x: T): T {
    throw new Error('Method not implemented.');
  }

  clip<T extends Tensor>(x: T, min: number, max: number): T {
    const xMin = this.minimum(x, scalar(max));
    return this.maximum(xMin, scalar(min)) as T;
  }

  abs<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Abs', x) as T;
  }

  sigmoid<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Sigmoid', x) as T;
  }

  sin<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Sin', x) as T;
  }

  cos<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Cos', x) as T;
  }

  tan<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Tan', x) as T;
  }

  asin<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Asin', x) as T;
  }

  acos<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Acos', x) as T;
  }

  atan<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Atan', x) as T;
  }

  sinh<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Sinh', x) as T;
  }

  cosh<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Cosh', x) as T;
  }

  tanh<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Tanh', x) as T;
  }

  mod(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', a.dtype)];
    return this.executeOpSingleOutput('FloorMod', opAttrs, [a, b]);
  }
  round<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Round', x) as T;
  }
  sign<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Sign', x) as T;
  }
  rsqrt<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Rsqrt', x) as T;
  }
  reciprocal<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Reciprocal', x) as T;
  }
  asinh<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Asinh', x) as T;
  }
  acosh<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Acosh', x) as T;
  }
  atanh<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Atanh', x) as T;
  }

  erf<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Erf', x) as T;
  }

  squaredDifference(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [this.createTypeOpAttr('T', a.dtype)];
    return this.executeOpSingleOutput('SquaredDifference', opAttrs, [a, b]);
  }

  expm1<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Expm1', x) as T;
  }

  softplus<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('Softplus', x) as T;
  }

  atan2<T extends Tensor>(a: T, b: T): T {
    const opAttrs = [this.createTypeOpAttr('T', a.dtype)];
    return this.executeOpSingleOutput('Atan2', opAttrs, [a, b]) as T;
  }

  step<T extends Tensor>(x: T, alpha: number): T {
    const dtype = x.dtype;
    const nans = this.isNaN(x);
    const stepNoNans = this.where(
        this.greater(x, scalar(0, dtype)), ones(x.shape),
        fill(x.shape, alpha, dtype), dtype);
    return this.where(nans, x, stepNoNans, dtype) as T;
  }

  conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'use_cudnn_on_gpu', type: this.binding.TF_ATTR_BOOL, value: true},
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations},
    ];
    return this.executeOpSingleOutput('Conv2D', opAttrs, [x, filter]) as
        Tensor4D;
  }

  conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', 'float32'),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'use_cudnn_on_gpu', type: this.binding.TF_ATTR_BOOL, value: true},
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];
    const inputSizes = tensor1d(convInfo.inShape, 'int32');
    return this.executeOpSingleOutput(
               'Conv2DBackpropInput', opAttrs, [inputSizes, filter, dy]) as
        Tensor4D;
  }

  conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', 'float32'),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'use_cudnn_on_gpu', type: this.binding.TF_ATTR_BOOL, value: true},
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];
    const filterSizes = tensor1d(convInfo.filterShape, 'int32');
    return this.executeOpSingleOutput(
               'Conv2DBackpropFilter', opAttrs, [x, filterSizes, dy]) as
        Tensor4D;
  }

  depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', 'float32'),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];

    const inputSizes = tensor1d(convInfo.inShape, 'int32');
    return this.executeOpSingleOutput(
               'DepthwiseConv2dNativeBackpropInput', opAttrs,
               [inputSizes, filter, dy]) as Tensor4D;
  }

  depthwiseConv2DDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', 'float32'),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];
    const filterSizes = tensor1d(convInfo.filterShape, 'int32');
    return this.executeOpSingleOutput(
               'DepthwiseConv2dNativeBackpropFilter', opAttrs,
               [x, filterSizes, dY]) as Tensor4D;
  }

  depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const dilations = [1, convInfo.dilationHeight, convInfo.dilationWidth, 1];
    const opAttrs = [
      this.createTypeOpAttr('T', input.dtype),
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'dilations', type: this.binding.TF_ATTR_INT, value: dilations}
    ];
    return this.executeOpSingleOutput(
               'DepthwiseConv2dNative', opAttrs, [input, filter]) as Tensor4D;
  }

  maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding}, {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      }
    ];
    return this.executeOpSingleOutput('MaxPool', opAttrs, [x]) as Tensor4D;
  }

  maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo):
      Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding type was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
    ];
    return this.executeOpSingleOutput('MaxPoolGrad', opAttrs, [x, y, dy]) as
        Tensor4D;
  }

  avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
    ];
    return this.executeOpSingleOutput('AvgPool', opAttrs, [x]) as Tensor4D;
  }

  avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D {
    if (convInfo.padInfo.type !== 'VALID' && convInfo.padInfo.type !== 'SAME') {
      throw new Error(
          `TF Backend supports only 'valid' and 'same' padding ` +
          `while padding type was ${convInfo.padInfo.type}`);
    }
    const ksize = [1, convInfo.filterHeight, convInfo.filterWidth, 1];
    const strides = [1, convInfo.strideHeight, convInfo.strideWidth, 1];
    const padding = convInfo.padInfo.type;
    const dataFormat = convInfo.dataFormat === 'channelsLast' ? 'NHWC' : 'NCHW';
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'ksize', type: this.binding.TF_ATTR_INT, value: ksize},
      {name: 'strides', type: this.binding.TF_ATTR_INT, value: strides},
      {name: 'padding', type: this.binding.TF_ATTR_STRING, value: padding},
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
    ];
    const origInputShape = tensor1d(x.shape, 'int32');
    return this.executeOpSingleOutput(
               'AvgPoolGrad', opAttrs, [origInputShape, dy]) as Tensor4D;
  }

  reshape<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]):
      Tensor<R> {
    const shapeTensor = tensor1d(shape, 'int32');

    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tshape', shapeTensor.dtype)
    ];
    return this.executeOpSingleOutput('Reshape', opAttrs, [x, shapeTensor]) as
        Tensor<R>;
  }

  cast<T extends Tensor>(x: T, dtype: DataType): T {
    const opAttrs = [
      this.createTypeOpAttr('SrcT', x.dtype),
      this.createTypeOpAttr('DstT', dtype)
    ];
    return this.executeOpSingleOutput('Cast', opAttrs, [x]) as T;
  }

  tile<T extends Tensor>(x: T, reps: number[]): T {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tmultiples', 'int32')
    ];
    const multiples = tensor1d(reps, 'int32');
    return this.executeOpSingleOutput('Tile', opAttrs, [x, multiples]) as T;
  }

  pad<T extends Tensor>(
      x: T, paddings: Array<[number, number]>, constantValue: number): T {
    // Bind tensor values
    const paddingsTensor = tensor2d(paddings, [paddings.length, 2], 'int32');
    const constantTensor = scalar(constantValue, x.dtype);

    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tpaddings', paddingsTensor.dtype)
    ];

    return this.executeOpSingleOutput(
               'PadV2', opAttrs, [x, paddingsTensor, constantTensor]) as T;
  }

  transpose<T extends Tensor>(x: T, perm: number[]): T {
    const permTensor = tensor1d(perm, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tperm', 'int32')
    ];
    return this.executeOpSingleOutput('Transpose', opAttrs, [x, permTensor]) as
        T;
  }

  gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T {
    const axisTensor = scalar(axis, 'int32');
    const opAttrs = [
      this.createTypeOpAttr('Tparams', x.dtype),
      this.createTypeOpAttr('Tindices', indices.dtype),
      this.createTypeOpAttr('Taxis', 'int32')
    ];
    return this.executeOpSingleOutput(
               'GatherV2', opAttrs, [x, indices, axisTensor]) as T;
  }

  resizeBilinear(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {
        name: 'align_corners',
        type: this.binding.TF_ATTR_BOOL,
        value: alignCorners
      },
    ];
    const size = tensor1d([newHeight, newWidth], 'int32');
    return this.executeOpSingleOutput('ResizeBilinear', opAttrs, [x, size]) as
        Tensor4D;
  }

  resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean):
      Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype), {
        name: 'align_corners',
        type: this.binding.TF_ATTR_BOOL,
        value: alignCorners
      }
    ];
    return this.executeOpSingleOutput('ResizeBilinearGrad', opAttrs, [dy, x]) as
        Tensor4D;
  }

  resizeNearestNeighbor(
      x: Tensor4D, newHeight: number, newWidth: number,
      alignCorners: boolean): Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {
        name: 'align_corners',
        type: this.binding.TF_ATTR_BOOL,
        value: alignCorners
      },
    ];
    const size = tensor1d([newHeight, newWidth], 'int32');
    return this.executeOpSingleOutput(
               'ResizeNearestNeighbor', opAttrs, [x, size]) as Tensor4D;
  }

  resizeNearestNeighborBackprop(
      dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype), {
        name: 'align_corners',
        type: this.binding.TF_ATTR_BOOL,
        value: alignCorners
      }
    ];
    const [, origHeight, origWidth, ] = x.shape;
    const size = tensor1d([origHeight, origWidth], 'int32');
    return this.executeOpSingleOutput(
               'ResizeNearestNeighborGrad', opAttrs, [dy, size]) as Tensor4D;
  }

  batchNormalization(
      x: Tensor4D, mean: Tensor1D|Tensor4D, variance: Tensor1D|Tensor4D,
      varianceEpsilon: number, scale?: Tensor1D|Tensor4D,
      offset?: Tensor1D|Tensor4D): Tensor4D {
    if (mean.rank > 1) {
      // Fused batch norm doesn't work with high-dim mean/var/scale/offset.
      let inv = rsqrt(variance.add(scalar(varianceEpsilon)));
      if (scale != null) {
        inv = inv.mul(scale);
      }
      const xNorm = x.sub(mean).mul(inv) as Tensor4D;
      return offset != null ? xNorm.add(offset) : xNorm;
    }
    const dataFormat = 'NHWC';
    const depth = x.shape[3];
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {
        name: 'epsilon',
        type: this.binding.TF_ATTR_FLOAT,
        value: varianceEpsilon
      },
      {
        name: 'data_format',
        type: this.binding.TF_ATTR_STRING,
        value: dataFormat
      },
      {name: 'is_training', type: this.binding.TF_ATTR_BOOL, value: false},
    ];
    const numOutputs = 5;
    if (scale == null) {
      scale = fill([depth], 1) as Tensor1D;
    }
    if (offset == null) {
      offset = fill([depth], 0) as Tensor1D;
    }
    return this.executeOpMultipleOutputs(
               'FusedBatchNorm', opAttrs, [x, scale, offset, mean, variance],
               numOutputs)[0] as Tensor4D;
  }

  localResponseNormalization4D(
      x: Tensor4D, radius: number, bias: number, alpha: number,
      beta: number): Tensor4D {
    const opAttrs = [
      this.createTypeOpAttr('T', x.dtype),
      {name: 'depth_radius', type: this.binding.TF_ATTR_INT, value: radius},
      {name: 'bias', type: this.binding.TF_ATTR_FLOAT, value: bias},
      {name: 'alpha', type: this.binding.TF_ATTR_FLOAT, value: alpha},
      {name: 'beta', type: this.binding.TF_ATTR_FLOAT, value: beta},
    ];
    return this.executeOpSingleOutput('LRN', opAttrs, [x]) as Tensor4D;
  }

  multinomial(
      logits: Tensor2D, normalized: boolean, numSamples: number,
      seed: number): Tensor2D {
    if (normalized) {
      throw new Error(
          'TF Node backend does not support normalized logits ' +
          'passed to multinomial');
    }
    const opAttrs = [
      this.createTypeOpAttr('T', logits.dtype),
      this.createTypeOpAttr('output_dtype', 'int32'),
      {name: 'seed', type: this.binding.TF_ATTR_INT, value: seed},
      {name: 'seed2', type: this.binding.TF_ATTR_INT, value: seed * seed},
    ];
    return this.executeOpSingleOutput(
               'Multinomial', opAttrs, [logits, scalar(numSamples, 'int32')]) as
        Tensor2D;
  }

  oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number):
      Tensor2D {
    const depthTensor = scalar(depth, 'int32');
    const onValueTensor = scalar(onValue, 'int32');
    const offValueTensor = scalar(offValue, 'int32');

    const opAttrs = [
      {name: 'axis', type: this.binding.TF_ATTR_INT, value: -1},
      this.createTypeOpAttr('T', indices.dtype),
      this.createTypeOpAttr('TI', indices.dtype)
    ];

    return this.executeOpSingleOutput('OneHot', opAttrs, [
      indices, depthTensor, onValueTensor, offValueTensor
    ]) as Tensor2D;
  }

  cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean):
      Tensor {
    const axisTensor = scalar(axis, 'int32');
    const opAttrs = [
      {name: 'exclusive', type: this.binding.TF_ATTR_BOOL, value: exclusive},
      {name: 'reverse', type: this.binding.TF_ATTR_BOOL, value: reverse},
      this.createTypeOpAttr('T', x.dtype),
      this.createTypeOpAttr('Tidx', 'int32')
    ];
    return this.executeOpSingleOutput('Cumsum', opAttrs, [x, axisTensor]);
  }

  fromPixels(
      pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
      numChannels: number): Tensor3D {
    if (pixels == null) {
      throw new Error('pixels passed to tf.fromPixels() can not be null');
    }
    // tslint:disable-next-line:no-any
    if ((pixels as any).getContext == null) {
      throw new Error(
          'When running in node, pixels must be an HTMLCanvasElement ' +
          'like the one returned by the `canvas` npm package');
    }
    const vals: Uint8ClampedArray =
        // tslint:disable-next-line:no-any
        (pixels as any)
            .getContext('2d')
            .getImageData(0, 0, pixels.width, pixels.height)
            .data;
    let values: Int32Array;
    if (numChannels === 4) {
      values = new Int32Array(vals);
    } else {
      const numPixels = pixels.width * pixels.height;
      values = new Int32Array(numPixels * numChannels);
      for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
          values[i * numChannels + channel] = vals[i * 4 + channel];
        }
      }
    }
    const outShape: [number, number, number] =
        [pixels.height, pixels.width, numChannels];
    return tensor3d(values, outShape, 'int32');
  }

  memory() {
    // Due to automatic garbage collection, the numbers are unreliable.
    // TODO: Since there is finalization in C, count the true
    // number of undisposed tensors.
    return {unreliable: true};
  }

  async time(f: () => void): Promise<BackendTimingInfo> {
    const start = process.hrtime();
    f();
    // hrtime() returns tuple of [seconds, nanoseconds], and we need to return
    // milliseconds.
    const elapsed = process.hrtime(start);
    return {kernelMs: elapsed[0] * 1000 + elapsed[1] / 1000000};
  }

  isNaN<T extends Tensor>(x: T): T {
    return this.executeOpSingleInput('IsNan', x) as T;
  }
}

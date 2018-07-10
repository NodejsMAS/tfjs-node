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

import * as tfc from '@tensorflow/tfjs-core';
import {BaseOpProgram} from './ops_base';

// import {DataType, Scalar, Tensor2D} from '@tensorflow/tfjs-core';
// import {TFEOpAttr, TFJSBinding} from '../tfjs_binding';

// export class ConcatV2 extends BaseThing {
//   constructor(
//       protected binding: TFJSBinding, protected a: Tensor2D,
//       protected b: Tensor2D, protected axis: Scalar) {
//     super(binding);

//     const opAttrs = [
//       {name: 'N', type: this.binding.TF_ATTR_INT, value: 2},
//       this.createTypeOpAttr('Tidx', 'int32'),
//       this.createTypeOpAttr('T', a.dtype)
//     ];
//   }
// }

export class ConcatV2OpProgram extends BaseOpProgram {
  constructor(values: tfc.Tensor, axis: tfc.Tensor) {
    super('ConcatV2');

    this.addInput(values);
    this.addInput(axis);

    // this.addOpAttr({name: 'N', type: this.backend.bind})
    // TODO add op attrs here?
  }
}

export function concat_v2(values: tfc.Tensor, axis: tfc.Tensor): tfc.Tensor {
  return null;
}

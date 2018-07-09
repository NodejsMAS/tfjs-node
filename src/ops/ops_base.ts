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

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';
import {TFEOpAttr} from '../tfjs_binding';

export abstract class BaseOpProgram {
  protected opAttrs: TFEOpAttr[];
  protected inputs: tfc.Tensor[];
  protected outputSize: number;
  protected backend: NodeJSKernelBackend;

  constructor(protected opName: string) {
    // Only load the binding once and at least reference the backend for now.
    this.backend = (tfc.ENV.findBackend('tensorflow') as NodeJSKernelBackend);
    this.outputSize = 0;
  }

  addInput(t: tfc.Tensor): void {
    this.inputs.push(t);
  }

  addOutput(t: tfc.Tensor): void {
    this.outputSize++;
  }

  addOpAttr(opAttr: TFEOpAttr): void {
    this.opAttrs.push(opAttr);
  }

  execute(): tfc.Tensor|tfc.Tensor[] {
    if (this.outputSize === 1) {
      return this.backend.executeOpSingleOutput(
          this.opName, this.opAttrs, this.inputs);
    } else {
      return this.backend.executeOpMultipleOutputs(
          this.opName, this.opAttrs, this.inputs, this.outputSize);
    }
  }
}

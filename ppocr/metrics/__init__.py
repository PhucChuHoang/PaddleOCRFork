# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ["build_metric"]

from .det_metric import DetMetric, DetFCEMetric
from .rec_metric import RecMetric, CNTMetric, CANMetric, LaTeXOCRMetric, RecMetricTopK
from .cls_metric import ClsMetric
from .e2e_metric import E2EMetric
from .distillation_metric import DistillationMetric
from .table_metric import TableMetric
from .kie_metric import KIEMetric
from .vqa_token_ser_metric import VQASerTokenMetric
from .vqa_token_re_metric import VQAReTokenMetric
from .sr_metric import SRMetric
from .ct_metric import CTMetric


def build_metric(config):
    support_dict = [
        "DetMetric",
        "DetFCEMetric",
        "RecMetric",
        "RecMetricTopK",
        "ClsMetric",
        "E2EMetric",
        "DistillationMetric",
        "TableMetric",
        "KIEMetric",
        "VQASerTokenMetric",
        "VQAReTokenMetric",
        "SRMetric",
        "CTMetric",
        "CNTMetric",
        "CANMetric",
        "LaTeXOCRMetric",
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class

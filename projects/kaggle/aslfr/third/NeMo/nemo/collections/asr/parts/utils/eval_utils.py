# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Tuple

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.utils import logging


def clean_label(_str: str, num_to_words: bool = True, langid="en") -> str:
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    """
    replace_with_space = [char for char in '/?*\",.:=?_{|}~¨«·»¡¿„…‧‹›≪≫!:;ː→']
    replace_with_blank = [char for char in '`¨´‘’“”`ʻ‘’“"‘”']
    replace_with_apos = [char for char in '‘’ʻ‘’‘']
    _str = _str.strip()
    _str = _str.lower()
    for i in replace_with_blank:
        _str = _str.replace(i, "")
    for i in replace_with_space:
        _str = _str.replace(i, " ")
    for i in replace_with_apos:
        _str = _str.replace(i, "'")
    if num_to_words:
        if langid == "en":
            _str = convert_num_to_words(_str, langid="en")
        else:
            logging.info(
                "Currently support basic num_to_words in English only. Please use Text Normalization to convert other languages! Skipping!"
            )

    ret = " ".join(_str.split())
    return ret


def convert_num_to_words(_str: str, langid: str = "en") -> str:
    """
    Convert digits to corresponding words. Note this is a naive approach and could be replaced with text normalization.
    """
    if langid == "en":
        num_to_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        _str = _str.strip()
        words = _str.split()
        out_str = ""
        num_word = []
        for word in words:
            if word.isdigit():
                num = int(word)
                while num:
                    digit = num % 10
                    digit_word = num_to_words[digit]
                    num_word.append(digit_word)
                    num = int(num / 10)
                    if not (num):
                        num_str = ""
                        num_word = num_word[::-1]
                        for ele in num_word:
                            num_str += ele + " "
                        out_str += num_str + " "
                        num_word.clear()
            else:
                out_str += word + " "
        out_str = out_str.strip()
    else:
        raise ValueError(
            "Currently support basic num_to_words in English only. Please use Text Normalization to convert other languages!"
        )
    return out_str


def cal_write_wer(
    pred_manifest: str = None,
    pred_text_attr_name: str = "pred_text",
    clean_groundtruth_text: bool = False,
    langid: str = 'en',
    use_cer: bool = False,
    output_filename: str = None,
) -> Tuple[str, dict, str]:
    """ 
    Calculate wer, inserion, deletion and substitution rate based on groundtruth text and pred_text_attr_name (pred_text) 
    We use WER in function name as a convention, but Error Rate (ER) currently support Word Error Rate (WER) and Character Error Rate (CER)
    """
    samples = []
    hyps = []
    refs = []
    eval_metric = "cer" if use_cer else "wer"

    with open(pred_manifest, 'r') as fp:
        for line in fp:
            sample = json.loads(line)

            if 'text' not in sample:
                logging.info(
                    "ground-truth text is not present in manifest! Cannot calculate Word Error Rate. Returning!"
                )
                return None, None, eval_metric

            hyp = sample[pred_text_attr_name]
            ref = sample['text']

            if clean_groundtruth_text:
                ref = clean_label(ref, langid=langid)

            wer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
                hypotheses=[hyp], references=[ref], use_cer=use_cer
            )
            sample[eval_metric] = wer  # evaluatin metric, could be word error rate of character error rate
            sample['tokens'] = tokens  # number of word/characters/tokens
            sample['ins_rate'] = ins_rate  # insertion error rate
            sample['del_rate'] = del_rate  # deletion error rate
            sample['sub_rate'] = sub_rate  # substitution error rate

            samples.append(sample)
            hyps.append(hyp)
            refs.append(ref)

    total_wer, total_tokens, total_ins_rate, total_del_rate, total_sub_rate = word_error_rate_detail(
        hypotheses=hyps, references=refs, use_cer=use_cer
    )

    if not output_filename:
        output_manifest_w_wer = pred_manifest
    else:
        output_manifest_w_wer = output_filename

    with open(output_manifest_w_wer, 'w') as fout:
        for sample in samples:
            json.dump(sample, fout)
            fout.write('\n')
            fout.flush()

    total_res = {
        "samples": len(samples),
        "tokens": total_tokens,
        eval_metric: total_wer,
        "ins_rate": total_ins_rate,
        "del_rate": total_del_rate,
        "sub_rate": total_sub_rate,
    }
    return output_manifest_w_wer, total_res, eval_metric

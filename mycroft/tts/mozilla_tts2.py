# Copyright 2020 Mycroft AI Inc.
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
#

import requests

from .tts import TTS, TTSValidator
from mycroft.configuration import Configuration


import base64
import json
import math
import os
import re
from urllib import parse

from requests_futures.sessions import FuturesSession
from requests.exceptions import (
    ReadTimeout, ConnectionError, ConnectTimeout, HTTPError
)

from mycroft.util.file_utils import get_cache_directory
from mycroft.util.log import LOG
from .mimic_tts import VISIMES
from .remote_tts import RemoteTTSException, RemoteTTSTimeoutException

# Heuristic value, caps character length of a chunk of text to be spoken as a
# work around for current Mimic2 implementation limits.
_max_sentence_size = 170


def _break_chunks(l, n):
    """Yield successive n-sized chunks

    Args:
        l (list): text (str) to split
        chunk_size (int): chunk size
    """
    for i in range(0, len(l), n):
        yield " ".join(l[i:i + n])


def _split_by_chunk_size(text, chunk_size):
    """Split text into word chunks by chunk_size size

    Args:
        text (str): text to split
        chunk_size (int): chunk size

    Returns:
        list: list of text chunks
    """
    text_list = text.split()

    if len(text_list) <= chunk_size:
        return [text]

    if chunk_size < len(text_list) < (chunk_size * 2):
        return list(_break_chunks(
            text_list,
            int(math.ceil(len(text_list) / 2))
        ))
    elif (chunk_size * 2) < len(text_list) < (chunk_size * 3):
        return list(_break_chunks(
            text_list,
            int(math.ceil(len(text_list) / 3))
        ))
    elif (chunk_size * 3) < len(text_list) < (chunk_size * 4):
        return list(_break_chunks(
            text_list,
            int(math.ceil(len(text_list) / 4))
        ))
    else:
        return list(_break_chunks(
            text_list,
            int(math.ceil(len(text_list) / 5))
        ))


def _split_by_punctuation(chunks, puncs):
    """Splits text by various punctionations
    e.g. hello, world => [hello, world]

    Args:
        chunks (list or str): text (str) to split
        puncs (list): list of punctuations used to split text

    Returns:
        list: list with split text
    """
    if isinstance(chunks, str):
        out = [chunks]
    else:
        out = chunks

    for punc in puncs:
        splits = []
        for t in out:
            # Split text by punctuation, but not embedded punctuation.  E.g.
            # Split:  "Short sentence.  Longer sentence."
            # But not at: "I.B.M." or "3.424", "3,424" or "what's-his-name."
            splits += re.split(r'(?<!\.\S)' + punc + r'\s', t)
        out = splits
    return [t.strip() for t in out]


def _add_punctuation(text):
    """Add punctuation at the end of each chunk.

    Mimic2 expects some form of punctuation at the end of a sentence.
    """
    punctuation = ['.', '?', '!', ';']
    if len(text) >= 1 and text[-1] not in punctuation:
        return text + '.'
    else:
        return text


def _sentence_chunker(text):
    """Split text into smaller chunks for TTS generation.

    NOTE: The smaller chunks are needed due to current Mimic2 TTS limitations.
    This stage can be removed once Mimic2 can generate longer sentences.

    Args:
        text (str): text to split
        chunk_size (int): size of each chunk
        split_by_punc (bool, optional): Defaults to True.

    Returns:
        list: list of text chunks
    """
    if len(text) <= _max_sentence_size:
        return [_add_punctuation(text)]

    # first split by punctuations that are major pauses
    first_splits = _split_by_punctuation(
        text,
        puncs=[r'\.', r'\!', r'\?', r'\:', r'\;']
    )

    # if chunks are too big, split by minor pauses (comma, hyphen)
    second_splits = []
    for chunk in first_splits:
        if len(chunk) > _max_sentence_size:
            second_splits += _split_by_punctuation(chunk,
                                                   puncs=[r'\,', '--', '-'])
        else:
            second_splits.append(chunk)

    # if chunks are still too big, chop into pieces of at most 20 words
    third_splits = []
    for chunk in second_splits:
        if len(chunk) > _max_sentence_size:
            third_splits += _split_by_chunk_size(chunk, 20)
        else:
            third_splits.append(chunk)

    return [_add_punctuation(chunk) for chunk in third_splits]


# ================================================================== Mozilla

class MozillaTTS2(TTS):
    def __init__(self, lang="en-us", config=None):
        if config is None:
            self.config = Configuration.get().get("tts", {}).get("mozilla2", {})
        else:
            self.config = config
        super(MozillaTTS2, self).__init__(lang, self.config,
                                         MozillaTTSValidator(self))
        self.url = self.config['url'] + "/api/tts"
        self.type = 'wav'

    def _preprocess_sentence(self, sentence):
        """Split sentence in chunks better suited for mimic2. """
        return _sentence_chunker(sentence)
        
    def get_tts(self, sentence, wav_file):
        response = requests.get(self.url, params={'text': sentence})
        with open(wav_file, 'wb') as f:
            f.write(response.content)
        return (wav_file, None)  # No phonemes


class MozillaTTSValidator(TTSValidator):
    def __init__(self, tts):
        super(MozillaTTSValidator, self).__init__(tts)

    def validate_dependencies(self):
        pass

    def validate_lang(self):
        # TODO
        pass

    def validate_connection(self):
        url = self.tts.config['url']
        response = requests.get(url)
        if not response.status_code == 200:
            raise ConnectionRefusedError

    def get_tts_class(self):
        return MozillaTTS2

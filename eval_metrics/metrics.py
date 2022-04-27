# MIT License
#
# Copyright (c) 2019 Richard Csaky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
import numpy as np
import copy
from scipy.spatial import distance
from nltk.translate import bleu_score, nist_score, meteor_score
from collections import Counter, defaultdict
import six
import torch
import os
import sys
import pandas as pd
import time
from eval_metrics.util import (
    get_model,
    get_tokenizer,
    get_idf_dict,
    bert_cos_score_idf,
    lang2model,
    model2layers,
    get_hash,
)



def score(
    cands,
    refs,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False
):
    """
    BERTScore metric.
    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have
                  multiple references, the returned score of this candidate is
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert lang is not None or model_type is not None, "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv")
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
            else:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}", file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

    if return_hash:
        return tuple(
            [
                out,
                get_hash(model_type, num_layers, idf, rescale_with_baseline,
                         use_custom_baseline=use_custom_baseline,
                         use_fast_tokenizer=use_fast_tokenizer),
            ]
        )

    return out


class BertScoreMetrics():
    def __init__(self):
        self.metrics = {'Bert-P': [], 'Bert-R': [], 'Bert-F1': []}

    def calculate_metrics(self, refs, reply):
        with open(refs, 'r') as f:
            with open(refs, 'r') as f:
                gold = f.readlines()
                gold = [g.strip() for g in gold]
            with open(reply, 'r') as f:
                test = f.readlines()
                test = [t.strip() for t in test]
        P, R, F1 = score(test, gold, lang='en', rescale_with_baseline=True)
        self.metrics['Bert-P'].append(P.mean())
        self.metrics['Bert-R'].append(R.mean())
        self.metrics['Bert-F1'].append(F1.mean())

    def update_metrics(self, resp, gt, source):
        pass

# Converts frequency dict to probabilities.
def build_distro(distro, path, vocab=None, probs=False, tri_quad=False):
    with open(path, encoding='utf-8') as file:
        for line in file:
            words = line.split()
            word_count = len(words)
            for i, word in enumerate(words):
                if vocab:
                    word = word if vocab.get(word) else '<unk>'
                w_in_dict = distro['uni'].get(word)
                distro['uni'][word] = distro['uni'][word] + 1 if w_in_dict else 1

                # Bigrams.
                if i < word_count - 1:
                    if vocab:
                        word2 = words[i + 1] if vocab.get(words[i + 1]) else '<unk>'
                    bi = (word, word2)
                    bigram_in_dict = distro['bi'].get(bi)
                    distro['bi'][bi] = distro['bi'][bi] + 1 if bigram_in_dict else 1
                if tri_quad:
                    # Triple.
                    if i < word_count - 2:
                        if vocab:
                            word3 = words[i + 2] if vocab.get(words[i + 2]) else '<unk>'
                        tri = (word, word2, word3)
                        trigram_in_dict = distro['tri'].get(tri)
                        distro['tri'][tri] = distro['tri'][tri] + 1 if trigram_in_dict else 1

                    # Triple.
                    if i < word_count - 3:
                        if vocab:
                            word4 = words[i + 3] if vocab.get(words[i + 3]) else '<unk>'
                        quad = (word, word2, word3, word4)
                        quadgram_in_dict = distro['quad'].get(quad)
                        distro['quad'][quad] = distro['quad'][quad] + 1 if quadgram_in_dict else 1

    if probs:
        distro['uni'] = convert_to_probs(distro['uni'])
        distro['bi'] = convert_to_probs(distro['bi'])
        if tri_quad:
            distro['tri'] = convert_to_probs(distro['tri'])
            distro['quad'] = convert_to_probs(distro['quad'])

# Go through a file and build word and bigram frequencies.
def convert_to_probs(freq_dict):
    num_words = sum(list(freq_dict.values()))
    return dict([(key, val / num_words) for key, val in freq_dict.items()])


# https://www.aclweb.org/anthology/P02-1040
class BleuMetrics():
    def __init__(self):
        self.metrics = {'bleu-1': [], 'bleu-2': [], 'bleu-3': [], 'bleu-4': []}

    def calculate_metrics(self, refs, reply):
        with open(refs, 'r') as f:
            gold = f.readlines()
            gold = [[g.strip().split()] for g in gold]
        with open(reply, 'r') as f:
            test = f.readlines()
            test = [t.strip().split() for t in test]
        self.metrics['bleu-1'].append(bleu_score.corpus_bleu(gold, test, weights=[1, 0, 0, 0]) * 100)
        self.metrics['bleu-2'].append(bleu_score.corpus_bleu(gold, test, weights=[0.5, 0.5, 0, 0]) * 100)
        self.metrics['bleu-3'].append(bleu_score.corpus_bleu(gold, test, weights=[0.33, 0.33, 0.33, 0]) * 100)
        self.metrics['bleu-4'].append(bleu_score.corpus_bleu(gold, test, weights=[0.25, 0.25, 0.25, 0.25]) * 100)
    # Calculate evaluate for one example.
    def update_metrics(self, resp, gt, source):
        pass

class MeteorMetrics():
    def __init__(self):
        self.metrics = {'meteor': []}

    def calculate_metrics(self, refs, reply):
        with open(refs, 'r') as f:
            with open(refs, 'r') as f:
                gold = f.readlines()
                gold = [g.strip() for g in gold]
            with open(reply, 'r') as f:
                test = f.readlines()
                test = [t.strip() for t in test]
        for g, t in zip(gold, test):
            self.metrics['meteor'].append(meteor_score.meteor_score(g, t))
        self.metrics['meteor'] = [sum(self.metrics['meteor']) / len(self.metrics['meteor'])]

    def update_metrics(self, resp, gt, source):
        pass

class NistMetrics():
    def __init__(self):
        self.metrics = {'nist': []}

    def calculate_metrics(self, refs, reply):
        with open(refs, 'r') as f:
            with open(refs, 'r') as f:
                gold = f.readlines()
                gold = [[g.strip().split()] for g in gold]
            with open(reply, 'r') as f:
                test = f.readlines()
                test = [t.strip().split() for t in test]
        self.metrics['nist'].append(nist_score.corpus_nist(gold, test))

    def update_metrics(self, resp, gt, source):
        pass

# https://www.aclweb.org/anthology/N16-1014
class DistinctMetrics():
    def __init__(self, vocab):
        '''
        Params:
          :vocab: Vocabulary dictionary.
        '''
        self.vocab = vocab
        self.metrics = {'distinct-1': [],
                        'distinct-2': [],
                        'distinct-3': [],
                        'distinct-4': []}

    # Calculate the distinct value for a distribution.
    def distinct(self, distro):
        return len(distro) / sum(list(distro.values()))

    # Calculate distinct evaluate for a given file.
    def calculate_metrics(self, filename):
        test_distro = {'uni': {}, 'bi': {}, 'tri': {}, 'quad':{}}
        build_distro(test_distro, filename, self.vocab, tri_quad=True)

        self.metrics['distinct-1'].append(round(self.distinct(test_distro['uni']) * 100, 2))
        self.metrics['distinct-2'].append(round(self.distinct(test_distro['bi']) * 100, 2))
        self.metrics['distinct-3'].append(round(self.distinct(test_distro['tri']) * 100, 2))
        self.metrics['distinct-4'].append(round(self.distinct(test_distro['quad']) * 100, 2))

    def update_metrics(self, a, s, d):
        pass

# https://arxiv.org/abs/1905.05471
class DivergenceMetrics():
    def __init__(self, vocab, gt_path):
        '''
        Params:
          :vocab: Vocabulary dictionary.
          :gt_path: Path to ground truth file.
        '''
        self.vocab = vocab
        self.gt_path = gt_path

        self.metrics = {'unigram-kl-div': [],
                        'bigram-kl-div': []}

    # Calculate kl divergence between between two distributions for a sentence.
    def update_metrics(self, resp, gt_words, source):
        '''
        Params:
          :resp_words: Response word list.
          :gt_words: Ground truth word list.
          :source_words: Source word list.
        '''
        uni_div = []
        bi_div = []
        word_count = len(gt_words)

        for i, word in enumerate(gt_words):
            if self.uni_distros['model'].get(word):
                word = word if self.vocab.get(word) else '<unk>'
                uni_div.append(math.log(self.uni_distros['gt'][word] /
                                        self.uni_distros['model'][word], 2))

            if i < word_count - 1:
                word2 = gt_words[i + 1] if self.vocab.get(gt_words[i + 1]) else '<unk>'
                bigram = (word, word2)
                if self.bi_distros['model'].get(bigram):
                    bi_div.append(math.log(self.bi_distros['gt'][bigram] /
                                           self.bi_distros['model'][bigram], 2))

        # Exclude divide by zero errors.
        if uni_div:
            self.metrics['unigram-kl-div'].append(sum(uni_div) / len(uni_div))
        if bi_div:
            self.metrics['bigram-kl-div'].append(sum(bi_div) / len(bi_div))

    # Get the distributions for test and ground truth data.
    def setup(self, filename):
        '''
        Params:
          :filename: Path to test responses.
        '''
        self.test_distro = {'uni': {}, 'bi': {}}
        self.gt_distro = {'uni': {}, 'bi': {}}
        build_distro(self.test_distro, filename, self.vocab)
        build_distro(self.gt_distro, self.gt_path, self.vocab)

        # Only keep intersection of test and ground truth distros.
        test, true = self.filter_distros(self.test_distro['uni'],
                                         self.gt_distro['uni'])
        self.uni_distros = {'model': test, 'gt': true}
        test, true = self.filter_distros(self.test_distro['bi'],
                                         self.gt_distro['bi'])
        self.bi_distros = {'model': test, 'gt': true}

    # Filter test and ground truth distributions, only keep intersection.
    def filter_distros(self, test, true):
        '''
        Params:
          :test: Test distribution.
          :true: Ground truth distribution.
        '''
        intersection = set.intersection(set(test.keys()), set(true.keys()))

        def probability_distro(distro):
            distro = dict(distro)
            for key in list(distro.keys()):
                if key not in intersection:
                    del distro[key]
            return convert_to_probs(distro)

        test = probability_distro(test)
        true = probability_distro(true)
        return test, true


# https://arxiv.org/pdf/1809.06873.pdf
class EmbeddingMetrics():
    def __init__(self, vocab, distro, emb_dim, average=True):
        '''
        Params:
          :vocab: Vocabulary dictionary.
          :ditro: Train distribution.
          :emb_dim: Embedding dimension for word vectors.
          :average: Whether embedding-average should be computed.
        '''
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.distro = distro
        self.average = average

        self.metrics = {'embedding-average': [],
                        'embedding-extrema': [],
                        'embedding-greedy': []}

    # Calculate embedding evaluate.
    def update_metrics(self, resp_words, gt_words, source_words):
        '''
        Params:
          :resp_words: Response word list.
          :gt_words: Ground truth word list.
          :source_words: Source word list.
        '''
        if self.average:
            avg_resp = self.avg_embedding(resp_words)
            avg_gt = self.avg_embedding(gt_words)

            # Check for zero vectors and compute cosine similarity.
            if np.count_nonzero(avg_resp) and np.count_nonzero(avg_gt):
                self.metrics['embedding-average'].append(
                    1 - distance.cosine(avg_gt, avg_resp))

        # Compute extrema embedding metric.
        extrema_resp = self.extrema_embedding(resp_words)
        extrema_gt = self.extrema_embedding(gt_words)
        if np.count_nonzero(extrema_resp) and np.count_nonzero(extrema_gt):
            self.metrics['embedding-extrema'].append(
                1 - distance.cosine(extrema_resp, extrema_gt))

        # Compute greedy embedding metric.
        one_side = self.greedy_embedding(gt_words, resp_words)
        other_side = self.greedy_embedding(resp_words, gt_words)

        if one_side and other_side:
            self.metrics['embedding-greedy'].append((one_side + other_side) / 2)

    # Calculate the average word embedding of a sentence.
    def avg_embedding(self, words):
        vectors = []
        for word in words:
            vector = self.vocab.get(word)
            prob = self.distro.get(word)
            if vector:
                if prob:
                    vectors.append(vector[0] * 0.001 / (0.001 + prob))
                else:
                    vectors.append(vector[0] * 0.001 / (0.001 + 0))

        if vectors:
            return np.sum(np.array(vectors), axis=0) / len(vectors)
        else:
            return np.zeros(self.emb_dim)

    # Calculate the extrema embedding of a sentence.
    def extrema_embedding(self, words):
        vector = np.zeros(self.emb_dim)
        for word in words:
            vec = self.vocab.get(word)
            if vec:
                for i in range(self.emb_dim):
                    if abs(vec[0][i]) > abs(vector[i]):
                        vector[i] = vec[0][i]
        return vector

    # Calculate the greedy embedding from one side.
    def greedy_embedding(self, words1, words2):
        y_vec = np.zeros((self.emb_dim, 1))
        x_count = 0
        y_count = 0
        cos_sim = 0
        for word in words2:
            vec = self.vocab.get(word)
            if vec:
                norm = np.linalg.norm(vec[0])
                vector = vec[0] / norm if norm else vec[0]
                y_vec = np.hstack((y_vec, (vector.reshape((self.emb_dim, 1)))))
                y_count += 1

        for word in words1:
            vec = self.vocab.get(word)
            if vec:
                norm = np.linalg.norm(vec[0])
                if norm:
                    cos_sim += np.max(
                        (vec[0] / norm).reshape((1, self.emb_dim)).dot(y_vec))
                    x_count += 1

        if x_count > 0 and y_count > 0:
            return cos_sim / x_count


# https://arxiv.org/pdf/1809.06873.pdf
class CoherenceMetrics(EmbeddingMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {'coherence': []}

    # Calculate coherence for one example.
    def update_metrics(self, resp_words, gt_words, source_words):
        '''
        Params:
          :resp_words: Response word list.
          :gt_words: Ground truth word list.
          :source_words: Source word list.
        '''
        avg_source = self.avg_embedding(source_words)
        avg_resp = self.avg_embedding(resp_words)

        # Check for zero vectors and compute cosine similarity.
        if np.count_nonzero(avg_resp) and np.count_nonzero(avg_source):
            self.metrics['coherence'].append(
                1 - distance.cosine(avg_source, avg_resp))

# http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf
class EntropyMetrics():
    def __init__(self, vocab, distro):
        '''
        Params:
          :vocab: Vocabulary dictionary.
          :ditro: Train distribution.
        '''
        self.vocab = vocab
        self.distro = distro

        self.metrics = {'per-unigram-entropy': [],
                        'per-bigram-entropy': [],
                        'utterance-unigram-entropy': [],
                        'utterance-bigram-entropy': []}

    # Update evaluate for one example.
    def update_metrics(self, resp_words, gt_words, source_words):
        '''
        Params:
          :resp_words: Response word list.
          :gt_words: Ground truth word list.
          :source_words: Source word list.
        '''
        uni_entropy = []
        bi_entropy = []
        word_count = len(resp_words)
        for i, word in enumerate(resp_words):
            # Calculate unigram entropy.
            word = word if self.vocab.get(word) else '<unk>'
            probability = self.distro['uni'].get(word)
            if probability:
                uni_entropy.append(math.log(probability, 2))

            # Calculate bigram entropy.
            if i < word_count - 1:
                w = resp_words[i + 1] if self.vocab.get(resp_words[i + 1]) else '<unk>'
                probability = self.distro['bi'].get((word, w))
                if probability:
                    bi_entropy.append(math.log(probability, 2))

        # Check if lists are empty.
        if uni_entropy:
            entropy = -sum(uni_entropy)
            self.metrics['per-unigram-entropy'].append(entropy / len(uni_entropy))
            self.metrics['utterance-unigram-entropy'].append(entropy)
        if bi_entropy:
            entropy = -sum(bi_entropy)
            self.metrics['per-bigram-entropy'].append(entropy / len(bi_entropy))
            self.metrics['utterance-bigram-entropy'].append(entropy)

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)


class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in six.iteritems(ref)]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in six.iteritems(cnts):
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in six.iteritems(vec_hyp[n]):
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)

class CIDErMetrics():
    def __init__(self):
        self._n = 4
        self._sigma = 6.0
        self.metrics = {'CIDEr': []}

    def calculate_metrics(self, refs, reply):
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        with open(refs, 'r') as f:
            with open(refs, 'r') as f:
                gold = f.readlines()
                gold = [[g.strip()] for g in gold]
            with open(reply, 'r') as f:
                test = f.readlines()
                test = [[t.strip()] for t in test]

        for t, g in zip(test, gold):
            cider_scorer += (t[0], g)

        (score, scores) = cider_scorer.compute_score()
        self.metrics['CIDEr'].append(score)

    def update_metrics(self, a, s, d):
        pass


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        merged_ref_ngram_counts = Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches
    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
            else:
                precisions[i] = 0.0
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0
    ratio = max(float(translation_length) / reference_length, 1e-10)
    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return bleu, precisions, bp, ratio, translation_length, reference_length


def calculate_cumulative_ngram_bleu(refs, hypos):
    gram1 = bleu_score.corpus_bleu(refs, hypos, weights=(1, 0, 0, 0), auto_reweigh=True)
    gram2 = bleu_score.corpus_bleu(refs, hypos, weights=(0.5, 0.5, 0, 0), auto_reweigh=True)
    gram3 = bleu_score.corpus_bleu(refs, hypos, weights=(0.3333, 0.3333, 0.3333, 0), auto_reweigh=True)
    gram4 = bleu_score.corpus_bleu(refs, hypos, weights=(0.25, 0.25, 0.25, 0.25), auto_reweigh=True)
    # return [100 * g for g in [gram1, gram2, gram3, gram4]]
    return [gram1, gram2, gram3, gram4]
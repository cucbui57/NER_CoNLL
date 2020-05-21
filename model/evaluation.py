from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from functools import reduce
from tqdm import tqdm


class Evaluator(object):
    """
    A class to evaluate a dataset on multiple
    metrics. Uses generator tricks to pass batch predictions
    to all the metric functions without recomputing
    """

    def __init__(self, name_dataset_evaluator, data_iter, *metrics):
        """
        Parameters:
            data_iter: Dataset iterator derived from torchtext.data.Iterator
            metric_fns: List of metric objects extending the Metric class
        """
        self.name_dataset_evaluator = name_dataset_evaluator
        self.data_iter = data_iter
        self.metrics = metrics or []

    def evaluate(self, model):
        """
        Evaluates the model on the given metrics for a dataset.
        Returns: A dict containing all the metrics
        """

        file_out = open("../module_evaluate/evaluate.txt", 'w')
        self.data_iter.init_epoch()
        model.eval()
        for m in self.metrics:
            m.reset()

        total_loss = 0

        prog_iter = tqdm(self.data_iter, leave=False)
        tag_vocab = model.vocabs[2]
        word_vocab = model.vocabs[0]
        # file_out.write(model.vocabs[2])
        with torch.no_grad():
            for batch in prog_iter:
                loss, predictions = model.loss(batch, compute_predictions=True)

                # list_word = []
                #
                # for sentence in batch.input_word:
                #     tmp = sentence.tolist()
                #     for word in tmp:
                #         print(type(word))
                #         print(word)
                #         list_word.append(word_vocab.itos[word[0]])
                # print(list_word)

                for item in predictions:
                    # print(item)
                    # print(tag_vocab.itos[item.tolist()])
                    for i in item.tolist():
                        file_out.write(tag_vocab.itos[i] + '\t')
                file_out.write('\n')
                for m in self.metrics:
                    m.evaluate(batch, loss, predictions)
                total_loss += float(loss)

                prog_iter.set_description('Evaluating data {}'.format(self.name_dataset_evaluator))

            results = {'loss': total_loss / len(self.data_iter)}
            for m in self.metrics:
                r = m.results(total_loss)
                if not isinstance(r, dict):
                    raise ValueError(
                        '{}.results() should return a dict containing metrics'.format(m.__class__.__name__))
                results.update(r)

        return results

    def evaluate_multi_task(self, model):
        model.eval()
        self.data_iter.init_epoch()

        list_metrics = self.metrics[0]
        for e_metrics in list_metrics:
            for m in e_metrics:
                m.reset()

        total_loss_combine = 0
        list_total_loss_each_task = [0.0] * model.number_task

        prog_iter = tqdm(self.data_iter, leave=False)

        # TODO we have bug in evaluation model
        with torch.no_grad():
            for batch in prog_iter:
                loss_combine, list_loss_full_tasks, list_predictions_full_tasks = model.loss(batch)

                # get result metric of each task
                for id_x, e_metrics in enumerate(list_metrics):
                    for m in e_metrics:
                        batch_data = getattr(batch, model.list_target_name[id_x])
                        loss_each_task = list_loss_full_tasks[id_x]
                        predictions_each_task = list_predictions_full_tasks[id_x]

                        m.evaluate_each_labels(batch_data, loss_each_task, predictions_each_task)

                list_total_loss_each_task[id_x] += float(list_loss_full_tasks[id_x])

                total_loss_combine += float(loss_combine)

                prog_iter.set_description('Evaluating data {}'.format(self.name_dataset_evaluator))

            list_results = [{'loss': (list_total_loss_each_task[id_x] / len(self.data_iter))}
                            for id_x in range(len(list_metrics))]

            for id_x, e_metrics in enumerate(list_metrics):
                for m in e_metrics:
                    r = m.results(list_total_loss_each_task[id_x])
                    if not isinstance(r, dict):
                        raise ValueError(
                            '{}.results() should return a dict containing metrics'.format(m.__class__.__name__))
                    list_results[id_x].update(r)

        return loss_combine, list_results


class Metrics(object):
    """
    Base metrics class that Evaluator calls to evaluate a metric on a dataset.
    Derived classes should implement the reset(), evaluate() and results() methods
    """

    def reset(self):
        """
        Called to reset counters, accumulators etc
        """
        raise NotImplementedError('{} must implement reset()'.format(self.__class__.__name__))

    def evaluate_each_labels(self, e_labels, loss, prediction):
        """
        Called to evaluate metrics for one batch
        Parameters:
            batch: A single minibatch. Instance of torchtext.data.Batch
            loss: Loss Tensor
            prediction: Model prediction
        """
        raise NotImplementedError('{} must implement evaluate()'.format(self.__class__.__name__))

    def evaluate(self, batch, loss, prediction):
        """
        Called to evaluate metrics for one batch
        Parameters:
            batch: A single minibatch. Instance of torchtext.data.Batch
            loss: Loss Tensor
            prediction: Model prediction
        """
        raise NotImplementedError('{} must implement evaluate()'.format(self.__class__.__name__))

    def results(self, total_loss):
        """
        Called to retrieve the final metrics.
        Parameters:
            total_loss: Total loss on the dataset
        Return:
            Must return a dict mapping metric names to values
        """


class BasicMetrics(Metrics):
    """
    A basic metrics class that computes symbol
    and sequence accuracy. It is meant to be passed to the Evaluator
    """

    def __init__(self, output_vocab, ignore_symbols=['<unk>', '<pad>']):
        """
        Parameters:
            output_vocab: Instance of torchtext.Vocab
            ignore_symbols: List of symbols to ignore while calculating symbol accuracy
                            e.g. padding, unknown etc
        """
        self.ignore_ids = [output_vocab.stoi[s] for s in (ignore_symbols or [])]

    def reset(self):
        self.total_correct_examples = 0
        self.total_correct_symbols = 0
        self.total_symbols = 0
        self.total_examples = 0

    def evaluate_each_labels(self, e_labels, loss, predictions):
        eq = torch.eq(predictions, e_labels).type_as(loss)
        self.total_correct_examples += int(eq.prod(1).sum())
        self.total_examples += e_labels.shape[0]

        # Mark elements to discard as 1
        discard = reduce(lambda x, y: x + y,
                         map(lambda ignore: ignore == e_labels, self.ignore_ids))
        # Elements which stayed zero are the ones we want
        mask = (discard == 0).float()

        # Mask correct predictions to remove ignored ones
        self.total_correct_symbols += int((eq * mask).sum())
        # Also track total symbols
        self.total_symbols += int(mask.sum())

    def evaluate(self, batch, loss, predictions):
        eq = torch.eq(predictions, batch.label).type_as(loss)
        self.total_correct_examples += int(eq.prod(1).sum())
        self.total_examples += batch.label.shape[0]

        # Mark elements to discard as 1
        discard = reduce(lambda x, y: x + y,
                         map(lambda ignore: ignore == batch.label, self.ignore_ids))
        # Elements which stayed zero are the ones we want
        mask = (discard == 0).float()

        # Mask correct predictions to remove ignored ones
        self.total_correct_symbols += int((eq * mask).sum())
        # Also track total symbols
        self.total_symbols += int(mask.sum())

    def results(self, total_loss):
        """
        Returns: A dict containing the following:
            loss: Total loss on the dataset
            acc: Symbol accuracy
            acc-seq: Sequence accuracy
        """
        return {
            'acc': self.total_correct_symbols / self.total_symbols,
            'acc-seq': self.total_correct_examples / self.total_examples
        }


def convert_iob_to_segments(sequence, mapping, begin_tag='B', inside_tag='I'):
    """
    Convert an IOB tag sequence to segments (O tags are excluded). Used
    by iob_metrics_fn
    Parameters:
        sequence: List of tag ids
        mapping: Maps ids to tags (B/I prefix + O etc.)
        begin_tag: Label for begin tag (default 'B')
        inside_tag: Label for inside tag (default 'I')
    Returns:
        A set of segment tuples of the form
        (segment value, segment start index, segment end index)
    """
    segments = []
    segment, segment_start, segment_end = None, None, None
    relevant_tags = set([begin_tag, inside_tag])

    mapping_sequence = map(lambda s: mapping[s].split('-', 1), sequence)
    # print(list(mapping_sequence))

    for i, tok in enumerate(mapping_sequence):
        if tok[0] in relevant_tags:
            if segment is None:
                segment, segment_start = tok[1], i
            elif tok[0] == begin_tag or tok[1] != segment:
                segments.append((segment, segment_start, segment_end))
                segment, segment_start = tok[1], i
            segment_end = i

    if segment is not None:
        segments.append((segment, segment_start, segment_end))

    return set(segments)
    # except:
    #     print(sequence)
    #     print(mapping)


class IOBMetrics(Metrics):
    """
    Compute precision, recall, F1
    for sequence tagging tasks following IOB schema. Meant
    to be passed to the Evaluator only
    """

    def __init__(self, tag_vocab):
        """
        Parameters:
            tag_vocab: Instance of torchtext.Vocab
        """
        self.tag_mappings = tag_vocab.itos

    def reset(self):
        self.correct_seg_count = 0
        self.pred_seg_count = 0
        self.total_seg_count = 0

    def evaluate_each_labels(self, e_labels, loss, predictions):
        for j in range(predictions.shape[0]):
            pred_indices = list(predictions[j, :].data)
            target_indices = list(e_labels[j, :].data)
            # print(pred_indices)
            pred_segs = convert_iob_to_segments(pred_indices, self.tag_mappings)
            target_segs = convert_iob_to_segments(target_indices, self.tag_mappings)

            correct_segs = pred_segs & target_segs

            self.correct_seg_count += len(correct_segs)
            self.pred_seg_count += len(pred_segs)
            self.total_seg_count += len(target_segs)

    def evaluate(self, batch, loss, predictions):
        for j in range(predictions.shape[0]):
            pred_indices = list(predictions[j, :].data)
            target_indices = list(batch.label[j, :].data)

            pred_segs = convert_iob_to_segments(pred_indices, self.tag_mappings)
            target_segs = convert_iob_to_segments(target_indices, self.tag_mappings)

            correct_segs = pred_segs & target_segs

            self.correct_seg_count += len(correct_segs)
            self.pred_seg_count += len(pred_segs)
            self.total_seg_count += len(target_segs)

    def results(self, total_loss):
        """
        Returns: A dict containing the following:
            precision: Tag precision
            recall: Tag recall
            F1: Tag F1
        """
        precision, recall, f1 = 0, 0, 0
        if self.correct_seg_count > 0:
            precision = self.correct_seg_count / self.pred_seg_count
            recall = self.correct_seg_count / self.total_seg_count
            f1 = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'F1': f1
        }
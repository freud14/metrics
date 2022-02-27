##############
Module metrics
##############

.. include:: ../links.rst

**********
Base class
**********

The base ``Metric`` class is an abstract base class that are used as the building block for all other Module
metrics.

.. autoclass:: torchmetrics.Metric
    :noindexentry:


*****************
Basic Aggregation
*****************

Torchmetrics comes with a number of metrics for aggregation of basic statistics: mean, max, min etc. of
either tensors or native python floats.

CatMetric
~~~~~~~~~

.. autoclass:: torchmetrics.CatMetric
    :noindexentry:

MaxMetric
~~~~~~~~~

.. autoclass:: torchmetrics.MaxMetric
    :noindexentry:

MeanMetric
~~~~~~~~~~

.. autoclass:: torchmetrics.MeanMetric
    :noindexentry:

MinMetric
~~~~~~~~~

.. autoclass:: torchmetrics.MinMetric
    :noindexentry:

SumMetric
~~~~~~~~~

.. autoclass:: torchmetrics.SumMetric
    :noindexentry:

*****
Audio
*****

For the purposes of audio metrics, inputs (predictions, targets) must have the same size.
If the input is 1D tensors the output will be a scalar. If the input is multi-dimensional with shape ``[...,time]``
the metric will be computed over the ``time`` dimension.

.. doctest::

    >>> import torch
    >>> from torchmetrics import SignalNoiseRatio
    >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
    >>> snr = SignalNoiseRatio()
    >>> snr(preds, target)
    tensor(16.1805)

PerceptualEvaluationSpeechQuality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality

PermutationInvariantTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.PermutationInvariantTraining
    :noindexentry:

SignalDistortionRatio
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.SignalDistortionRatio
    :noindexentry:

ScaleInvariantSignalDistortionRatio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ScaleInvariantSignalDistortionRatio
    :noindexentry:

ScaleInvariantSignalNoiseRatio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ScaleInvariantSignalNoiseRatio
    :noindexentry:

SignalNoiseRatio
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.SignalNoiseRatio
    :noindexentry:

ShortTimeObjectiveIntelligibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.audio.stoi.ShortTimeObjectiveIntelligibility
    :noindexentry:


**************
Classification
**************

Input types
~~~~~~~~~~~

For the purposes of classification metrics, inputs (predictions and targets) are split
into these categories (``N`` stands for the batch size and ``C`` for number of classes):

.. csv-table:: \*dtype ``binary`` means integers that are either 0 or 1
    :header: "Type", "preds shape", "preds dtype", "target shape", "target dtype"
    :widths: 20, 10, 10, 10, 10

    "Binary", "(N,)", "``float``", "(N,)", "``binary``\*"
    "Multi-class", "(N,)", "``int``", "(N,)", "``int``"
    "Multi-class with logits or probabilities", "(N, C)", "``float``", "(N,)", "``int``"
    "Multi-label", "(N, ...)", "``float``", "(N, ...)", "``binary``\*"
    "Multi-dimensional multi-class", "(N, ...)", "``int``", "(N, ...)", "``int``"
    "Multi-dimensional multi-class with logits or probabilities", "(N, C, ...)", "``float``", "(N, ...)", "``int``"

.. note::
    All dimensions of size 1 (except ``N``) are "squeezed out" at the beginning, so
    that, for example, a tensor of shape ``(N, 1)`` is treated as ``(N, )``.

When predictions or targets are integers, it is assumed that class labels start at 0, i.e.
the possible class labels are 0, 1, 2, 3, etc. Below are some examples of different input types

.. testcode::

    # Binary inputs
    binary_preds  = torch.tensor([0.6, 0.1, 0.9])
    binary_target = torch.tensor([1, 0, 2])

    # Multi-class inputs
    mc_preds  = torch.tensor([0, 2, 1])
    mc_target = torch.tensor([0, 1, 2])

    # Multi-class inputs with probabilities
    mc_preds_probs  = torch.tensor([[0.8, 0.2, 0], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    mc_target_probs = torch.tensor([0, 1, 2])

    # Multi-label inputs
    ml_preds  = torch.tensor([[0.2, 0.8, 0.9], [0.5, 0.6, 0.1], [0.3, 0.1, 0.1]])
    ml_target = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, 0]])


Using the multiclass parameter
------------------------------

In some cases, you might have inputs which appear to be (multi-dimensional) multi-class
but are actually binary/multi-label - for example, if both predictions and targets are
integer (binary) tensors. Or it could be the other way around, you want to treat
binary/multi-label inputs as 2-class (multi-dimensional) multi-class inputs.

For these cases, the metrics where this distinction would make a difference, expose the
``multiclass`` argument. Let's see how this is used on the example of
:class:`~torchmetrics.StatScores` metric.

First, let's consider the case with label predictions with 2 classes, which we want to
treat as binary.

.. testcode::

   from torchmetrics.functional import stat_scores

   # These inputs are supposed to be binary, but appear as multi-class
   preds  = torch.tensor([0, 1, 0])
   target = torch.tensor([1, 1, 0])

As you can see below, by default the inputs are treated
as multi-class. We can set ``multiclass=False`` to treat the inputs as binary -
which is the same as converting the predictions to float beforehand.

.. doctest::

    >>> stat_scores(preds, target, reduce='macro', num_classes=2)
    tensor([[1, 1, 1, 0, 1],
            [1, 0, 1, 1, 2]])
    >>> stat_scores(preds, target, reduce='macro', num_classes=1, multiclass=False)
    tensor([[1, 0, 1, 1, 2]])
    >>> stat_scores(preds.float(), target, reduce='macro', num_classes=1)
    tensor([[1, 0, 1, 1, 2]])

Next, consider the opposite example: inputs are binary (as predictions are probabilities),
but we would like to treat them as 2-class multi-class, to obtain the metric for both classes.

.. testcode::

   preds  = torch.tensor([0.2, 0.7, 0.3])
   target = torch.tensor([1, 1, 0])

In this case we can set ``multiclass=True``, to treat the inputs as multi-class.

.. doctest::

    >>> stat_scores(preds, target, reduce='macro', num_classes=1)
    tensor([[1, 0, 1, 1, 2]])
    >>> stat_scores(preds, target, reduce='macro', num_classes=2, multiclass=True)
    tensor([[1, 1, 1, 0, 1],
            [1, 0, 1, 1, 2]])

Accuracy
~~~~~~~~

.. autoclass:: torchmetrics.Accuracy
    :noindexentry:

AveragePrecision
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.AveragePrecision
    :noindexentry:

AUC
~~~

.. autoclass:: torchmetrics.AUC
    :noindexentry:

AUROC
~~~~~

.. autoclass:: torchmetrics.AUROC
    :noindexentry:

BinnedAveragePrecision
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.BinnedAveragePrecision
    :noindexentry:

BinnedPrecisionRecallCurve
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.BinnedPrecisionRecallCurve
    :noindexentry:

BinnedRecallAtFixedPrecision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.BinnedRecallAtFixedPrecision
    :noindexentry:

CalibrationError
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.CalibrationError
    :noindexentry:

CohenKappa
~~~~~~~~~~

.. autoclass:: torchmetrics.CohenKappa
    :noindexentry:

ConfusionMatrix
~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ConfusionMatrix
    :noindexentry:

F1Score
~~~~~~~

.. autoclass:: torchmetrics.F1Score
    :noindexentry:

FBetaScore
~~~~~~~~~~

.. autoclass:: torchmetrics.FBetaScore
    :noindexentry:

HammingDistance
~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.HammingDistance
    :noindexentry:

HingeLoss
~~~~~~~~~

.. autoclass:: torchmetrics.HingeLoss
    :noindexentry:

JaccardIndex
~~~~~~~~~~~~

.. autoclass:: torchmetrics.JaccardIndex
    :noindexentry:

KLDivergence
~~~~~~~~~~~~

.. autoclass:: torchmetrics.KLDivergence
    :noindexentry:

MatthewsCorrCoef
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MatthewsCorrCoef
    :noindexentry:

Precision
~~~~~~~~~

.. autoclass:: torchmetrics.Precision
    :noindexentry:

PrecisionRecallCurve
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.PrecisionRecallCurve
    :noindexentry:

Recall
~~~~~~

.. autoclass:: torchmetrics.Recall
    :noindexentry:


ROC
~~~

.. autoclass:: torchmetrics.ROC
    :noindexentry:


Specificity
~~~~~~~~~~~

.. autoclass:: torchmetrics.Specificity
    :noindexentry:


StatScores
~~~~~~~~~~

.. autoclass:: torchmetrics.StatScores
    :noindexentry:

*****
Image
*****

Image quality metrics can be used to access the quality of synthetic generated images from machine
learning algorithms such as `Generative Adverserial Networks (GANs) <https://en.wikipedia.org/wiki/Generative_adversarial_network>`_.

FrechetInceptionDistance
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.image.fid.FrechetInceptionDistance
    :noindexentry:

InceptionScore
~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.image.inception.InceptionScore
    :noindexentry:

KernelInceptionDistance
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.image.kid.KernelInceptionDistance
    :noindexentry:

LearnedPerceptualImagePatchSimilarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity
    :noindexentry:

MultiScaleStructuralSimilarityIndexMeasure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MultiScaleStructuralSimilarityIndexMeasure
    :noindexentry:

PeakSignalNoiseRatio
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.PeakSignalNoiseRatio
    :noindexentry:

StructuralSimilarityIndexMeasure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.StructuralSimilarityIndexMeasure
    :noindexentry:

UniversalImageQualityIndex
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.UniversalImageQualityIndex
    :noindexentry:

*********
Detection
*********

Object detection metrics can be used to evaluate the predicted detections with given groundtruth detections on images.

MeanAveragePrecision
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.detection.map.MeanAveragePrecision
    :noindexentry:

**********
Regression
**********

CosineSimilarity
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.CosineSimilarity
    :noindexentry:


ExplainedVariance
~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ExplainedVariance
    :noindexentry:


MeanAbsoluteError
~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MeanAbsoluteError
    :noindexentry:


MeanAbsolutePercentageError
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MeanAbsolutePercentageError
    :noindexentry:


MeanSquaredError
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MeanSquaredError
    :noindexentry:


MeanSquaredLogError
~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MeanSquaredLogError
    :noindexentry:


PearsonCorrCoef
~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.PearsonCorrCoef
    :noindexentry:


R2Score
~~~~~~~

.. autoclass:: torchmetrics.R2Score
    :noindexentry:


SpearmanCorrCoef
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.SpearmanCorrCoef
    :noindexentry:

SymmetricMeanAbsolutePercentageError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.SymmetricMeanAbsolutePercentageError
    :noindexentry:


TweedieDevianceScore
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.TweedieDevianceScore
    :noindexentry:


*********
Retrieval
*********

Input details
~~~~~~~~~~~~~

For the purposes of retrieval metrics, inputs (indexes, predictions and targets) must have the same size
(``N`` stands for the batch size) and the following types:

.. csv-table::
    :header: "indexes shape", "indexes dtype", "preds shape", "preds dtype", "target shape", "target dtype"
    :widths: 10, 10, 10, 10, 10, 10

    "(N,...)", "``long``", "(N,...)", "``float``", "(N,...)", "``long`` or ``bool``"

.. note::
    All dimensions are flattened at the beginning, so
    that, for example, a tensor of shape ``(N, M)`` is treated as ``(N * M, )``.

In Information Retrieval you have a query that is compared with a variable number of documents. For each pair ``(Q_i, D_j)``,
a score is computed that measures the relevance of document ``D`` w.r.t. query ``Q``. Documents are then sorted by score
and you hope that relevant documents are scored higher. ``target`` contains the labels for the documents (relevant or not).

Since a query may be compared with a variable number of documents, we use ``indexes`` to keep track of which scores belong to
the set of pairs ``(Q_i, D_j)`` having the same query ``Q_i``.

.. note::
    `Retrieval` metrics are only intended to be used globally. This means that the average of the metric over each batch can be quite different
    from the metric computed on the whole dataset. For this reason, we suggest to compute the metric only when all the examples
    has been provided to the metric. When using `Pytorch Lightning`, we suggest to use ``on_step=False``
    and ``on_epoch=True`` in ``self.log`` or to place the metric calculation in ``training_epoch_end``, ``validation_epoch_end`` or ``test_epoch_end``.

.. doctest::

    >>> from torchmetrics import RetrievalMAP
    >>> # functional version works on a single query at a time
    >>> from torchmetrics.functional import retrieval_average_precision

    >>> # the first query was compared with two documents, the second with three
    >>> indexes = torch.tensor([0, 0, 1, 1, 1])
    >>> preds = torch.tensor([0.8, -0.4, 1.0, 1.4, 0.0])
    >>> target = torch.tensor([0, 1, 0, 1, 1])

    >>> map = RetrievalMAP() # or some other retrieval metric
    >>> map(preds, target, indexes=indexes)
    tensor(0.6667)

    >>> # the previous instruction is roughly equivalent to
    >>> res = []
    >>> # iterate over indexes of first and second query
    >>> for indexes in ([0, 1], [2, 3, 4]):
    ...     res.append(retrieval_average_precision(preds[indexes], target[indexes]))
    >>> torch.stack(res).mean()
    tensor(0.6667)


RetrievalMAP
~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalMAP
    :noindexentry:


RetrievalMRR
~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalMRR
    :noindexentry:


RetrievalPrecision
~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalPrecision
    :noindexentry:


RetrievalRPrecision
~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalRPrecision
    :noindexentry:


RetrievalRecall
~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalRecall
    :noindexentry:


RetrievalFallOut
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalFallOut
    :noindexentry:


RetrievalNormalizedDCG
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalNormalizedDCG
    :noindexentry:


RetrievalHitRate
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.RetrievalHitRate
    :noindexentry:

****
Text
****

BERTScore
~~~~~~~~~~

.. autoclass:: torchmetrics.text.bert.BERTScore
    :noindexentry:

BLEUScore
~~~~~~~~~

.. autoclass:: torchmetrics.BLEUScore
    :noindexentry:

CharErrorRate
~~~~~~~~~~~~~

.. autoclass:: torchmetrics.CharErrorRate
    :noindexentry:

CHRFScore
~~~~~~~~~

.. autoclass:: torchmetrics.CHRFScore
    :noindexentry:

ExtendedEditDistance
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ExtendedEditDistance
    :noindexentry:

MatchErrorRate
~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MatchErrorRate
    :noindexentry:

ROUGEScore
~~~~~~~~~~

.. autoclass:: torchmetrics.text.rouge.ROUGEScore
    :noindexentry:

SacreBLEUScore
~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.SacreBLEUScore
    :noindexentry:

SQuAD
~~~~~

.. autoclass:: torchmetrics.SQuAD
    :noindexentry:

TranslationEditRate
~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.TranslationEditRate
    :noindexentry:

WordErrorRate
~~~~~~~~~~~~~

.. autoclass:: torchmetrics.WordErrorRate
    :noindexentry:

WordInfoLost
~~~~~~~~~~~~

.. autoclass:: torchmetrics.WordInfoLost
    :noindexentry:

WordInfoPreserved
~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.WordInfoPreserved
    :noindexentry:


********
Wrappers
********

Modular wrapper metrics are not metrics in themself, but instead take a metric and alter the internal logic
of the base metric.

BootStrapper
~~~~~~~~~~~~

.. autoclass:: torchmetrics.BootStrapper
    :noindexentry:

ClasswiseWrapper
~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.ClasswiseWrapper
    :noindexentry:

MetricTracker
~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MetricTracker
    :noindexentry:

MinMaxMetric
~~~~~~~~~~~~

.. autoclass:: torchmetrics.MinMaxMetric
    :noindexentry:

MultioutputWrapper
~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.MultioutputWrapper
    :noindexentry:

from math import log2


def calcule_itr(targets, accuracy, avg_time):
    """
    % Calculate information transfer rate (ITR) for brain-computer interface
    % (BCI) [2]
    % function [ itr ] = itr(n, p, t)
    %
    % Input:
    %   n   : # of targets
    %   p   : Target identification accuracy (0 <= p <= 1)
    %   t   : Averaged time for a selection [s]
    %
    % Output:
    %   itr : Information transfer rate [bits/min]
    %
    % Reference:
    %   [1] M. Cheng, X. Gao, S. Gao, and D. Xu,
    %       "Design and Implementation of a Brain-Computer Interface With High
    %        Transfer Rates",
    %       IEEE Trans. Biomed. Eng. 49, 1181-1186, 2002.
    %
    % Masaki Nakanishi, 22-Dec-2017
    % Swartz Center for Computational Neuroscience, Institute for Neural
    % Computation, University of California San Diego
    % E-mail: masaki@sccn.ucsd.edu
    """

    if accuracy < 0 or 1 < accuracy:
        print('stats:itr:BadInputValue', 'Accuracy need to be between 0 and 1.')
        itr = 0
    elif accuracy < 1 / targets:
        print('stats:itr:BadInputValue', 'The ITR might be incorrect because the accuracy < chance level.')
        itr = 0
    elif accuracy == 1:
        itr = log2(targets) * 60 / avg_time
    else:
        itr = (log2(targets) + accuracy * log2(accuracy) + (1 - accuracy) * log2(
            (1 - accuracy) / (targets - 1))) * 60 / avg_time
    return itr

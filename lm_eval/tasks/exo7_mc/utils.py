import numpy as np


def process_results(doc, results):

    if not results:
        # Return 0.0 for accuracy if no scores were generated for this document
        return {"acc": 0.0}

    ll, _ = zip(*results)
    ll = np.array(ll)

    # Convert log-likelihoods to probabilities.
    probs = np.exp(ll)

    # Normalize probabilities.
    probs_norm = probs / np.sum(probs)

    labels = np.array(doc["targets"]["labels"])
    # Compute the normalized probability mass for the correct answer.
    pm_true = np.sum(probs_norm[labels == 1])

    return {"acc": pm_true}
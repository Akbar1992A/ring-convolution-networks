"""
Experiments for Ring Convolution Networks
Reproducing results from the paper
"""

import numpy as np
import argparse
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time

from ring_convolution import RingConvolutionNetwork, FastRingLayer


def load_mnist_data(n_samples=5000):
    """Load MNIST dataset."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Subsample for faster experiments
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices] / 255.0  # Normalize
    y = y[indices].astype(int)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_random_projection(X, y, use_ring=False):
    """Test with random projection (no training)."""
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    n_trials = 10
    
    accuracies = []
    
    for _ in range(n_trials):
        if use_ring:
            # Ring weights
            model = RingConvolutionNetwork(n_features, 128, n_classes)
            predictions = []
            for x in X:
                output = model(x)
                predictions.append(np.argmax(output))
        else:
            # Standard weights
            W1 = np.random.randn(128, n_features) * 0.1
            W2 = np.random.randn(n_classes, 128) * 0.1
            predictions = []
            for x in X:
                h = np.maximum(0, np.dot(W1, x))  # ReLU
                output = np.dot(W2, h)
                predictions.append(np.argmax(output))
        
        accuracy = np.mean(np.array(predictions) == y) * 100
        accuracies.append(accuracy)
    
    return np.mean(accuracies), np.std(accuracies)


def compare_architectures(dataset='mnist'):
    """Compare RCN with standard neural networks."""
    print(f"\nComparing architectures on {dataset}")
    print("=" * 50)
    
    if dataset == 'mnist':
        X_train, X_test, y_train, y_test = load_mnist_data()
        test_data, test_labels = X_test[:1000], y_test[:1000]
    elif dataset == 'random':
        test_data = np.random.randn(1000, 784)
        test_labels = np.random.randint(0, 10, 1000)
    elif dataset == 'smooth':
        # Generate smooth signals
        test_data = []
        test_labels = []
        for i in range(1000):
            freq = np.random.randint(1, 10)
            signal = np.sin(np.linspace(0, freq*2*np.pi, 784))
            test_data.append(signal)
            test_labels.append(freq)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
    
    # Test standard network
    print("\nStandard Neural Network:")
    start_time = time.time()
    std_acc, std_std = test_random_projection(test_data, test_labels, use_ring=False)
    std_time = time.time() - start_time
    print(f"  Accuracy: {std_acc:.2f}% ± {std_std:.2f}%")
    print(f"  Time: {std_time:.3f}s")
    
    # Test RCN
    print("\nRing Convolution Network:")
    start_time = time.time()
    rcn_acc, rcn_std = test_random_projection(test_data, test_labels, use_ring=True)
    rcn_time = time.time() - start_time
    print(f"  Accuracy: {rcn_acc:.2f}% ± {rcn_std:.2f}%")
    print(f"  Time: {rcn_time:.3f}s")
    
    # Calculate improvement
    improvement = (rcn_acc - std_acc) / std_acc * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    
    return std_acc, rcn_acc, improvement


def main():
    parser = argparse.ArgumentParser(description='RCN Experiments')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'random', 'smooth'],
                       help='Dataset to test on')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare with baseline on all datasets')
    
    args = parser.parse_args()
    
    print("Ring Convolution Networks - Experiments")
    print("=" * 50)
    
    if args.compare_baseline:
        # Run comparison on all datasets
        results = {}
        for dataset in ['random', 'smooth', 'mnist']:
            std_acc, rcn_acc, improvement = compare_architectures(dataset)
            results[dataset] = {
                'standard': std_acc,
                'rcn': rcn_acc,
                'improvement': improvement
            }
        
        # Print summary table
        print("\n" + "=" * 50)
        print("SUMMARY OF RESULTS")
        print("=" * 50)
        print(f"{'Dataset':<15} {'Standard NN':<12} {'RCN':<12} {'Improvement':<12}")
        print("-" * 50)
        for dataset, res in results.items():
            print(f"{dataset:<15} {res['standard']:<12.1f} {res['rcn']:<12.1f} {res['improvement']:+.1f}%")
    else:
        # Run on single dataset
        compare_architectures(args.dataset)


if __name__ == "__main__":
    main()

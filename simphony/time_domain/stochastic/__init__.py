"""Stochastic (Gaussian process) simulation backend for time-domain photonic
circuits."""

from simphony.time_domain.stochastic.gaussian_process import (
    autocorrelation_to_covariance,
    covariance_blocks_to_matrix,
    covariance_matrix_to_blocks,
    covariance_to_autocorrelation,
    gaussian_process_response,
    propagate_autocorrelation,
    propagate_mean,
    white_noise_covariance,
)

__all__ = [
    "covariance_to_autocorrelation",
    "autocorrelation_to_covariance",
    "propagate_mean",
    "propagate_autocorrelation",
    "gaussian_process_response",
    "white_noise_covariance",
    "covariance_blocks_to_matrix",
    "covariance_matrix_to_blocks",
]
